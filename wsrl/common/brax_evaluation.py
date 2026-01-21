"""
Evaluation utilities for Brax environments.

Provides both Gym-style evaluation (for wrapped Brax envs) and
native Brax evaluation using Brax's built-in approach with jax.lax.scan.

Based on brax.training.acting but modified to return full trajectories.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.envs.base import PipelineEnv, State as BraxState
from brax.training import types
from brax.training.types import PRNGKey, Policy, PolicyParams


# ============================================================================
# Transition data structure (from brax.training.types)
# ============================================================================

class Transition(NamedTuple):
    """Container for a single transition with trajectory data."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray  # 1 - done
    next_observation: jnp.ndarray


class TrajectoryData(NamedTuple):
    """Container for trajectory data collected during evaluation."""
    observations: jnp.ndarray      # (num_envs, max_steps, obs_dim)
    actions: jnp.ndarray           # (num_envs, max_steps, action_dim)
    rewards: jnp.ndarray           # (num_envs, max_steps)
    discounts: jnp.ndarray         # (num_envs, max_steps) - 1-done
    next_observations: jnp.ndarray # (num_envs, max_steps, obs_dim)
    episode_lengths: jnp.ndarray   # (num_envs,)
    episode_returns: jnp.ndarray   # (num_envs,)


# ============================================================================
# Actor step and unroll functions (adapted from brax.training.acting)
# ============================================================================

def actor_step(
    env: envs.Env,
    env_state: BraxState,
    policy: Callable[[jnp.ndarray, PRNGKey], Tuple[jnp.ndarray, Any]],
    key: PRNGKey,
) -> Tuple[BraxState, Transition]:
    """
    Collect data for a single step.
    
    Adapted from brax.training.acting.actor_step.
    
    Args:
        env: Brax environment (batched)
        env_state: Current environment state
        policy: Policy function (obs, key) -> (actions, extras)
        key: Random key
        
    Returns:
        next_state: Next environment state
        transition: Transition data for this step
    """
    actions, _ = policy(env_state.obs, key)
    next_state = env.step(env_state, actions)
    return next_state, Transition(
        observation=env_state.obs,
        action=actions,
        reward=next_state.reward,
        discount=1 - next_state.done,
        next_observation=next_state.obs,
    )


def generate_unroll(
    env: envs.Env,
    env_state: BraxState,
    policy: Callable[[jnp.ndarray, PRNGKey], Tuple[jnp.ndarray, Any]],
    key: PRNGKey,
    unroll_length: int,
) -> Tuple[BraxState, Transition]:
    """
    Collect trajectories of given unroll_length using jax.lax.scan.
    
    Adapted from brax.training.acting.generate_unroll.
    
    Args:
        env: Brax environment (batched)
        env_state: Initial environment state
        policy: Policy function (obs, key) -> (actions, extras)
        key: Random key
        unroll_length: Number of steps to unroll
        
    Returns:
        final_state: Final environment state
        transitions: Stacked transitions (unroll_length, num_envs, ...)
    """
    def scan_step(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        next_state, transition = actor_step(env, state, policy, current_key)
        return (next_state, next_key), transition

    (final_state, _), transitions = jax.lax.scan(
        scan_step, (env_state, key), (), length=unroll_length
    )
    return final_state, transitions


# ============================================================================
# Evaluator with trajectory collection
# ============================================================================

class EvaluatorWithTrajectories:
    """
    Evaluator that collects full trajectories using Brax's native approach.
    
    Based on brax.training.acting.Evaluator but modified to return trajectories.
    """
    
    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[jnp.ndarray, PRNGKey], Tuple[jnp.ndarray, Any]],
        num_eval_envs: int,
        episode_length: int,
        key: PRNGKey,
    ):
        """
        Initialize evaluator.
        
        Args:
            eval_env: Batched Brax environment for evaluation
            eval_policy_fn: Policy function (obs, key) -> (actions, extras)
            num_eval_envs: Number of parallel evaluation environments
            episode_length: Maximum episode length
            key: Random key
        """
        self._key = key
        self._num_eval_envs = num_eval_envs
        self._episode_length = episode_length
        
        # Wrap with EvalWrapper to track episode metrics
        self._eval_env = envs.training.EvalWrapper(eval_env)
        self._policy_fn = eval_policy_fn
        
        # JIT compile the evaluation unroll
        def _generate_eval_unroll(key: PRNGKey) -> Tuple[BraxState, Transition]:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = self._eval_env.reset(reset_keys)
            return generate_unroll(
                self._eval_env,
                eval_first_state,
                eval_policy_fn,
                key,
                unroll_length=episode_length,
            )
        
        self._generate_eval_unroll = jax.jit(_generate_eval_unroll)
    
    def evaluate(self) -> Tuple[Dict[str, float], TrajectoryData]:
        """
        Run evaluation and collect trajectories.
        
        Returns:
            stats: Dictionary of evaluation statistics
            trajectories: TrajectoryData with collected trajectories
        """
        self._key, unroll_key = jax.random.split(self._key)
        
        # Generate unroll
        final_state, transitions = self._generate_eval_unroll(unroll_key)
        
        # Get episode metrics from EvalWrapper
        eval_metrics = final_state.info['eval_metrics']
        
        # Transpose from (steps, num_envs, ...) to (num_envs, steps, ...)
        trajectories = TrajectoryData(
            observations=jnp.transpose(transitions.observation, (1, 0, 2)),
            actions=jnp.transpose(transitions.action, (1, 0, 2)),
            rewards=jnp.transpose(transitions.reward, (1, 0)),
            discounts=jnp.transpose(transitions.discount, (1, 0)),
            next_observations=jnp.transpose(transitions.next_observation, (1, 0, 2)),
            episode_lengths=eval_metrics.episode_steps,
            episode_returns=eval_metrics.episode_metrics['reward'],
        )
        
        # Compute statistics
        stats = {
            'average_return': float(jnp.mean(trajectories.episode_returns)),
            'std_return': float(jnp.std(trajectories.episode_returns)),
            'min_return': float(jnp.min(trajectories.episode_returns)),
            'max_return': float(jnp.max(trajectories.episode_returns)),
            'average_length': float(jnp.mean(trajectories.episode_lengths)),
        }
        
        return stats, trajectories


# ============================================================================
# Helper functions
# ============================================================================

def _get_brax_env(env) -> PipelineEnv:
    """Extract raw Brax PipelineEnv from a potentially wrapped environment."""
    if isinstance(env, PipelineEnv):
        return env
    
    if hasattr(env, 'brax_env'):
        return env.brax_env
    
    if hasattr(env, 'env'):
        return _get_brax_env(env.env)
    
    if hasattr(env, 'unwrapped'):
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'brax_env'):
            return unwrapped.brax_env
    
    raise ValueError(
        f"Could not extract Brax PipelineEnv from {type(env)}. "
        "Expected a BraxToGymWrapper or raw PipelineEnv."
    )


def make_policy_fn_for_brax(
    wsrl_policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray, PRNGKey], Tuple[jnp.ndarray, Dict]]:
    """
    Wrap a wsrl-style policy function to match Brax's expected signature.
    
    wsrl policy: (obs) -> actions
    Brax policy: (obs, key) -> (actions, extras)
    
    Args:
        wsrl_policy_fn: Policy function that takes observations and returns actions
        
    Returns:
        Brax-compatible policy function
    """
    def brax_policy(obs: jnp.ndarray, key: PRNGKey) -> Tuple[jnp.ndarray, Dict]:
        actions = wsrl_policy_fn(obs)
        return actions, {}
    
    return brax_policy


# ============================================================================
# Main evaluation function
# ============================================================================

def evaluate_with_trajectories_jit_multi_episode(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env,  # Can be Gym-wrapped or raw PipelineEnv
    num_episodes: int,
    episode_length: int,
    rng: jax.Array,
    clip_action: float = jnp.inf,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate policy for exactly num_episodes using Brax's native jit and lax.scan.
    
    This uses Brax's native evaluation approach with EvalWrapper for accurate
    episode metrics, adapted to return full trajectories.
    
    Compatible with the standard evaluate_with_trajectories signature:
    (policy_fn, env, num_episodes) -> (stats, trajectories)
    
    Args:
        policy_fn: Policy function that takes batched observations
                   and returns batched actions.
        env: Brax environment (can be Gym-wrapped or raw PipelineEnv)
        num_episodes: Number of episodes to evaluate
        episode_length: Maximum episode length
        rng: JAX random key
        clip_action: Action clipping value (default: no clipping)
        
    Returns:
        stats: Dictionary of evaluation statistics
        trajectories: List of trajectory dicts (compatible with standard format)
    """
    # Extract raw Brax env from wrapper if needed
    brax_env = _get_brax_env(env)
    
    # Wrap environment for training (handles batching)
    wrapped_env = envs.training.wrap(
        brax_env,
        episode_length=episode_length,
        action_repeat=1,
    )
    
    # Create Brax-compatible policy with action clipping
    def clipped_policy(obs: jnp.ndarray, key: PRNGKey) -> Tuple[jnp.ndarray, Dict]:
        actions = policy_fn(obs)
        actions = jnp.clip(actions, -clip_action, clip_action)
        return actions, {}
    
    # Create evaluator
    evaluator = EvaluatorWithTrajectories(
        eval_env=wrapped_env,
        eval_policy_fn=clipped_policy,
        num_eval_envs=num_episodes,
        episode_length=episode_length,
        key=rng,
    )
    
    # Run evaluation
    stats, traj_data = evaluator.evaluate()
    
    # Convert TrajectoryData to list of dicts format
    trajectories = []
    for i in range(num_episodes):
        ep_len = int(traj_data.episode_lengths[i])
        traj = {
            'observations': np.array(traj_data.observations[i, :ep_len]),
            'actions': np.array(traj_data.actions[i, :ep_len]),
            'rewards': np.array(traj_data.rewards[i, :ep_len]),
            'dones': 1.0 - np.array(traj_data.discounts[i, :ep_len]),
            'masks': np.array(traj_data.discounts[i, :ep_len]),
            'next_observations': np.array(traj_data.next_observations[i, :ep_len]),
        }
        trajectories.append(traj)
    
    return stats, trajectories


def evaluate_brax_wrapped(
    policy_fn: Callable,
    env,
    num_episodes: int,
    normalizer: Optional[Any] = None,
    clip_action: float = np.inf,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate policy on a Brax environment wrapped with BraxToGymWrapper.
    
    This uses the Gym-compatible interface, so the env should be wrapped
    with BraxToGymWrapper (or created via make_brax_env).
    
    Args:
        policy_fn: Function that takes observation and returns action
        env: Gym-wrapped Brax environment
        num_episodes: Number of evaluation episodes
        normalizer: Optional normalizer for observations
        clip_action: Action clipping value
        
    Returns:
        stats: Dictionary of evaluation statistics
        trajectories: List of trajectory dictionaries
    """
    trajectories = []
    stats = defaultdict(list)
    
    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        done = False
        
        while not done:
            # Normalize observation if normalizer provided
            if normalizer is not None:
                normalized_obs = normalizer(observation)
            else:
                normalized_obs = observation
            
            action = policy_fn(normalized_obs)
            action = np.clip(action, -clip_action, clip_action)
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=reward,
                dones=done,
                infos=info,
                masks=1 - terminated,
            )
            
            for k, v in transition.items():
                trajectory[k].append(v)
            
            observation = next_observation
        
        trajectories.append(dict(trajectory))
    
    # Compute stats
    returns = [np.sum(t['rewards']) for t in trajectories]
    lengths = [len(t['rewards']) for t in trajectories]
    
    stats = {
        'average_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'average_length': np.mean(lengths),
    }
    
    return stats, trajectories


def evaluate_brax_native(
    brax_env,
    policy_params,
    normalizer_params,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    seed: int = 0,
    episode_length: int = 1000,
    num_eval_envs: int = 128,
) -> Dict[str, float]:
    """
    Evaluate policy on Brax environment using Brax's native Evaluator.
    
    This provides the same evaluation as Brax's training scripts, ensuring
    consistency when comparing pre-trained and finetuned policies.
    
    Args:
        brax_env: Base Brax environment (not wrapped)
        policy_params: Policy network parameters
        normalizer_params: Normalizer parameters (RunningStatisticsState)
        hidden_layer_sizes: Hidden layer sizes for policy network
        seed: Random seed
        episode_length: Maximum episode length
        num_eval_envs: Number of parallel evaluation environments
        
    Returns:
        Dictionary of evaluation metrics
    """
    from brax import envs
    from brax.training.acme import running_statistics
    from brax.training.agents.sac import networks as sac_networks
    from brax.training.agents.sac.networks import make_sac_networks
    from brax.training.acting import Evaluator
    import flax.linen as nn
    
    wrap_for_training = envs.training.wrap
    
    rng = jax.random.PRNGKey(seed)
    obs_size = brax_env.observation_size
    action_size = brax_env.action_size
    
    # Create the same network architecture as used during training
    normalize_fn = running_statistics.normalize
    
    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=nn.relu,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)
    
    # Wrap environment for evaluation
    eval_env = wrap_for_training(
        brax_env,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=None,
    )
    
    # Create evaluator
    evaluator = Evaluator(
        eval_env,
        partial(make_policy, deterministic=True),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=1,
        key=rng,
    )
    
    # Run evaluation
    eval_metrics = evaluator.run_evaluation(
        (normalizer_params, policy_params),
        training_metrics={},
    )

    print(f"Eval metrics: {eval_metrics}")
    
    return {
        'episode_reward': float(eval_metrics['eval/episode_reward']),
        'episode_length': float(eval_metrics['eval/avg_episode_length']),
    }


def evaluate_with_brax_and_gym(
    agent,
    brax_env,
    gym_env,
    normalizer: Optional[Any] = None,
    normalizer_params=None,
    policy_params=None,
    num_episodes: int = 10,
    brax_num_eval_envs: int = 128,
    episode_length: int = 1000,
    seed: int = 0,
    argmax: bool = True,
) -> Dict[str, float]:
    """
    Evaluate agent on both Brax (native) and Gym-wrapped environments.
    
    This is useful for comparing results and debugging discrepancies.
    
    Args:
        agent: wsrl agent
        brax_env: Base Brax environment
        gym_env: Gym-wrapped environment (for wsrl-style evaluation)
        normalizer: Optional BraxNormalizer for Gym evaluation
        normalizer_params: Brax normalizer params for native evaluation
        policy_params: Brax policy params for native evaluation
        num_episodes: Number of Gym evaluation episodes
        brax_num_eval_envs: Number of parallel Brax evaluation envs
        episode_length: Maximum episode length
        seed: Random seed
        argmax: Whether to use deterministic actions
        
    Returns:
        Dictionary with both Brax and Gym evaluation results
    """
    from wsrl.utils.brax_utils import make_normalized_policy_fn
    
    results = {}
    
    # Gym-style evaluation
    policy_fn = make_normalized_policy_fn(agent, normalizer, argmax=argmax)
    gym_stats, _ = evaluate_brax_wrapped(
        policy_fn,
        gym_env,
        num_episodes=num_episodes,
        normalizer=None,  # Already applied in policy_fn
    )
    results['gym_return'] = gym_stats['average_return']
    results['gym_std'] = gym_stats['std_return']
    
    # Brax native evaluation (if params provided)
    if normalizer_params is not None and policy_params is not None:
        brax_metrics = evaluate_brax_native(
            brax_env,
            policy_params,
            normalizer_params,
            seed=seed,
            episode_length=episode_length,
            num_eval_envs=brax_num_eval_envs,
        )
        results['brax_return'] = brax_metrics['episode_reward']
    
    return results
