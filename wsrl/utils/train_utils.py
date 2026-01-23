from collections.abc import Mapping
from typing import Optional, Tuple

import numpy as np
import os
from etils import epath
from brax import envs as brax_envs
from absl import logging


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        if isinstance(batches[0][key], Mapping):
            # to concatenate batch["observations"]["image"], etc.
            concatenated[key] = concatenate_batches([batch[key] for batch in batches])
        else:
            concatenated[key] = np.concatenate(
                [batch[key] for batch in batches], axis=0
            ).astype(np.float32)
    return concatenated


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        if isinstance(batch[key], Mapping):
            # to index into batch["observations"]["image"], etc.
            indexed[key] = index_batch(batch[key], indices)
        else:
            indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    indices = np.random.randint(batch["rewards"].shape[0], size=size)
    return index_batch(batch, indices)


# ============================================================================
# Simple Replay Buffer (no Gym dependency)
# ============================================================================

class SimpleReplayBuffer:
    """Simple replay buffer that doesn't depend on Gym spaces."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        seed: int = 0,
    ):
        self._capacity = capacity
        self._size = 0
        self._insert_index = 0
        self._rng = np.random.default_rng(seed)
        
        # Pre-allocate arrays
        self._observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._masks = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
    
    def __len__(self):
        return self._size
    
    def insert(self, transition: dict):
        """Insert a single transition."""
        idx = self._insert_index
        self._observations[idx] = transition["observations"]
        self._next_observations[idx] = transition["next_observations"]
        self._actions[idx] = transition["actions"]
        self._rewards[idx] = transition["rewards"]
        self._masks[idx] = transition["masks"]
        self._dones[idx] = transition["dones"]
        
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def insert_batch(self, transitions: dict):
        """Insert a batch of transitions."""
        batch_size = len(transitions["observations"])
        for i in range(batch_size):
            self.insert({
                "observations": transitions["observations"][i],
                "next_observations": transitions["next_observations"][i],
                "actions": transitions["actions"][i],
                "rewards": transitions["rewards"][i],
                "masks": transitions["masks"][i],
                "dones": transitions["dones"][i],
            })
    
    def sample(self, batch_size: int) -> dict:
        """Sample a random batch."""
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observations": self._observations[indices],
            "next_observations": self._next_observations[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "masks": self._masks[indices],
            "dones": self._dones[indices],
        }


# ============================================================================
# Brax Environment Helper
# ============================================================================

def make_brax_training_env(
    env_name: str,
    backend: str = "generalized",
    episode_length: int = 1000,
    mjcf_path: Optional[str] = None,
    wandb_logger=None,
):
    """Create a Brax environment wrapped for training.
    
    Args:
        env_name: Name of the Brax environment
        backend: Brax backend to use
        episode_length: Maximum episode length
        mjcf_path: Optional path to custom MJCF file
        wandb_logger: Optional wandb logger for saving MJCF files
    
    Returns:
        Tuple of (base_env, wrapped_env)
    """
    if "s2r" in env_name:
        if mjcf_path is None:
            mjcf_path = os.path.join(epath.resource_path('data_gen'), f"assets/mjcf/{env_name}.xml")
        else:
            mjcf_path = mjcf_path
        if wandb_logger is not None:
            wandb_logger.save_file(mjcf_path, f"mjcf/{env_name}.xml")
        base_env = brax_envs.get_environment(
            env_name=env_name,
            backend="generalized",
            mjcf_path=mjcf_path
        )
    else:
        base_env = brax_envs.get_environment(env_name=env_name, backend=backend)
    wrapped_env = brax_envs.training.wrap(
        base_env,
        episode_length=episode_length,
        action_repeat=1,
    )
    return base_env, wrapped_env


# ============================================================================
# Hypernet Policy Improvement
# ============================================================================

def improve_policy_with_hypernet(
    current_agent,
    hypernet_loader,
    brax_normalizer_params,
    current_rewards: float,
    target_rewards: float,
    seed: int,
    loss_cond=None
):
    """Use hypernet to generate improved policy parameters.
    
    Args:
        current_agent: Current agent with policy to improve
        hypernet_loader: HypernetLoader instance
        brax_normalizer_params: Brax normalizer parameters
        current_rewards: Current reward value
        target_rewards: Target reward value
        seed: Random seed
        loss_cond: Optional loss condition
    
    Returns:
        Agent with improved policy parameters
    """
    from wsrl.utils.brax_utils import (
        convert_wsrl_actor_to_brax_policy,
        load_brax_policy_to_wsrl_agent,
    )
    
    if hypernet_loader is None:
        return current_agent
    
    logging.info("=" * 80)
    logging.info("IMPROVING POLICY WITH HYPERNET")
    logging.info("=" * 80)
    logging.info(f"Current rewards: {current_rewards:.2f}")
    logging.info(f"Target rewards: {target_rewards:.2f}")
    
    # Extract current policy parameters from agent
    wsrl_actor_params = current_agent.state.params['modules_actor']
    current_brax_policy_params = convert_wsrl_actor_to_brax_policy(wsrl_actor_params)
    
    # Generate improved parameters with hypernet
    improved_brax_policy_params = hypernet_loader.generate_parameters(
        policy_normalizer=brax_normalizer_params,
        policy_params=current_brax_policy_params,
        current_rewards=current_rewards,
        target_rewards=target_rewards,
        loss_cond=loss_cond,
        seed=seed,
    )
    
    # Load improved parameters into agent
    improved_agent = load_brax_policy_to_wsrl_agent(
        current_agent,
        improved_brax_policy_params,
        brax_normalizer_params=brax_normalizer_params,
        absorb_normalizer=False,
    )
    
    logging.info("Successfully generated and loaded improved parameters!")
    logging.info("=" * 80)
    
    return improved_agent


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_and_log_brax(
    wsrl_agent,
    normalizer_params,
    env,
    step_number: int,
    brax_hidden_dims: list,
    seed: int,
    num_eval_envs: int = 128,
    episode_length: int = 1000,
    wandb_logger=None,
):
    """Evaluate agent using native Brax and log results.
    
    Args:
        wsrl_agent: Wsrl agent to evaluate
        normalizer_params: Brax normalizer parameters
        env: Brax environment
        step_number: Current training step
        brax_hidden_dims: Hidden layer dimensions for policy
        seed: Random seed
        num_eval_envs: Number of parallel evaluation environments
        episode_length: Episode length for evaluation
        wandb_logger: Optional wandb logger
    
    Returns:
        Dictionary with evaluation metrics
    """
    from wsrl.utils.brax_utils import convert_wsrl_actor_to_brax_policy
    from wsrl.common.brax_evaluation import evaluate_brax_native
    
    # Extract policy params from agent and convert to Brax format
    wsrl_actor_params = wsrl_agent.state.params['modules_actor']
    current_brax_policy_params = convert_wsrl_actor_to_brax_policy(wsrl_actor_params)
    
    brax_metrics = evaluate_brax_native(
        env,
        current_brax_policy_params,
        normalizer_params,
        hidden_layer_sizes=tuple(brax_hidden_dims),
        seed=seed,
        episode_length=episode_length,
        num_eval_envs=num_eval_envs,
    )
    
    eval_info = {
        "brax_native_return": brax_metrics["episode_reward"],
        "brax_native_length": brax_metrics["episode_length"],
    }
    if wandb_logger is not None:
        wandb_logger.log({"evaluation": eval_info}, step=step_number)
    return eval_info
