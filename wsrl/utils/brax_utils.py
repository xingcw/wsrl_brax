"""
Utilities for loading and converting Brax SAC checkpoints to wsrl agents.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import numpy as np
import logging
from wsrl.agents.sac import SACAgent

from wsrl.envs.brax_dataset import (
    convert_brax_normalizer_to_dict,
    load_brax_sac_checkpoint,
)

from brax.training import types
from brax.training.agents.sac import networks as sac_networks
from brax.training import networks
from brax.training.acme import running_statistics
from brax import envs
from flax import linen as nn
from typing import Sequence
from brax.training import distribution


def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = nn.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
) -> sac_networks.SACNetworks:
  """Make SAC networks."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
  )
  q_network = networks.make_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=(256, 256),
      activation=activation,
      layer_norm=q_network_layer_norm,
  )
  return sac_networks.SACNetworks(
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution,
  )


def load_brax_checkpoint_to_wsrl_agent(
    agent,
    ckpt_path: str,
    checkpoint_idx: int = -1,
    absorb_normalizer: bool = False,
) -> Tuple[Any, Optional[Dict[str, np.ndarray]]]:
    """
    Load a Brax SAC checkpoint into a wsrl SACAgent.
    
    This function handles the conversion between Brax parameter naming conventions
    and wsrl conventions, mapping the policy network weights appropriately.
    
    Args:
        agent: wsrl SACAgent instance (already initialized)
        ckpt_path: Path to Brax checkpoint
        checkpoint_idx: Which checkpoint to load if multiple are stacked
        absorb_normalizer: Whether to absorb normalizer into network weights
        
    Returns:
        agent: Updated agent with loaded weights
        normalizer: Dict with normalizer params if not absorbed, else None
    """
    # Load Brax checkpoint
    normalizer_params, policy_params, _ = load_brax_sac_checkpoint(
        ckpt_path, checkpoint_idx
    )
    
    # Extract the actual params dict
    if 'params' in policy_params:
        brax_policy = policy_params['params']
    else:
        brax_policy = policy_params
    
    # Get current agent params
    current_params = agent.state.params    
    new_actor_params = _convert_brax_to_wsrl_mlp(
        brax_policy, 
        current_params['actor'],
        absorb_normalizer=absorb_normalizer,
        normalizer_params=normalizer_params if absorb_normalizer else None,
    )
    
    # Update agent params
    new_params = current_params.copy({'actor': new_actor_params})
    new_state = agent.state.replace(params=new_params)
    new_agent = agent.replace(state=new_state)
    
    # Return normalizer if not absorbed
    normalizer = None
    if not absorb_normalizer:
        normalizer = convert_brax_normalizer_to_dict(normalizer_params)
    
    return new_agent, normalizer


def _convert_brax_to_wsrl_mlp(
    brax_params: Dict[str, Any],
    wsrl_actor_params: Dict[str, Any],
    absorb_normalizer: bool = False,
    normalizer_params: Any = None,
) -> Dict[str, Any]:
    """
    Convert Brax MLP parameters to wsrl actor parameter structure.
    
    Both use (input_dim, output_dim) convention for kernels, so no transpose needed.
    
    For Brax-compatible networks, the structure is:
    - network/hidden_0, network/hidden_1, hidden_2 (mean), log_std
    
    For standard wsrl networks:
    - network/layers_0, network/layers_1, mean_layer
    
    Args:
        brax_params: Brax policy parameters
        wsrl_actor_params: Current wsrl actor parameters (for structure reference)
        absorb_normalizer: Whether to absorb normalizer into first layer
        normalizer_params: Normalizer parameters (required if absorbing)
        
    Returns:
        New actor parameters with Brax weights loaded
    """
    import copy
    
    new_params = copy.deepcopy(wsrl_actor_params)
    network_params = new_params['network']
    
    layer_mapping = [
        ('hidden_0', 'hidden_0'),
        ('hidden_1', 'hidden_1'),
    ]
    
    for brax_name, wsrl_name in layer_mapping:
        kernel = brax_params[brax_name]['kernel']
        bias = brax_params[brax_name]['bias']
        
        if absorb_normalizer and brax_name == 'hidden_0' and normalizer_params is not None:
            kernel, bias = _absorb_normalizer(kernel, bias, normalizer_params)

        network_params[wsrl_name]['kernel'] = jnp.array(kernel)
        network_params[wsrl_name]['bias'] = jnp.array(bias)
        logging.info(f"[Brax to wsrl] Loaded {wsrl_name} to {brax_name}")
    
    if 'hidden_2' in brax_params:
        kernel = brax_params['hidden_2']['kernel']
        bias = brax_params['hidden_2']['bias']
        new_params['hidden_2']['kernel'] = jnp.array(kernel)
        new_params['hidden_2']['bias'] = jnp.array(bias)
        logging.info(f"[Brax to wsrl] Loaded hidden_2")
    
    return new_params


def _absorb_normalizer(
    kernel: np.ndarray,
    bias: np.ndarray,
    normalizer_params,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Absorb normalizer into layer weights.
    
    Given normalized_input = (input - mean) / std:
    output = W @ normalized_input + b
           = W @ (input - mean) / std + b
           = (W / std) @ input + (b - W @ mean / std)
    
    So: W_new = W / std, b_new = b - W @ mean / std
    """
    mean = np.array(normalizer_params.mean).flatten()
    std = np.array(normalizer_params.std).flatten()
    
    # kernel shape: (input_dim, output_dim)
    # Divide each row by corresponding std
    kernel_new = kernel / std[:, np.newaxis]
    
    # bias shape: (output_dim,)
    # b_new = b - (mean / std) @ W = b - (mean / std).T @ W
    bias_new = bias - (mean / std) @ kernel
    
    return kernel_new, bias_new


class BraxNormalizer:
    """
    Wrapper class for applying Brax-style observation normalization.
    
    This can be used to normalize observations before passing to the policy
    when the normalizer was not absorbed into the network weights.
    
    JAX-compatible: works with both numpy arrays and JAX arrays, and can be
    used inside JIT-compiled functions.
    """
    
    def __init__(self, mean: np.ndarray, std: np.ndarray, clip: float = 10.0):
        """
        Initialize normalizer.
        
        Args:
            mean: Mean values for each observation dimension
            std: Standard deviation values for each dimension
            clip: Clip normalized values to [-clip, clip]
        """
        # Store as JAX arrays for JIT compatibility
        self.mean = jnp.array(mean, dtype=jnp.float32).flatten()
        self.std = jnp.array(std, dtype=jnp.float32).flatten()
        self.clip = clip
        
        # Avoid division by zero
        self.std = jnp.maximum(self.std, 1e-8)
    
    def __call__(self, obs):
        """Normalize observation. Works with both numpy and JAX arrays."""
        # Ensure JAX array (avoid conversion if already JAX)
        if not isinstance(obs, jnp.ndarray):
            obs = jnp.array(obs)
        normalized = (obs - self.mean) / self.std
        return jnp.clip(normalized, -self.clip, self.clip)
    
    def normalize(self, obs):
        """Normalize observation (alias for __call__)."""
        return self(obs)
    
    @classmethod
    def from_dict(cls, normalizer_dict: Dict[str, np.ndarray], clip: float = 10.0):
        """Create normalizer from dict with 'mean' and 'std' keys."""
        return cls(normalizer_dict['mean'], normalizer_dict['std'], clip=clip)
    
    @classmethod
    def from_brax_params(cls, normalizer_params, clip: float = 10.0):
        """Create normalizer from Brax RunningStatisticsState."""
        return cls(
            np.array(normalizer_params.mean),
            np.array(normalizer_params.std),
            clip=clip,
        )


def make_normalized_policy_fn(agent, normalizer: Optional[BraxNormalizer] = None, argmax: bool = True):
    """
    Create a policy function that optionally applies normalization.
    
    Args:
        agent: wsrl agent
        normalizer: Optional BraxNormalizer to apply before policy
        argmax: Whether to use deterministic (mode) actions
        
    Returns:
        policy_fn: Function that takes observation and returns action
    """
    def policy_fn(observation, seed=None):
        if normalizer is not None:
            observation = normalizer(observation)
        return agent.sample_actions(observation, seed=seed, argmax=argmax)
    
    return policy_fn


def load_brax_policy_to_wsrl_agent(
    agent,
    brax_policy_params: Dict[str, Any],
    brax_normalizer_params: Any = None,
    absorb_normalizer: bool = False,
) -> Any:
    """
    Load Brax policy parameters into a wsrl SACAgent's actor.
    
    This is a simpler version that takes already-loaded Brax params.
    
    Args:
        agent: wsrl SACAgent instance (already initialized)
        brax_policy_params: Brax policy parameters dict
        brax_normalizer_params: Brax normalizer params (optional, for absorbing)
        absorb_normalizer: Whether to absorb normalizer into first layer
        
    Returns:
        agent: Updated agent with loaded policy weights
    """
    # Extract the actual params dict
    brax_policy = brax_policy_params['params']
    current_params = agent.state.params
    
    # Convert Brax params to wsrl format
    new_actor_params = _convert_brax_to_wsrl_mlp(
        brax_policy,
        current_params['modules_actor'],
        absorb_normalizer=absorb_normalizer,
        normalizer_params=brax_normalizer_params if absorb_normalizer else None,
    )
    
    # Update agent params
    new_params = deepcopy(current_params)
    new_params['modules_actor'] = new_actor_params
    new_state = agent.state.replace(params=new_params)
    
    return agent.replace(state=new_state)


def convert_wsrl_actor_to_brax_policy(
    wsrl_actor_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert wsrl actor parameters back to Brax policy parameter format.
    
    This is the inverse of _convert_brax_to_wsrl_mlp, used for evaluation.
    
    Args:
        wsrl_actor_params: wsrl actor parameters from agent.state.params['modules_actor']
        
    Returns:
        Brax-style policy parameters dict with structure:
        {
            'hidden_0': {'kernel': ..., 'bias': ...},
            'hidden_1': {'kernel': ..., 'bias': ...},
            'hidden_2': {'kernel': ..., 'bias': ...},
        }
    """
    brax_params = {}
    
    network_params = wsrl_actor_params['network']
    
    # Brax-compatible structure: direct mapping
    layer_mapping = [
        ('hidden_0', 'hidden_0'),
        ('hidden_1', 'hidden_1'),
    ]
    
    for wsrl_name, brax_name in layer_mapping:
        # Convert to numpy for Brax format (this is only for saving/evaluation, so CPU is fine)
        brax_params[brax_name] = {
            'kernel': np.array(network_params[wsrl_name]['kernel']),
            'bias': np.array(network_params[wsrl_name]['bias']),
        }
        logging.info(f"[wsrl to Brax] Loaded {wsrl_name} to {brax_name}")
    
    # Handle output layer (hidden_2)
    if 'hidden_2' in wsrl_actor_params:
        output_layer = wsrl_actor_params['hidden_2']
        brax_params['hidden_2'] = {
            'kernel': np.array(output_layer['kernel']),
            'bias': np.array(output_layer['bias']),
        }
        logging.info(f"[wsrl to Brax] Loaded hidden_2")
    
    brax_policy_params = {
        "params": brax_params
    }
    
    return brax_policy_params


def load_brax_q_network_to_wsrl_agent(
    agent,
    q_network_params: Dict[str, Any],
    critic_ensemble_size: int = 2,
) -> Any:
    """
    Load Brax Q-network parameters into a wsrl SACAgent's critic.
    
    This function maps Brax's Q-network parameters to wsrl's critic structure.
    Note: Brax typically uses a single Q-network (with 2 outputs for min Q),
    while wsrl uses an ensemble of Q-networks. This function replicates the
    Brax Q-network across the ensemble.
    
    Args:
        agent: wsrl SACAgent instance (already initialized)
        q_network_params: Dict with 'q_params' and optionally 'target_q_params'
        critic_ensemble_size: Number of critics in wsrl ensemble
        
    Returns:
        agent: Updated agent with loaded Q-network weights
    """
    import copy
    
    q_params = q_network_params['q_params']['params']
    target_q_params = q_network_params['target_q_params']['params']
    
    # Get current agent params
    current_params = copy.deepcopy(agent.state.params)
    current_target_params = copy.deepcopy(agent.state.target_params)
    
    new_critic_params = _convert_brax_q_to_wsrl_critic(
        q_params,
        current_params['modules_critic'],
        critic_ensemble_size,
    )
    
    # Update params
    current_params['modules_critic'] = new_critic_params
    
    new_target_critic_params = _convert_brax_q_to_wsrl_critic(
        target_q_params,
        current_target_params['modules_critic'],
        critic_ensemble_size,
    )
    current_target_params['modules_critic'] = new_target_critic_params
    
    # Create new state with updated params
    new_state = agent.state.replace(
        params=current_params,
        target_params=current_target_params,
    )
    
    return agent.replace(state=new_state)


def _convert_brax_q_to_wsrl_critic(
    brax_q_params: Dict[str, Any],
    wsrl_critic_params: Dict[str, Any],
    critic_ensemble_size: int,
) -> Dict[str, Any]:
    """
    Convert Brax Q-network parameters to wsrl critic parameter structure.
    
    Brax Q-network has structure: hidden_0, hidden_1, hidden_2 (with 2 outputs for Q1, Q2)
    
    For Brax-compatible wsrl critic:
    - network/critic_ensemble/hidden_0, hidden_1, hidden_2
    
    For standard wsrl critic:
    - network/critic_ensemble/layers_0, layers_1, Dense_0 (output)
    
    The ensemble dimension is the first axis of each parameter.
    
    Args:
        brax_q_params: Brax Q-network parameters
        wsrl_critic_params: Current wsrl critic parameters (for structure reference)
        critic_ensemble_size: Number of critics in wsrl ensemble
        
    Returns:
        New critic parameters with Brax weights loaded
    """
    import copy
    
    new_params = copy.deepcopy(wsrl_critic_params)
    ensemble_params = new_params['network']
    
    layer_mapping = [
        ('hidden_0', 'hidden_0'),
        ('hidden_1', 'hidden_1'),
    ]
    # concatenate the brax_q_params weights
    brax_q_params_weights = {
        'hidden_0': {'kernel': None, 'bias': None},
        'hidden_1': {'kernel': None, 'bias': None},
        'hidden_2': {'kernel': None, 'bias': None},
    }

    # Stack all layers with ensemble dimension (entire BraxCritic is ensemblized)
    for brax_name in ['hidden_0', 'hidden_1', 'hidden_2']:
        brax_q_params_weights[brax_name]['kernel'] = np.stack([
            brax_q_params[f"MLP_{i}"][brax_name]['kernel']
            for i in range(critic_ensemble_size)], axis=0)
        brax_q_params_weights[brax_name]['bias'] = np.stack([
            brax_q_params[f"MLP_{i}"][brax_name]['bias']
            for i in range(critic_ensemble_size)], axis=0)

    for brax_name, wsrl_name in layer_mapping:
        ensemble_params[wsrl_name]['kernel'] = jnp.array(brax_q_params_weights[brax_name]['kernel'])
        ensemble_params[wsrl_name]['bias'] = jnp.array(brax_q_params_weights[brax_name]['bias'])
        logging.info(f"[Brax to wsrl] Loaded {wsrl_name} to {brax_name}")

    # hidden_2 also has ensemble dimension since entire BraxCritic is ensemblized
    new_params['hidden_2'] = {
        'kernel': jnp.array(brax_q_params_weights['hidden_2']['kernel']),
        'bias': jnp.array(brax_q_params_weights['hidden_2']['bias'])
    }
    logging.info(f"[Brax to wsrl] Loaded hidden_2")
    return new_params


def verify_brax_wsrl_equivalence(
    brax_env,
    brax_policy_params: Dict[str, Any],
    brax_q_params: Dict[str, Any],
    brax_normalizer_params: Any,
    wsrl_agent: Any,
    num_steps: int = 100,
    num_envs: int = 1,
    seed: int = 0,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Verify that wsrl agent performs exactly the same as Brax policy/Q-network.
    
    This function collects trajectories step-by-step using the same Brax environment
    state for both policies, comparing actions and Q-values at each step.
    
    Args:
        brax_env: Brax environment (base, not wrapped)
        brax_policy_params: Brax policy parameters dict
        brax_q_params: Brax Q-network parameters dict (with 'q_params' key) - REQUIRED
        brax_normalizer_params: Brax normalizer parameters
        wsrl_agent: wsrl SACAgent instance (with loaded Brax weights)
        num_steps: Number of steps to compare
        num_envs: Number of parallel environments
        seed: Random seed
        tolerance: Tolerance for numerical differences
        
    Returns:
        Dictionary with verification results:
        - 'policy_match': bool, whether actions match
        - 'q_match': bool, whether Q-values match
        - 'action_diffs': list of action differences per step
        - 'q_diffs': list of Q-value differences per step
        - 'max_action_diff': float, maximum action difference
        - 'max_q_diff': float, maximum Q-value difference
    """
    
    # Create Brax SAC networks
    obs_size = brax_env.observation_size
    action_size = brax_env.action_size

    wsrl_agent: SACAgent
    
    actor_params = wsrl_agent.state.params['modules_actor']
    network_params = actor_params['network']
    hidden_dims = [network_params['hidden_0']['kernel'].shape[-1], network_params['hidden_1']['kernel'].shape[-1]]
    
    normalize_fn = running_statistics.normalize
    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        hidden_layer_sizes=hidden_dims,
        activation=nn.relu
    )
    
    # Create Brax policy inference function
    make_policy = sac_networks.make_inference_fn(sac_network)
    brax_policy_fn = make_policy((brax_normalizer_params, brax_policy_params), deterministic=True)
    
    q_params = brax_q_params['q_params']
    target_q_params = brax_q_params['target_q_params']
    
    def brax_q_fn(obs, actions):
        """Get Q-values from Brax Q-network."""
        q_values = sac_network.q_network.apply(brax_normalizer_params, q_params, obs, actions)
        # Brax Q-network outputs 2 Q-values (Q1, Q2), take min
        if q_values.ndim > 1 and q_values.shape[-1] == 2:
            q_values = jnp.min(q_values, axis=-1)
        return q_values
    
    def wsrl_q_fn(obs, actions):
        """Get Q-values from wsrl agent's critic."""
        normalized_obs = brax_normalizer(obs)
        q_values = wsrl_agent.forward_critic(normalized_obs, actions, rng=None, train=False)
        if q_values.ndim > 1:
            q_values = jnp.min(q_values, axis=0)
        return q_values

    def brax_target_q_fn(obs, actions):
        """Get target Q-values from Brax Q-network."""
        q_values = sac_network.q_network.apply(brax_normalizer_params, target_q_params, obs, actions)
        if q_values.ndim > 1 and q_values.shape[-1] == 2:
            q_values = jnp.min(q_values, axis=-1)
        return q_values

    def wsrl_target_q_fn(obs, actions):
        """Get target Q-values from wsrl agent's critic."""
        normalized_obs = brax_normalizer(obs)
        q_values = wsrl_agent.forward_target_critic(normalized_obs, actions, rng=None, train=False)
        if q_values.ndim > 1:
            q_values = jnp.min(q_values, axis=0)
        return q_values
    
    # Wrap environment for training
    wrapped_env = envs.training.wrap(
        brax_env,
        episode_length=1000,
        action_repeat=1,
    )
    
    # Initialize environment
    rng = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(rng, num_envs)
    env_state = wrapped_env.reset(reset_keys)

    logging.info(f"Verification using dtype: {env_state.obs.dtype}")
    
    # Create wsrl policy function (with normalizer)
    brax_normalizer = BraxNormalizer.from_brax_params(brax_normalizer_params)
    
    def wsrl_policy_fn(obs, key):
        """Get actions from wsrl agent."""
        normalized_obs = brax_normalizer(obs)
        actions = wsrl_agent.sample_actions(normalized_obs, argmax=True)
        return actions, {}
    
    # Collect step-by-step comparisons
    action_diffs = []
    q_diffs = []
    target_q_diffs = []
    policy_match = True
    q_match = True
    target_q_match = True
    
    step_fn = jax.jit(wrapped_env.step)

    from tqdm import tqdm
    for step in tqdm(range(num_steps), total=num_steps, desc="Verifying Brax-wsrl equivalence"):
        obs = env_state.obs
        
        # Get actions from both policies
        rng, brax_key, wsrl_key = jax.random.split(rng, 3)
        brax_actions, _ = brax_policy_fn(obs, brax_key)
        wsrl_actions, _ = wsrl_policy_fn(obs, wsrl_key)
        
        # Compare actions
        action_diff = jnp.abs(brax_actions - wsrl_actions)
        max_action_diff = jnp.max(action_diff)
        action_diffs.append(float(max_action_diff))
        
        if max_action_diff > tolerance:
            policy_match = False
            logging.warning(
                f"Step {step}: Action mismatch! Max diff: {max_action_diff:.6f}, "
                f"Mean diff: {jnp.mean(action_diff):.6f}"
            )
        
        brax_q = brax_q_fn(obs, brax_actions)
        wsrl_q = wsrl_q_fn(obs, brax_actions)
        
        # Compare Q-values
        q_diff = jnp.abs(brax_q - wsrl_q)
        max_q_diff = jnp.max(q_diff)
        q_diffs.append(float(max_q_diff))
        
        if max_q_diff > tolerance:
            q_match = False
            logging.warning(
                f"Step {step}: Q-value mismatch! Max diff: {max_q_diff:.6f}, "
                f"Mean diff: {jnp.mean(q_diff):.6f}"
            )
        
        brax_target_q = brax_target_q_fn(obs, brax_actions)
        wsrl_target_q = wsrl_target_q_fn(obs, brax_actions)
        
        # Compare target Q-values
        target_q_diff = jnp.abs(brax_target_q - wsrl_target_q)
        max_target_q_diff = jnp.max(target_q_diff)
        target_q_diffs.append(float(max_target_q_diff))
        
        if max_target_q_diff > tolerance:
            target_q_match = False
            logging.warning(
                f"Step {step}: Q-value mismatch! Max diff: {max_q_diff:.6f}, "
                f"Mean diff: {jnp.mean(target_q_diff):.6f}"
            )
        
        # Step environment using Brax actions (for consistency)
        env_state = step_fn(env_state, brax_actions)

        if not policy_match or not q_match or not target_q_match:
            raise ValueError("Verification failed")
    
    results = {
        'policy_match': policy_match,
        'q_match': q_match,
        'action_diffs': action_diffs,
        'q_diffs': q_diffs,
        'max_action_diff': max(action_diffs) if action_diffs else 0.0,
        'max_q_diff': max(q_diffs) if q_diffs else 0.0,
        'mean_action_diff': np.mean(action_diffs) if action_diffs else 0.0,
        'mean_q_diff': np.mean(q_diffs) if q_diffs else 0.0,
        'max_target_q_diff': max(target_q_diffs) if target_q_diffs else 0.0,
        'mean_target_q_diff': np.mean(target_q_diffs) if target_q_diffs else 0.0,
    }
    
    logging.info(f"Verification complete:")
    logging.info(f"  Policy match: {policy_match} (max diff: {results['max_action_diff']:.6f})")
    logging.info(f"  Q-network match: {q_match} (max diff: {results['max_q_diff']:.6f})")
    logging.info(f"  Target Q-network match: {target_q_match} (max diff: {results['max_target_q_diff']:.6f})")
    
    return results