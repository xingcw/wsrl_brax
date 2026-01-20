"""
Utilities for loading and converting Brax SAC checkpoints to wsrl agents.
"""

import os
import pickle
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from wsrl.envs.brax_dataset import (
    convert_brax_normalizer_to_dict,
    load_brax_sac_checkpoint,
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
    
    # Map Brax policy params to wsrl actor params
    # Brax naming: hidden_0, hidden_1, hidden_2
    # wsrl MLP naming: layers_0, layers_1, ... (based on hidden_dims)
    
    # Build mapping from Brax layer names to wsrl layer indices
    brax_to_wsrl_mapping = {
        'hidden_0': 'layers_0',
        'hidden_1': 'layers_1',
        'hidden_2': 'layers_2',  # This is the output layer (mean)
    }
    
    # Note: wsrl Policy has structure: encoder -> network (MLP) -> output heads
    # The network MLP has layers named 'layers_0', 'layers_1', etc.
    
    # For now, we only load the policy (actor) weights, not the critic
    # The critic will be trained from scratch during finetuning
    
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
    Main work is mapping layer names and handling the different nesting.
    
    Args:
        brax_params: Brax policy parameters
        wsrl_actor_params: Current wsrl actor parameters (for structure reference)
        absorb_normalizer: Whether to absorb normalizer into first layer
        normalizer_params: Normalizer parameters (required if absorbing)
        
    Returns:
        New actor parameters with Brax weights loaded
    """
    import copy
    
    # Make a copy of wsrl params to modify
    new_params = copy.deepcopy(wsrl_actor_params)
    
    # The wsrl Policy structure:
    # - network/layers_0/kernel, network/layers_0/bias
    # - network/layers_1/kernel, network/layers_1/bias
    # - mean_layer/kernel, mean_layer/bias (or similar output layer)
    
    # Extract network params
    if 'network' in new_params:
        network_params = new_params['network']
    else:
        network_params = new_params
    
    # Map Brax layers
    layer_mapping = [
        ('hidden_0', 'layers_0'),
        ('hidden_1', 'layers_1'),
    ]
    
    for brax_name, wsrl_name in layer_mapping:
        if brax_name in brax_params and wsrl_name in network_params:
            kernel = np.array(brax_params[brax_name]['kernel'])
            bias = np.array(brax_params[brax_name]['bias'])
            
            # Absorb normalizer into first layer if requested
            if absorb_normalizer and brax_name == 'hidden_0' and normalizer_params is not None:
                kernel, bias = _absorb_normalizer(kernel, bias, normalizer_params)
            
            network_params[wsrl_name]['kernel'] = jnp.array(kernel)
            network_params[wsrl_name]['bias'] = jnp.array(bias)
    
    # Handle output layer (mean layer for actor)
    # Brax uses hidden_2 for mean output in SAC
    if 'hidden_2' in brax_params:
        # Find the mean output layer in wsrl params
        if 'mean_layer' in new_params:
            new_params['mean_layer']['kernel'] = jnp.array(brax_params['hidden_2']['kernel'])
            new_params['mean_layer']['bias'] = jnp.array(brax_params['hidden_2']['bias'])
        elif 'MeanHead_0' in new_params:
            new_params['MeanHead_0']['kernel'] = jnp.array(brax_params['hidden_2']['kernel'])
            new_params['MeanHead_0']['bias'] = jnp.array(brax_params['hidden_2']['bias'])
    
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
    
    q_params = q_network_params.get('q_params')
    target_q_params = q_network_params.get('target_q_params')
    
    if q_params is None:
        raise ValueError("q_network_params must contain 'q_params'")
    
    # Extract the actual params dict
    if 'params' in q_params:
        brax_q = q_params['params']
    else:
        brax_q = q_params
    
    # Get current agent params
    current_params = copy.deepcopy(agent.state.params)
    current_target_params = copy.deepcopy(agent.state.target_params)
    
    # Map Brax Q-network to wsrl critic
    # Brax Q-network: hidden_0, hidden_1, hidden_2 (output)
    # wsrl critic: critic/network/critic_ensemble/layers_0, layers_1, etc.
    
    new_critic_params = _convert_brax_q_to_wsrl_critic(
        brax_q,
        current_params['critic'],
        critic_ensemble_size,
    )
    
    # Update params
    current_params['critic'] = new_critic_params
    
    # Also update target params if available
    if target_q_params is not None:
        if 'params' in target_q_params:
            brax_target_q = target_q_params['params']
        else:
            brax_target_q = target_q_params
        
        new_target_critic_params = _convert_brax_q_to_wsrl_critic(
            brax_target_q,
            current_target_params['critic'],
            critic_ensemble_size,
        )
        current_target_params['critic'] = new_target_critic_params
    else:
        # Use the same params for target if not provided
        current_target_params['critic'] = new_critic_params
    
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
    wsrl critic has structure: network/critic_ensemble/layers_0, layers_1, etc.
    with the ensemble dimension as the first axis of each parameter.
    
    Args:
        brax_q_params: Brax Q-network parameters
        wsrl_critic_params: Current wsrl critic parameters (for structure reference)
        critic_ensemble_size: Number of critics in wsrl ensemble
        
    Returns:
        New critic parameters with Brax weights loaded
    """
    import copy
    
    new_params = copy.deepcopy(wsrl_critic_params)
    
    # Navigate to the network params in wsrl structure
    # Structure: critic -> network -> critic_ensemble -> layers_X
    if 'network' in new_params:
        network_params = new_params['network']
        if 'critic_ensemble' in network_params:
            ensemble_params = network_params['critic_ensemble']
        else:
            ensemble_params = network_params
    else:
        ensemble_params = new_params
    
    # Map Brax hidden layers to wsrl layers
    # Brax: hidden_0, hidden_1, hidden_2
    # wsrl: layers_0, layers_1, Dense_0 (output)
    layer_mapping = [
        ('hidden_0', 'layers_0'),
        ('hidden_1', 'layers_1'),
    ]
    
    for brax_name, wsrl_name in layer_mapping:
        if brax_name in brax_q_params and wsrl_name in ensemble_params:
            kernel = np.array(brax_q_params[brax_name]['kernel'])
            bias = np.array(brax_q_params[brax_name]['bias'])
            
            # wsrl ensemble expects shape (ensemble_size, ...) for each param
            # Replicate the Brax params across the ensemble
            kernel_ensemble = np.stack([kernel] * critic_ensemble_size, axis=0)
            bias_ensemble = np.stack([bias] * critic_ensemble_size, axis=0)
            
            ensemble_params[wsrl_name]['kernel'] = jnp.array(kernel_ensemble)
            ensemble_params[wsrl_name]['bias'] = jnp.array(bias_ensemble)
    
    # Handle output layer (Dense_0 in wsrl)
    if 'hidden_2' in brax_q_params:
        # Brax hidden_2 outputs 2 Q-values (Q1, Q2) with shape (hidden, 2)
        # We need to split this for the ensemble
        kernel = np.array(brax_q_params['hidden_2']['kernel'])  # (hidden, 2)
        bias = np.array(brax_q_params['hidden_2']['bias'])  # (2,)
        
        # For wsrl, each ensemble member outputs 1 Q-value
        # Shape should be (ensemble_size, hidden, 1)
        if kernel.shape[-1] == 2 and critic_ensemble_size == 2:
            # Split the 2 Q-value outputs into 2 ensemble members
            kernel_ensemble = kernel[:, :, np.newaxis]  # This won't work correctly
            # Actually, reshape (hidden, 2) -> (2, hidden, 1)
            kernel_ensemble = np.transpose(kernel)[:, :, np.newaxis]  # (2, hidden, 1)
            bias_ensemble = bias[:, np.newaxis]  # (2, 1)
        else:
            # Replicate single output across ensemble
            kernel_single = kernel[:, :1] if kernel.shape[-1] > 1 else kernel
            bias_single = bias[:1] if bias.shape[-1] > 1 else bias
            kernel_ensemble = np.stack([kernel_single] * critic_ensemble_size, axis=0)
            bias_ensemble = np.stack([bias_single] * critic_ensemble_size, axis=0)
        
        # Find the output layer in wsrl params
        output_layer_name = None
        for name in ensemble_params.keys():
            if 'Dense' in name or name == 'output':
                output_layer_name = name
                break
        
        if output_layer_name and output_layer_name in ensemble_params:
            ensemble_params[output_layer_name]['kernel'] = jnp.array(kernel_ensemble)
            ensemble_params[output_layer_name]['bias'] = jnp.array(bias_ensemble)
    
    return new_params
