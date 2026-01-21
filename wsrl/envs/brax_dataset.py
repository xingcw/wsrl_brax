"""
Utilities for loading Brax SAC checkpoints and creating datasets for finetuning.
"""

import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def load_brax_sac_checkpoint(
    ckpt_path: str,
    checkpoint_idx: int = -1,
    load_q_network: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load a Brax SAC checkpoint from disk.
    
    Brax SAC checkpoints typically contain:
    - normalizer_params: Running statistics for observation normalization
    - policy_params: Policy network parameters
    - q_params (optional): Q-network parameters (if saved with save_q_network=True)
    
    The checkpoint may contain multiple training snapshots stacked along axis 0.
    
    Args:
        ckpt_path: Path to the checkpoint directory or file
        checkpoint_idx: Index of checkpoint to load if multiple are stacked.
                       Use -1 for the last checkpoint.
        load_q_network: Whether to also load Q-network parameters (for wsrl finetuning)
    
    Returns:
        normalizer_params: Dict with 'mean' and 'std' for observation normalization
        policy_params: Dict with policy network parameters
        eval_metrics: Optional evaluation metrics if available
        q_network_params: Optional Q-network parameters if load_q_network=True
    """
    # Handle both directory and file paths
    if os.path.isdir(ckpt_path):
        ckpt_file = os.path.join(ckpt_path, "sac_params.pkl")
    else:
        ckpt_file = ckpt_path
    
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    
    with open(ckpt_file, "rb") as f:
        sac_params = pickle.load(f)
    
    # Unpack the checkpoint
    normalizer_params_all, policy_params_all = sac_params
    
    # Extract specific checkpoint if multiple are stacked
    def extract_checkpoint(params, idx):
        """Extract a single checkpoint from stacked parameters."""
        return jax.tree_util.tree_map(
            lambda x: x[idx] if isinstance(x, (np.ndarray, jnp.ndarray)) and x.ndim > 1 else x,
            params
        )
    
    mean_shape = np.array(normalizer_params_all.mean).shape
    if len(mean_shape) > 1:
        # Multiple checkpoints stacked - extract the requested one
        normalizer_params = extract_checkpoint(normalizer_params_all, checkpoint_idx)
        policy_params = extract_checkpoint(policy_params_all, checkpoint_idx)
    else:
        # Single checkpoint
        normalizer_params = normalizer_params_all
        policy_params = policy_params_all
    
    # Try to load eval metrics if available
    eval_metrics = None
    eval_metrics_file = os.path.join(os.path.dirname(ckpt_file), "sac_metrics.pkl")
    if os.path.exists(eval_metrics_file):
        with open(eval_metrics_file, "rb") as f:
            eval_metrics = pickle.load(f)
    
    logging.info(f"Loaded checkpoint from {ckpt_file} with checkpoint index {checkpoint_idx}")
    logging.info(f"Eval metrics: {eval_metrics}")
    
    # Try to load Q-network params if requested
    q_network_params = None
    if load_q_network:
        q_params_file = os.path.join(os.path.dirname(ckpt_file), "sac_q_params.pkl")
        with open(q_params_file, "rb") as f:
            q_network_data = pickle.load(f)
        
        # Extract q_params at the specified checkpoint index
        q_params_all = q_network_data['q_params']
        target_q_params = q_network_data['target_q_params']       
        
        q_params = extract_checkpoint(q_params_all, checkpoint_idx)
        target_q_params = extract_checkpoint(target_q_params, checkpoint_idx)
        
        q_network_params = {
            'q_params': q_params,
            'target_q_params': target_q_params,
        }
        logging.info(f"Loaded Q-network params from {q_params_file} with checkpoint index {checkpoint_idx}")
    
    return normalizer_params, policy_params, eval_metrics, q_network_params


def convert_brax_normalizer_to_dict(normalizer_params) -> Dict[str, np.ndarray]:
    """
    Convert Brax normalizer parameters to a simple dict format.
    
    Args:
        normalizer_params: Brax RunningStatisticsState with mean/std attributes
        
    Returns:
        Dict with 'mean' and 'std' as numpy arrays
    """
    return {
        'mean': np.array(normalizer_params.mean, dtype=np.float32),
        'std': np.array(normalizer_params.std, dtype=np.float32),
    }


def convert_brax_policy_to_wsrl_format(
    policy_params: Dict[str, Any],
    architecture: list = None,
) -> Dict[str, np.ndarray]:
    """
    Convert Brax MLP policy parameters to wsrl-compatible format.
    
    Brax uses (input_dim, output_dim) for kernel shapes.
    wsrl/Flax also uses (input_dim, output_dim), so no transpose needed.
    
    The main conversion is flattening the nested dict structure to match
    the wsrl parameter naming convention.
    
    Args:
        policy_params: Dict with Brax policy parameters
        architecture: List of (layer_name, param_type) tuples defining the architecture.
                     If None, auto-detects from the params structure.
    
    Returns:
        Dict with wsrl-compatible parameter names
    """
    # Default architecture for Brax SAC MLP
    if architecture is None:
        architecture = [
            ('hidden_0', 'kernel'),
            ('hidden_0', 'bias'),
            ('hidden_1', 'kernel'),
            ('hidden_1', 'bias'),
            ('hidden_2', 'kernel'),
            ('hidden_2', 'bias'),
        ]
    
    # Extract the params dict (may be nested under 'params')
    if 'params' in policy_params:
        params = policy_params['params']
    else:
        params = policy_params
    
    # Convert to wsrl format
    wsrl_params = {}
    
    for layer_name, param_type in architecture:
        if layer_name in params and param_type in params[layer_name]:
            key = f'{layer_name}/{param_type}'
            wsrl_params[key] = np.array(params[layer_name][param_type], dtype=np.float32)
    
    return wsrl_params


def absorb_normalizer_into_policy(
    normalizer_params,
    policy_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Absorb observation normalization into the first layer of the policy network.
    
    This "bakes in" the normalization into the network weights, allowing you to
    use the policy without an explicit normalization step.
    
    Given normalized_obs = (obs - mean) / std, we modify the first layer:
    - New W = W / std (element-wise division along input dimension)
    - New b = b - mean @ W / std
    
    Args:
        normalizer_params: Normalizer with mean/std attributes
        policy_params: Policy parameters with first layer named 'hidden_0'
        
    Returns:
        Modified policy parameters with normalizer absorbed
    """
    import copy
    
    # Get normalizer stats
    mean = np.array(normalizer_params.mean).reshape(1, -1)
    std = np.array(normalizer_params.std).reshape(1, -1)
    
    # Make a copy to avoid modifying original
    params = copy.deepcopy(policy_params)
    
    # Get first layer params (may be nested under 'params')
    if 'params' in params:
        layer_params = params['params']['hidden_0']
    else:
        layer_params = params['hidden_0']
    
    w = np.array(layer_params['kernel'])  # (input_dim, output_dim)
    b = np.array(layer_params['bias']).reshape(1, -1)  # (1, output_dim)
    
    # Absorb normalization: W_new = W / std, b_new = b - (mean / std) @ W
    w_new = w / std.T  # Divide each row by corresponding std
    b_new = b - (mean / std) @ w
    
    # Update params
    layer_params['kernel'] = w_new
    layer_params['bias'] = b_new.flatten()
    
    return params


def create_dummy_dataset_from_brax_env(
    env_name: str,
    backend: str = "generalized",
    num_samples: int = 1000,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Create a dummy dataset from a Brax environment for agent initialization.
    
    This is useful when you want to initialize a wsrl agent with the correct
    shapes but don't have a pre-existing dataset.
    
    Args:
        env_name: Name of the Brax environment
        backend: Brax backend
        num_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        Dataset dict with observations, actions, etc.
    """
    from brax import envs
    
    # Create environment
    env = envs.get_environment(env_name=env_name, backend=backend)
    
    obs_size = env.observation_size
    action_size = env.action_size
    
    # Create random data with correct shapes
    rng = np.random.RandomState(seed)
    
    return {
        'observations': rng.randn(num_samples, obs_size).astype(np.float32),
        'actions': rng.uniform(-1, 1, (num_samples, action_size)).astype(np.float32),
        'next_observations': rng.randn(num_samples, obs_size).astype(np.float32),
        'rewards': rng.randn(num_samples).astype(np.float32),
        'masks': np.ones(num_samples, dtype=np.float32),
        'dones': np.zeros(num_samples, dtype=np.float32),
    }
