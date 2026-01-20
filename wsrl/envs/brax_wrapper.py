"""
Brax-to-Gym wrapper that makes Brax environments compatible with the Gym interface.
This allows using Brax environments in existing training pipelines designed for Gym.
"""

from typing import Any, Dict, Optional, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.training import types
from gym import spaces


class BraxToGymWrapper(gym.Env):
    """
    Wraps a Brax environment to provide a Gym-compatible interface.
    
    This wrapper handles:
    - Converting between Brax's functional API and Gym's stateful API
    - Converting JAX arrays to NumPy arrays
    - Managing the internal state of the Brax environment
    - Providing proper observation/action spaces
    - JIT compiling env functions for fast execution
    
    Args:
        env_name: Name of the Brax environment (e.g., 'ant', 'halfcheetah', 'humanoid')
        backend: Brax backend to use ('generalized', 'spring', 'positional', 'mjx')
        episode_length: Maximum episode length
        action_repeat: Number of times to repeat each action
        seed: Random seed for environment
    """
    
    metadata = {"render.modes": ["rgb_array"]}
    
    def __init__(
        self,
        env_name: str = "ant",
        backend: str = "generalized",
        episode_length: int = 1000,
        action_repeat: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        
        self.env_name = env_name
        self.backend = backend
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self._seed = seed
        
        # Create the base Brax environment
        self._brax_env = envs.get_environment(env_name=env_name, backend=backend)
        
        # Note: We don't use envs.training.wrap because it adds vectorization
        # that expects batched inputs. Instead, we use the base env directly
        # and handle episode length ourselves.
        self._wrapped_env = self._brax_env
        
        # JIT compile the environment functions for fast execution
        self._jit_reset = jax.jit(self._brax_env.reset)
        self._jit_step = jax.jit(self._brax_env.step)
        
        # Initialize JAX random key
        self._rng = jax.random.PRNGKey(seed)
        
        # Internal state (will be set on reset)
        self._state: Optional[types.State] = None
        self._step_count = 0
        
        # Define observation and action spaces
        obs_size = self._brax_env.observation_size
        action_size = self._brax_env.action_size
        
        # Brax typically has observations in [-inf, inf] range
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        
        # Brax actions are typically in [-1, 1] range
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_size,),
            dtype=np.float32,
        )
        
        # For compatibility with d4rl-style normalization
        self._max_episode_steps = episode_length
    
    @property
    def brax_env(self):
        """Access the underlying Brax environment."""
        return self._brax_env
    
    @property
    def wrapped_env(self):
        """Access the wrapped Brax environment (with episode length handling)."""
        return self._wrapped_env
    
    def seed(self, seed: Optional[int] = None):
        """Set the random seed."""
        if seed is not None:
            self._seed = seed
            self._rng = jax.random.PRNGKey(seed)
        return [self._seed]
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Returns:
            observation: Initial observation
            info: Dictionary with additional info
        """
        if seed is not None:
            self.seed(seed)
        
        # Split the RNG for this reset
        self._rng, reset_rng = jax.random.split(self._rng)
        
        # Reset the Brax environment (JIT compiled)
        self._state = self._jit_reset(reset_rng)
        self._step_count = 0
        
        # Convert observation to numpy
        obs = np.array(self._state.obs, dtype=np.float32)
        
        info = {
            "brax_state": self._state,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode terminated (e.g., agent fell)
            truncated: Whether episode was truncated (e.g., max steps reached)
            info: Dictionary with additional info
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")
        
        # Convert action to JAX array if needed
        if isinstance(action, np.ndarray):
            action = jnp.array(action)
        
        # Step the environment (JIT compiled)
        self._state = self._jit_step(self._state, action)
        self._step_count += 1
        
        # Convert outputs to numpy
        obs = np.array(self._state.obs, dtype=np.float32)
        reward = float(self._state.reward)
        
        # Determine if episode is done
        # Check for done flag from Brax state
        done = bool(self._state.done) if hasattr(self._state, 'done') else False
        
        # Truncation happens when we hit max episode length
        truncated = self._step_count >= self._max_episode_steps
        terminated = done and not truncated
        
        # Force done if truncated
        if truncated:
            done = True
        
        info = {
            "brax_state": self._state,
            "step_count": self._step_count,
        }
        
        # Add metrics from Brax state if available
        if hasattr(self._state, 'metrics'):
            for k, v in self._state.metrics.items():
                try:
                    info[k] = float(v) if hasattr(v, 'item') else v
                except (TypeError, ValueError):
                    pass  # Skip non-numeric metrics
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = "rgb_array"):
        """Render the environment (if supported by the Brax backend)."""
        if self._state is None:
            return None
        
        if mode == "rgb_array":
            # Brax rendering support varies by backend
            try:
                from brax.io import image
                return image.render(self._brax_env.sys, self._state.pipeline_state)
            except (ImportError, AttributeError):
                return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def get_normalized_score(self, reward: float) -> float:
        """
        Return normalized score for compatibility with d4rl-style evaluation.
        
        For Brax envs, we just return the raw reward as there's no standard
        normalization scheme like d4rl.
        """
        return reward


class BraxScaledRewardWrapper(gym.Wrapper):
    """
    Wrapper to scale and bias rewards for Brax environments.
    Works with the new Gym API (5-tuple returns).
    """
    def __init__(self, env, scale: float = 1.0, bias: float = 0.0):
        super().__init__(env)
        self.scale = scale
        self.bias = bias
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward * self.scale + self.bias
        return obs, reward, terminated, truncated, info


class BraxRecordEpisodeStatistics(gym.Wrapper):
    """
    Record episode statistics for Brax environments.
    Works with the new Gym API (5-tuple returns).
    """
    def __init__(self, env):
        super().__init__(env)
        self._episode_return = 0.0
        self._episode_length = 0
    
    def reset(self, **kwargs):
        self._episode_return = 0.0
        self._episode_length = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += reward
        self._episode_length += 1
        
        if terminated or truncated:
            info['episode'] = {
                'r': self._episode_return,
                'l': self._episode_length,
            }
        
        return obs, reward, terminated, truncated, info


def make_brax_env(
    env_name: str,
    backend: str = "generalized",
    episode_length: int = 1000,
    action_repeat: int = 1,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    scale_and_clip_action: bool = True,
    action_clip_lim: float = 0.99999,
    seed: int = 0,
) -> gym.Env:
    """
    Create a Brax environment wrapped for use with wsrl training.
    
    Args:
        env_name: Name of the Brax environment
        backend: Brax backend ('generalized', 'spring', 'positional', 'mjx')
        episode_length: Maximum episode length
        action_repeat: Number of times to repeat each action
        reward_scale: Scale factor for rewards
        reward_bias: Bias to add to rewards
        scale_and_clip_action: Whether to scale and clip actions
        action_clip_lim: Action clipping limit
        seed: Random seed
        
    Returns:
        Wrapped Gym-compatible environment
    """
    # Create base Brax-to-Gym wrapper
    env = BraxToGymWrapper(
        env_name=env_name,
        backend=backend,
        episode_length=episode_length,
        action_repeat=action_repeat,
        seed=seed,
    )
    
    # Apply action scaling/clipping if requested
    # Note: We use custom wrappers that work with the new Gym API (5-tuple)
    if scale_and_clip_action:
        # RescaleAction and ClipAction work with new API
        env = gym.wrappers.RescaleAction(env, -action_clip_lim, action_clip_lim)
        env = gym.wrappers.ClipAction(env)
    
    # Apply reward scaling if needed
    if reward_scale != 1.0 or reward_bias != 0.0:
        env = BraxScaledRewardWrapper(env, reward_scale, reward_bias)
    
    # Add episode statistics recording
    env = BraxRecordEpisodeStatistics(env)
    
    return env
