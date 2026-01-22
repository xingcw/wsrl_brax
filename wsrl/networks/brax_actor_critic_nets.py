"""
Brax-compatible Policy and Critic networks.

These use hidden_0, hidden_1, hidden_2 naming to match Brax SAC parameter structure.
"""

from typing import Optional

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax

from wsrl.common.initialization import init_fns
from wsrl.networks.brax_mlp import BraxMLP
from wsrl.networks.actor_critic_nets import TanhMultivariateNormalDiag


class BraxPolicy(nn.Module):
    """
    Brax-compatible Policy network.
    
    Structure:
    - encoder (optional, None for Brax)
    - BraxMLP with hidden_0, hidden_1 (2 hidden layers)
    - hidden_2: Mean output layer
    - std_head: Log std output layer (for exp parameterization)
    
    This matches Brax's policy network parameter structure.
    """
    encoder: Optional[nn.Module]
    network: BraxMLP
    action_dim: int
    init_final: Optional[float] = None
    std_parameterization: str = "exp"  # "exp", "softplus", "fixed", or "uniform"
    std_min: Optional[float] = 1e-5
    std_max: Optional[float] = 10.0
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None
    kernel_init_type: Optional[str] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type] if self.kernel_init_type else None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, train: bool = False
    ) -> distrax.Distribution:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        # Forward through BraxMLP (hidden_0, hidden_1)
        outputs = self.network(obs_enc, train=train)

        # Mean output layer: hidden_2 (Brax-compatible naming)
        init_fn = self.init_fn() if self.init_fn else None
        if init_fn:
            logits = nn.Dense(self.action_dim * 2, kernel_init=init_fn(), name="hidden_2")(outputs)
            means, log_stds = jnp.split(logits, 2, axis=-1)
        else:
            logits = nn.Dense(self.action_dim * 2, name="hidden_2")(outputs)
            means, log_stds = jnp.split(logits, 2, axis=-1)
        
        # Std parameterization
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                stds = jnp.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = nn.softplus(log_stds)
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            assert self.std_parameterization == "fixed"
            if type(self.fixed_std) == list:
                stds = jnp.array(self.fixed_std)
            else:
                assert isinstance(
                    self.fixed_std, (int, float)
                ), "fixed std must be a number"
                stds = jnp.array([self.fixed_std] * self.action_dim)

        # Clip stds to avoid numerical instability
        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.tanh_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )

        return distribution


class BraxCritic(nn.Module):
    """
    Brax-compatible Critic network.
    
    Structure:
    - encoder (optional, None for Brax)
    - BraxMLP with hidden_0, hidden_1 (2 hidden layers)
    - hidden_2: Q-value output layer
    
    This matches Brax's Q-network parameter structure.
    """
    encoder: Optional[nn.Module]
    network: BraxMLP
    init_final: Optional[float] = None
    kernel_init_type: Optional[str] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type] if self.kernel_init_type else None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)
        
        # jax.debug.print("[Brax Critic] obs_enc: {obs_enc}", obs_enc=obs_enc)
        # jax.debug.print("[Brax Critic] actions: {actions}", actions=actions)
        inputs = jnp.concatenate([obs_enc, actions], -1)
        outputs = self.network(inputs, train=train)
        # jax.debug.print("[Brax Critic] outputs: {outputs}", outputs=outputs)
        
        # Q-value output layer: hidden_2 (Brax-compatible naming)
        init_fn = self.init_fn() if self.init_fn else None
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
                name="hidden_2"
            )(outputs)
        elif init_fn:
            value = nn.Dense(1, kernel_init=init_fn(), name="hidden_2")(outputs)
        else:
            value = nn.Dense(1, name="hidden_2")(outputs)
        # jax.debug.print("[Brax Critic] value: {value}", value=value)
        return jnp.squeeze(value, -1)
