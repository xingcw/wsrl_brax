"""
Brax-compatible MLP networks that use hidden_0, hidden_1, hidden_2 naming.

This matches the parameter structure used by Brax SAC training.
"""

from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from wsrl.common.initialization import init_fns


class BraxMLP(nn.Module):
    """
    Brax-compatible MLP with hidden_0, hidden_1 naming for hidden layers.
    
    For a 2-layer MLP (2 hidden layers), this creates:
    - hidden_0: First hidden layer
    - hidden_1: Second hidden layer
    
    The output layer (hidden_2) is handled separately by the parent network.
    """
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.relu
    activate_final: bool = True
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    kernel_init_type: Optional[str] = None
    kernel_scale_final: Optional[float] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type] if self.kernel_init_type else None

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)
        
        init_fn = self.init_fn() if self.init_fn else None

        # Create hidden layers with Brax naming: hidden_0, hidden_1
        for i, size in enumerate(self.hidden_dims):
            layer_name = f"hidden_{i}"
            
            # Use custom init if available, otherwise default
            if init_fn:
                if i + 1 == len(self.hidden_dims) and self.kernel_scale_final is not None:
                    x = nn.Dense(size, kernel_init=init_fn(self.kernel_scale_final), name=layer_name)(x)
                else:
                    x = nn.Dense(size, kernel_init=init_fn(), name=layer_name)(x)
            else:
                x = nn.Dense(size, name=layer_name)(x)

            # Normalization and activation after each layer
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = activations(x)
        return x
