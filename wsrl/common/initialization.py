from typing import Optional

import flax.linen as nn


def var_scaling_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def orthogonal_init(scale: Optional[float] = None):
    """GPU-safe replacement for orthogonal init.
    
    orthogonal() uses QR decomposition which fails on RTX 5090 (Blackwell)
    due to cuSolver incompatibility. Using lecun_normal as safe alternative.
    """
    return nn.initializers.lecun_normal()


def xavier_normal_init():
    """Xavier normal using variance_scaling (GPU-safe).
    
    Note: nn.initializers.xavier_normal() internally uses orthogonal,
    which fails on Blackwell GPUs. Using variance_scaling equivalent.
    """
    return nn.initializers.variance_scaling(1.0, "fan_avg", "truncated_normal")


def kaiming_init():
    """Kaiming/He normal using variance_scaling (GPU-safe).
    
    Note: nn.initializers.kaiming_normal() internally uses orthogonal,
    which fails on Blackwell GPUs. Using variance_scaling equivalent.
    """
    return nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")


def xavier_uniform_init():
    return nn.initializers.xavier_uniform()


def lecun_normal_init():
    return nn.initializers.lecun_normal()


init_fns = {
    None: lecun_normal_init,  # Default to GPU-safe initializer
    "var_scaling": var_scaling_init,
    "orthogonal": orthogonal_init,
    "xavier_normal": xavier_normal_init,
    "kaiming": kaiming_init,
    "xavier_uniform": xavier_uniform_init,
    "lecun_normal": lecun_normal_init,
}
