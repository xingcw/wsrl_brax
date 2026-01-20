"""
Configuration for SAC agent with Brax environments.
This wraps the SAC config with agent_kwargs as expected by finetune.py.
"""

from ml_collections import ConfigDict

from experiments.configs.sac_config import get_config as get_sac_config


def get_config(config_string=None):
    """
    Get Brax SAC configuration.
    
    Args:
        config_string: Optional config variant (not used, for compatibility)
        
    Returns:
        ConfigDict with agent_kwargs containing SAC configuration
    """
    # Get base SAC config
    sac_config = get_sac_config()
    
    # Wrap in agent_kwargs as expected by finetune.py
    config = ConfigDict(
        dict(
            agent_kwargs=sac_config.to_dict(),
        )
    )
    
    return config
