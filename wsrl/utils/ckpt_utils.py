"""Checkpoint utilities for preemption-safe checkpointing with GCS support."""

from etils import epath
from absl import logging
import orbax.checkpoint as ocp


def _extract_wandb_run_id(v):
    """Extract a string wandb run ID from a possibly wrapped/array value (e.g. numpy scalar)."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip():
        return v.strip()
    try:
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None


def load_preemption_checkpoint(checkpoint_dir: str):
    """Load checkpoint from GCS or local path if it exists.
    
    Returns:
        tuple: (agent_state, step, wandb_run_id, replay_buffer_state, ckpt_manager) or (None, 0, None, None, None) if no checkpoint
    """
    if not checkpoint_dir:
        return None, 0, None, None, None
    
    ckpt_path = epath.Path(checkpoint_dir)
    
    # Create checkpoint manager for future saves (with auto-cleanup)
    checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=3)
    ckpt_manager = ocp.CheckpointManager(ckpt_path, options=checkpoint_options)
    
    try:
        # Try CheckpointManager first (new format)
        latest_step = ckpt_manager.latest_step()
        
        if latest_step is not None:
            logging.info(f"Found checkpoint at step {latest_step} (CheckpointManager format)")
            try:
                restored = ckpt_manager.restore(latest_step)
                # Handle StandardSave format: dict, dict-like, or wrapped
                if isinstance(restored, dict):
                    ckpt_data = restored
                elif hasattr(restored, 'get') and callable(getattr(restored, 'get')):
                    ckpt_data = restored
                else:
                    ckpt_data = {'agent': restored, 'step': latest_step, 'wandb_run_id': None, 'replay_buffer': None}
                
                agent_state = ckpt_data.get('agent', None) if isinstance(ckpt_data, dict) else None
                step = ckpt_data.get('step', latest_step) if isinstance(ckpt_data, dict) else latest_step
                wandb_run_id = _extract_wandb_run_id(ckpt_data.get('wandb_run_id', None) if isinstance(ckpt_data, dict) else None)
                replay_buffer_state = ckpt_data.get('replay_buffer', None) if isinstance(ckpt_data, dict) else None
                
                logging.info(f"Successfully restored checkpoint from step {step}")
                if wandb_run_id:
                    logging.info(f"Found wandb run ID: {wandb_run_id}")
                
                return agent_state, int(step), wandb_run_id, replay_buffer_state, ckpt_manager
            except Exception as e:
                logging.warning(f"Failed to restore with CheckpointManager, trying PyTreeCheckpointer: {e}")
        
        # Fallback to PyTreeCheckpointer for backward compatibility
        checkpointer = ocp.PyTreeCheckpointer()
        latest_step = checkpointer.latest_step(ckpt_path)
        
        if latest_step is not None:
            logging.info(f"Found checkpoint at step {latest_step} (PyTreeCheckpointer format)")
            restored = checkpointer.restore(ckpt_path / str(latest_step))
            
            agent_state = restored.get('agent', None)
            step = restored.get('step', latest_step)
            wandb_run_id = _extract_wandb_run_id(restored.get('wandb_run_id', None))
            replay_buffer_state = restored.get('replay_buffer', None)
            
            logging.info(f"Successfully restored checkpoint from step {step}")
            if wandb_run_id:
                logging.info(f"Found wandb run ID: {wandb_run_id}")
            
            return agent_state, int(step), wandb_run_id, replay_buffer_state, ckpt_manager
        else:
            logging.info(f"No checkpoint found in {checkpoint_dir}")
            return None, 0, None, None, ckpt_manager
    except Exception as e:
        logging.warning(f"Failed to load checkpoint: {e}")
        return None, 0, None, None, ckpt_manager


def save_preemption_checkpoint(checkpoint_dir: str, agent, step: int, wandb_run_id: str, replay_buffer=None, ckpt_manager=None):
    """Save checkpoint to GCS or local path for preemption recovery.
    
    Args:
        checkpoint_dir: Path to save checkpoint (supports GCS)
        agent: The agent to save
        step: Current training step
        wandb_run_id: Wandb run ID for resume
        replay_buffer: Optional replay buffer to save
        ckpt_manager: Optional checkpoint manager (will create one if not provided)
    """
    if not checkpoint_dir:
        return
    
    ckpt_path = epath.Path(checkpoint_dir)
    
    try:
        # Use checkpoint manager for automatic cleanup
        if ckpt_manager is None:
            checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=3)
            ckpt_manager = ocp.CheckpointManager(ckpt_path, options=checkpoint_options)
        
        # Prepare checkpoint data
        ckpt_data = {
            'agent': agent,
            'step': step,
            'wandb_run_id': wandb_run_id,
        }
        
        # Optionally include replay buffer (can be large)
        if replay_buffer is not None:
            try:
                ckpt_data['replay_buffer'] = {
                    'observations': replay_buffer._observations[:replay_buffer._size],
                    'next_observations': replay_buffer._next_observations[:replay_buffer._size],
                    'actions': replay_buffer._actions[:replay_buffer._size],
                    'rewards': replay_buffer._rewards[:replay_buffer._size],
                    'masks': replay_buffer._masks[:replay_buffer._size],
                    'dones': replay_buffer._dones[:replay_buffer._size],
                    'size': replay_buffer._size,
                    'insert_index': replay_buffer._insert_index,
                }
            except Exception as e:
                logging.warning(f"Failed to save replay buffer: {e}")
        
        # Save using checkpoint manager (automatically handles cleanup)
        ckpt_manager.save(
            step=step,
            args=ocp.args.StandardSave(ckpt_data)
        )
        
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
