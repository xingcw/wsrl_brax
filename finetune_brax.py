"""
Finetuning script for Brax environments with wsrl.
Uses pure Brax (no Gym wrapper) for faster vectorized training.

Note: To use Q-network loading, the Brax checkpoint must be saved with save_q_network=True
in the train_sac_brax.py script.
"""

import os
import shutil
import warnings

# Suppress noisy runtime logs before importing JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logging (ERROR only)
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC warnings

# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging as python_logging
# Reduce logging verbosity BEFORE importing JAX
python_logging.getLogger("jax").setLevel(python_logging.ERROR)
python_logging.getLogger("jax._src").setLevel(python_logging.ERROR)
python_logging.getLogger("brax").setLevel(python_logging.WARNING)
python_logging.getLogger("orbax").setLevel(python_logging.WARNING)

# Also suppress absl logging for io.py messages
import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)

import jax
import jax.numpy as jnp
# Suppress JAX info logs
jax.config.update("jax_log_compiles", False)
import numpy as np
import random
import torch
import tqdm
from absl import app, flags, logging
import orbax.checkpoint as ocp
from ml_collections import config_flags

from brax import envs as brax_envs

from wsrl.agents import agents
from wsrl.common.wandb import WandBLogger
from wsrl.utils.timer_utils import Timer
from wsrl.utils.train_utils import (
    subsample_batch,
    SimpleReplayBuffer,
    make_brax_training_env,
    improve_policy_with_hypernet,
    evaluate_and_log_brax,
)
from wsrl.utils.ckpt_utils import load_preemption_checkpoint, save_preemption_checkpoint

# Brax-specific imports
from wsrl.envs.brax_dataset import (
    create_dummy_dataset_from_brax_env,
    load_brax_sac_checkpoint,
)
from wsrl.utils.brax_utils import (
    BraxNormalizer,
    load_brax_q_network_to_wsrl_agent,
    load_brax_policy_to_wsrl_agent,
    verify_brax_wsrl_equivalence,
)

from data_gen.mujoco.cartpole_brax import InvertedPendulum
from data_gen.mujoco.ant_brax import Ant
from data_gen.mujoco.halfcheetah_brax import Halfcheetah
from data_gen.mujoco.hopper_brax import Hopper
brax_envs.register_environment('ant_s2r', Ant)
brax_envs.register_environment('cartpole', InvertedPendulum)
brax_envs.register_environment('halfcheetah_s2r', Halfcheetah)
brax_envs.register_environment('hopper_s2r', Hopper)

# Hypernet imports
from utils.hypernet_utils import HypernetLoader

FLAGS = flags.FLAGS

# Brax environment settings
flags.DEFINE_string("brax_env", "ant", "Brax environment name (e.g., ant, halfcheetah, humanoid)")
flags.DEFINE_string("brax_backend", "generalized", "Brax backend (generalized, spring, positional, mjx)")
flags.DEFINE_string("brax_mjcf_path", "", "Path to custom MJCF file for Brax environment")
flags.DEFINE_integer("brax_episode_length", 1000, "Maximum episode length for Brax env")
flags.DEFINE_string("brax_ckpt_path", "", "Path to Brax SAC checkpoint to load for initialization")
flags.DEFINE_integer("brax_ckpt_idx", -1, "Checkpoint index to load (-1 for latest)")
flags.DEFINE_list("brax_hidden_dims", ["256", "256"], "Hidden layer sizes for policy network")
flags.DEFINE_integer("brax_num_eval_envs", 128, "Number of parallel envs for Brax native evaluation")
flags.DEFINE_bool("load_brax_q_network", False, "Load Q-network from Brax checkpoint for wsrl finetuning")

# Hypernet settings
flags.DEFINE_string("hypernet_config_path", "", "Path to hypernet config file (YAML)")
flags.DEFINE_string("hypernet_ckpt_path", "", "Path to hypernet checkpoint directory")
flags.DEFINE_string("hypernet_ckpt_step", "best", "Hypernet checkpoint step (best/number)")
flags.DEFINE_bool("hypernet_improve_init", False, "Use hypernet to improve initial policy before finetuning")
flags.DEFINE_float("hypernet_target_reward", None, "Target reward for hypernet (default: 1.5x current)")
flags.DEFINE_integer(
    "resume_ckpt_axis_size",
    16,
    "Resume checkpoint sharding axis divisibility constraint (forwarded to utils.fsdp_utils.shard_module_params).",
)

# Reward settings
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias.")
flags.DEFINE_float(
    "clip_action",
    0.99999,
    "Clip actions to be between [-n, n]. This is needed for tanh policies.",
)

# Training settings
flags.DEFINE_integer("num_online_steps", 500_000, "Number of online training steps.")
flags.DEFINE_integer("warmup_steps", 0, "Number of warmup steps before performing updates")
flags.DEFINE_integer("num_envs", 1, "Number of parallel training environments")

# Agent settings
flags.DEFINE_string("agent", "sac", "RL agent to use (sac recommended for Brax)")
flags.DEFINE_integer("utd", 1, "Update-to-data ratio of the critic")
flags.DEFINE_integer("batch_size", 256, "Batch size for training")
flags.DEFINE_integer("replay_buffer_capacity", int(2e6), "Replay buffer capacity")

# Experiment house keeping
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "save_dir",
    os.path.expanduser("~/wsrl_log"),
    "Directory to save the logs and checkpoints",
)
flags.DEFINE_string("resume_path", "", "Path to resume from (wsrl checkpoint)")
flags.DEFINE_string("checkpoint_dir", "", "Directory for preemption-safe checkpoints (supports GCS, e.g., gs://bucket/path). Auto-resumes if checkpoint exists.")
flags.DEFINE_integer("log_interval", 5_000, "Log every n steps")
flags.DEFINE_integer("eval_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("save_interval", 100_000, "Save every n steps.")
flags.DEFINE_integer("checkpoint_interval", 10_000, "Save preemption-safe checkpoint every n steps (for auto-resume)")
flags.DEFINE_integer("n_eval_trajs", 20, "Number of trajectories for evaluation.")
flags.DEFINE_bool("deterministic_eval", True, "Whether to use deterministic evaluation")

# Wandb settings
flags.DEFINE_string("exp_name", "", "Experiment name for wandb logging")
flags.DEFINE_string("project", "wsrl_brax", "Wandb project folder")
flags.DEFINE_string("group", None, "Wandb group of the experiment")
flags.DEFINE_bool("debug", False, "If true, no logging to wandb")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    """
    House keeping
    """
    # ==========================================================================
    # SEED EVERYTHING FIRST
    # ==========================================================================
    logging.info(f"Setting all random seeds to {FLAGS.seed}")
    
    # Seed Python's random module
    random.seed(FLAGS.seed)
    
    # Seed NumPy
    np.random.seed(FLAGS.seed)
    
    # Seed PyTorch
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
        torch.cuda.manual_seed_all(FLAGS.seed)
    
    # Seed JAX (will be split later for different purposes)
    # This is just to ensure deterministic initialization
    os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
    
    logging.info("All random number generators seeded successfully")
    # ==========================================================================
    
    brax_hidden_dims = [int(d) for d in FLAGS.brax_hidden_dims]
    
    # Minimum steps before updates (need some data in buffer)
    min_steps_to_update = FLAGS.batch_size

    """
    Try to load checkpoint for resumption (if checkpoint_dir is provided)
    """
    resumed_agent_state = None
    resumed_step = 0
    resumed_wandb_run_id = None
    resumed_replay_buffer_state = None
    preemption_ckpt_manager = None
    
    if FLAGS.checkpoint_dir:
        logging.info("=" * 80)
        logging.info("CHECKING FOR EXISTING CHECKPOINT")
        logging.info("=" * 80)
        resumed_agent_state, resumed_step, resumed_wandb_run_id, \
            resumed_replay_buffer_state, preemption_ckpt_manager = \
                load_preemption_checkpoint(FLAGS.checkpoint_dir)
        if resumed_agent_state is not None:
            logging.info(f"Will resume from step {resumed_step}")
        logging.info("=" * 80)
    
    """
    Wandb and logging
    """
    # Build descriptive experiment name (WITHOUT seed for grouping)
    group_parts = []
    
    # Add user-provided exp_name if specified
    if FLAGS.exp_name:
        group_parts.append(FLAGS.exp_name)
    
    # Add env name
    group_parts.append(FLAGS.brax_env)
    
    # Add MJCF filename if custom MJCF is provided
    if FLAGS.brax_mjcf_path:
        mjcf_filename = os.path.splitext(os.path.basename(FLAGS.brax_mjcf_path))[0]
        group_parts.append(mjcf_filename)
    
    # Add agent type
    group_parts.append(FLAGS.agent)
    
    # Add checkpoint info if loading from Brax checkpoint
    if FLAGS.brax_ckpt_path:
        brax_ckpt_basename = os.path.basename(FLAGS.brax_ckpt_path.rstrip("/"))
        group_parts.append(brax_ckpt_basename)
        ckpt_id = FLAGS.brax_ckpt_idx if FLAGS.brax_ckpt_idx >= 0 else "latest"
        group_parts.append(f"ckpt{ckpt_id}")
    
    # Add hypernet indicator if using hypernet for init improvement
    if FLAGS.hypernet_improve_init and FLAGS.hypernet_config_path and FLAGS.hypernet_ckpt_path:
        group_parts.append("hypernet_init")
    
    # Group name: everything EXCEPT seed (for aggregating across seeds)
    group_name = "_".join(group_parts) if group_parts else "default_group"
    
    # Run name: include seed for individual run identification
    run_name_parts = group_parts + [f"seed{FLAGS.seed}"]
    exp_descriptor = "_".join(run_name_parts)
    
    # Use user-provided group if specified, otherwise use auto-generated group
    final_group = FLAGS.group if FLAGS.group is not None else group_name
    
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": FLAGS.project or "wsrl_brax",
            "group": final_group,  # This groups runs together in wandb
            "exp_descriptor": exp_descriptor,  # This is the individual run name
            "tags": [f"seed_{FLAGS.seed}", FLAGS.brax_env, FLAGS.agent],  # Add tags for easy filtering
        }
    )
    
    # Add FLAGS to variant for full tracking
    variant = FLAGS.config.to_dict()
    variant.update({
        "seed": FLAGS.seed,
        "brax_env": FLAGS.brax_env,
        "agent": FLAGS.agent,
        "num_online_steps": FLAGS.num_online_steps,
        "batch_size": FLAGS.batch_size,
        "utd": FLAGS.utd,
    })
    
    # If resuming, use the saved run ID to continue the same wandb run
    if resumed_wandb_run_id:
        logging.info(f"Resuming wandb run: {resumed_wandb_run_id}")
        # Override the experiment_id to use the resumed ID
        wandb_config.update({
            "unique_identifier": "",  # Will be set to resumed ID
            "experiment_id": resumed_wandb_run_id,
        })
        
        # We need to initialize wandb manually for resume
        import wandb
        wandb_output_dir = os.path.expanduser("~/wandb_logs")
        os.makedirs(wandb_output_dir, exist_ok=True)
        
        wandb.init(
            config=variant,
            project=wandb_config['project'],
            entity=wandb_config.get('entity', None),
            group=wandb_config['group'],
            dir=wandb_output_dir,
            id=resumed_wandb_run_id,
            resume="must",
            save_code=True,
            mode="disabled" if FLAGS.debug else "online",
        )
        
        # Create a simple wrapper to match WandBLogger interface
        class ResumedWandBLogger:
            def __init__(self, run, config):
                from omegaconf import DictConfig
                self.run = run
                self.config = DictConfig(config)
                
            def log(self, data: dict, step: int = None):
                from wsrl.common.wandb import _recursive_flatten_dict
                data_flat = _recursive_flatten_dict(data)
                data = {k: v for k, v in zip(*data_flat)}
                wandb.log(data, step=step)
            
            def save_file(self, local_path: str, run_path: str = None, policy: str = "now"):
                if run_path is None:
                    run_path = os.path.basename(local_path)
                run_dir = self.run.dir or os.getcwd()
                dst_path = os.path.join(run_dir, run_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(local_path, dst_path)
                wandb.save(dst_path, base_path=run_dir, policy=policy)
                return dst_path
        
        wandb_logger = ResumedWandBLogger(wandb.run, wandb_config)
    else:
        wandb_logger = WandBLogger(
            wandb_config=wandb_config,
            variant=variant,
            random_str_in_identifier=True,
            disable_online_logging=FLAGS.debug,
        )
    
    # Save wandb run ID for future resumption
    current_wandb_run_id = None
    if hasattr(wandb_logger, 'run') and wandb_logger.run is not None:
        current_wandb_run_id = wandb_logger.run.id
        logging.info(f"Wandb run ID: {current_wandb_run_id}")
    else:
        logging.warning("Could not get wandb run ID")
    
    logging.info("=" * 80)
    logging.info(f"EXPERIMENT CONFIGURATION")
    logging.info("=" * 80)
    logging.info(f"Project: {wandb_config['project']}")
    logging.info(f"Group: {final_group}")
    logging.info(f"Run Name: {exp_descriptor}")
    logging.info(f"Seed: {FLAGS.seed}")
    if FLAGS.checkpoint_dir:
        logging.info(f"Preemption Checkpoint: {FLAGS.checkpoint_dir}")
        logging.info(f"Checkpoint Interval: {FLAGS.checkpoint_interval} steps")
        if resumed_step > 0:
            logging.info(f"Resuming from: Step {resumed_step}")
    logging.info("=" * 80)

    save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )
    
    # Initialize Orbax checkpoint manager for saving
    checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=3)
    ckpt_manager = ocp.CheckpointManager(save_dir, options=checkpoint_options)

    """
    Create Brax environments (pure Brax, no Gym wrapper)
    """
    logging.info(f"Creating Brax environment: {FLAGS.brax_env} with backend {FLAGS.brax_backend}")
    
    # Create base env (for native evaluation) and wrapped env (for training)
    brax_base_env, brax_train_env = make_brax_training_env(
        env_name=FLAGS.brax_env,
        backend=FLAGS.brax_backend,
        episode_length=FLAGS.brax_episode_length,
        mjcf_path=FLAGS.brax_mjcf_path if FLAGS.brax_mjcf_path else None,
        wandb_logger=wandb_logger
    )
    
    obs_dim = brax_base_env.observation_size
    action_dim = brax_base_env.action_size
    logging.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    """
    Create dummy dataset for agent initialization
    """
    logging.info("Creating dummy dataset for agent initialization")
    dataset = create_dummy_dataset_from_brax_env(
        env_name=FLAGS.brax_env,
        backend=FLAGS.brax_backend,
        num_samples=FLAGS.batch_size,
        seed=FLAGS.seed,
    )

    """
    Replay buffer (simple, no Gym dependency)
    """
    replay_buffer = SimpleReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=FLAGS.replay_buffer_capacity,
        seed=FLAGS.seed,
    )
    
    # Restore replay buffer from checkpoint if available
    if resumed_replay_buffer_state is not None:
        logging.info("Restoring replay buffer from checkpoint...")
        try:
            buffer_size = resumed_replay_buffer_state['size']
            replay_buffer._observations[:buffer_size] = resumed_replay_buffer_state['observations']
            replay_buffer._next_observations[:buffer_size] = resumed_replay_buffer_state['next_observations']
            replay_buffer._actions[:buffer_size] = resumed_replay_buffer_state['actions']
            replay_buffer._rewards[:buffer_size] = resumed_replay_buffer_state['rewards']
            replay_buffer._masks[:buffer_size] = resumed_replay_buffer_state['masks']
            replay_buffer._dones[:buffer_size] = resumed_replay_buffer_state['dones']
            replay_buffer._size = buffer_size
            replay_buffer._insert_index = resumed_replay_buffer_state['insert_index']
            logging.info(f"Replay buffer restored with {buffer_size} transitions")
        except Exception as e:
            logging.warning(f"Failed to restore replay buffer: {e}. Starting with empty buffer.")

    """
    Initialize agent
    """
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    example_batch = subsample_batch(dataset, FLAGS.batch_size)
    
    # Update config with Brax-specific hidden dims
    FLAGS.config.agent_kwargs.policy_network_kwargs.hidden_dims = brax_hidden_dims
    FLAGS.config.agent_kwargs.critic_network_kwargs.hidden_dims = (256, 256)
    logging.info(f"Using hidden dims {brax_hidden_dims} for policy")
    
    agent = agents[FLAGS.agent].create_brax_compatible(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        critic_network_kwargs=FLAGS.config.agent_kwargs['critic_network_kwargs'],
        policy_network_kwargs=FLAGS.config.agent_kwargs['policy_network_kwargs'],
        policy_kwargs=FLAGS.config.agent_kwargs.get('policy_kwargs', {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        }),
        critic_ensemble_size=FLAGS.config.agent_kwargs.get('critic_ensemble_size', 2),
        critic_subsample_size=FLAGS.config.agent_kwargs.get('critic_subsample_size', None),
        temperature_init=FLAGS.config.agent_kwargs.get('temperature_init', 1.0),
        **{k: v for k, v in FLAGS.config.agent_kwargs.items() 
            if k not in ['critic_network_kwargs', 'policy_network_kwargs', 
                        'policy_kwargs', 'critic_ensemble_size', 
                        'critic_subsample_size', 'temperature_init']},
    )

    # Load wsrl checkpoint if specified (for transfer learning, not resumption)
    if FLAGS.resume_path != "":
        assert os.path.exists(FLAGS.resume_path), "resume path does not exist"
        restore_path = os.path.join(FLAGS.resume_path, "default")
        if os.path.exists(restore_path):
            agent = ocp.StandardCheckpointer().restore(restore_path, target=agent)
        else:
            agent = ocp.StandardCheckpointer().restore(FLAGS.resume_path, target=agent)
        logging.info(f"Restored agent from {FLAGS.resume_path}")
    
    # Restore from preemption checkpoint if it exists (overrides resume_path)
    if resumed_agent_state is not None:
        logging.info("Restoring agent state from preemption checkpoint...")
        agent = resumed_agent_state
        logging.info(f"Agent restored from step {resumed_step}")

    """
    Load Hypernet and generate parameters (if specified and used for init improvement)
    """
    hypernet_loader = None
    
    if FLAGS.hypernet_improve_init and FLAGS.hypernet_config_path != "" and FLAGS.hypernet_ckpt_path != "":
        # Warn if both hypernet and Brax checkpoint are specified
        if FLAGS.brax_ckpt_path != "":
            logging.warning(
                "Both hypernet and Brax checkpoint specified. "
                "Brax checkpoint will override hypernet-generated parameters."
            )
        
        logging.info("=" * 80)
        logging.info("HYPERNET INITIALIZATION")
        logging.info("=" * 80)
        
        # Load hypernet
        hypernet_loader = HypernetLoader(
            config_path=FLAGS.hypernet_config_path,
            checkpoint_path=FLAGS.hypernet_ckpt_path,
            checkpoint_step=FLAGS.hypernet_ckpt_step,
            seed=FLAGS.seed,
            ckpt_axis_size=FLAGS.resume_ckpt_axis_size,
        )
        
        logging.info("=" * 80)

    """
    Load Brax checkpoint (normalizer, policy, and optionally Q-network)
    """
    brax_normalizer = None
    brax_normalizer_params = None
    brax_q_network_params = None
    
    if FLAGS.brax_ckpt_path != "":
        logging.info(f"Loading Brax checkpoint from: {FLAGS.brax_ckpt_path}")
        
        # Load Brax SAC checkpoint (with Q-network if requested)
        brax_normalizer_params, brax_policy_params, brax_eval_metrics, brax_q_network_params = load_brax_sac_checkpoint(
            FLAGS.brax_ckpt_path,
            checkpoint_idx=FLAGS.brax_ckpt_idx,
            load_q_network=True,  # Always try to load Q-network for evaluation
        )
        
        if brax_eval_metrics is not None:
            stored_rewards = brax_eval_metrics.get('eval/episode_reward', None)
            if stored_rewards is not None:
                logging.info(f"Brax checkpoint stored return: {stored_rewards:.2f}")
        
        # Create normalizer for observation preprocessing
        brax_normalizer = BraxNormalizer.from_brax_params(brax_normalizer_params)
        logging.info("Using Brax normalizer for observation preprocessing")
        
        # Load Brax policy into wsrl agent
        logging.info("Loading Brax policy into wsrl agent's actor...")
        agent = load_brax_policy_to_wsrl_agent(
            agent,
            brax_policy_params,
            brax_normalizer_params=brax_normalizer_params,
            absorb_normalizer=False,  # Don't absorb, we'll use normalizer separately
        )
        logging.info("Successfully loaded Brax policy parameters")
        
        logging.info("Loading Brax Q-network into wsrl agent's critic...")
        agent = load_brax_q_network_to_wsrl_agent(
            agent,
            brax_q_network_params,
            critic_ensemble_size=FLAGS.config.agent_kwargs['critic_ensemble_size'],
        )
        logging.info("Successfully loaded Brax Q-network parameters")
        
        # Verify that loaded weights match Brax policy/Q-network exactly
        logging.info("Verifying loaded weights match Brax policy/Q-network...")
        prev_matmul_precision = jax.config.jax_default_matmul_precision
        jax.config.update("jax_default_matmul_precision", "float32")
        _ = verify_brax_wsrl_equivalence(
            brax_env=brax_base_env,
            brax_policy_params=brax_policy_params,
            brax_q_params=brax_q_network_params,
            brax_normalizer_params=brax_normalizer_params,
            wsrl_agent=agent,
            num_steps=100,
            num_envs=FLAGS.num_envs,
            seed=FLAGS.seed,
            tolerance=2e-3,
        )
        jax.config.update("jax_default_matmul_precision", prev_matmul_precision)
        logging.info("Verification passed!")

    """
    Training loop
    """
    timer = Timer()
    
    # Start from resumed step if available, otherwise from agent's step
    if resumed_step > 0:
        step = resumed_step
        logging.info(f"Resuming training from step {step}")
    else:
        step = int(agent.state.step)  # 0 for new agents, or load from pre-trained
    
    # Initialize environment state
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, FLAGS.num_envs)
    env_state = brax_train_env.reset(reset_keys)
    env_step = jax.jit(brax_train_env.step)

    """
    Use hypernet to improve initial policy (if requested)
    """
    if FLAGS.hypernet_improve_init:
        # Evaluate current policy first
        logging.info("Evaluating initial policy before hypernet improvement...")
        initial_eval = evaluate_and_log_brax(
            agent, 
            brax_normalizer_params, 
            brax_base_env, 
            step,
            brax_hidden_dims,
            FLAGS.seed,
            num_eval_envs=FLAGS.brax_num_eval_envs,
            episode_length=200,
            wandb_logger=None,
        )
        current_reward = initial_eval['brax_native_return']
        target_reward = FLAGS.hypernet_target_reward

        full_eval_res = evaluate_and_log_brax(
            agent, 
            brax_normalizer_params, 
            brax_base_env, 
            step,
            brax_hidden_dims,
            FLAGS.seed,
            num_eval_envs=FLAGS.brax_num_eval_envs,
            episode_length=1000,
            wandb_logger=wandb_logger,
        )
        logging.info(f"Full evaluation result: {full_eval_res}")
        
        # Improve policy with hypernet
        agent = improve_policy_with_hypernet(
            current_agent=agent,
            hypernet_loader=hypernet_loader,
            brax_normalizer_params=brax_normalizer_params,
            current_rewards=current_reward,
            target_rewards=target_reward,
            seed=FLAGS.seed,
            loss_cond=None
        )
        
        # Evaluate improved policy
        logging.info("Evaluating improved policy after hypernet...")
        improved_eval = evaluate_and_log_brax(
            agent, 
            brax_normalizer_params, 
            brax_base_env, 
            step,
            brax_hidden_dims,
            FLAGS.seed,
            num_eval_envs=FLAGS.brax_num_eval_envs,
            episode_length=200,
            wandb_logger=None,
        )
        improved_reward = improved_eval['brax_native_return']
        logging.info(f"Reward improvement: {current_reward:.2f} → "
        f"{improved_reward:.2f} (Δ={improved_reward-current_reward:+.2f})")

    logging.info(f"Starting online training for {FLAGS.num_online_steps} steps")

    for _ in tqdm.tqdm(range(step, FLAGS.num_online_steps)):
        timer.tick("total")

        """
        Env Step
        """
        with timer.context("env step"):
            rng, action_rng = jax.random.split(rng)
            
            # Get current observation (shape: [num_envs, obs_dim])
            obs = env_state.obs
            
            # Normalize observation for policy if using Brax normalizer
            if brax_normalizer is not None:
                policy_obs = brax_normalizer(obs)
            else:
                policy_obs = obs
            
            # Sample action
            action = agent.sample_actions(policy_obs, seed=action_rng)
            action = jnp.clip(action, -FLAGS.clip_action, FLAGS.clip_action)
            
            # Step environment
            next_env_state = env_step(env_state, action)
            
            # Scale reward
            reward = next_env_state.reward * FLAGS.reward_scale + FLAGS.reward_bias
            
            # Store transitions in replay buffer (convert to numpy)
            transitions = {
                "observations": np.array(obs),
                "next_observations": np.array(next_env_state.obs),
                "actions": np.array(action),
                "rewards": np.array(reward),
                "masks": np.array(1.0 - next_env_state.done),
                "dones": np.array(next_env_state.done),
            }
            
            # Insert transitions (handles both single env and batched)
            if FLAGS.num_envs == 1:
                # Squeeze batch dimension for single env
                for k, v in transitions.items():
                    transitions[k] = v.squeeze(0) if v.ndim > 1 or (v.ndim == 1 and 
                    k in ["rewards", "masks", "dones"]) else v[0]
                replay_buffer.insert(transitions)
            else:
                replay_buffer.insert_batch(transitions)
            
            env_state = next_env_state

        """
        Updates
        """
        with timer.context("update"):
            if step <= max(FLAGS.warmup_steps, min_steps_to_update):
                # No updates during warmup
                pass
            else:
                # Sample batch from replay buffer
                batch = replay_buffer.sample(FLAGS.batch_size)
                
                # Normalize observations if using Brax normalizer
                if brax_normalizer is not None:
                    batch = batch.copy()
                    # Convert to JAX arrays once, then normalize (more efficient)
                    obs = jnp.array(batch["observations"])
                    next_obs = jnp.array(batch["next_observations"])
                    batch["observations"] = brax_normalizer(obs)
                    batch["next_observations"] = brax_normalizer(next_obs)
                    # Convert other batch items to JAX if needed
                    for key in ["actions", "rewards", "masks", "dones"]:
                        if key in batch and not isinstance(batch[key], jnp.ndarray):
                            batch[key] = jnp.array(batch[key])

                # Update agent
                if FLAGS.utd > 1:
                    agent, update_info = agent.update_high_utd(
                        batch,
                        utd_ratio=FLAGS.utd,
                    )
                else:
                    agent, update_info = agent.update(batch)

        """
        Advance Step
        """
        step += 1

        """
        Evaluation
        """
        eval_steps = (1, FLAGS.num_online_steps)  # First and last step
        if step % FLAGS.eval_interval == 0 or step in eval_steps:
            logging.info("Evaluating...")
            with timer.context("evaluation"):
                eval_info = evaluate_and_log_brax(
                    agent,
                    brax_normalizer_params,
                    brax_base_env,
                    step,
                    brax_hidden_dims,
                    FLAGS.seed,
                    num_eval_envs=FLAGS.brax_num_eval_envs,
                    episode_length=FLAGS.brax_episode_length,
                    wandb_logger=wandb_logger
                )
                if eval_info:
                    logging.info(f"Step {step}: brax_return={eval_info['brax_native_return']:.2f}, brax_length={eval_info['brax_native_length']:.2f}")

        """
        Save Checkpoint
        """
        if step % FLAGS.save_interval == 0:
            logging.info("Saving checkpoint...")
            ckpt_manager.save(step, args=ocp.args.StandardSave(agent))
            logging.info(f"Saved checkpoint at step {step} to {save_dir}")
        
        """
        Save Preemption-Safe Checkpoint (for auto-resume after TPU preemption)
        """
        if FLAGS.checkpoint_dir and step % FLAGS.checkpoint_interval == 0:
            save_preemption_checkpoint(
                FLAGS.checkpoint_dir,
                agent,
                step,
                current_wandb_run_id,
                replay_buffer=None,  # Set to replay_buffer to save buffer state (increases checkpoint size)
                ckpt_manager=preemption_ckpt_manager
            )

        timer.tock("total")

        """
        Logging
        """
        if step % FLAGS.log_interval == 0:
            if "update_info" in locals():
                update_info = jax.device_get(update_info)
                wandb_logger.log({"training": update_info}, step=step)

            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

    # Final save and wait for checkpoint completion
    logging.info("Training complete. Saving final checkpoint...")
    ckpt_manager.save(step, args=ocp.args.StandardSave(agent))
    ckpt_manager.wait_until_finished()
    logging.info(f"Final checkpoint saved to {save_dir}")
    
    # Save final preemption checkpoint
    if FLAGS.checkpoint_dir:
        logging.info("Saving final preemption checkpoint...")
        save_preemption_checkpoint(
            FLAGS.checkpoint_dir,
            agent,
            step,
            current_wandb_run_id,
            replay_buffer=None,
            ckpt_manager=preemption_ckpt_manager
        )
        logging.info(f"Final preemption checkpoint saved to {FLAGS.checkpoint_dir}")


if __name__ == "__main__":
    # JAX configuration for distributed training
    torch.multiprocessing.set_start_method('spawn', force=True)

    # this setting is supposed to put here to stop torch from locking open files
    # otherwise, the number of open files will exceed the limit
    torch.multiprocessing.set_sharing_strategy('file_system')

    jax.distributed.initialize()

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    app.run(main)
