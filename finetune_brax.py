"""
Finetuning script for Brax environments with wsrl.
Uses pure Brax (no Gym wrapper) for faster vectorized training.

Note: To use Q-network loading, the Brax checkpoint must be saved with save_q_network=True
in the train_sac_brax.py script.
"""

import os
import warnings

# Suppress noisy runtime logs before importing JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logging (ERROR only)
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC warnings

# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Optional
from etils import epath
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
from wsrl.utils.train_utils import subsample_batch

# Brax-specific imports
from wsrl.envs.brax_dataset import (
    create_dummy_dataset_from_brax_env,
    load_brax_sac_checkpoint,
)
from wsrl.utils.brax_utils import (
    BraxNormalizer,
    load_brax_q_network_to_wsrl_agent,
    load_brax_policy_to_wsrl_agent,
    convert_wsrl_actor_to_brax_policy,
    verify_brax_wsrl_equivalence,
)
from wsrl.common.brax_evaluation import evaluate_brax_native

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
flags.DEFINE_integer("log_interval", 5_000, "Log every n steps")
flags.DEFINE_integer("eval_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("save_interval", 100_000, "Save every n steps.")
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


# ============================================================================
# Simple Replay Buffer (no Gym dependency)
# ============================================================================

class SimpleReplayBuffer:
    """Simple replay buffer that doesn't depend on Gym spaces."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        seed: int = 0,
    ):
        self._capacity = capacity
        self._size = 0
        self._insert_index = 0
        self._rng = np.random.default_rng(seed)
        
        # Pre-allocate arrays
        self._observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._masks = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
    
    def __len__(self):
        return self._size
    
    def insert(self, transition: dict):
        """Insert a single transition."""
        idx = self._insert_index
        self._observations[idx] = transition["observations"]
        self._next_observations[idx] = transition["next_observations"]
        self._actions[idx] = transition["actions"]
        self._rewards[idx] = transition["rewards"]
        self._masks[idx] = transition["masks"]
        self._dones[idx] = transition["dones"]
        
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def insert_batch(self, transitions: dict):
        """Insert a batch of transitions."""
        batch_size = len(transitions["observations"])
        for i in range(batch_size):
            self.insert({
                "observations": transitions["observations"][i],
                "next_observations": transitions["next_observations"][i],
                "actions": transitions["actions"][i],
                "rewards": transitions["rewards"][i],
                "masks": transitions["masks"][i],
                "dones": transitions["dones"][i],
            })
    
    def sample(self, batch_size: int) -> dict:
        """Sample a random batch."""
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observations": self._observations[indices],
            "next_observations": self._next_observations[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "masks": self._masks[indices],
            "dones": self._dones[indices],
        }


# ============================================================================
# Brax Environment Helper
# ============================================================================

def make_brax_training_env(
    env_name: str,
    backend: str = "generalized",
    episode_length: int = 1000,
    mjcf_path: Optional[str] = None,
    wandb_logger: Optional[WandBLogger] = None,
):
    """Create a Brax environment wrapped for training."""
    if "s2r" in env_name:
        if mjcf_path is None:
            mjcf_path = os.path.join(epath.resource_path('data_gen'), f"assets/mjcf/{env_name}.xml")
        else:
            mjcf_path = mjcf_path
        if wandb_logger is not None:
            wandb_logger.save_file(mjcf_path, f"mjcf/{env_name}.xml")
        base_env = brax_envs.get_environment(
            env_name=env_name,
            backend="generalized",
            mjcf_path=mjcf_path
        )
    else:
        base_env = brax_envs.get_environment(env_name=env_name, backend=backend)
    wrapped_env = brax_envs.training.wrap(
        base_env,
        episode_length=episode_length,
        action_repeat=1,
    )
    return base_env, wrapped_env


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
        ckpt_id = FLAGS.brax_ckpt_idx if FLAGS.brax_ckpt_idx >= 0 else "latest"
        group_parts.append(f"ckpt{ckpt_id}")
    
    # Add hypernet indicator if using hypernet
    if FLAGS.hypernet_config_path and FLAGS.hypernet_ckpt_path:
        hypernet_tag = "hypernet"
        if FLAGS.hypernet_improve_init:
            hypernet_tag += "_init"
        group_parts.append(hypernet_tag)
    
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
    
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=variant,
        random_str_in_identifier=True,
        disable_online_logging=FLAGS.debug,
    )
    
    logging.info("=" * 80)
    logging.info(f"EXPERIMENT CONFIGURATION")
    logging.info("=" * 80)
    logging.info(f"Project: {wandb_config['project']}")
    logging.info(f"Group: {final_group}")
    logging.info(f"Run Name: {exp_descriptor}")
    logging.info(f"Seed: {FLAGS.seed}")
    logging.info("=" * 80)

    save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )
    
    # Initialize Orbax checkpoint manager for saving
    checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=30)
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

    # Load wsrl checkpoint if specified
    if FLAGS.resume_path != "":
        assert os.path.exists(FLAGS.resume_path), "resume path does not exist"
        restore_path = os.path.join(FLAGS.resume_path, "default")
        if os.path.exists(restore_path):
            agent = ocp.StandardCheckpointer().restore(restore_path, target=agent)
        else:
            agent = ocp.StandardCheckpointer().restore(FLAGS.resume_path, target=agent)
        logging.info(f"Restored agent from {FLAGS.resume_path}")

    """
    Load Hypernet and generate parameters (if specified)
    """
    hypernet_loader = None
    
    if FLAGS.hypernet_config_path != "" and FLAGS.hypernet_ckpt_path != "":
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
    Helper function to use hypernet for parameter improvement
    """
    def improve_policy_with_hypernet(
        current_agent,
        hypernet_loader,
        brax_normalizer_params,
        current_rewards,
        target_rewards,
        loss_cond=None
    ):
        """Use hypernet to generate improved policy parameters."""
        if hypernet_loader is None:
            return current_agent
        
        logging.info("=" * 80)
        logging.info("IMPROVING POLICY WITH HYPERNET")
        logging.info("=" * 80)
        logging.info(f"Current rewards: {current_rewards:.2f}")
        logging.info(f"Target rewards: {target_rewards:.2f}")
        
        # Extract current policy parameters from agent
        wsrl_actor_params = current_agent.state.params['modules_actor']
        current_brax_policy_params = convert_wsrl_actor_to_brax_policy(wsrl_actor_params)
        
        # Generate improved parameters with hypernet
        improved_brax_policy_params = hypernet_loader.generate_parameters(
            policy_normalizer=brax_normalizer_params,
            policy_params=current_brax_policy_params,
            current_rewards=current_rewards,
            target_rewards=target_rewards,
            loss_cond=loss_cond,
            seed=FLAGS.seed,
        )
        
        # Load improved parameters into agent
        improved_agent = load_brax_policy_to_wsrl_agent(
            current_agent,
            improved_brax_policy_params,
            brax_normalizer_params=brax_normalizer_params,
            absorb_normalizer=False,
        )
        
        logging.info("✓ Successfully generated and loaded improved parameters!")
        logging.info("=" * 80)
        
        return improved_agent
    
    """
    Evaluation function using native Brax
    """
    def evaluate_and_log(
        wsrl_agent, 
        normalizer_params, 
        env, 
        step_number, 
        wandb_logger=None, 
        episode_length=1000
    ):
        # Extract policy params from agent and convert to Brax format
        wsrl_actor_params = wsrl_agent.state.params['modules_actor']
        current_brax_policy_params = convert_wsrl_actor_to_brax_policy(wsrl_actor_params)
        
        brax_metrics = evaluate_brax_native(
            env,
            current_brax_policy_params,
            normalizer_params,
            hidden_layer_sizes=tuple(brax_hidden_dims),
            seed=FLAGS.seed,
            episode_length=episode_length,
            num_eval_envs=FLAGS.brax_num_eval_envs,
        )
        
        eval_info = {
            "brax_native_return": brax_metrics["episode_reward"],
            "brax_native_length": brax_metrics["episode_length"],
        }
        if wandb_logger is not None:
            wandb_logger.log({"evaluation": eval_info}, step=step_number)
        return eval_info

    """
    Training loop
    """
    timer = Timer()
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
        initial_eval = evaluate_and_log(
            agent, 
            brax_normalizer_params, 
            brax_base_env, 
            step, 
            wandb_logger=None, 
            episode_length=200
        )
        current_reward = initial_eval['brax_native_return']
        target_reward = FLAGS.hypernet_target_reward

        full_eval_res = evaluate_and_log(
            agent, 
            brax_normalizer_params, 
            brax_base_env, 
            step, 
            wandb_logger=wandb_logger, 
            episode_length=1000
        )
        logging.info(f"Full evaluation result: {full_eval_res}")
        
        # Improve policy with hypernet
        agent = improve_policy_with_hypernet(
            current_agent=agent,
            hypernet_loader=hypernet_loader,
            brax_normalizer_params=brax_normalizer_params,
            current_rewards=current_reward,
            target_rewards=target_reward,
            loss_cond=None
        )
        
        # Evaluate improved policy
        logging.info("Evaluating improved policy after hypernet...")
        improved_eval = evaluate_and_log(
            agent, 
            brax_normalizer_params, 
            brax_base_env, 
            step, 
            wandb_logger=None, 
            episode_length=200
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
                eval_info = evaluate_and_log(agent, brax_normalizer_params, brax_base_env, step, wandb_logger)
                if eval_info:
                    logging.info(f"Step {step}: brax_return={eval_info['brax_native_return']:.2f}, brax_length={eval_info['brax_native_length']:.2f}")

        """
        Save Checkpoint
        """
        if step % FLAGS.save_interval == 0:
            logging.info("Saving checkpoint...")
            ckpt_manager.save(step, args=ocp.args.StandardSave(agent))
            logging.info(f"Saved checkpoint at step {step} to {save_dir}")

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
