"""
Finetuning script for Brax environments with wsrl.
Note: To use Q-network loading, the Brax checkpoint must be saved with save_q_network=True
in the train_sac_brax.py script.
"""

import os
import warnings

# Suppress XLA/JAX warnings before importing JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logging (ERROR only)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't preallocate GPU memory
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC warnings

# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial

import logging as python_logging
# Reduce logging verbosity BEFORE importing JAX
python_logging.getLogger("jax").setLevel(python_logging.ERROR)
python_logging.getLogger("jax._src").setLevel(python_logging.ERROR)
python_logging.getLogger("brax").setLevel(python_logging.WARNING)
python_logging.getLogger("orbax").setLevel(python_logging.WARNING)

# Also suppress absl logging for io.py messages
import absl.logging
absl.logging.set_verbosity(absl.logging.WARNING)

import jax
# Suppress JAX info logs
jax.config.update("jax_log_compiles", False)
import numpy as np
import tqdm
from absl import app, flags, logging
import orbax.checkpoint as ocp
from ml_collections import config_flags

from wsrl.agents import agents
from wsrl.common.wandb import WandBLogger
from wsrl.data.replay_buffer import ReplayBuffer
from wsrl.utils.timer_utils import Timer
from wsrl.utils.train_utils import subsample_batch

# Brax-specific imports
from wsrl.envs.brax_wrapper import make_brax_env
from wsrl.envs.brax_dataset import (
    create_dummy_dataset_from_brax_env,
    load_brax_sac_checkpoint,
)
from wsrl.utils.brax_utils import BraxNormalizer, load_brax_q_network_to_wsrl_agent
from wsrl.common.brax_evaluation import evaluate_brax_native, evaluate_with_trajectories_jit_multi_episode

FLAGS = flags.FLAGS

# Brax environment settings
flags.DEFINE_string("brax_env", "ant", "Brax environment name (e.g., ant, halfcheetah, humanoid)")
flags.DEFINE_string("brax_backend", "generalized", "Brax backend (generalized, spring, positional, mjx)")
flags.DEFINE_integer("brax_episode_length", 1000, "Maximum episode length for Brax env")
flags.DEFINE_string("brax_ckpt_path", "", "Path to Brax SAC checkpoint to load for initialization")
flags.DEFINE_integer("brax_ckpt_idx", -1, "Checkpoint index to load (-1 for latest)")
flags.DEFINE_list("brax_hidden_dims", ["256", "256"], "Hidden layer sizes for policy network")
flags.DEFINE_integer("brax_num_eval_envs", 128, "Number of parallel envs for Brax native evaluation")
flags.DEFINE_bool("brax_native_eval", True, "Also run Brax native evaluation for comparison")
flags.DEFINE_bool("load_brax_q_network", False, "Load Q-network from Brax checkpoint for wsrl finetuning")

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


def main(_):
    """
    House keeping
    """
    brax_hidden_dims = [int(d) for d in FLAGS.brax_hidden_dims]
    
    # Minimum steps before updates (need some data in buffer)
    min_steps_to_update = FLAGS.batch_size

    """
    Wandb and logging
    """
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": FLAGS.project or "wsrl_brax",
            "group": FLAGS.group or "brax_finetune",
            "exp_descriptor": f"{FLAGS.exp_name}_{FLAGS.brax_env}_{FLAGS.agent}_seed{FLAGS.seed}",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        random_str_in_identifier=True,
        disable_online_logging=FLAGS.debug,
    )

    save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )
    
    # Initialize Orbax checkpoint manager for saving
    checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=30)
    ckpt_manager = ocp.CheckpointManager(save_dir, options=checkpoint_options)

    """
    Create Brax environments
    """
    logging.info(f"Creating Brax environment: {FLAGS.brax_env} with backend {FLAGS.brax_backend}")
    
    # Import brax for base environment (needed for native evaluation)
    from brax import envs as brax_envs
    
    # Keep reference to base Brax env for native evaluation
    brax_base_env = brax_envs.get_environment(
        env_name=FLAGS.brax_env, 
        backend=FLAGS.brax_backend
    )
    
    # Create Gym-wrapped Brax environments
    finetune_env = make_brax_env(
        env_name=FLAGS.brax_env,
        backend=FLAGS.brax_backend,
        episode_length=FLAGS.brax_episode_length,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        scale_and_clip_action=True,
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed,
    )
    eval_env = make_brax_env(
        env_name=FLAGS.brax_env,
        backend=FLAGS.brax_backend,
        episode_length=FLAGS.brax_episode_length,
        reward_scale=1.0,  # No scaling for evaluation
        reward_bias=0.0,
        scale_and_clip_action=True,
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed + 1000,
    )

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
    Replay buffer
    """
    replay_buffer = ReplayBuffer(
        finetune_env.observation_space,
        finetune_env.action_space,
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
    FLAGS.config.agent_kwargs.critic_network_kwargs.hidden_dims = brax_hidden_dims
    logging.info(f"Using hidden dims {brax_hidden_dims} for policy")
    
    agent = agents[FLAGS.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        encoder_def=None,
        **FLAGS.config.agent_kwargs,
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
    Load Brax checkpoint (normalizer, policy, and optionally Q-network)
    """
    brax_normalizer = None
    brax_normalizer_params = None
    brax_policy_params = None
    brax_q_network_params = None
    
    if FLAGS.brax_ckpt_path != "":
        logging.info(f"Loading Brax checkpoint from: {FLAGS.brax_ckpt_path}")
        
        # Load Brax SAC checkpoint (with Q-network if requested)
        brax_normalizer_params, brax_policy_params, brax_eval_metrics, brax_q_network_params = load_brax_sac_checkpoint(
            FLAGS.brax_ckpt_path,
            checkpoint_idx=FLAGS.brax_ckpt_idx,
            load_q_network=FLAGS.load_brax_q_network,
        )
        
        if brax_eval_metrics is not None:
            stored_rewards = brax_eval_metrics.get('eval/episode_reward', None)
            logging.info(f"Brax checkpoint stored return: {stored_rewards:.2f}")
        
        # Create normalizer for observation preprocessing
        brax_normalizer = BraxNormalizer.from_brax_params(brax_normalizer_params)
        logging.info("Using Brax normalizer for observation preprocessing")
        
        # Load Q-network into wsrl agent if requested and available
        if FLAGS.load_brax_q_network and brax_q_network_params is not None:
            logging.info("Loading Brax Q-network into wsrl agent's critic...")
            try:
                agent = load_brax_q_network_to_wsrl_agent(
                    agent,
                    brax_q_network_params,
                    critic_ensemble_size=FLAGS.config.agent_kwargs.get('critic_ensemble_size', 2),
                )
                logging.info("Successfully loaded Brax Q-network parameters")
            except Exception as e:
                logging.warning(f"Failed to load Q-network parameters: {e}")
                logging.warning("Continuing with randomly initialized critic")
        elif FLAGS.load_brax_q_network:
            logging.warning("load_brax_q_network=True but no Q-network params found in checkpoint. "
                          "Make sure the Brax checkpoint was saved with save_q_network=True.")

    """
    Evaluation function
    """
    def evaluate_and_log_results(
        eval_env,
        policy_fn,
        eval_func,
        step_number,
        wandb_logger,
        n_eval_trajs=FLAGS.n_eval_trajs,
    ):
        # Wrap policy_fn to apply normalization if using Brax normalizer
        if brax_normalizer is not None:
            original_policy_fn = policy_fn
            def normalized_policy_fn(obs):
                normalized_obs = brax_normalizer(obs)
                return original_policy_fn(normalized_obs)
            effective_policy_fn = normalized_policy_fn
        else:
            effective_policy_fn = policy_fn

        stats, trajs = eval_func(
            effective_policy_fn,
            eval_env,
            n_eval_trajs,
        )

        eval_info = {
            "average_return": np.mean([np.sum(t["rewards"]) for t in trajs]),
            "average_traj_length": np.mean([len(t["rewards"]) for t in trajs]),
            "std_return": np.std([np.sum(t["rewards"]) for t in trajs]),
            "min_return": np.min([np.sum(t["rewards"]) for t in trajs]),
            "max_return": np.max([np.sum(t["rewards"]) for t in trajs]),
        }
        
        # Run native Brax evaluation for comparison if enabled and checkpoint loaded
        if (FLAGS.brax_native_eval and brax_base_env is not None and 
            brax_normalizer_params is not None and brax_policy_params is not None):
            try:
                brax_native_metrics = evaluate_brax_native(
                    brax_base_env,
                    brax_policy_params,
                    brax_normalizer_params,
                    hidden_layer_sizes=tuple(brax_hidden_dims),
                    seed=FLAGS.seed,
                    episode_length=FLAGS.brax_episode_length,
                    num_eval_envs=FLAGS.brax_num_eval_envs,
                )
                eval_info["brax_native_return"] = brax_native_metrics["episode_reward"]
            except Exception as e:
                logging.warning(f"Brax native evaluation failed: {e}")

        wandb_logger.log({"evaluation": eval_info}, step=step_number)
        
        return eval_info

    """
    Training loop
    """
    timer = Timer()
    step = int(agent.state.step)  # 0 for new agents, or load from pre-trained
    observation, info = finetune_env.reset()
    done = False

    logging.info(f"Starting online training for {FLAGS.num_online_steps} steps")

    for _ in tqdm.tqdm(range(step, FLAGS.num_online_steps)):
        timer.tick("total")

        """
        Env Step
        """
        with timer.context("env step"):
            rng, action_rng = jax.random.split(rng)
            
            # Normalize observation for policy if using Brax normalizer
            if brax_normalizer is not None:
                policy_obs = brax_normalizer(observation)
            else:
                policy_obs = observation
            
            action = agent.sample_actions(policy_obs, seed=action_rng)
            next_observation, reward, done, truncated, info = finetune_env.step(action)

            # Store raw observations in replay buffer (not normalized)
            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=reward,
                masks=1.0 - done,
                dones=1.0 if (done or truncated) else 0,
            )
            replay_buffer.insert(transition)

            observation = next_observation
            if done or truncated:
                observation, info = finetune_env.reset()
                done = False

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
                    batch["observations"] = brax_normalizer(batch["observations"])
                    batch["next_observations"] = brax_normalizer(batch["next_observations"])

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
                # Split rng for evaluation (so we get fresh randomness each eval)
                rng, eval_rng = jax.random.split(rng)
                
                policy_fn = partial(
                    agent.sample_actions, argmax=FLAGS.deterministic_eval
                )
                eval_func = partial(
                    evaluate_with_trajectories_jit_multi_episode, 
                    episode_length=FLAGS.brax_episode_length,
                    rng=eval_rng,
                    clip_action=FLAGS.clip_action
                )

                eval_info = evaluate_and_log_results(
                    eval_env=eval_env,
                    policy_fn=policy_fn,
                    eval_func=eval_func,
                    step_number=step,
                    wandb_logger=wandb_logger,
                )
                logging.info(f"Step {step}: avg_return={eval_info['average_return']:.2f}")

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

    # Final save
    logging.info("Training complete. Saving final checkpoint...")
    ckpt_manager.save(step, args=ocp.args.StandardSave(agent))
    logging.info(f"Final checkpoint saved to {save_dir}")


if __name__ == "__main__":
    app.run(main)
