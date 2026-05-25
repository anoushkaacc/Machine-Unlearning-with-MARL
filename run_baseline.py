import argparse
import csv
import os
import pickle
import random
import sys
import time

import numpy as np
import tensorflow.compat.v1 as tf

import maddpg.common.tf_util as U

tf.disable_v2_behavior()
os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")


def parse_args():
    parser = argparse.ArgumentParser("MADDPG baseline training on MPE simple_spread")
    parser.add_argument("--scenario", type=str, default="simple_spread", help="MPE scenario name")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes to run")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy for adversaries")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="batch size")
    parser.add_argument("--num-units", type=int, default=64, help="number of hidden units")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/baseline/", help="checkpoint directory")
    parser.add_argument("--save-rate", type=int, default=1000, help="save every N episodes")
    parser.add_argument("--log-dir", type=str, default="./logs/", help="directory for reward logs")
    parser.add_argument("--log-file", type=str, default="baseline_rewards.csv", help="episode reward CSV filename")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    return parser.parse_args()


def mlp_model(input_tensor, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    del rnn_cell
    with tf.variable_scope(scope, reuse=reuse):
        out = input_tensor
        out = fully_connected(out, num_outputs=num_units, scope="fc1", activation_fn=tf.nn.relu)
        out = fully_connected(out, num_outputs=num_units, scope="fc2", activation_fn=tf.nn.relu)
        out = fully_connected(out, num_outputs=num_outputs, scope="fc3", activation_fn=None)
        return out


def add_mpe_to_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(script_dir, "..", "multiagent-particle-envs-master")),
        os.path.normpath(os.path.join(script_dir, "..", "multiagent-particle-envs")),
        os.path.normpath(os.path.join(script_dir, "multiagent-particle-envs-master")),
        os.path.normpath(os.path.join(script_dir, "multiagent-particle-envs")),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            break


def fully_connected(input_tensor, num_outputs, scope, activation_fn=None):
    input_dim = input_tensor.get_shape().as_list()[-1]
    if input_dim is None:
        raise ValueError("Last input dimension must be known for fully_connected layer.")
    with tf.variable_scope(scope):
        weights = tf.get_variable(
            "weights",
            shape=[input_dim, num_outputs],
            initializer=tf.glorot_uniform_initializer(),
        )
        biases = tf.get_variable(
            "biases",
            shape=[num_outputs],
            initializer=tf.zeros_initializer(),
        )
        out = tf.matmul(input_tensor, weights) + biases
        if activation_fn is not None:
            out = activation_fn(out)
        return out


def set_global_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if hasattr(tf, "set_random_seed"):
        tf.set_random_seed(seed)
    elif hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
        tf.random.set_seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_env(env, seed):
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except TypeError:
            pass

    if hasattr(env, "action_space"):
        for action_space in env.action_space:
            if hasattr(action_space, "seed"):
                action_space.seed(seed)

    if hasattr(env, "observation_space"):
        for obs_space in env.observation_space:
            if hasattr(obs_space, "seed"):
                obs_space.seed(seed)

    if hasattr(env, "world"):
        if hasattr(env.world, "seed"):
            env.world.seed(seed)
        if hasattr(env.world, "np_random"):
            env.world.np_random = np.random.RandomState(seed)


def make_env(scenario_name):
    add_mpe_to_path()
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    return MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    from maddpg.trainer.maddpg import MADDPGAgentTrainer

    trainers = []
    for i in range(num_adversaries):
        trainers.append(
            MADDPGAgentTrainer(
                "agent_%d" % i,
                mlp_model,
                obs_shape_n,
                env.action_space,
                i,
                arglist,
                local_q_func=(arglist.adv_policy == "ddpg"),
            )
        )
    for i in range(num_adversaries, env.n):
        trainers.append(
            MADDPGAgentTrainer(
                "agent_%d" % i,
                mlp_model,
                obs_shape_n,
                env.action_space,
                i,
                arglist,
                local_q_func=(arglist.good_policy == "ddpg"),
            )
        )
    return trainers


def tag_for_episode(episode_id):
    if 2000 <= episode_id <= 3000:
        return "malicious"
    return "normal"


def save_replay_buffer(trainers, save_dir, episode_id):
    replay_path = os.path.join(save_dir, "replay_buffer_ep{}.pkl".format(episode_id))
    payload = {
        "episode": episode_id,
        "agents": [],
    }
    for idx, agent in enumerate(trainers):
        payload["agents"].append(
            {
                "agent_index": idx,
                "next_idx": agent.replay_buffer._next_idx,
                "maxsize": agent.replay_buffer._maxsize,
                "storage": agent.replay_buffer._storage,
            }
        )
    with open(replay_path, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return replay_path


def train(arglist):
    os.makedirs(arglist.save_dir, exist_ok=True)
    os.makedirs(arglist.log_dir, exist_ok=True)
    log_path = os.path.join(arglist.log_dir, arglist.log_file)

    set_global_seeds(arglist.seed)
    with U.single_threaded_session():
        env = make_env(arglist.scenario)
        seed_env(env, arglist.seed)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        saver = tf.train.Saver()
        U.initialize()

        with open(log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode", "avg_reward", "seed"])

            obs_n = env.reset()
            episode_step = 0
            episode_count = 0
            train_step = 0
            episode_total_reward = 0.0
            t_start = time.time()

            print(
                "Starting baseline training on {} for {} episodes (seed={})".format(
                    arglist.scenario, arglist.num_episodes, arglist.seed
                )
            )
            while episode_count < arglist.num_episodes:
                current_episode_id = episode_count + 1
                current_tag = tag_for_episode(current_episode_id)
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, _ = env.step(action_n)
                episode_step += 1
                terminal = episode_step >= arglist.max_episode_len

                for i, agent in enumerate(trainers):
                    agent.experience(
                        obs_n[i],
                        action_n[i],
                        rew_n[i],
                        new_obs_n[i],
                        done_n[i],
                        terminal,
                        episode_id=current_episode_id,
                        tag=current_tag,
                    )

                obs_n = new_obs_n
                step_reward = float(np.mean(rew_n))
                episode_total_reward += step_reward
                train_step += 1

                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    agent.update(trainers, train_step)

                if terminal or all(done_n):
                    episode_count += 1
                    avg_reward = episode_total_reward / float(episode_step)
                    writer.writerow([episode_count, avg_reward, arglist.seed])
                    csvfile.flush()

                    if episode_count % 100 == 0 or episode_count == 1:
                        print(
                            "episode: {}, avg_reward: {:.6f}, elapsed: {:.2f}s".format(
                                episode_count, avg_reward, time.time() - t_start
                            )
                        )

                    if episode_count % arglist.save_rate == 0:
                        U.save_state(arglist.save_dir, saver=saver)
                        print("Saved checkpoint at episode {}".format(episode_count))

                    if episode_count == 4000:
                        U.save_state(arglist.save_dir, saver=saver)
                        replay_path = save_replay_buffer(trainers, arglist.save_dir, episode_count)
                        print("Saved checkpoint at episode 4000")
                        print("Saved replay buffer to {}".format(replay_path))

                    obs_n = env.reset()
                    episode_step = 0
                    episode_total_reward = 0.0

        U.save_state(arglist.save_dir, saver=saver)
        replay_path = save_replay_buffer(trainers, arglist.save_dir, episode_count)
        print("Training complete. Final checkpoint saved to {}".format(arglist.save_dir))
        print("Final replay buffer saved to {}".format(replay_path))
        print("Episode reward log written to {}".format(log_path))


if __name__ == "__main__":
    args = parse_args()
    train(args)
