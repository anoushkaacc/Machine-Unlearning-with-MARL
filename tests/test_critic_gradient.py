import os
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import maddpg.common.tf_util as U
from run_baseline import add_mpe_to_path, get_trainers, make_env, set_global_seeds

add_mpe_to_path()
from maddpg.trainer.maddpg import compute_critic_gradient


def build_arglist():
    return SimpleNamespace(
        lr=1e-2,
        gamma=0.95,
        batch_size=32,
        num_units=64,
        max_episode_len=25,
        adv_policy="maddpg",
        good_policy="maddpg",
    )


def test_compute_critic_gradient_non_zero():
    args = build_arglist()
    set_global_seeds(123)

    with U.single_threaded_session():
        env = make_env("simple_spread")
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, num_adversaries=0, obs_shape_n=obs_shape_n, arglist=args)
        U.initialize()

        obs_batches = [[] for _ in range(env.n)]
        act_batches = [[] for _ in range(env.n)]
        target_q = []

        obs_n = env.reset()
        for _ in range(32):
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            new_obs_n, rew_n, done_n, _ = env.step(action_n)

            for i in range(env.n):
                obs_batches[i].append(obs_n[i])
                act_batches[i].append(action_n[i])

            # D_f target for critic; bootstrap term omitted for this focused gradient test.
            target_q.append(rew_n[0])
            obs_n = new_obs_n

            if all(done_n):
                obs_n = env.reset()

        D_f = {
            "obs_n": [np.array(x, dtype=np.float32) for x in obs_batches],
            "act_n": [np.array(x, dtype=np.float32) for x in act_batches],
            "target_q": np.array(target_q, dtype=np.float32),
        }

        grad_vec = compute_critic_gradient(trainers[0], D_f)
        grad_norm = float(np.linalg.norm(grad_vec))

        assert grad_vec.ndim == 1
        assert grad_vec.size > 0
        assert grad_norm > 0.0
        print("PASS: compute_critic_gradient returns non-zero gradient for D_f.")


if __name__ == "__main__":
    test_compute_critic_gradient_non_zero()
