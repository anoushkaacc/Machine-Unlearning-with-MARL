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

from maddpg.trainer.maddpg import (
    _get_actor_params,
    _parse_actor_transitions,
    _set_actor_params,
    actor_unlearn,
    compute_actor_loss,
)


def build_arglist():
    return SimpleNamespace(
        lr=1e-2,
        gamma=0.95,
        batch_size=32,
        num_units=64,
        max_episode_len=10,
        adv_policy="maddpg",
        good_policy="maddpg",
    )


def _build_split_batches(trainers, env, forget_episodes=(20, 30), total_episodes=50):
    records = []
    obs_n = env.reset()
    episode_step = 0
    episode_id = 1

    while episode_id <= total_episodes:
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        new_obs_n, rew_n, done_n, _ = env.step(action_n)
        episode_step += 1
        terminal = episode_step >= 10 or all(done_n)

        records.append(
            {
                "obs_n": [np.array(o, dtype=np.float32) for o in obs_n],
                "act_n": [np.array(a, dtype=np.float32) for a in action_n],
                "tag": "malicious" if forget_episodes[0] <= episode_id <= forget_episodes[1] else "retain",
            }
        )

        obs_n = new_obs_n
        if terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_id += 1

    def pack(tag):
        subset = [r for r in records if r["tag"] == tag]
        obs_n = []
        act_n = []
        for i in range(env.n):
            obs_n.append(np.array([r["obs_n"][i] for r in subset], dtype=np.float32))
            act_n.append(np.array([r["act_n"][i] for r in subset], dtype=np.float32))
        return {"obs_n": obs_n, "act_n": act_n}

    return pack("malicious"), pack("retain")


def test_actor_unlearn_small_kl_and_forget_drop():
    args = build_arglist()
    set_global_seeds(321)

    with U.single_threaded_session():
        env = make_env("simple_spread")
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, num_adversaries=0, obs_shape_n=obs_shape_n, arglist=args)
        U.initialize()
        agent = trainers[0]

        D_f, D_r = _build_split_batches(trainers, env)
        p_vars, theta_init = _get_actor_params(agent)
        base_f = compute_actor_loss(agent, D_f)
        obs_r_n, _ = _parse_actor_transitions(D_r)
        old_logits = np.array(agent.p_debug["p_values"](obs_r_n[agent.agent_index]), dtype=np.float32)

        success = False
        tried = []
        configs = [
            (2.0, 1e-3, 1e-2),
            (3.0, 1e-3, 5e-2),
            (4.0, 5e-4, 1e-1),
            (5.0, 5e-4, 2e-1),
        ]
        for lambda_forget, lr, kl_coeff in configs:
            _set_actor_params(agent, p_vars, theta_init)
            info = actor_unlearn(
                agent,
                D_f,
                D_r,
                lambda_forget=lambda_forget,
                lr=lr,
                kl_coeff=kl_coeff,
            )
            f_after = compute_actor_loss(agent, D_f)
            kl_after = float(agent.p_debug["kl_loss"](obs_r_n[agent.agent_index], old_logits))
            tried.append((lambda_forget, lr, kl_coeff, base_f, f_after, kl_after))

            if f_after > base_f and kl_after < 0.2:
                success = True
                break

        if not success:
            raise AssertionError("actor_unlearn criteria not met after sweep: {}".format(tried))

        print("PASS: actor_unlearn kept policy divergence small and reduced performance on D_f.")


if __name__ == "__main__":
    test_actor_unlearn_small_kl_and_forget_drop()
