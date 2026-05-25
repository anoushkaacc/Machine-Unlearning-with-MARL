import os
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import maddpg.common.tf_util as U
from run_baseline import get_trainers, make_env, set_global_seeds

add_mpe_path_done = False
if not add_mpe_path_done:
    from run_baseline import add_mpe_to_path

    add_mpe_to_path()
    add_mpe_path_done = True

from maddpg.trainer.maddpg import (
    _get_critic_params,
    _set_critic_params,
    compute_critic_loss,
    critic_unlearn,
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
                "target_q": float(rew_n[0]),
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
        target_q = np.array([r["target_q"] for r in subset], dtype=np.float32)
        return {"obs_n": obs_n, "act_n": act_n, "target_q": target_q}

    return pack("malicious"), pack("retain")


def test_critic_unlearn_increases_forget_loss_and_preserves_retain():
    args = build_arglist()
    set_global_seeds(123)

    with U.single_threaded_session():
        env = make_env("simple_spread")
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, num_adversaries=0, obs_shape_n=obs_shape_n, arglist=args)
        U.initialize()
        agent = trainers[0]

        D_f, D_r = _build_split_batches(trainers, env)
        q_vars, phi_init = _get_critic_params(agent)
        base_forget = compute_critic_loss(agent, D_f)
        base_retain = compute_critic_loss(agent, D_r)

        success = False
        tried = []
        for damping in [1e-3, 1e-2, 1e-1, 1.0]:
            _set_critic_params(agent, q_vars, phi_init)
            critic_unlearn(agent, D_f, eta=0.01, damping=damping, cg_iters=10, hvp_eps=1e-3)
            forget_after = compute_critic_loss(agent, D_f)
            retain_after = compute_critic_loss(agent, D_r)
            retain_rel = abs(retain_after - base_retain) / max(abs(base_retain), 1e-6)
            tried.append((damping, base_forget, forget_after, base_retain, retain_after, retain_rel))

            if forget_after > base_forget and retain_rel < 0.25:
                success = True
                break

        if not success:
            raise AssertionError("Unlearning criteria not met after damping sweep: {}".format(tried))

        print("PASS: critic_unlearn increased D_f loss while keeping D_r similar (with damping adjustment).")


if __name__ == "__main__":
    test_critic_unlearn_increases_forget_loss_and_preserves_retain()
