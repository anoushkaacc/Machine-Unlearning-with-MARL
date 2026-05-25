import copy
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

from maddpg.trainer.maddpg import critic_unlearn, recovery_train_on_retain


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


def test_recovery_curve_on_retain_only():
    args = build_arglist()
    set_global_seeds(101)

    with U.single_threaded_session():
        env = make_env("simple_spread")
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, num_adversaries=0, obs_shape_n=obs_shape_n, arglist=args)
        U.initialize()
        agent = trainers[0]

        D_f, D_r = _build_split_batches(trainers, env)
        D_f_frozen = copy.deepcopy(D_f)

        critic_unlearn(agent, D_f, eta=0.01, damping=0.1, cg_iters=10, hvp_eps=1e-3)
        curve = recovery_train_on_retain(
            agent,
            D_r,
            steps=500,
            batch_size=128,
            log_every=50,
            log_path=os.path.join(PROJECT_ROOT, "logs", "recovery_curve.csv"),
            seed=101,
        )

        assert len(curve) >= 3
        assert os.path.exists(os.path.join(PROJECT_ROOT, "logs", "recovery_curve.csv"))
        assert np.isfinite(curve[-1]["critic_loss_retain"])
        assert np.isfinite(curve[-1]["actor_loss_retain"])

        for i in range(len(D_f["obs_n"])):
            assert np.array_equal(D_f["obs_n"][i], D_f_frozen["obs_n"][i])
            assert np.array_equal(D_f["act_n"][i], D_f_frozen["act_n"][i])
        assert np.array_equal(D_f["target_q"], D_f_frozen["target_q"])

        print("PASS: retain-only recovery ran for 500 steps and logged a recovery curve with frozen D_f.")


if __name__ == "__main__":
    test_recovery_curve_on_retain_only()
