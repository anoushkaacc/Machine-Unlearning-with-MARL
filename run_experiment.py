import argparse
import csv
import os
import pickle
import random
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
import tensorflow.compat.v1 as tf

import maddpg.common.tf_util as U
from run_baseline import add_mpe_to_path, get_trainers, make_env, set_global_seeds

add_mpe_to_path()
from maddpg.trainer.maddpg import (
    actor_unlearn,
    critic_unlearn,
    recovery_train_on_retain,
)

tf.disable_v2_behavior()


def parse_args():
    parser = argparse.ArgumentParser("Baseline vs full retraining vs selective unlearning")
    parser.add_argument("--scenario", type=str, default="simple_spread")
    parser.add_argument("--max-episode-len", type=int, default=25)
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--num-adversaries", type=int, default=0)
    parser.add_argument("--good-policy", type=str, default="maddpg")
    parser.add_argument("--adv-policy", type=str, default="maddpg")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-units", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument(
        "--seed-list",
        type=str,
        default="",
        help="Comma-separated explicit seeds to run, e.g. '456,789'. Overrides --num-seeds when provided.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["critic_only", "actor_only", "full_selective", "all"],
        help="Selective unlearning ablation mode. Use 'all' to run all ablations.",
    )
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--retrain-steps", type=int, default=1000, help="retain-only retraining steps (500-2000)")
    parser.add_argument("--unlearn-recovery-steps", type=int, default=1000, help="retain-only recovery steps (500-2000)")
    parser.add_argument("--lambda-forget", type=float, default=3.0)
    parser.add_argument("--save-dir", type=str, default="./experiment_outputs/")
    parser.add_argument(
        "--selective-only",
        action="store_true",
        help="Skip baseline/full-retrain and run selective unlearning using saved baseline artifacts.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="",
        help="Root dir containing seed_* baseline artifacts. Defaults to --save-dir.",
    )
    return parser.parse_args()


def _to_trainer_args(arglist):
    return SimpleNamespace(
        lr=arglist.lr,
        gamma=arglist.gamma,
        batch_size=arglist.batch_size,
        num_units=arglist.num_units,
        max_episode_len=arglist.max_episode_len,
        adv_policy=arglist.adv_policy,
        good_policy=arglist.good_policy,
    )


def _tag_for_episode(episode_id):
    if 2000 <= episode_id <= 3000:
        return "malicious"
    return "normal"


def _softmax(x):
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _set_all_seeds(seed):
    seed_i = int(seed)
    set_global_seeds(seed_i)
    random.seed(seed_i)
    np.random.seed(seed_i)
    tf.set_random_seed(seed_i)


def _reset_env(env, seed=None):
    if seed is not None:
        try:
            return env.reset(seed=int(seed))
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(int(seed))
    return env.reset()


def _retain_batch_size(arglist, D_r):
    n_samples = int(D_r["target_q"].shape[0])
    if n_samples <= 0:
        raise ValueError("D_r is empty; cannot compute retain batch size.")
    return min(int(arglist.batch_size), n_samples)


def _safe_critic_unlearn(agent, D_f):
    attempts = [
        {"eta": 0.01, "damping": 1e-2, "cg_iters": 10, "hvp_eps": 1e-3},
        {"eta": 0.005, "damping": 5e-2, "cg_iters": 10, "hvp_eps": 5e-4},
        {"eta": 0.001, "damping": 1e-1, "cg_iters": 8, "hvp_eps": 1e-4},
    ]
    last_err = None
    for cfg in attempts:
        try:
            return critic_unlearn(agent, D_f, **cfg)
        except ValueError as exc:
            last_err = exc
            print("[Selective Unlearn] critic_unlearn retry after instability: {}".format(cfg))
    print("[Selective Unlearn] Warning: critic_unlearn skipped due to repeated non-finite loss: {}".format(last_err))
    return {"skipped": True}


def _extract_tagged_datasets(trainers):
    storages = [agent.replay_buffer._storage for agent in trainers]
    n = len(trainers)
    size = len(storages[0])
    for s in storages[1:]:
        if len(s) != size:
            raise ValueError("Replay buffers are misaligned across agents.")

    malicious_idx = []
    retain_idx = []
    for i, sample in enumerate(storages[0]):
        tag = sample[6] if len(sample) > 6 else "normal"
        if tag == "malicious":
            malicious_idx.append(i)
        else:
            retain_idx.append(i)

    def build(idxes):
        obs_n = [[] for _ in range(n)]
        act_n = [[] for _ in range(n)]
        target_q = []
        for idx in idxes:
            target_q.append(float(storages[0][idx][2]))
            for agent_i in range(n):
                obs_n[agent_i].append(np.array(storages[agent_i][idx][0], copy=False))
                act_n[agent_i].append(np.array(storages[agent_i][idx][1], copy=False))
        return {
            "obs_n": [np.array(x, dtype=np.float32) for x in obs_n],
            "act_n": [np.array(x, dtype=np.float32) for x in act_n],
            "target_q": np.array(target_q, dtype=np.float32),
        }

    return build(malicious_idx), build(retain_idx)


def _fallback_forget_split(D_r, forget_fraction=0.1):
    n = int(D_r["target_q"].shape[0])
    if n <= 1:
        raise ValueError("Not enough samples to build fallback forget/retain split.")
    k = max(1, int(n * forget_fraction))
    idx_f = np.arange(0, k)
    idx_r = np.arange(k, n)
    if idx_r.size == 0:
        idx_r = np.arange(0, n - 1)
        idx_f = np.array([n - 1])

    D_f = {
        "obs_n": [obs[idx_f] for obs in D_r["obs_n"]],
        "act_n": [act[idx_f] for act in D_r["act_n"]],
        "target_q": D_r["target_q"][idx_f],
    }
    D_r_new = {
        "obs_n": [obs[idx_r] for obs in D_r["obs_n"]],
        "act_n": [act[idx_r] for act in D_r["act_n"]],
        "target_q": D_r["target_q"][idx_r],
    }
    return D_f, D_r_new


def _evaluate_avg_reward(env, trainers, num_episodes, max_episode_len, eval_seed=None):
    episode_rewards = []
    for ep in range(1, num_episodes + 1):
        episode_seed = None if eval_seed is None else int(eval_seed) + ep - 1
        obs_n = _reset_env(env, seed=episode_seed)
        total = 0.0
        steps = 0
        while steps < max_episode_len:
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            obs_n, rew_n, done_n, _ = env.step(action_n)
            total += float(np.mean(rew_n))
            steps += 1
            if all(done_n):
                break
        episode_rewards.append(total / float(max(steps, 1)))
        if ep == 1 or ep == num_episodes or ep % 10 == 0:
            print("[Eval] episode {}/{}".format(ep, num_episodes))
    return float(np.mean(episode_rewards))


def _evaluate_malicious_reward_proxy(trainers, D_f, max_points=2048):
    if D_f["target_q"].size == 0:
        return 0.0
    n = len(trainers)
    k = min(int(max_points), int(D_f["target_q"].shape[0]))
    obs_n = [obs[:k] for obs in D_f["obs_n"]]
    act_n = []
    for i in range(n):
        acts_i = [trainers[i].action(obs) for obs in obs_n[i]]
        act_n.append(np.array(acts_i, dtype=np.float32))
    q_vals = trainers[0].q_debug["q_values"](*(obs_n + act_n))
    return float(np.mean(q_vals))


def _compute_policy_divergence_to_baseline(trainers, baseline_logits):
    kls = []
    for i, agent in enumerate(trainers):
        obs_i = baseline_logits[i]["obs"]
        old_logits = baseline_logits[i]["logits"]
        new_logits = np.array(agent.p_debug["p_values"](obs_i), dtype=np.float32)
        p_old = _softmax(old_logits)
        p_new = _softmax(new_logits)
        kl = np.sum(p_old * (np.log(p_old + 1e-8) - np.log(p_new + 1e-8)), axis=1)
        kls.append(float(np.mean(kl)))
    return float(np.mean(kls))


def _collect_baseline_logits(trainers, D_r, max_points=4096):
    logits = []
    for i, agent in enumerate(trainers):
        obs_i = D_r["obs_n"][i][:max_points]
        logits.append(
            {
                "obs": np.array(obs_i, dtype=np.float32),
                "logits": np.array(agent.p_debug["p_values"](obs_i), dtype=np.float32),
            }
        )
    return logits


def _flatten_agent_trainable_params(agent):
    vars_all = sorted(U.scope_vars(agent.name, trainable_only=True), key=lambda v: v.name)
    if len(vars_all) == 0:
        return np.zeros((0,), dtype=np.float32)
    vals = U.get_session().run(vars_all)
    return np.concatenate([np.reshape(v, (-1,)) for v in vals]).astype(np.float32)


def _baseline_artifact_path(seed_dir):
    return os.path.join(seed_dir, "baseline_artifacts.pkl")


def _save_baseline_artifacts(seed_dir, baseline_ckpt, D_f, D_r, baseline_logits, baseline_metrics):
    payload = {
        "baseline_ckpt": baseline_ckpt,
        "D_f": D_f,
        "D_r": D_r,
        "baseline_logits": baseline_logits,
        "baseline_metrics": baseline_metrics,
    }
    path = _baseline_artifact_path(seed_dir)
    with open(path, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _load_baseline_artifacts(seed_dir):
    path = _baseline_artifact_path(seed_dir)
    if not os.path.exists(path):
        raise FileNotFoundError("Missing baseline artifacts: {}".format(path))
    with open(path, "rb") as fp:
        payload = pickle.load(fp)
    required = ["baseline_ckpt", "D_f", "D_r", "baseline_logits", "baseline_metrics"]
    for key in required:
        if key not in payload:
            raise ValueError("Baseline artifact missing key '{}' in {}".format(key, path))
    return payload


def _train_baseline(arglist):
    tf.reset_default_graph()
    trainer_args = _to_trainer_args(arglist)
    baseline_ckpt = os.path.join(arglist.save_dir, "baseline", "model")
    os.makedirs(os.path.dirname(baseline_ckpt), exist_ok=True)

    with U.single_threaded_session():
        print("[Baseline] Starting baseline training for {} episodes...".format(arglist.num_episodes))
        t0 = time.time()
        _set_all_seeds(arglist.seed)
        env = make_env(arglist.scenario)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, min(env.n, arglist.num_adversaries), obs_shape_n, trainer_args)
        saver = tf.train.Saver()
        U.initialize()

        obs_n = _reset_env(env, seed=arglist.seed)
        ep_step = 0
        ep_count = 0
        train_step = 0
        ep_reward = 0.0
        ep_reward_hist = []
        while ep_count < arglist.num_episodes:
            current_episode = ep_count + 1
            tag = _tag_for_episode(current_episode)
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            new_obs_n, rew_n, done_n, _ = env.step(action_n)
            ep_step += 1
            terminal = ep_step >= arglist.max_episode_len

            for i, agent in enumerate(trainers):
                agent.experience(
                    obs_n[i],
                    action_n[i],
                    rew_n[i],
                    new_obs_n[i],
                    done_n[i],
                    terminal,
                    episode_id=current_episode,
                    tag=tag,
                )

            obs_n = new_obs_n
            ep_reward += float(np.mean(rew_n))
            train_step += 1
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                agent.update(trainers, train_step)

            if terminal or all(done_n):
                ep_count += 1
                ep_reward_hist.append(ep_reward / float(max(ep_step, 1)))
                if ep_count == 1 or ep_count % 100 == 0 or ep_count == arglist.num_episodes:
                    recent = ep_reward_hist[-100:]
                    print(
                        "[Baseline] episode {}/{} | mean_reward(last {}): {:.4f} | elapsed: {:.1f}s".format(
                            ep_count,
                            arglist.num_episodes,
                            len(recent),
                            float(np.mean(recent)),
                            time.time() - t0,
                        )
                    )
                obs_n = _reset_env(env, seed=arglist.seed + ep_count)
                ep_step = 0
                ep_reward = 0.0

        U.save_state(baseline_ckpt, saver=saver)
        print("[Baseline] Checkpoint saved: {}".format(baseline_ckpt))
        D_f, D_r = _extract_tagged_datasets(trainers)
        if int(D_f["target_q"].shape[0]) == 0:
            print("[Baseline] Warning: D_f is empty (no malicious-tagged episodes in this run).")
            print("[Baseline] Applying fallback split: first 10% of D_r as D_f for downstream phases.")
            D_f, D_r = _fallback_forget_split(D_r, forget_fraction=0.1)
        print(
            "[Baseline] Dataset split complete | D_f: {} samples | D_r: {} samples".format(
                int(D_f["target_q"].shape[0]), int(D_r["target_q"].shape[0])
            )
        )
        baseline_logits = _collect_baseline_logits(trainers, D_r)
        metrics = {
            "avg_reward": _evaluate_avg_reward(
                env,
                trainers,
                arglist.eval_episodes,
                arglist.max_episode_len,
                eval_seed=arglist.seed + 10000,
            ),
            "malicious_reward": _evaluate_malicious_reward_proxy(trainers, D_f),
            "policy_divergence": 0.0,
        }
        print("[Baseline] Metrics: {}".format(metrics))
    return baseline_ckpt, D_f, D_r, baseline_logits, metrics, train_step


def _run_full_retraining_short(arglist, D_f, D_r, baseline_logits):
    tf.reset_default_graph()
    trainer_args = _to_trainer_args(arglist)
    ckpt = os.path.join(".", "experiment_outputs", "full_retrain_short", "model")
    curve_path = os.path.join(".", "experiment_outputs", "full_retrain_short", "recovery_curve.csv")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    with U.single_threaded_session():
        print("[Full Retrain] Starting retain-only retraining...")
        t0 = time.time()
        _set_all_seeds(arglist.seed + 1)
        env = make_env(arglist.scenario)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, min(env.n, arglist.num_adversaries), obs_shape_n, trainer_args)
        saver = tf.train.Saver()
        U.initialize()

        recovery_curve = []
        for i, agent in enumerate(trainers):
            print(
                "[Full Retrain] Agent {}/{} recovery training for {} steps...".format(
                    i + 1, len(trainers), arglist.retrain_steps
                )
            )
            agent_t0 = time.time()
            recovery_train_on_retain(
                agent,
                D_r,
                steps=arglist.retrain_steps,
                batch_size=_retain_batch_size(arglist, D_r),
                log_every=50,
                log_path=curve_path if i == 0 else None,
                seed=arglist.seed + 1,
            )
            print(
                "[Full Retrain] Agent {}/{} done in {:.1f}s".format(
                    i + 1, len(trainers), time.time() - agent_t0
                )
            )

        U.save_state(ckpt, saver=saver)
        print("[Full Retrain] Checkpoint saved: {}".format(ckpt))
        metrics = {
            "avg_reward": _evaluate_avg_reward(
                env,
                trainers,
                arglist.eval_episodes,
                arglist.max_episode_len,
                eval_seed=arglist.seed + 11000,
            ),
            "malicious_reward": float("nan"),
            "policy_divergence": _compute_policy_divergence_to_baseline(trainers, baseline_logits),
        }
        print("[Full Retrain] Metrics: {}".format(metrics))
        print("[Full Retrain] Total elapsed: {:.1f}s".format(time.time() - t0))
    return metrics


def _run_full_retraining(arglist, D_r, baseline_logits, baseline_train_steps):
    tf.reset_default_graph()
    trainer_args = _to_trainer_args(arglist)
    ckpt = os.path.join(arglist.save_dir, "full_retrain_true", "model")
    curve_path = os.path.join(arglist.save_dir, "full_retrain_true", "recovery_curve.csv")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    with U.single_threaded_session():
        print("[Full Retrain True] Starting true retain-only retraining from scratch...")
        t0 = time.time()
        _set_all_seeds(arglist.seed + 1)
        env = make_env(arglist.scenario)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, min(env.n, arglist.num_adversaries), obs_shape_n, trainer_args)
        saver = tf.train.Saver()
        U.initialize()

        # Match baseline training length using the same number of gradient updates.
        full_retrain_steps = int(baseline_train_steps)
        recovery_curve = []
        for i, agent in enumerate(trainers):
            print(
                "[Full Retrain True] Agent {}/{} retain-only training for {} steps...".format(
                    i + 1, len(trainers), full_retrain_steps
                )
            )
            agent_t0 = time.time()
            curve_i = recovery_train_on_retain(
                agent,
                D_r,
                steps=full_retrain_steps,
                batch_size=_retain_batch_size(arglist, D_r),
                log_every=50,
                log_path=curve_path if i == 0 else None,
                seed=arglist.seed + 1,
            )
            if i == 0:
                recovery_curve = curve_i
            print(
                "[Full Retrain True] Agent {}/{} done in {:.1f}s".format(
                    i + 1, len(trainers), time.time() - agent_t0
                )
            )

        U.save_state(ckpt, saver=saver)
        print("[Full Retrain True] Checkpoint saved: {}".format(ckpt))
        metrics = {
            "avg_reward": _evaluate_avg_reward(
                env,
                trainers,
                arglist.eval_episodes,
                arglist.max_episode_len,
                eval_seed=arglist.seed + 12000,
            ),
            "malicious_reward": float("nan"),
            "policy_divergence": _compute_policy_divergence_to_baseline(trainers, baseline_logits),
            "recovery_curve": recovery_curve,
        }
        print("[Full Retrain True] Metrics: {}".format(metrics))
        print("[Full Retrain True] Total elapsed: {:.1f}s".format(time.time() - t0))
    return metrics


def _run_selective_unlearning(arglist, baseline_ckpt, D_f, D_r, baseline_logits, mode):
    tf.reset_default_graph()
    trainer_args = _to_trainer_args(arglist)
    ckpt = os.path.join(arglist.save_dir, "selective_unlearn_{}".format(mode), "model")
    curve_path = os.path.join(arglist.save_dir, "selective_unlearn_{}".format(mode), "recovery_curve.csv")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    with U.single_threaded_session():
        print("[Selective Unlearn:{}] Starting selective unlearning...".format(mode))
        t0 = time.time()
        _set_all_seeds(arglist.seed + 2)
        env = make_env(arglist.scenario)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, min(env.n, arglist.num_adversaries), obs_shape_n, trainer_args)
        saver = tf.train.Saver()
        U.initialize()
        U.load_state(baseline_ckpt)
        print("[Selective Unlearn:{}] Loaded baseline checkpoint: {}".format(mode, baseline_ckpt))
        baseline_param_vecs = [_flatten_agent_trainable_params(agent) for agent in trainers]

        recovery_curve = []
        for i, agent in enumerate(trainers):
            print("[Selective Unlearn:{}] Agent {}/{} unlearning...".format(mode, i + 1, len(trainers)))
            agent_t0 = time.time()
            if mode in ("critic_only", "full_selective"):
                _safe_critic_unlearn(agent, D_f)
            if mode in ("actor_only", "full_selective"):
                actor_unlearn(agent, D_f, D_r, lambda_forget=arglist.lambda_forget, lr=1e-3, kl_coeff=5e-2)
            print(
                "[Selective Unlearn:{}] Agent {}/{} retain recovery for {} steps...".format(
                    mode, i + 1, len(trainers), arglist.unlearn_recovery_steps
                )
            )
            curve_i = recovery_train_on_retain(
                agent,
                D_r,
                steps=arglist.unlearn_recovery_steps,
                batch_size=_retain_batch_size(arglist, D_r),
                log_every=50,
                log_path=curve_path if i == 0 else None,
                seed=arglist.seed + 2,
            )
            if i == 0:
                recovery_curve = curve_i
            print(
                "[Selective Unlearn:{}] Agent {}/{} done in {:.1f}s".format(
                    mode, i + 1, len(trainers), time.time() - agent_t0
                )
            )

        U.save_state(ckpt, saver=saver)
        print("[Selective Unlearn:{}] Checkpoint saved: {}".format(mode, ckpt))
        post_param_vecs = [_flatten_agent_trainable_params(agent) for agent in trainers]
        per_agent_shift = []
        for base_vec, post_vec in zip(baseline_param_vecs, post_param_vecs):
            if base_vec.shape != post_vec.shape:
                raise ValueError("Parameter vector shape mismatch during parameter_distance computation.")
            per_agent_shift.append(float(np.linalg.norm(post_vec - base_vec, ord=2)))
        mean_param_shift = float(np.mean(per_agent_shift)) if per_agent_shift else 0.0
        metrics = {
            "avg_reward": _evaluate_avg_reward(
                env,
                trainers,
                arglist.eval_episodes,
                arglist.max_episode_len,
                eval_seed=arglist.seed + 13000,
            ),
            "malicious_reward": _evaluate_malicious_reward_proxy(trainers, D_f),
            "policy_divergence": _compute_policy_divergence_to_baseline(trainers, baseline_logits),
            "parameter_distance": mean_param_shift,
            "recovery_curve": recovery_curve,
        }
        print("[Selective Unlearn:{}] Metrics: {}".format(mode, metrics))
        print("[Selective Unlearn:{}] mean_param_shift: {:.6f}".format(mode, mean_param_shift))
        print("[Selective Unlearn:{}] Total elapsed: {:.1f}s".format(mode, time.time() - t0))
    return metrics


def _save_comparison_table(path, baseline_metrics, full_metrics, unlearn_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        ("baseline", baseline_metrics),
        ("full_retrain_without_Df", full_metrics),
        ("selective_unlearning", unlearn_metrics),
    ]
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["method", "avg_reward", "reward_on_malicious_scenario", "policy_divergence"])
        for name, m in rows:
            writer.writerow([name, m["avg_reward"], m["malicious_reward"], m["policy_divergence"]])


def _safe_div(num, den, default=0.0):
    if abs(den) < 1e-12:
        return default
    return num / den


def _build_extended_metrics(baseline_metrics, full_metrics, unlearn_metrics, baseline_time, full_time, unlearn_time):
    forget_drop_full = baseline_metrics["malicious_reward"] - full_metrics["malicious_reward"]
    forget_drop_unlearn = baseline_metrics["malicious_reward"] - unlearn_metrics["malicious_reward"]
    retain_loss_unlearn = abs(unlearn_metrics["avg_reward"] - baseline_metrics["avg_reward"])
    retain_loss_full = abs(full_metrics["avg_reward"] - baseline_metrics["avg_reward"])
    retain_relative_unlearn = _safe_div(retain_loss_unlearn, abs(baseline_metrics["avg_reward"]), default=0.0)
    retain_relative_full = _safe_div(retain_loss_full, abs(baseline_metrics["avg_reward"]), default=0.0)
    gap_vs_full = abs(unlearn_metrics["avg_reward"] - full_metrics["avg_reward"])
    compute_ratio = _safe_div(unlearn_time, full_time, default=0.0)
    compute_speedup = _safe_div(full_time, unlearn_time, default=0.0)

    baseline_row = {
        "method": "baseline",
        "avg_reward": baseline_metrics["avg_reward"],
        "malicious_reward": baseline_metrics["malicious_reward"],
        "forget_drop": 0.0,
        "retain_loss": 0.0,
        "retain_relative": 0.0,
        "policy_divergence": baseline_metrics["policy_divergence"],
        "compute_time": baseline_time,
    }
    full_row = {
        "method": "full_retrain",
        "avg_reward": full_metrics["avg_reward"],
        "malicious_reward": full_metrics["malicious_reward"],
        "forget_drop": forget_drop_full,
        "retain_loss": retain_loss_full,
        "retain_relative": retain_relative_full,
        "policy_divergence": full_metrics["policy_divergence"],
        "compute_time": full_time,
    }
    unlearn_row = {
        "method": "selective_unlearning",
        "avg_reward": unlearn_metrics["avg_reward"],
        "malicious_reward": unlearn_metrics["malicious_reward"],
        "forget_drop": forget_drop_unlearn,
        "retain_loss": retain_loss_unlearn,
        "retain_relative": retain_relative_unlearn,
        "policy_divergence": unlearn_metrics["policy_divergence"],
        "compute_time": unlearn_time,
    }
    summary = {
        "forget_drop_selective": forget_drop_unlearn,
        "forget_drop_full": forget_drop_full,
        "retain_loss_selective": retain_loss_unlearn,
        "retain_loss_full": retain_loss_full,
        "retain_relative_selective": retain_relative_unlearn,
        "retain_relative_full": retain_relative_full,
        "gap_vs_full": gap_vs_full,
        "compute_ratio": compute_ratio,
        "compute_speedup": compute_speedup,
    }
    return [baseline_row, full_row, unlearn_row], summary


def _save_extended_metrics(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "method",
                "avg_reward",
                "malicious_reward",
                "forget_drop",
                "retain_loss",
                "retain_relative",
                "policy_divergence",
                "compute_time",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["method"],
                    row["avg_reward"],
                    row["malicious_reward"],
                    row["forget_drop"],
                    row["retain_loss"],
                    row["retain_relative"],
                    row["policy_divergence"],
                    row["compute_time"],
                ]
            )


def _save_seed_results(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "seed",
                "baseline_avg_reward",
                "full_avg_reward",
                "selective_avg_reward",
                "baseline_malicious_reward",
                "full_malicious_reward",
                "selective_malicious_reward",
                "retain_loss",
                "policy_divergence",
                "forget_drop",
                "parameter_distance",
            ]
        )
        writer.writerow(
            [
                row["seed"],
                row["baseline_avg_reward"],
                row["full_avg_reward"],
                row["selective_avg_reward"],
                row["baseline_malicious_reward"],
                row["full_malicious_reward"],
                row["selective_malicious_reward"],
                row["retain_loss"],
                row["policy_divergence"],
                row["forget_drop"],
                row["parameter_distance"],
            ]
        )


def _save_aggregated_results(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "mean_avg_reward",
                "std_avg_reward",
                "mean_retain_loss",
                "std_retain_loss",
                "mean_policy_divergence",
                "std_policy_divergence",
                "mean_forget_drop",
                "std_forget_drop",
                "mean_param_shift",
            ]
        )
        writer.writerow(
            [
                row["mean_avg_reward"],
                row["std_avg_reward"],
                row["mean_retain_loss"],
                row["std_retain_loss"],
                row["mean_policy_divergence"],
                row["std_policy_divergence"],
                row["mean_forget_drop"],
                row["std_forget_drop"],
            ]
        )


def _save_ablation_results(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "mode",
                "mean_avg_reward",
                "std_avg_reward",
                "mean_retain_loss",
                "std_retain_loss",
                "mean_policy_divergence",
                "std_policy_divergence",
                "mean_forget_drop",
                "std_forget_drop",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["mode"],
                    row["mean_avg_reward"],
                    row["std_avg_reward"],
                    row["mean_retain_loss"],
                    row["std_retain_loss"],
                    row["mean_policy_divergence"],
                    row["std_policy_divergence"],
                    row["mean_forget_drop"],
                    row["std_forget_drop"],
                    row["mean_param_shift"],
                ]
            )


def _metric_mean_std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(np.mean(vals)), float(np.std(vals))


def _mean_curve(curves, key):
    valid_curves = [c for c in curves if c]
    if not valid_curves:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    step_grid = np.array(sorted({int(p["step"]) for c in valid_curves for p in c}), dtype=np.float64)
    series = []
    for curve in valid_curves:
        steps = np.array([int(p["step"]) for p in curve], dtype=np.float64)
        vals = np.array([float(p[key]) for p in curve], dtype=np.float64)
        series.append(np.interp(step_grid, steps, vals))
    return step_grid, np.mean(np.vstack(series), axis=0)


def _make_publication_plots(plot_dir, baseline_rows, full_rows, per_mode_rows, full_curves, per_mode_curves):
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    mode_order = ["critic_only", "actor_only", "full_selective"]
    active_modes = [m for m in mode_order if m in per_mode_rows]
    mode_labels = {
        "critic_only": "Critic Only",
        "actor_only": "Actor Only",
        "full_selective": "Full Selective",
    }

    baseline_avg_mean, baseline_avg_std = _metric_mean_std(baseline_rows, "avg_reward")
    full_avg_mean, full_avg_std = _metric_mean_std(full_rows, "avg_reward")
    baseline_time_mean, baseline_time_std = _metric_mean_std(baseline_rows, "compute_time")
    full_time_mean, full_time_std = _metric_mean_std(full_rows, "compute_time")
    full_retain_mean, full_retain_std = _metric_mean_std(full_rows, "retain_loss")
    full_policy_mean, full_policy_std = _metric_mean_std(full_rows, "policy_divergence")

    def _save_bar(path, labels, means, stds, ylabel, title):
        x = np.arange(len(labels))
        fig, ax = plt.subplots()
        ax.bar(x, means, yerr=stds, capsize=4, color="#2a6f97")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    avg_labels = ["Baseline", "Full Retrain"] + [mode_labels[m] for m in active_modes]
    avg_means = [baseline_avg_mean, full_avg_mean]
    avg_stds = [baseline_avg_std, full_avg_std]
    for mode in active_modes:
        m, s = _metric_mean_std(per_mode_rows[mode], "selective_avg_reward")
        avg_means.append(m)
        avg_stds.append(s)
    _save_bar(
        os.path.join(plot_dir, "avg_reward_comparison.png"),
        avg_labels,
        avg_means,
        avg_stds,
        "Average Reward",
        "Avg Reward Comparison",
    )

    retain_labels = ["Full Retrain"] + [mode_labels[m] for m in active_modes]
    retain_means = [full_retain_mean]
    retain_stds = [full_retain_std]
    for mode in active_modes:
        m, s = _metric_mean_std(per_mode_rows[mode], "retain_loss")
        retain_means.append(m)
        retain_stds.append(s)
    _save_bar(
        os.path.join(plot_dir, "retention_loss_comparison.png"),
        retain_labels,
        retain_means,
        retain_stds,
        "Retention Loss",
        "Retention Loss Comparison",
    )

    forget_labels = [mode_labels[m] for m in active_modes]
    forget_means = []
    forget_stds = []
    for mode in active_modes:
        m, s = _metric_mean_std(per_mode_rows[mode], "forget_drop")
        forget_means.append(m)
        forget_stds.append(s)
    _save_bar(
        os.path.join(plot_dir, "forget_drop_comparison.png"),
        forget_labels,
        forget_means,
        forget_stds,
        "Forget Drop",
        "Forget Drop Comparison",
    )

    policy_labels = ["Full Retrain"] + [mode_labels[m] for m in active_modes]
    policy_means = [full_policy_mean]
    policy_stds = [full_policy_std]
    for mode in active_modes:
        m, s = _metric_mean_std(per_mode_rows[mode], "policy_divergence")
        policy_means.append(m)
        policy_stds.append(s)
    _save_bar(
        os.path.join(plot_dir, "policy_divergence_comparison.png"),
        policy_labels,
        policy_means,
        policy_stds,
        "Policy Divergence",
        "Policy Divergence Comparison",
    )

    time_labels = ["Baseline", "Full Retrain"] + [mode_labels[m] for m in active_modes]
    time_means = [baseline_time_mean, full_time_mean]
    time_stds = [baseline_time_std, full_time_std]
    for mode in active_modes:
        m, s = _metric_mean_std(per_mode_rows[mode], "compute_time")
        time_means.append(m)
        time_stds.append(s)
    _save_bar(
        os.path.join(plot_dir, "compute_time_comparison.png"),
        time_labels,
        time_means,
        time_stds,
        "Compute Time (s)",
        "Compute Time Comparison",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    step_full_c, mean_full_c = _mean_curve(full_curves, "critic_loss_retain")
    step_full_a, mean_full_a = _mean_curve(full_curves, "actor_loss_retain")
    if step_full_c.size > 0:
        axes[0].plot(step_full_c, mean_full_c, linewidth=2.0, label="Full Retrain", color="#1d3557")
    if step_full_a.size > 0:
        axes[1].plot(step_full_a, mean_full_a, linewidth=2.0, label="Full Retrain", color="#1d3557")
    mode_colors = {
        "critic_only": "#2a9d8f",
        "actor_only": "#e76f51",
        "full_selective": "#f4a261",
    }
    for mode in active_modes:
        steps_c, means_c = _mean_curve(per_mode_curves[mode], "critic_loss_retain")
        steps_a, means_a = _mean_curve(per_mode_curves[mode], "actor_loss_retain")
        if steps_c.size > 0:
            axes[0].plot(steps_c, means_c, linewidth=2.0, label=mode_labels[mode], color=mode_colors[mode])
        if steps_a.size > 0:
            axes[1].plot(steps_a, means_a, linewidth=2.0, label=mode_labels[mode], color=mode_colors[mode])
    axes[0].set_title("Recovery Curves (Critic)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Critic Loss on Retain")
    axes[1].set_title("Recovery Curves (Actor)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Actor Loss on Retain")
    for ax in axes:
        ax.grid(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "recovery_curves_mean_across_seeds.png"), dpi=300)
    plt.close(fig)


def _paired_pvalue(a_vals, b_vals):
    a = np.array(a_vals, dtype=np.float64)
    b = np.array(b_vals, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 2:
        return float("nan")
    _, p = ttest_rel(a[mask], b[mask])
    return float(p)


def _save_statistical_tests(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        for line in lines:
            fp.write(line + "\n")


def main():
    args = parse_args()
    if args.retrain_steps < 500 or args.retrain_steps > 2000:
        raise ValueError("--retrain-steps must be in [500, 2000]")
    if args.unlearn_recovery_steps < 500 or args.unlearn_recovery_steps > 2000:
        raise ValueError("--unlearn-recovery-steps must be in [500, 2000]")
    seed_pool = [123, 456, 789, 1011, 2022]
    if args.seed_list.strip():
        seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
        if len(seeds) == 0:
            raise ValueError("--seed-list was provided but no valid seeds were parsed.")
    else:
        if args.num_seeds < 1 or args.num_seeds > len(seed_pool):
            raise ValueError("--num-seeds must be in [1, {}]".format(len(seed_pool)))
        seeds = seed_pool[: args.num_seeds]
    modes = ["critic_only", "actor_only", "full_selective"] if args.mode == "all" else [args.mode]
    artifacts_root = args.artifacts_dir if args.artifacts_dir else args.save_dir

    os.makedirs(args.save_dir, exist_ok=True)
    _set_all_seeds(args.seed)
    print("[Run] Starting experiment with config:")
    print(
        "[Run] scenario={} episodes={} eval_episodes={} retrain_steps={} unlearn_recovery_steps={} num_seeds={}".format(
            args.scenario,
            args.num_episodes,
            args.eval_episodes,
            args.retrain_steps,
            args.unlearn_recovery_steps,
            args.num_seeds,
        )
    )
    per_mode_rows = {m: [] for m in modes}
    per_mode_curves = {m: [] for m in modes}
    baseline_rows = []
    full_rows = []
    full_curves = []

    for seed in seeds:
        print("=== SEED {} ===".format(seed))
        run_args = SimpleNamespace(**vars(args))
        run_args.seed = seed
        run_args.save_dir = os.path.join(args.save_dir, "seed_{}".format(seed))
        os.makedirs(run_args.save_dir, exist_ok=True)
        if args.selective_only:
            seed_artifact_dir = os.path.join(artifacts_root, "seed_{}".format(seed))
            payload = _load_baseline_artifacts(seed_artifact_dir)
            baseline_ckpt = payload["baseline_ckpt"]
            D_f = payload["D_f"]
            D_r = payload["D_r"]
            baseline_logits = payload["baseline_logits"]
            baseline_metrics = payload["baseline_metrics"]
            baseline_time = float("nan")
            full_metrics = {
                "avg_reward": float("nan"),
                "malicious_reward": float("nan"),
                "policy_divergence": float("nan"),
                "recovery_curve": [],
            }
            full_time = float("nan")
            print("[Run] selective-only: loaded baseline artifacts for seed {} from {}".format(seed, seed_artifact_dir))
        else:
            t_baseline = time.time()
            baseline_ckpt, D_f, D_r, baseline_logits, baseline_metrics, baseline_train_steps = _train_baseline(run_args)
            baseline_time = time.time() - t_baseline
            artifact_path = _save_baseline_artifacts(
                run_args.save_dir, baseline_ckpt, D_f, D_r, baseline_logits, baseline_metrics
            )
            print("[Run] Saved baseline artifacts: {}".format(artifact_path))

            t_full = time.time()
            full_metrics = _run_full_retraining(run_args, D_r, baseline_logits, baseline_train_steps)
            full_time = time.time() - t_full

        baseline_rows.append(
            {
                "seed": seed,
                "avg_reward": baseline_metrics["avg_reward"],
                "compute_time": baseline_time,
            }
        )
        full_rows.append(
            {
                "seed": seed,
                "avg_reward": full_metrics["avg_reward"],
                "retain_loss": abs(full_metrics["avg_reward"] - baseline_metrics["avg_reward"])
                if np.isfinite(full_metrics["avg_reward"])
                else float("nan"),
                "policy_divergence": full_metrics["policy_divergence"],
                "compute_time": full_time,
            }
        )
        full_curves.append(full_metrics.get("recovery_curve", []))

        for mode in modes:
            t_unlearn = time.time()
            unlearn_metrics = _run_selective_unlearning(run_args, baseline_ckpt, D_f, D_r, baseline_logits, mode)
            unlearn_time = time.time() - t_unlearn

            table_path = os.path.join(run_args.save_dir, "comparison_table_{}.csv".format(mode))
            _save_comparison_table(table_path, baseline_metrics, full_metrics, unlearn_metrics)
            extended_rows, _ = _build_extended_metrics(
                baseline_metrics,
                full_metrics,
                unlearn_metrics,
                baseline_time,
                full_time,
                unlearn_time,
            )
            extended_path = os.path.join(run_args.save_dir, "extended_metrics_{}.csv".format(mode))
            _save_extended_metrics(extended_path, extended_rows)

            seed_row = {
                "seed": seed,
                "baseline_avg_reward": baseline_metrics["avg_reward"],
                "full_avg_reward": full_metrics["avg_reward"],
                "selective_avg_reward": unlearn_metrics["avg_reward"],
                "baseline_malicious_reward": baseline_metrics["malicious_reward"],
                "full_malicious_reward": full_metrics["malicious_reward"],
                "selective_malicious_reward": unlearn_metrics["malicious_reward"],
                "retain_loss": abs(unlearn_metrics["avg_reward"] - baseline_metrics["avg_reward"]),
                "policy_divergence": unlearn_metrics["policy_divergence"],
                "forget_drop": baseline_metrics["malicious_reward"] - unlearn_metrics["malicious_reward"],
                "parameter_distance": unlearn_metrics.get("parameter_distance", float("nan")),
                "compute_time": unlearn_time,
            }
            per_mode_rows[mode].append(seed_row)
            per_mode_curves[mode].append(unlearn_metrics.get("recovery_curve", []))
            seed_results_path = os.path.join(args.save_dir, "results_seed_{}_{}.csv".format(seed, mode))
            _save_seed_results(seed_results_path, seed_row)
            print("Saved seed results to {}".format(seed_results_path))

    ablation_rows = []
    for mode in modes:
        rows = per_mode_rows[mode]
        avg_reward_vals = np.array([r["selective_avg_reward"] for r in rows], dtype=np.float64)
        retain_loss_vals = np.array([r["retain_loss"] for r in rows], dtype=np.float64)
        policy_div_vals = np.array([r["policy_divergence"] for r in rows], dtype=np.float64)
        forget_drop_vals = np.array([r["forget_drop"] for r in rows], dtype=np.float64)
        param_shift_vals = np.array([r["parameter_distance"] for r in rows], dtype=np.float64)

        ablation_rows.append(
            {
                "mode": mode,
                "mean_avg_reward": float(np.mean(avg_reward_vals)),
                "std_avg_reward": float(np.std(avg_reward_vals)),
                "mean_retain_loss": float(np.mean(retain_loss_vals)),
                "std_retain_loss": float(np.std(retain_loss_vals)),
                "mean_policy_divergence": float(np.mean(policy_div_vals)),
                "std_policy_divergence": float(np.std(policy_div_vals)),
                "mean_forget_drop": float(np.mean(forget_drop_vals)),
                "std_forget_drop": float(np.std(forget_drop_vals)),
                "mean_param_shift": float(np.nanmean(param_shift_vals)),
            }
        )

    ablation_path = os.path.join(args.save_dir, "ablation_results.csv")
    _save_ablation_results(ablation_path, ablation_rows)

    if len(ablation_rows) == 1:
        aggregated = ablation_rows[0]
    else:
        aggregated = next(r for r in ablation_rows if r["mode"] == "full_selective")
    aggregated_path = os.path.join(args.save_dir, "aggregated_results.csv")
    _save_aggregated_results(aggregated_path, aggregated)

    print("=== MULTI-SEED ABLATION SUMMARY (N={}) ===".format(len(seeds)))
    for row in ablation_rows:
        print("--- Mode: {} ---".format(row["mode"]))
        print("Avg Reward: {:.6f} +/- {:.6f}".format(row["mean_avg_reward"], row["std_avg_reward"]))
        print("Retain Loss: {:.6f} +/- {:.6f}".format(row["mean_retain_loss"], row["std_retain_loss"]))
        print(
            "Policy Divergence: {:.6f} +/- {:.6f}".format(
                row["mean_policy_divergence"], row["std_policy_divergence"]
            )
        )
        print("Forget Drop: {:.6f} +/- {:.6f}".format(row["mean_forget_drop"], row["std_forget_drop"]))
        print("mean_param_shift: {:.6f}".format(row["mean_param_shift"]))
    plot_dir = os.path.join(".", "experiment_outputs", "final_plots")
    _make_publication_plots(plot_dir, baseline_rows, full_rows, per_mode_rows, full_curves, per_mode_curves)
    print("Saved publication plots to {}".format(plot_dir))

    selective_mode = "full_selective" if "full_selective" in per_mode_rows else modes[0]
    selective_rows = per_mode_rows[selective_mode]
    baseline_map = {int(r["seed"]): r for r in baseline_rows}
    full_map = {int(r["seed"]): r for r in full_rows}

    sel_retain = []
    full_retain = []
    base_retain = []
    sel_forget = []
    full_forget = []
    base_forget = []
    sel_policy = []
    full_policy = []
    base_policy = []
    for row in selective_rows:
        seed = int(row["seed"])
        if seed not in full_map or seed not in baseline_map:
            continue
        sel_retain.append(float(row["retain_loss"]))
        full_retain.append(float(full_map[seed]["retain_loss"]))
        base_retain.append(0.0)
        sel_forget.append(float(row["forget_drop"]))
        full_forget.append(float("nan"))
        base_forget.append(0.0)
        sel_policy.append(float(row["policy_divergence"]))
        full_policy.append(float(full_map[seed]["policy_divergence"]))
        base_policy.append(0.0)

    p_sel_vs_full_retain = _paired_pvalue(sel_retain, full_retain)
    p_sel_vs_base_retain = _paired_pvalue(sel_retain, base_retain)
    p_sel_vs_full_forget = _paired_pvalue(sel_forget, full_forget)
    p_sel_vs_base_forget = _paired_pvalue(sel_forget, base_forget)
    p_sel_vs_full_policy = _paired_pvalue(sel_policy, full_policy)
    p_sel_vs_base_policy = _paired_pvalue(sel_policy, base_policy)

    stats_lines = [
        "Paired t-tests (mode: {})".format(selective_mode),
        "Selective vs Full Retrain | retain_loss p-value: {}".format(p_sel_vs_full_retain),
        "Selective vs Baseline | retain_loss p-value: {}".format(p_sel_vs_base_retain),
        "Selective vs Full Retrain | forget_drop p-value: {}".format(p_sel_vs_full_forget),
        "Selective vs Baseline | forget_drop p-value: {}".format(p_sel_vs_base_forget),
        "Selective vs Full Retrain | policy_divergence p-value: {}".format(p_sel_vs_full_policy),
        "Selective vs Baseline | policy_divergence p-value: {}".format(p_sel_vs_base_policy),
    ]
    stats_path = os.path.join(args.save_dir, "statistical_tests.txt")
    _save_statistical_tests(stats_path, stats_lines)
    print("=== STATISTICAL TESTS ===")
    for line in stats_lines[1:]:
        print(line)
    print("Saved statistical tests to {}".format(stats_path))
    print("Saved ablation results to {}".format(ablation_path))
    print("Saved aggregated results to {}".format(aggregated_path))


if __name__ == "__main__":
    main()
