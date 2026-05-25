import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


def parse_args():
    parser = argparse.ArgumentParser("Merge per-seed experiment outputs without retraining")
    parser.add_argument("--input-dir", type=str, default="./experiment_outputs_123_456_789_full/")
    parser.add_argument("--output-dir", type=str, default="./experiment_outputs_merged/")
    parser.add_argument("--seeds", type=str, default="123,456,789")
    parser.add_argument("--modes", type=str, default="critic_only,actor_only,full_selective")
    return parser.parse_args()


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _read_single_row_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing file: {}".format(path))
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if len(rows) == 0:
        raise ValueError("No data rows in {}".format(path))
    return rows[0]


def _read_extended_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing file: {}".format(path))
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    out = {}
    for r in rows:
        out[r["method"]] = {k: _to_float(v) for k, v in r.items() if k != "method"}
    return out


def _read_curve(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    out = []
    for r in rows:
        out.append(
            {
                "step": int(_to_float(r["step"])),
                "critic_loss_retain": _to_float(r["critic_loss_retain"]),
                "actor_loss_retain": _to_float(r["actor_loss_retain"]),
            }
        )
    return out


def _mean_std(vals):
    a = np.array(vals, dtype=np.float64)
    return float(np.nanmean(a)), float(np.nanstd(a))


def _paired_pvalue(a_vals, b_vals):
    a = np.array(a_vals, dtype=np.float64)
    b = np.array(b_vals, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 2:
        return float("nan")
    _, p = ttest_rel(a[mask], b[mask])
    return float(p)


def _save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _mean_curve(curves, key):
    valid = [c for c in curves if c]
    if not valid:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    step_grid = np.array(sorted({int(p["step"]) for c in valid for p in c}), dtype=np.float64)
    stacked = []
    for c in valid:
        steps = np.array([int(p["step"]) for p in c], dtype=np.float64)
        vals = np.array([_to_float(p[key]) for p in c], dtype=np.float64)
        mask = np.isfinite(steps) & np.isfinite(vals)
        if int(np.sum(mask)) < 2:
            continue
        stacked.append(np.interp(step_grid, steps[mask], vals[mask]))
    if len(stacked) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return step_grid, np.nanmean(np.vstack(stacked), axis=0)


def _make_plots(plot_dir, baseline_rows, full_rows, per_mode_rows, full_curves, per_mode_curves):
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
    labels = {"critic_only": "Critic Only", "actor_only": "Actor Only", "full_selective": "Full Selective"}

    b_avg_m, b_avg_s = _mean_std([r["avg_reward"] for r in baseline_rows])
    f_avg_m, f_avg_s = _mean_std([r["avg_reward"] for r in full_rows])
    b_t_m, b_t_s = _mean_std([r["compute_time"] for r in baseline_rows])
    f_t_m, f_t_s = _mean_std([r["compute_time"] for r in full_rows])
    f_r_m, f_r_s = _mean_std([r["retain_loss"] for r in full_rows])
    f_p_m, f_p_s = _mean_std([r["policy_divergence"] for r in full_rows])

    def save_bar(filename, xlabels, means, stds, ylabel, title):
        x = np.arange(len(xlabels))
        fig, ax = plt.subplots()
        ax.bar(x, means, yerr=stds, capsize=4, color="#2a6f97")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.close(fig)

    avg_labels = ["Baseline", "Full Retrain"] + [labels[m] for m in active_modes]
    avg_means = [b_avg_m, f_avg_m]
    avg_stds = [b_avg_s, f_avg_s]
    for m in active_modes:
        mm, ss = _mean_std([r["selective_avg_reward"] for r in per_mode_rows[m]])
        avg_means.append(mm)
        avg_stds.append(ss)
    save_bar("avg_reward_comparison.png", avg_labels, avg_means, avg_stds, "Average Reward", "Avg Reward Comparison")

    r_labels = ["Full Retrain"] + [labels[m] for m in active_modes]
    r_means = [f_r_m]
    r_stds = [f_r_s]
    for m in active_modes:
        mm, ss = _mean_std([r["retain_loss"] for r in per_mode_rows[m]])
        r_means.append(mm)
        r_stds.append(ss)
    save_bar("retention_loss_comparison.png", r_labels, r_means, r_stds, "Retention Loss", "Retention Loss Comparison")

    fg_labels = [labels[m] for m in active_modes]
    fg_means = []
    fg_stds = []
    for m in active_modes:
        mm, ss = _mean_std([r["forget_drop"] for r in per_mode_rows[m]])
        fg_means.append(mm)
        fg_stds.append(ss)
    save_bar("forget_drop_comparison.png", fg_labels, fg_means, fg_stds, "Forget Drop", "Forget Drop Comparison")

    p_labels = ["Full Retrain"] + [labels[m] for m in active_modes]
    p_means = [f_p_m]
    p_stds = [f_p_s]
    for m in active_modes:
        mm, ss = _mean_std([r["policy_divergence"] for r in per_mode_rows[m]])
        p_means.append(mm)
        p_stds.append(ss)
    save_bar(
        "policy_divergence_comparison.png",
        p_labels,
        p_means,
        p_stds,
        "Policy Divergence",
        "Policy Divergence Comparison",
    )

    t_labels = ["Baseline", "Full Retrain"] + [labels[m] for m in active_modes]
    t_means = [b_t_m, f_t_m]
    t_stds = [b_t_s, f_t_s]
    for m in active_modes:
        mm, ss = _mean_std([r["compute_time"] for r in per_mode_rows[m]])
        t_means.append(mm)
        t_stds.append(ss)
    save_bar("compute_time_comparison.png", t_labels, t_means, t_stds, "Compute Time (s)", "Compute Time Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    c_steps, c_vals = _mean_curve(full_curves, "critic_loss_retain")
    a_steps, a_vals = _mean_curve(full_curves, "actor_loss_retain")
    if c_steps.size > 0:
        axes[0].plot(c_steps, c_vals, label="Full Retrain", linewidth=2.0, color="#1d3557")
    if a_steps.size > 0:
        axes[1].plot(a_steps, a_vals, label="Full Retrain", linewidth=2.0, color="#1d3557")
    colors = {"critic_only": "#2a9d8f", "actor_only": "#e76f51", "full_selective": "#f4a261"}
    for m in active_modes:
        ms_c_steps, ms_c_vals = _mean_curve(per_mode_curves[m], "critic_loss_retain")
        ms_a_steps, ms_a_vals = _mean_curve(per_mode_curves[m], "actor_loss_retain")
        if ms_c_steps.size > 0:
            axes[0].plot(ms_c_steps, ms_c_vals, label=labels[m], linewidth=2.0, color=colors[m])
        if ms_a_steps.size > 0:
            axes[1].plot(ms_a_steps, ms_a_vals, label=labels[m], linewidth=2.0, color=colors[m])
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


def main():
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    if len(seeds) == 0:
        raise ValueError("No seeds parsed from --seeds")
    if len(modes) == 0:
        raise ValueError("No modes parsed from --modes")

    per_mode_rows = {m: [] for m in modes}
    per_mode_curves = {m: [] for m in modes}
    baseline_rows = []
    full_rows = []
    full_curves = []

    for seed in seeds:
        ext = _read_extended_rows(
            os.path.join(args.input_dir, "seed_{}".format(seed), "extended_metrics_full_selective.csv")
        )
        b = ext["baseline"]
        f = ext["full_retrain"]
        baseline_rows.append({"seed": seed, "avg_reward": b["avg_reward"], "compute_time": b["compute_time"]})
        full_rows.append(
            {
                "seed": seed,
                "avg_reward": f["avg_reward"],
                "retain_loss": f["retain_loss"],
                "policy_divergence": f["policy_divergence"],
                "compute_time": f["compute_time"],
            }
        )
        full_curves.append(
            _read_curve(os.path.join(args.input_dir, "seed_{}".format(seed), "full_retrain_true", "recovery_curve.csv"))
        )
        for mode in modes:
            row = _read_single_row_csv(os.path.join(args.input_dir, "results_seed_{}_{}.csv".format(seed, mode)))
            parsed = {k: _to_float(v) for k, v in row.items()}
            ext_mode = _read_extended_rows(
                os.path.join(args.input_dir, "seed_{}".format(seed), "extended_metrics_{}.csv".format(mode))
            )
            if "selective_unlearning" in ext_mode:
                parsed["compute_time"] = ext_mode["selective_unlearning"].get("compute_time", float("nan"))
            else:
                parsed["compute_time"] = float("nan")
            per_mode_rows[mode].append(parsed)
            per_mode_curves[mode].append(
                _read_curve(
                    os.path.join(
                        args.input_dir, "seed_{}".format(seed), "selective_unlearn_{}".format(mode), "recovery_curve.csv"
                    )
                )
            )

    ablation_rows = []
    for mode in modes:
        rows = per_mode_rows[mode]
        ablation_rows.append(
            {
                "mode": mode,
                "mean_avg_reward": _mean_std([r["selective_avg_reward"] for r in rows])[0],
                "std_avg_reward": _mean_std([r["selective_avg_reward"] for r in rows])[1],
                "mean_retain_loss": _mean_std([r["retain_loss"] for r in rows])[0],
                "std_retain_loss": _mean_std([r["retain_loss"] for r in rows])[1],
                "mean_policy_divergence": _mean_std([r["policy_divergence"] for r in rows])[0],
                "std_policy_divergence": _mean_std([r["policy_divergence"] for r in rows])[1],
                "mean_forget_drop": _mean_std([r["forget_drop"] for r in rows])[0],
                "std_forget_drop": _mean_std([r["forget_drop"] for r in rows])[1],
                "mean_param_shift": _mean_std([r.get("parameter_distance", float("nan")) for r in rows])[0],
            }
        )

    ablation_path = os.path.join(args.output_dir, "ablation_results.csv")
    _save_csv(
        ablation_path,
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
            "mean_param_shift",
        ],
        [
            [
                r["mode"],
                r["mean_avg_reward"],
                r["std_avg_reward"],
                r["mean_retain_loss"],
                r["std_retain_loss"],
                r["mean_policy_divergence"],
                r["std_policy_divergence"],
                r["mean_forget_drop"],
                r["std_forget_drop"],
                r["mean_param_shift"],
            ]
            for r in ablation_rows
        ],
    )

    aggregated = next((r for r in ablation_rows if r["mode"] == "full_selective"), ablation_rows[0])
    aggregated_path = os.path.join(args.output_dir, "aggregated_results.csv")
    _save_csv(
        aggregated_path,
        [
            "mean_avg_reward",
            "std_avg_reward",
            "mean_retain_loss",
            "std_retain_loss",
            "mean_policy_divergence",
            "std_policy_divergence",
            "mean_forget_drop",
            "std_forget_drop",
        ],
        [
            [
                aggregated["mean_avg_reward"],
                aggregated["std_avg_reward"],
                aggregated["mean_retain_loss"],
                aggregated["std_retain_loss"],
                aggregated["mean_policy_divergence"],
                aggregated["std_policy_divergence"],
                aggregated["mean_forget_drop"],
                aggregated["std_forget_drop"],
            ]
        ],
    )

    selective_mode = "full_selective" if "full_selective" in per_mode_rows else modes[0]
    selective_rows = per_mode_rows[selective_mode]
    full_by_seed = {int(r["seed"]): r for r in full_rows}
    sel_retain = []
    full_retain = []
    base_retain = []
    sel_forget = []
    full_forget = []
    base_forget = []
    sel_policy = []
    full_policy = []
    base_policy = []
    for r in selective_rows:
        s = int(r["seed"])
        if s not in full_by_seed:
            continue
        sel_retain.append(float(r["retain_loss"]))
        full_retain.append(float(full_by_seed[s]["retain_loss"]))
        base_retain.append(0.0)
        sel_forget.append(float(r["forget_drop"]))
        full_forget.append(float("nan"))
        base_forget.append(0.0)
        sel_policy.append(float(r["policy_divergence"]))
        full_policy.append(float(full_by_seed[s]["policy_divergence"]))
        base_policy.append(0.0)

    lines = [
        "Paired t-tests (mode: {})".format(selective_mode),
        "Selective vs Full Retrain | retain_loss p-value: {}".format(_paired_pvalue(sel_retain, full_retain)),
        "Selective vs Baseline | retain_loss p-value: {}".format(_paired_pvalue(sel_retain, base_retain)),
        "Selective vs Full Retrain | forget_drop p-value: {}".format(_paired_pvalue(sel_forget, full_forget)),
        "Selective vs Baseline | forget_drop p-value: {}".format(_paired_pvalue(sel_forget, base_forget)),
        "Selective vs Full Retrain | policy_divergence p-value: {}".format(_paired_pvalue(sel_policy, full_policy)),
        "Selective vs Baseline | policy_divergence p-value: {}".format(_paired_pvalue(sel_policy, base_policy)),
    ]
    stats_path = os.path.join(args.output_dir, "statistical_tests.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(stats_path, "w") as fp:
        for line in lines:
            fp.write(line + "\n")

    plot_dir = os.path.join(args.output_dir, "final_plots")
    _make_plots(plot_dir, baseline_rows, full_rows, per_mode_rows, full_curves, per_mode_curves)

    print("Saved merged ablation results to {}".format(ablation_path))
    print("Saved merged aggregated results to {}".format(aggregated_path))
    print("Saved merged statistical tests to {}".format(stats_path))
    print("Saved merged plots to {}".format(plot_dir))


if __name__ == "__main__":
    main()
