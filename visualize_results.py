import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Visualize MARL unlearning experiment results")
    parser.add_argument("--input-dir", type=str, default="./experiment_outputs/")
    return parser.parse_args()


def _load_extended_metrics(path):
    if not os.path.exists(path):
        raise FileNotFoundError("extended_metrics.csv not found at {}".format(path))
    df = pd.read_csv(path)
    expected = {
        "method",
        "avg_reward",
        "malicious_reward",
        "forget_drop",
        "retain_ratio",
        "policy_divergence",
        "compute_time",
    }
    missing = expected.difference(set(df.columns))
    if missing:
        raise ValueError("extended_metrics.csv missing columns: {}".format(sorted(missing)))
    return df


def _bar_plot(methods, values, ylabel, title, save_path, color):
    plt.figure(figsize=(6, 4))
    plt.bar(methods, values, color=color)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_recovery_curve_if_exists(input_dir):
    full_path = os.path.join(input_dir, "full_retrain", "recovery_curve.csv")
    unlearn_path = os.path.join(input_dir, "selective_unlearn", "recovery_curve.csv")
    if not (os.path.exists(full_path) and os.path.exists(unlearn_path)):
        return

    full_df = pd.read_csv(full_path)
    unlearn_df = pd.read_csv(unlearn_path)
    if "step" not in full_df.columns or "step" not in unlearn_df.columns:
        return
    if "critic_loss_retain" not in full_df.columns or "critic_loss_retain" not in unlearn_df.columns:
        return

    plt.figure(figsize=(7, 4))
    plt.plot(full_df["step"], full_df["critic_loss_retain"], label="Full Retrain", linewidth=2)
    plt.plot(unlearn_df["step"], unlearn_df["critic_loss_retain"], label="Selective Unlearn", linewidth=2)
    plt.xlabel("steps", fontsize=11)
    plt.ylabel("loss", fontsize=11)
    plt.title("Recovery Curve Comparison", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, "recovery_curve_comparison.png"), dpi=300)
    plt.close()


def _paper_summary_figure(df, input_dir):
    order = ["baseline", "full_retrain", "selective_unlearning"]
    sub_df = df.set_index("method").reindex(order).reset_index()
    methods = sub_df["method"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(methods, sub_df["avg_reward"], color="#4C78A8")
    axes[0, 0].set_title("Reward Comparison", fontsize=11)

    axes[0, 1].bar(methods, sub_df["malicious_reward"], color="#F58518")
    axes[0, 1].set_title("Forget Effect", fontsize=11)

    pd_df = sub_df[sub_df["method"].isin(["full_retrain", "selective_unlearning"])]
    axes[1, 0].bar(pd_df["method"], pd_df["policy_divergence"], color="#54A24B")
    axes[1, 0].set_title("Policy Divergence", fontsize=11)

    ct_df = sub_df[sub_df["method"].isin(["full_retrain", "selective_unlearning"])]
    axes[1, 1].bar(ct_df["method"], ct_df["compute_time"], color="#E45756")
    axes[1, 1].set_title("Compute Cost", fontsize=11)

    for ax in axes.flatten():
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Stability-Preserving Selective Unlearning in Centralized-Critic MARL", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(input_dir, "paper_summary_figure.png"), dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    ext_path = os.path.join(input_dir, "extended_metrics.csv")
    df = _load_extended_metrics(ext_path)

    order = ["baseline", "full_retrain", "selective_unlearning"]
    sub_df = df.set_index("method").reindex(order).reset_index()

    _bar_plot(
        sub_df["method"].tolist(),
        sub_df["avg_reward"].tolist(),
        ylabel="avg_reward",
        title="Reward Comparison",
        save_path=os.path.join(input_dir, "reward_comparison.png"),
        color="#4C78A8",
    )
    _bar_plot(
        sub_df["method"].tolist(),
        sub_df["malicious_reward"].tolist(),
        ylabel="malicious_reward",
        title="Malicious Scenario Reward",
        save_path=os.path.join(input_dir, "forget_effect.png"),
        color="#F58518",
    )

    pd_df = sub_df[sub_df["method"].isin(["full_retrain", "selective_unlearning"])]
    _bar_plot(
        pd_df["method"].tolist(),
        pd_df["policy_divergence"].tolist(),
        ylabel="policy_divergence",
        title="Policy Divergence",
        save_path=os.path.join(input_dir, "policy_divergence.png"),
        color="#54A24B",
    )
    _bar_plot(
        pd_df["method"].tolist(),
        pd_df["compute_time"].tolist(),
        ylabel="compute_time (s)",
        title="Compute Time Comparison",
        save_path=os.path.join(input_dir, "compute_cost.png"),
        color="#E45756",
    )

    _paper_summary_figure(df, input_dir)
    _plot_recovery_curve_if_exists(input_dir)
    print("Saved visualizations in {}".format(input_dir))


if __name__ == "__main__":
    main()
