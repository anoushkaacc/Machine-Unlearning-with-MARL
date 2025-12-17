import pandas
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
game_types = ["grid_world", "aircraft_landing"]
retain_rewards = {
    "grid_world": [25.16008949499254, 25.140105583106703],
    "aircraft_landing": [30.615856195298097, 30.57562285308169]
}

fig, axes = plt.subplots(1, len(game_types), figsize=(15, 7.5))
legend_labels = []  # To store legend labels for both subplots

for i, game_type in enumerate(game_types):
    df = pandas.read_csv(f"trained_result/{game_type}/combined_result.csv")
    data = {
        "Model Utility": df["Model Utility"].tolist(),
        "Forget Quality": df["Forget Quality"].tolist(),
        "Methodology": df["Methodology"].tolist(),
    }

    ax = axes[i]
    ax.set_xlabel('Model Utility', fontsize=18)
    ax.set_ylabel('Forget Quality\n(log p-value)', fontsize=18)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    if game_type == "grid_world":
        ax.axline((0, -0.333), (26, -0.333), color="purple", linestyle="dashed", label="Before Unlearning")
        ax.text(-0.115, 0.425, '-0.33', transform=ax.transAxes, fontsize=18, color='purple')
        ax.set_xlim(25.05, 25.2)
        ax.set_ylim(-0.620, 0.020)
    else:
        ax.axline((0, -0.33), (26, -0.33), color="purple", linestyle="dashed", label="Before Unlearning")
        ax.text(-0.13, 0.05, '-0.33', transform=ax.transAxes, fontsize=18, color='purple')
        ax.set_xlim(30.15, 31.75)
        ax.set_ylim(-0.35, 0.003)

    sns.lineplot(data=data, x="Model Utility", y="Forget Quality", hue="Methodology", sort=False, ax=ax)
    ax.scatter(x=retain_rewards[game_type][0], y=0, label="LFS", marker="*", c="red")
    ax.scatter(x=retain_rewards[game_type][1], y=0, label="Non-transfer LFS", marker="*", c="green")
    sns.scatterplot(data=data, x="Model Utility", y="Forget Quality", hue="Methodology",s=[20, 40, 60, 80, 20, 40, 60, 80, 100], legend=False,ax=ax)

    ax.set_title(game_type.capitalize(), fontsize=18)
    ax.legend().set_visible(False)



handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=[0.5, 1.1],loc='upper center',ncol=5,fontsize=18)

# plt.legend(bbox_to_anchor=(1, 1.2), ncol=5, fontsize=10)
plt.tight_layout()
plt.savefig("merged_plots.pdf", format="pdf", bbox_inches="tight")
plt.show()
