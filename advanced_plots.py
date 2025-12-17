import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

sns.set(style="darkgrid")

# Load Data
try:
    with open("trained_data/s3-dec-cRewards-Unlearn-epoch-10.pkl", "rb") as f:
        unlearn_rewards = pickle.load(f)
except FileNotFoundError:
    print("Data file not found.")
    exit()

# Prepare Data Frame for Line Plot
epochs = list(range(1, len(unlearn_rewards) + 1))
df_line = pd.DataFrame({"Episode": epochs, "Reward": unlearn_rewards})

# 1. Training Curve (Rolling Average)
plt.figure(figsize=(10, 6))
sns.lineplot(x="Episode", y="Reward", data=df_line, alpha=0.3, color="blue", label="Raw Reward")
# Calculate rolling average for smoothness
df_line["Rolling_Reward"] = df_line["Reward"].rolling(window=50).mean()
sns.lineplot(x="Episode", y="Rolling_Reward", data=df_line, color="darkblue", linewidth=2, label="Rolling Avg (50 eps)")

plt.title("Unlearning Phase: Agent Performance Over Time")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.savefig("unlearning_curve.pdf", bbox_inches='tight')
print("Generated unlearning_curve.pdf")

# 2. Comparison Box Plot (Simulated Data for Valid Env vs Unlearning Env)
# We take the last 100 episodes as "Unlearning Performance" (Map 0)
# We take a sample of "Normal Performance" from the baseline data we know (~25)
# Since the raw rewards are cumulative (around -4000 to +100 range), we'll use the raw data directly.

# Create synthetic "Retained Map" data based on the "Model Utility" average we know (approx 25 per step * 10 steps = 250? No, let's just use the unlearning data distribution)
# Actually, the unlearning rewards are very negative (hitting obstacles). 
# Let's show the distribution of rewards during the unlearning process to show variance.

plt.figure(figsize=(8, 6))
sns.boxplot(y=unlearn_rewards[-200:], color="orange") # Last 200 episodes
plt.title("Reward Distribution (Final 200 Unlearning Episodes)")
plt.ylabel("Reward per Episode")
plt.savefig("reward_distribution.pdf", bbox_inches='tight')
print("Generated reward_distribution.pdf")
