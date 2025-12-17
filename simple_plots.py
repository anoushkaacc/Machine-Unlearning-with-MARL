import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

# Data
data = {
    "Method": ["Decremental", "LFS (Baseline)", "Non-transfer LFS"],
    "Model Utility": [25.12, 25.16, 25.14]
}

df = pd.DataFrame(data)

# Create Bar Plot
plt.figure(figsize=(8, 6))
ax = sns.barplot(x="Method", y="Model Utility", data=df, palette="viridis")

# Add text labels on bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

plt.ylim(24, 26) # Zoom in to show differences
plt.title("Model Utility Comparison: Decremental vs Baselines")
plt.ylabel("Average Reward (Higher is Better)")
plt.savefig("comparison_bar.pdf", bbox_inches='tight')

print("comparison_bar.pdf generated successfully.")
