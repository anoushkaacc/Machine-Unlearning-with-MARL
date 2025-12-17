import pandas
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
game_type = "grid_world"
retain_reward = [25.16008949499254,25.140105583106703] if game_type == "grid_world" else [30.615856195298097,30.57562285308169]

df = pandas.read_csv(f"trained_result/{game_type}/combined_result.csv")
data = {
    "Model Utility":df["Model Utility"].tolist(),
    "Forget Quality":df["Forget Quality"].tolist(),
    "Methodology":df["Methodology"].tolist(),
}

fig, ax = plt.subplots()

plt.xlabel('Model Utility',fontsize=16)
plt.ylabel('Forget Quality\n(log p-value)',fontsize=16)
plt.tick_params(axis="x",labelsize=16)
plt.tick_params(axis="y",labelsize=16)

if game_type == "grid_world":
    plt.axline((0, -0.333), (26, -0.333), color="purple", linestyle="dashed", label="Before Unlearning")
    plt.text(-0.12, 0.425, '-0.33', transform=plt.gca().transAxes, fontsize=15, color='purple')

    plt.xlim(25.05,25.2)
    plt.ylim(-0.620,0.020)

else:
    plt.axline((0, -0.33), (26, -0.33), color="purple", linestyle="dashed", label="Before Unlearning")
    plt.text(-0.13, 0.05, '-0.33', transform=plt.gca().transAxes, fontsize=15, color='purple')

    plt.xlim(30.15,31.75)
    plt.ylim(-0.35,0.003)
sns.lineplot(
    data=data, x="Model Utility", y="Forget Quality",hue="Methodology",sort=False
)

plt.scatter(x=retain_reward[0],y=0,label="LFS",marker="*",c = "red")
plt.scatter(x=retain_reward[0],y=0,label="LFS",marker="*",c = "red")
plt.scatter(x=retain_reward[1],y=0,label = "Non-transfer LFS",marker="*", c = "green")

# plt.legend(loc='best',fontsize=10)
plt.legend(bbox_to_anchor=(1.25, 1.2),ncol=5,fontsize=10)
# sns.scatterplot(data=data,x="Model Utility", y="Forget Quality",hue="Methodology",s=[20,40,60,80,20,40,60,80,100],legend=False)
sns.scatterplot(data=data,x="Model Utility", y="Forget Quality",hue="Methodology",legend=False)

if game_type == "grid_world":
    plt.title("Grid World",fontsize=16)
else:
    plt.title("Aircraft Landing",fontsize=16)

fig.savefig(f"{game_type}.pdf",format="pdf",bbox_inches="tight")

# fig.show()
