# %% read data
import pandas as pd

df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None)
# 1 area A,
# 2 perimeter P,
# 3 compactness C = 4*pi*A/P^2,
# 4 length of kernel,
# 5 width of kernel,
# 6 asymmetry coefficient
# 7 length of kernel groove.
# 8 target
df.columns = [
    "area",
    "perimeter",
    "compactness",
    "length_kernel",
    "width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]


# %%
df.describe()


#%%
import seaborn as sns


sns.scatterplot(
    x="area",
    y="asymmetry_coefficient",
    data=df,
    hue="target",
    legend="full",
)


# %% also try lmplot and pairplot
import matplotlib.pyplot as plt
sns.lmplot(x="compactness", y="perimeter", data=df, hue="target", legend="full", ci=True)
plt.title('Scatter Plot: Compactness vs Perimeter')
plt.show()

#%%
sns.pairplot(df, hue="target")
plt.suptitle('PairPlot for all Features with Hue as Target', y=1.02)

#%%
sns.boxplot(x="target", y="area", data=df)
plt.title('Boxplot: Area Distribution Across Different Targets')
plt.show()


# %% determine the best numbmer of clusters
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score


x = df.drop("target", axis=1)
y = df["target"]
inertia = {}
homogeneity = {}

# %%
# use kmeans to loop over candidate number of clusters 
# store inertia and homogeneity score in each iteration

for n_clusters in range(1, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x)
    labels = kmeans.labels_
    inertia[n_clusters] = kmeans.inertia_
    homogeneity[n_clusters] = homogeneity_score(y, labels)

print("Number of Clusters\tInertia\tHomogeneity")
for n_clusters in range(1, 10):
    print(f"{n_clusters}\t\t\t{inertia[n_clusters]}\t{homogeneity[n_clusters]}")

plt.figure(figsize=(12, 6))
plt.plot(list(inertia.keys()), list(inertia.values()), marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
# %% 
ax = sns.lineplot(
    x=list(inertia.keys()),
    y=list(inertia.values()),
    color="blue",
    label="inertia",
    legend=None,
)
ax.set_ylabel("inertia")
ax.twinx()
ax = sns.lineplot(
    x=list(homogeneity.keys()),
    y=list(homogeneity.values()),
    color="red",
    label="homogeneity",
    legend=None,
)
ax.set_ylabel("homogeneity")
ax.figure.legend()


# %%
