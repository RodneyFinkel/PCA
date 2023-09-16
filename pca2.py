import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

wine = load_wine()

df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

df["target"] = wine.target

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(df.drop(columns=["target"]))

pca = PCA(n_components=2)

vecs = pca.fit_transform(X_std)

reduced_df = pd.DataFrame(data=vecs, columns=['Principal Component 1', 'Principal Component 2'])
final_df = pd.concat([reduced_df, df[['target']]], axis=1)

plt.figure(figsize=(8, 6))
targets = list(set(final_df['target']))
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    idx = final_df['target'] == target
    plt.scatter(final_df.loc[idx, 'Principal Component 1'], final_df.loc[idx, 'Principal Component 2'], c=color, s=50)

plt.legend(targets, title="Target", loc='upper right')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on the Wine Dataset')
plt.show()