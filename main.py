import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('payment_fraud.csv')
df = df.sample(frac=0.1, random_state=1)

df.describe()

kolich = ['accountAgeDays','numItems','localTime','paymentMethodAgeDays']
kach = ['paymentMethod']

print(len(df),len(df.drop_duplicates()))

df = df.drop_duplicates()
print(df.isnull().sum())

df = pd.get_dummies(df, columns = kach)

from sklearn. preprocessing import StandardScaler
ss = StandardScaler()
df1 = df.copy()
ss.fit(df1)
df1.iloc[:, :] = ss.transform(df1)
df1

# Выбросы
for i in df1:
    mean = np.mean(df1[i])
    std = np.std(df1[i])
    interval = [mean - 3 * std, mean + 3 * std]
    for index,val in enumerate(df1[i]):
        if not (interval[0] <= val <= interval[1]):
            df1[i][index] = np.nan
print(df1.isnull().sum())
df1 = df1.dropna()
print(df1.isnull().sum())

df2 = df1.copy()
df2.columns = [f'pc{i}' for i in range(1,len(df1.columns)+1)]

from sklearn.decomposition import PCA
pca = PCA(svd_solver="full")
df2.iloc[:, :] = pca.fit_transform(df2.iloc[:, :])
pca.components_

plt.figure()
plt.grid()
plt.scatter(df2['pc1'], df2['pc2'],  edgecolor='black', lw=.1, cmap='jet')
plt.title("Rotated projected points [2D]")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.axis('equal')
plt.show()

from scipy.cluster.hierarchy import dendrogram, ward
linkage_array = ward(df2)
plt.figure(figsize=(20, 10))
dendrogram(linkage_array, truncate_mode='level', no_labels=True, p=10)
plt.title("Dendrogram")
plt.show()


import SimpSOM as sps

net = sps.somNet(20, 20, df2, PBC=True)