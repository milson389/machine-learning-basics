import pandas as pd
from sqlalchemy import true
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('datasets/Mall_Customers.csv')
df = df.rename(columns={'Gender':'gender', 'Age': 'age', 'Annual Income (k$)':'annual_income', 'Spending Score (1-100)': 'spending_score'})
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)

X = df.drop(['CustomerID', 'gender'], axis=1)

clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)


print(clusters)

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
plt.show()