import pandas as pd
from sqlalchemy import true
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/Mall_Customers.csv')
df = df.rename(columns={'Gender':'gender', 'Age': 'age', 'Annual Income (k$)':'annual_income', 'Spending Score (1-100)': 'spending_score'})
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)

X = df.drop(['CustomerID', 'gender'], axis=1)

km5 = KMeans(n_clusters=5).fit(X)

X['Labels'] = km5.labels_

plt.figure(figsize=(8,4))
sns.scatterplot(X['annual_income'], X['spending_score'], hue=X['Labels'], palette=sns.color_palette('hls', 5))
plt.title('KMeans 5 Clusters')
plt.show()