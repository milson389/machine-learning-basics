import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


df = pd.read_csv('datasets/diabetes.csv')

X = df[df.columns[:8]]
y = df['Outcome']

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))