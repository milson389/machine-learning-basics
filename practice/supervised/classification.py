import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


iris = pd.read_csv('datasets/iris.csv')
iris.drop('Id', axis=1, inplace=True)

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

tree_model = DecisionTreeClassifier()
tree_model = tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

acc_score = round(accuracy_score(y_pred, y_test), 3)

print('Accuracy: ', acc_score)

print(tree_model.predict([[6.2, 3.4, 5.4, 2.3]])[0])