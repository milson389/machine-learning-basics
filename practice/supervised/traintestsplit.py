import sklearn
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split, cross_val_score


iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

clf = tree.DecisionTreeClassifier()

scores = cross_val_score(clf, x, y, cv=5)
print(scores)