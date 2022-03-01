from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = DecisionTreeClassifier().fit(X_train, y_train)

pca = PCA(n_components=2)
pca_attributes = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

model2 = DecisionTreeClassifier().fit(X_train_pca, y_train)
print(model2.score(X_test_pca, y_test))
