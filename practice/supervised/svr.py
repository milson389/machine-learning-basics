import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('datasets/Salary_Data.csv')

X = df['YearsExperience']
y = df['Salary']

X = X[:, np.newaxis]

model = SVR()
parameters = {
    'kernel':['rbf'],
    'C': [1000, 10000, 100000],
    'gamma': [0.5, 0.05, 0.005]
}

grid_search = GridSearchCV(model, parameters)
grid_search.fit(X, y)
print(grid_search.best_params_)

model2 = SVR(C=100000, gamma=0.005, kernel='rbf')
model2.fit(X, y)

plt.scatter(X, y)
plt.plot(X, model2.predict(X))
plt.show()