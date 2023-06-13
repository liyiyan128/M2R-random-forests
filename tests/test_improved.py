import numpy as np
from sklearn.datasets import load_iris
from sklearn import model_selection
import importlib
rf = importlib.import_module("random-forests")


iris = load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
W = np.array(X, dtype=object)
cat = ["s" for _ in range(X.shape[0])]
for i in range(X.shape[0]):
    if X[i, 3] <= 0.8:
        cat[i] = "a"
    elif 0.8 < X[i, 3] <= 1.6:
        cat[i] = "b"
    elif X[i, 3] > 1.6:
        cat[i] = "c"
W[:, 3] = cat
W_train, W_test, z_train, z_test = model_selection.train_test_split(W, y)


# Test on continuous data.
decision_tree = rf.ImprovedDecisionTree()
decision_tree.fit(X_train, y_train)
decision_tree._grow(X_train, y_train)
y_predicted = decision_tree.predict(X_test)


# Test on categorical data.
decision_tree2 = rf.ImprovedDecisionTree()
decision_tree2.fit(W_train, z_train, feature_type=[0, 0, 0, 1])
decision_tree2._grow(W_train, z_train)
z_predicted = decision_tree2.predict(W_test)
