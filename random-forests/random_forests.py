import numpy as np
from scipy.stats import bootstrap
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier:
    """This is an implementation of a random forest classifier.

    Attributes
    ----------

    """
    def __init__(self, n_trees=100, min_sample_split=2, max_depth=None,
                 n_features=None):
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : dataframe
            The training dataset.
        y : array-like
            The labels.
        """
        pass

    def predict(self, X):
        pass
