import numpy as np
# from scipy.stats import bootstrap
from .decision_trees import DecisionTree


class RandomForest:
    """This is an implementation of a random forest.

    Attributes
    ----------
    n_trees : int
        The number of decision trees.
    max_depth : int, default=100
        The maximum number of splits in a decision tree.
    min_leaf_size : int, default=1
        The minimum size of a leaf node.
    n_cadidates : int, default=None
        The number of candidate splits.
    criterion : string, default="gini"
        The criterion when growing a decision tree.
    feature_type : array_like, default="continuous"
        An array consists of types of features,
        continuous: 0 or categorical: 1.

    Parameters
    ----------
    X : ndarray
        The dataset.
        The rows of `X` are data points and the columns corespond to features.
    y : array_like
        The labels.
    """
    def __init__(self, n_trees=10, max_depth=100, min_leaf_size=1,
                 n_candidates=10, criterion="gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_candidates = n_candidates
        self.criterion = criterion
        self.forest = []

    def fit(self, X, y, n_features):
        """Fit the random forest."""
        for _ in range(self.n_trees):
            tree = DecisionTree(self.max_depth, self.min_leaf_size,
                                self.n_candidates, self.criterion)
            X_bootstrap, y_bootstrap = bootstrap(X, y)
            tree.fit(X_bootstrap, y_bootstrap)
            self.forest.append(tree)

    def predict(self, X):
        """Return the predicted labels for `X`."""
        votes = np.array([tree.predict(X) for tree in self.forest])
        return self._majority_vote(votes)

    def _majority_vote(self, votes):
        """Return the majority vote for `votes`."""
        pass


def bootstrap(X, y):
    """Bootstrap sample."""
    n = X.shape[0]
    idx = np.random.choice(n, size=n, replace=True)
    return X[idx], y[idx]
