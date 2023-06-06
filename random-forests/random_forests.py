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
    m_features : int, default=None
        The number of features restricted to use when growing a decision tree.
    feature_type : array_like, default="continuous"
        An array consists of types of features,
        continuous: 0 or categorical: 1.
    forest
        The random forest.

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

    def fit(self, X, y, m_features=None, feature_type="continuous"):
        """Fit the random forest."""
        if not m_features:
            self.m_features = X.shape[1]
        else:
            self.m_features = m_features
        # Ensembling decision trees.
        for _ in range(self.n_trees):
            tree = DecisionTree(self.max_depth, self.min_leaf_size,
                                self.n_candidates, self.criterion)
            # Bootstrap sample.
            X_bootstrap, y_bootstrap = bootstrap(X, y, self.m_features)
            tree.fit(X_bootstrap, y_bootstrap, feature_type)
            self.forest.append(tree)

    def predict(self, X):
        """Return the predicted labels for `X`."""
        votes = np.array([tree.predict(X) for tree in self.forest])
        return self._majority_vote(votes)

    def _majority_vote(self, votes):
        """Return the majority vote for `votes`."""
        pass


def bootstrap(X, y, m_features):
    """Return a bootstrap sample and the corresponding labels.

    Restrict the bootstrap sample to m randomly chosen features.
    """
    n_samples, n_features = X.shape
    bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
    feature_idx = np.zeros(n_features, dtype=bool)
    for i in np.random.choice(n_features, size=m_features):
        feature_idx[i] = 1
    return X[bootstrap_idx][feature_idx], y[bootstrap_idx]
