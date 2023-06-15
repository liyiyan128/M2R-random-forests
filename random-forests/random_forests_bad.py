import numpy as np
# from scipy.stats import bootstrap
from .decision_trees import DecisionTree
from .decision_trees import majority_vote


class BadRandomForest:
    """This is an implementation of a random forest.

    Attributes
    ----------
    n_trees : int, default=10
        The number of decision trees.
    max_depth : int, default=100
        The maximum number of splits in a decision tree.
    min_leaf_size : int, default=1
        The minimum size of a leaf node.
    n_cadidates : int, default=10
        The number of candidate splits.
    criterion : string, default="gini"
        The criterion when growing a decision tree.
    m_features : int, default=None
        The number of features considered at each split.
        If an int then must have 1 <= m_features <= X.shape[1].
        If m_features=None then considers all features.
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

    def fit(self, X, y, feature_type="continuous", m_features=None):
        """Fit the random forest."""
        # Ensembling decision trees.
        self.forest = []
        for _ in range(self.n_trees):
            tree = DecisionTree(self.max_depth, self.min_leaf_size,
                                self.n_candidates, self.criterion)
            tree.fit(X, y, feature_type, m_features)
            self.forest.append(tree)

    def predict(self, X):
        """Return the predicted labels for `X`."""
        predictions = np.array([tree.predict(X) for tree in self.forest])
        return majority_votes(predictions)


def majority_votes(predictions):
    """Return the majority votes for `predictions`."""
    majority_votes = []
    for col in predictions.T:
        majority_votes.append(majority_vote(col))
    return np.array(majority_votes)


def misclassification_rate(y_tilde, y):
    """Return the proportion of misclassifications."""
    return sum(np.logical_not(y_tilde == y)) / len(y)
