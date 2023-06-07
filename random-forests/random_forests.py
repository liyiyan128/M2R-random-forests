import numpy as np
# from scipy.stats import bootstrap
from .decision_trees import DecisionTree
from .decision_trees import majority_vote


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
        1 <= m_features <= X.shape[1]
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
        if not m_features:
            for _ in range(self.n_trees):
                tree = DecisionTree(self.max_depth, self.min_leaf_size,
                                    self.n_candidates, self.criterion)
                # Bootstrap sample.
                X_bootstrap, y_bootstrap = bootstrap(X, y)
                tree.fit(X_bootstrap, y_bootstrap, feature_type)
                self.forest.append(tree)
        else:
            # Checking parameters
            if not 1 <= m_features <= X.shape[1]:
                raise ValueError("must have 1 <= m_features <= X.shape[1]")
            if isinstance(feature_type, str):
                if feature_type == "continuous":
                    feature_type = np.zeros(X.shape[1])
                elif feature_type == "categorical":
                    feature_type = np.ones(X.shape[1])
                else:
                    raise ValueError(f"{feature_type} is not valid for"
                                     + "feature_type parameter.")
            else:
                if len(feature_type) == X.shape[1]:
                    feature_type = np.array(feature_type)
                else:
                    raise ValueError("feature_type has wrong length")
            # Fitting ensemble of trees
            for _ in range(self.n_trees):
                tree = DecisionTree(self.max_depth, self.min_leaf_size,
                                    self.n_candidates, self.criterion)
                # Bootstrap sample.
                X_bootstrap, y_bootstrap = bootstrap(X, y)
                # Select the features to use
                feature_choice = np.random.choice(np.arange(X.shape[1]),
                                                  size=m_features,
                                                  replace=False)
                barr = np.zeros(X.shape[1], dtype=bool)
                barr[feature_choice] = True
                # Fit the tree but only using random m_features
                tree.fit(X_bootstrap[:, barr], y_bootstrap, feature_type[barr])
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


def bootstrap(X, y):
    """Return a bootstrap sample and the corresponding labels."""
    n_samples = X.shape[0]
    bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[bootstrap_idx], y[bootstrap_idx]
