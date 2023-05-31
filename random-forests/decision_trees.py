import numpy as np


class Node:  # Linden, Koya
    """This is a class for a node.

    Attributes
    ----------
    feature : numerical or string
        The feature of the node.
    threshold : numerical or string
        The threshold when splitting a node.
    left : Node, default=None
        The left node of this node.
    right : Node, default=None
        The right node of this node.
    data : array_like
        The data in the node.
    """
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, *, data=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.data = data

    def is_leaf(self):
        return self.data is not None


class DecisionTree:  # Yiyan, Alex, Chengdong
    """This is an implementation of a decision tree.

    Attributes
    ----------
    max_depth : int, default=100
        The maximum number of splits.
    min_leaf_size : int, default=1
        The minimum size of a leaf node.
    n_cadidates : int, default=None
        The number of candidates in a split.
    criterion : string, default="gini"
        The criterion when growing a decision tree.
    tree
        The decision tree.
    n_labels : int
        The number of distinct labels in the training dataset.
    n_features : int
        The number of features in the training dataset.
    feature_type : array_like
        An array consists of types of features.
        (Continuous=0 or categorical=1)

    Parameters
    ----------
    X : ndarray
        The dataset.
        The rows of `X` are data points and the columns corespond to features.
    y : array_like
        The labels.
    """
    def __init__(self, max_depth=100, min_leaf_size=1,
                 n_candidates=None, criterion="gini"):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_candidates = n_candidates
        self.criterion = criterion
        self.tree = None
        self.n_labels = None
        self.n_features = None
        self.feature_type = None

    def fit(self, X, y, feature_type=None):
        """Fit the training dataset `X` and the labels `y`
        by the decision tree."""
        self.n_labels = len(np.unique(y))
        self.n_features = X.shape[1]
        self.feature_type = feature_type
        self.tree = self._grow(X, y)

    def predict(self, X):
        """Return the predicted labels `y` for dataset `X`."""
        pass

    def _grow(self, X, y, depth=0):
        # Stopping conditions.
        # Stop growing and return a leaf node.
        if depth >= self.max_depth:
            return Node(data=self._majority_vote(y))
        # Stop if leaf_size <= min.
        ## TODO

        # Find the best splitting feature and threshhold
        # using greedy approach
        feature, threshold = self._cutpoint(X, y)

        # Split the data using the best cutpoint.
        # Create two boolean arrays to slice left and right data.
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        # Increment depth and call _grow() recursively.
        # left_data = X[left_idx]
        # right_data = X[right_idx]
        left_node = self._grow(X[left_idx], y[left_idx], depth+1)
        right_node = self._grow(X[right_idx], y[right_idx], depth+1)

        return Node(feature, threshold, left_node, right_node)

    def _cutpoint(self, X, y):
        """Return the feature and the threshold of the cutpoint
        of a split using greedy approach.

        For each feature, randomly choose a threshold
        """
        best_feature = None
        best_threhold = None
        best_score = None

        for i in range(self.n_features):
            threholds = []
            scores = []
            # If categorical.
            if self.feature_type[i]:
                # Categorical split.
                ## sudo
                # for _ in range(self.n_candidates):
                #     randomly choose a threshold
                #     thresholds.append(threshold)
                #     compute score
                #     scores.append(score)
                pass
            # Otherwise continuous.
            # ...

            ## sudo
            # score = best(scores)
            # threshold = thresholds[scores.index(score)]

            # Initialise best_score.
            if best_score is None:
                best_score = score
                pass
            # If score is better than best_score:
            #    best_score = score
            #    best_feature = feature[i]
            #    best_threshold = threshold

        return best_feature, best_threhold

    def _majority_vote(self, y):
        """Return the most common label in `y`.

        In case of a tie, choose randomly.
        """
        pass

    def _traverse(self, x, node):
        """Traverse the decision tree with data point `x`
        from node `node`."""
        # if node.is_leaf():
        #     return majority_vote

        # if <= threshold:
        #     return self._traverse(x, node.left)
        # else:
        pass


def gini_index(y):
    # G = sum(p_m_k(1 - p_m_k)), 1 <= k <= K
    # np.bincount counts the number of occurences of each value in y.
    # len(y) = number of classes.
    ps = np.bincount(y)/len(y)
    return np.sum(ps * (1 - ps))


def classification_error_rate(y):
    pass
