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


class DecisionTree:  # Yiyan, Alex, chengdong
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
    data_range : array_like
        An array consists of range for each feature.
    feature_type : array_like, default=None
        An array consists of types of features,
        continuous: 0 or categorical: 1.
        (If unspecified, take data as continuous.)

    Parameters
    ----------
    X : ndarray
        The dataset.
        The rows of `X` are data points and the columns corespond to features.
    y : array_like
        The labels.
    """
    def __init__(self, max_depth=100, min_leaf_size=1,
                 n_candidates=100, criterion="gini"):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_candidates = n_candidates
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y, data_range=None, feature_type=None):
        """Fit the training dataset `X` and the labels `y`
        by the decision tree."""
        self.n_labels = len(np.unique(y))
        self.n_features = X.shape[1]
        self.feature_type = feature_type
        self.data_range = data_range
        # If data_range=None unspecified,
        # find the range for each feature.
        if not data_range:
            self.data_range = np.empty(self.n_features, dtype=object)
            for i in range(self.n_features):
                try:
                    # Categorical.
                    if self.feature_type[i]:
                        self.data_range[i] = np.unique(X[:, i])
                    # Continuous otherwise.
                    self.data_range[i] = np.array([X[:, i].min(), X[:, i].max()])
                # feature_type=None, unspecified.
                except TypeError:
                    self.data_range[i] = np.array([X[:, i].min(), X[:, i].max()])

        self.tree = self._grow(X, y)

    def predict(self, X):
        """Return the predicted labels `y` for dataset `X`."""
        y = [self._traverse(x, self.tree) for x in X]
        return np.array(y)

    def _grow(self, X, y, depth=0):
        # Stopping conditions.
        # Stop growing and return a leaf node.
        if depth >= self.max_depth:
            return Node(data=self._majority_vote(y))
        # Stop if leaf_size <= min_leaf_size.
        if len(y) <= self.min_leaf_size:
            return Node(data=self._majority_vote(y))
        # Stop if `y` contains only one unique label.
        if len(np.unique(y)) == 1:
            return Node(data=self._majority_vote(y))

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

        For each feature, randomly choose a threshold.
        """
        best_score = None
        best_feature = None
        best_threshold = None

        for i in range(self.n_features):
            # Categorical.
            if self.feature_type[i]:
                # Randomly choose thresholds of size=n_candidates.
                thresholds = np.random.choice(self.data_range[i],
                                              self.n_candidates)
            # Otherwise continuous.
            else:
                lo, hi = self.data_range[i]
                thresholds = np.random.uniform(lo, hi, self.n_candidates)

            X_col = X[:, i]
            scores = np.array([self._criterion(X_col, y, threshold)
                              for threshold in thresholds])
            score = scores.min()
            threshold = thresholds[np.argmin(scores)]

            # Initialise best_score.
            if not best_score:
                best_score = score
            # If score is better than best_score.
            if score <= best_score:
                best_score = score
                best_feature = i
                best_threshold = threshold

        return best_feature, best_threshold

    def _criterion(self, X_col, y, threshold):
        '''Compute the score using specified criterion.'''
        if self.criterion == "gini":
            criterion = gini_index
        if self.criterion == "classification_error_rate":
            criterion = classification_error_rate

        # Split data `X_col` by `threshold`.
        left_idx = X_col <= threshold
        right_idx = ~left_idx

        left_score = criterion(y[left_idx])
        right_score = criterion(y[right_idx])
        # Weighted average.
        rt = (left_score*len(left_idx)
              + right_score*len(right_idx)) / (len(y) + 1e-16)
        return rt

    def _majority_vote(self, y):
        """Return the most common label in `y`.

        In case of a tie, choose randomly.
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        max_indices = np.where(counts == max_count)[0]
        if len(max_indices) == 1:
            # Only one label with the maximum count, return it.
            return unique_labels[max_indices[0]]
        else:
            # Multiple labels with the same maximum count, choose randomly.
            random_index = np.random.choice(max_indices)
            return unique_labels[random_index]

    def _traverse(self, x, node):
        """Traverse the decision tree with data point `x`
        from node `node`."""
        if node.is_leaf():
            return node.data
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)  # Traverse left subtree.
        else:
            return self._traverse(x, node.right)  # Traverse right subtree.


def gini_index(y):
    """Return the gini index for labels `y`."""
    # G = sum(p_m_k(1 - p_m_k)), 1 <= k <= K
    ps = len(np.unique(y)) / (len(y) + 1e-16)
    return np.sum(ps * (1 - ps))


def classification_error_rate(y):
    """Return the classification error rate for labels `y`."""
    if len(y) == 0:
        return 0.0

    counts = np.bincount(y)
    error_rate = 1 - np.max(counts)/(len(y) + 1e-16)
    return error_rate
