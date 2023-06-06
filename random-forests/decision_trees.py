import numpy as np


class Node:
    """This is a class for a node.

    Attributes
    ----------
    feature : numerical or string
        The feature used to split the node.
    threshold : numerical or string
        The threshold used to split the node.
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


class DecisionTree:
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
    n_features : int
        The number of features in the training dataset.
    data_range : array_like
        An array consists of the range of the current data for each feature.
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
    def __init__(self, max_depth=100, min_leaf_size=1,
                 n_candidates=10, criterion="gini"):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_candidates = n_candidates
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y, feature_type="continuous"):
        """Fit the training dataset `X` and the labels `y`
        by the decision tree."""
        self.n_features = X.shape[1]
        self.feature_type = []
        # Initialise `feature_type`.
        if isinstance(feature_type, str):
            if feature_type == "continuous":
                self.feature_type = [0 for _ in range(self.n_features)]
            if feature_type == "categorical":
                self.feature_type = [1 for _ in range(self.n_features)]
        else:
            self.feature_type = feature_type
        # Initialise `data_range`.
        self.data_range = np.empty(self.n_features, dtype=object)
        # Grow the decision tree.
        self.tree = self._grow(X, y)

    def predict(self, X):
        """Return the predicted labels `y` for dataset `X`."""
        y = [self._traverse(x, self.tree) for x in X]
        return np.array(y)

    def _grow(self, X, y, depth=0):
        """Grow the decision tree.

        Grow the decision tree by recursively calling _grow
        until hitting one of the stopping conditions.
        Instantiate and return a leaf node.
        """
        # Stopping conditions.
        # If depth >= math_depth,
        # or leaf_size <= min_leaf_size,
        # or `y` contains only one unique label.
        valid_cols_barr = valid_cols(X)
        if (depth >= self.max_depth
                or len(y) <= self.min_leaf_size
                or len(np.unique(y)) == 1
                or not np.any(valid_cols_barr)):
            return Node(data=self._majority_vote(y))

        # Find the best splitting feature and threshhold
        # using greedy approach.
        feature, threshold = self._cutpoint(X, y, valid_cols_barr)
        # Split the data using the best cutpoint.
        # Create two boolean arrays to slice left and right data.
        if self.feature_type[feature]:  # If categorical.
            left_idx = np.isin(X[:, feature], threshold)
        else:
            left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        # Increment depth and call _grow() recursively.
        # left_data = X[left_idx]
        # right_data = X[right_idx]
        left_node = self._grow(X[left_idx], y[left_idx], depth+1)
        right_node = self._grow(X[right_idx], y[right_idx], depth+1)

        return Node(feature, threshold, left_node, right_node)

    def _cutpoint(self, X, y, v_cols):
        """Return the feature and the threshold of the cutpoint
        of a split using greedy approach.

        For each feature, randomly sample thresholds.
        """
        best_score = float('inf')
        best_feature = None
        best_threshold = None

        features = np.arange(self.n_features)[v_cols]
        for _ in range(self.n_candidates):
            # Choose a feature randomly.
            feature = np.random.choice(features)
            # Categorical split.
            if self.feature_type[feature]:
                score, threshold = self._split_categorical(X, y, feature)
            # Continuous split.
            else:
                score, threshold = self._split_continuous(X, y, feature)
            # Update best feature and threshold
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold
        return best_feature, best_threshold

    def _split_continuous(self, X, y, feature):
        """Split continuous data."""
        X_col = X[:, feature]
        # Find the range of the data for the feature.
        self.data_range[feature] = np.array([X_col.min(), X_col.max()])
        lo, hi = self.data_range[feature]
        # Randomly choose a threshold.
        threshold = np.random.uniform(lo, hi)
        score = self._criterion(X_col, y, feature, threshold)
        return score, threshold

    def _split_categorical(self, X, y, feature):
        """Split categorical data."""
        X_col = X[:, feature]
        # Find the range of the data for the feature.
        self.data_range[feature] = np.unique(X_col)
        n_category = len(self.data_range[feature])
        while True:
            # Create a random boolean array for random subset slicing.
            subset_idx = np.random.choice([True, False], n_category)
            # Avoid empty set and the set itself.
            if subset_idx.any() and not subset_idx.all():
                break
        threshold = self.data_range[feature][subset_idx]
        score = self._criterion(X_col, y, feature, threshold)
        return score, threshold

    def _criterion(self, X_col, y, feature, threshold):
        '''Compute the score using specified criterion.'''
        # Split `X_col` by `threshold`.
        if self.feature_type[feature]:  # If categorical.
            left_idx = np.isin(X_col, threshold)
        else:
            left_idx = X_col <= threshold
        right_idx = ~left_idx
        if self.criterion == "gini_weighted":
            left_score = gini_index(y[left_idx])
            right_score = gini_index(y[right_idx])
            # Weighted average.
            rt = (left_score*len(left_idx)
                  + right_score*len(right_idx)) / len(y)
        elif self.criterion == "gini":
            left_score = gini_index(y[left_idx])
            right_score = gini_index(y[right_idx])
            # Weighted average.
            rt = left_score + right_score
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

        if self.feature_type[node.feature]:  # If categorical.
            if np.isin(x[node.feature], node.threshold):
                return self._traverse(x, node.left)  # Traverse left subtree.
            else:
                return self._traverse(x, node.right)  # Traverse right subtree.
        else:
            if x[node.feature] <= node.threshold:
                return self._traverse(x, node.left)  # Traverse left subtree.
            else:
                return self._traverse(x, node.right)  # Traverse right subtree.


def gini_index(y):
    """Return the gini index for labels `y`."""
    # G = sum(p_m_k(1 - p_m_k)), 1 <= k <= K

    ps = np.unique(y, return_counts=True)[1] / len(y)
    return np.sum(ps * (1 - ps))


def valid_cols(data):
    """Return boolean array of columns where not all entries are the same
        as well as label column and no_valid_cols is True if all columns which
        are not label have all entries the same."""
    barr = np.empty(data.shape[1], dtype=bool)
    for j in range(data.shape[1]):
        barr[j] = not len(np.unique(data[:, j])) == 1
    return barr
