import numpy as np
from .decision_trees import Node
from .decision_trees import gini_index
from .decision_trees import valid_cols
from .decision_trees import majority_vote


class ImprovedDecisionTree:
    """This is an implementation of a decision tree.

    Attributes
    ----------
    max_depth : int, default=100
        The maximum number of splits.
    min_leaf_size : int, default=1
        The minimum size of a leaf node.
    n_cadidates : int, default=10
        The number of candidates in a split.
    criterion : string, default="gini"
        The criterion when growing a decision tree.
    tree
        The decision tree.
    n_features : int
        The number of features in the training dataset.
    feature_type : array_like, default="continuous"
        An array consists of types of features,
        continuous: 0 or categorical: 1.
    m_features : int, default=None
        The number of features considered at each split.
        If an int then must have 1 <= m_features <= X.shape[1].
        If m_features=None then considers all features.
    Parameters
    ----------
    X : ndarray
        The dataset.
        The rows of `X` are data points and the columns corespond to features.
    y : array_like
        The labels.
    """
    def __init__(self, max_depth=100, min_leaf_size=1, n_candidates=10,
                 criterion="gini"):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_candidates = n_candidates
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y, feature_type="continuous", m_features=None):
        """Fit the training dataset `X` and the labels `y`
        by the decision tree."""
        self.n_features = X.shape[1]
        # Initialise `m_features`.
        if m_features is None:
            self.m_features = self.n_features
        elif m_features <= 0 or m_features > self.n_features:
            raise ValueError("1 <= m_features <= " + f"{self.n_features}")
        else:
            self.m_features = m_features
        # Initialise `feature_type`.
        self.feature_type = []
        if isinstance(feature_type, str):
            if feature_type == "continuous":
                self.feature_type = np.zeros(self.n_features)
            if feature_type == "categorical":
                self.feature_type = np.ones(self.n_features)
        else:
            self.feature_type = feature_type
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
        # or if all rows of X are the same so cannot split
        v_cols = valid_cols(X)
        if (depth >= self.max_depth
                or len(y) <= self.min_leaf_size
                or len(np.unique(y)) == 1
                or not np.any(v_cols)):
            return Node(data=majority_vote(y))

        # Find the best splitting feature and threshhold
        # using greedy approach.
        feature, threshold = self._cutpoint(X, y, v_cols)
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
        # Restrict features.
        features = np.arange(self.n_features)[v_cols]
        if len(features) > self.m_features:
            features = np.random.choice(features, size=self.m_features,
                                        replace=False)
        # Split.
        for feature in features:
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
        lo, hi = X_col.min(), X_col.max()
        # Randomly choose thresholds of size=`n_candidate`.
        thresholds = np.random.uniform(lo, hi, self.n_candidates)
        scores = np.array([self._criterion(X_col, y, feature, threshold)
                          for threshold in thresholds])
        score = scores.min()
        threshold = thresholds[np.argmin(scores)]
        return score, threshold

    def _split_categorical(self, X, y, feature):
        """Split categorical data."""
        X_col = X[:, feature]
        # Find the different categories for the feature.
        categories = np.unique(X_col)
        n_category = len(categories)
        # Create an array of random subset indices.
        subset_idx = np.empty(self.n_candidates, dtype=object)
        for i in range(self.n_candidates):
            while True:
                # Create a random boolean array for subset slicing.
                idx = np.random.choice([True, False], n_category)
                # Avoid empty set and the set itself.
                if idx.any() and not idx.all():
                    break
            subset_idx[i] = idx
        thresholds = np.array([categories[idx] for idx in subset_idx],
                              dtype=object)
        scores = np.array([self._criterion(X_col, y, feature, threshold)
                          for threshold in thresholds])
        score = scores.min()
        threshold = thresholds[np.argmin(scores)]
        return score, threshold

    def _criterion(self, X_col, y, feature, threshold):
        '''Compute the score using specified criterion.'''
        # Split `X_col` by `threshold`.
        if self.feature_type[feature]:  # If categorical.
            left_idx = np.isin(X_col, threshold)
        else:
            left_idx = X_col <= threshold
        right_idx = ~left_idx
        # Choose criterion.
        if self.criterion == "gini_weighted":
            left_score = gini_index(y[left_idx])
            right_score = gini_index(y[right_idx])
            # Weighted average.
            rt = (left_score*len(left_idx)
                  + right_score*len(right_idx)) / len(y)
        if self.criterion == "gini":
            left_score = gini_index(y[left_idx])
            right_score = gini_index(y[right_idx])
            # Sum.
            rt = left_score + right_score
        return rt

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
