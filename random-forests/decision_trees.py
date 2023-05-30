class Node:  # Linden, Koya
    """This is a class for a node.

    Attributes
    ----------
    feature : numerical or string
        The feature of the node.
    threshold : numerical or string
        The threshold when splitting a node.
    category : string
        The category of the node.
    left : Node, default=None
        The left node of this node.
    right : Node, default=None
        The right node of this node.
    data : array_like
        The data in the node.
    """
    def __init__(self, feature=None, threshold=None, category=None,
                 left=None, right=None, *, data=None):
        self.feature = feature
        self.threshold = threshold
        self.category = category
        self.left = left
        self.right = right
        self.data = data

    def _split(self):
        pass

    def _next(self):
        pass

    # def node_error(self):
    #     pass


class DecisionTree:  # Yiyan, Alex, chengdong
    """This is an implementation of a decision tree.

    Attributes
    ----------
    max_depth : int
        The maximum number of splits.
    min_leaf_size : int
        The minimum size of a leaf node.
    criterion : string, default="gini"
        The criterion when growing a decision tree.
    root : Node
        The root node of the decision tree.

    Parameters
    ----------
    X : ndarray
        The training dataset.
    y : array_like
        The labels.
    """
    def __init__(self, max_depth, min_leaf_size, criterion="gini"):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate(self):
        pass

    def _grow(self, X, y, max_depth):
        pass

    def _split(self):
        pass


def gini_index(y):
    pass


def classification_error_rate(y):
    pass
