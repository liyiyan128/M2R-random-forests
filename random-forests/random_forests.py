from .decision_trees import DecisionTree
import pandas as pd
import numpy as np


def choice(data):
    """For use with evaluate to select random mode if there are multiple."""
    return np.random.choice(data.dropna())


def bootstrap(data):
    """Return bootstrap sample of data."""
    nrows = data.shape[0]
    return data.iloc[np.random.randint(nrows, size=nrows), :]


class RandomForest:
    """Implements random forests algorithm.

    Attributes
    ----------
    data : pandas.core.frame.DataFrame
        The training data for the trees. Must be pre-prepared so that any
        categorical features of data have type category. E.g. if the data
        has a column "class" which is categorical, before passing the data
        to DecisionTree run data["class"] = data["class"].astype("category")
    label : string, the feature of data you want to predict.
        Must also be a column of data.
    m : int, Number of node splits to evaluate at each split
    max_depth : int, maximum number of splits in the DecisionTree
    ntrees : number of trees to grow in forest
    min_size : minimum region/data size of a node in a tree, any node with
        region size <= min_size will be turned into a terminal node.
    criterion : "gini" or "entropy" to use when scoring splits
    """
    def __init__(self, data, label, *, m=10, max_depth=10, ntrees=5,
                 min_size=1, criterion="gini"):
        if criterion not in {"gini", "entropy"}:
            raise TypeError(f"criterion {criterion} not understood, must be"
                            + "'gini' or 'entropy'.")
        self.data = data
        self.label = label
        self.m = m
        self.max_depth = max_depth
        self.ntrees = ntrees
        self.min_size = min_size
        self.criterion = criterion
        self.forest = self._fit()

    def _fit(self):
        """Fit the random forest."""
        return [DecisionTree(bootstrap(self.data), self.label, m=self.m,
                             max_depth=self.max_depth, min_size=self.min_size,
                             criterion=self.criterion)
                for i in range(self.ntrees)]

    def predict(self, obs):
        """Evaluate self given obs."""
        predictions = pd.DataFrame()
        for j, tree in enumerate(self.forest):
            predictions[j] = pd.Series(tree.predict(obs))
        return predictions.mode(axis=1).apply(choice, axis=1)

    # def test(self, tdata):
    #     """Given tdata with known labels returns info on
    #     # misclassifications."""
    #     labels = pd.DataFrame(tdata[self.label])
    #     labels["predicted"] = self.evaluate(tdata)
    #     labels["correct?"] = labels[self.label] == labels["predicted"]
