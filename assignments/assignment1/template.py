import numpy as np
import pandas as pd

np.random.seed(seed=0)


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    Parameters
    ----------
    y_true : array, shape = [n_instances]
        Actual target values.
    y_pred : array, shape = [n_instances]
        Predicted target values.

    Returns
    -------
    score : float
        Returns the fraction of correctly classified instances.
    """
    score = y_true == y_pred
    return np.average(score)


class Perceptron(object):
    """Perceptron."""

    def __init__(self):
        self.weight_ = None

    def _update_weight(self, x, y):
        """Updates the weight vector.

        The new weight w_{k+1} is computed as:
            w_{k+1} = w_k + y * x
        where x is the feature vector and y is the class value.

        Parameters
        ----------
        x : array, shape = [n_features + 1]
            Feature vector for instance to update.
        y : int
            Target value for instance to update.

        Returns
        -------
        self.weight_ : array, shape = [n_features + 1]
            Updated weight vector.
        """
        ### INSERT CODE HERE ####

    def _decision_function(self, x):
        """Decides class labels for instance.

        The predicted class label is computed as:
            sign(w^T * x)
        where w is the weight vector and x is the feature vector.

        Parameters
        ----------
        x : array, shape = [n_features + 1]
            Feature vector for instance to update.

        Returns
        -------
        C : int
            Predicted target value for instance.
        """
        ### INSERT CODE HERE ####

    def fit(self, X, y):
        """Fit the perceptron.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training data.
        y : array, shape = [n_instances, n_targets]
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        n_instances, n_features = np.shape(X)

        # Insert bias term.
        bias = np.negative(np.ones(shape=(n_instances, 1)))
        X = np.hstack((X, bias))

        # Initialize weight vector.
        self.weight_ = np.random.rand(n_features + 1)

        # Repeatedly iterate through all of the instances from first to last,
        # correcting each misclassified instance until no misclassified
        # instances remain. Print a line for each corrected instance.
        i = 0
        while True:
            for j in range(n_instances):
                ### INSERT CODE HERE ####

                if acc == 1.0:
                    return self

        return self


with open('output.txt', 'w') as f_out:
    df = pd.read_csv('digits_binary.csv')
    ### INSERT CODE HERE ####
