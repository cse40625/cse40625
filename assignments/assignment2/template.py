import numpy as np
import pandas as pd


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


def sigmoid(z):
    """Computes the sigmoid function."""
    return 1. / (1. + np.exp(-1. * z))


class LogisticRegression(object):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    Parameters
    ----------
    max_iter : int, optional (default=100)
        Maximum number of iterations taken for the solver to converge.
    learning_rate : float, optional (default=0.1)
        The learning rate for weight updates.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    weight_ : array
        Weight vector.

    References
    ----------
    .. [1] Y. S. Abu-Mostafa, M. Magdon-Ismail, and H-T Lin. "Learning from
           Data." AMLBook, 2012.
    """

    def __init__(self, max_iter=500, learning_rate=0.01, random_state=0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.weight_ = None

    def _compute_gradient(self, weights, X, y):
        """Compute the gradient for the logsitic regression error.

        Logistic regression minimizes cross-entropy error:
            E(w) = 1/N \sum_N ln(1 + exp(y_N * w.T * x_n)),
        which can be minimized by computing the gradient:
            grad = -1/N \sum_N ((y_n * x_n) / (1 + exp(y_N * w.T * x_n))),
        where N is the number of instances and w are the weights.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features + 1]
            Training data.
        y : array, shape = [n_instances, n_targets]
            Target values (assumed to be {-1, 1}).

        Returns
        -------
        grad : int
            The gradient of the error.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Return the gradient to use to update weight vector.
        # ================================================

    def _decision_function(self, X):
        """Decides target value for instance(s).

        The predicted target value for each instance x of X is computed as:
            theta(w.T * x)
        where theta is the logistic function, w is the weight vector, and x is
        the feature vector.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features + 1]
            Feature matrix or vector for instance(s) to update.

        Returns
        -------
        P : float
            Predicted target value for instance(s).
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Return the predicted target value.
        # ================================================

    def fit(self, X, y):
        """Fit logistic regression.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training data.
        y : array, shape = [n_instances, n_targets]
            Target values (assumed to be {-1, 1}).

        Returns
        -------
        self : returns an instance of self.
        """
        np.random.seed(seed=self.random_state)

        # ================ YOUR CODE HERE ================
        # Instructions: Insert a negative bias term and initialize the weights
        # with random samples from a Normal distribution with zero mean and
        # unit variance. Iterate up to max_iter times. Each iteration, compute
        # the gradient and use it to update the weight vector. Using the
        # updated weight vector, apply the decision function to generate the
        # predicted class probabilities. Use these probabilities to predict the
        # target class values. Each iteration, print the current accuracy score
        # each iteration. Halt if no misclassified instances remain.
        # ================================================

    def predict(self, X):
        """Predict target values for instances in X.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Instances.

        Returns
        -------
        y_pred : array, shape = [n_instances]
            Predicted target value for instances.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Insert a negative bias term. Use the probabilities
        # output by the decision function to decide the predicted (binary)
        # target value. Use a probability of 0.5 as the threshold, with
        # positive (+1) class predictions assigned to predicted probabilites
        # greater than 0.5 and negative (-1) class predictions assigned to
        # predicted probabilities less than 0.5.
        # ================================================


with open('output.txt', 'w') as f_out:
    df = pd.read_csv('digits_binary.csv')
    X = df.ix[:, :-1]
    y = df.ix[:, -1]

    # ================ YOUR CODE HERE ================
    # Instructions: Initialize and fit the logistic regression model.
    # ================================================