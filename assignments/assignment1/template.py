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


class Perceptron(object):
    """Perceptron.

    The perceptron, proposed by [1], is a supervised linear classifier that
    makes predictions by combining the feature vector with a set of weights and
    assigning binary output based on a classification threshold.

    This implementation inserts the threshold into the weight vector as a
    negative bias term (i.e., a feature column of -1). The weight vector is
    initialized with random samples from a uniform distribution over [0, 1).
    The model is updated only on misclassified instances, using the sign of the
    dot product of the weight and feature vectors as the target prediction.

    Parameters
    ----------
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
    .. [1] F. Rosenblatt. "The Perceptron: A Probabilistic Model for
           Information Storage and Organization in the Brain." Psychological
           Review 65 (6): 386-408, 1958.
    """

    def __init__(self, random_state=0):
        self.random_state = random_state

        self.weight_ = None

    def _update_weight(self, x, y):
        """Updates the weight vector.

        The new weight vector w_{k+1} is computed as:
            w_{k+1} = w_k + y * x
        where x is the feature vector and y is the target value.

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
        # ================ YOUR CODE HERE ================
        # Instructions: Return the updated weight vector.
        # ================================================

    def _decision_function(self, X):
        """Decides target value for instance(s).

        The predicted target value for each instance x of X is computed as:
            sign(w.T * x)
        where w is the weight vector and x is the feature vector.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features + 1]
            Feature matrix or vector for instance(s) to update.

        Returns
        -------
        C : int
            Predicted target value for instance(s).
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Return the predicted target value.
        # ================================================

    def fit(self, X, y):
        """Fit the perceptron.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training data.
        y : array, shape = [n_instances, n_targets]
            Target values (assumed to be {-1, 1}).

        Returns
        -------
        self : Returns an instance of self.
        """
        np.random.seed(seed=self.random_state)

        n_instances, n_features = np.shape(X)

        # Insert bias term.
        bias = np.negative(np.ones(shape=(n_instances, 1)))
        X_1 = np.hstack((X, bias))

        # Initialize weight vector.
        self.weight_ = np.random.rand(n_features + 1)

        # Repeatedly iterate through all of the instances from first to last.
        i = 0
        while True:
            for j in range(n_instances):
                # ================ YOUR CODE HERE ================
                # Instructions: Identify an instance misclassified by the
                # decision function, and use it to update the weight vector.
                # Print a line for each corrected instance with the line number
                # (starting from 0), a space, and the current fraction of
                # correctly classified instances to three decimal places. Halt
                # when all instances are correctly classified.
                # ================================================

                if acc == 1.0:
                    return self

        return self


with open('output.txt', 'w') as f_out:
    df = pd.read_csv('digits_binary.csv')
    # ================ YOUR CODE HERE ================
    # Instructions: Initialize and fit the perceptron.
    # ================================================
