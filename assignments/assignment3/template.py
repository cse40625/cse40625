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


def confusion_matrix(y_true, y_pred):
    """Confusion matrix.

    Parameters
    ----------
    y_true : array, shape = [n_instances]
        Actual target values.
    y_pred : array, shape = [n_instances]
        Predicted target values.

    Returns
    -------
    cm : DataFrame
        Returns the confusion matrix.
    """    
    return pd.crosstab(y_true, y_pred, 
                       rownames=['Actual'], colnames=['Predicted'])


class GaussianNB(object):
    """The Gaussian naive Bayes (GaussianNB) classifier.

    Gaussian naive Bayes is a specific formulation of the naive Bayes
    classifier that makes the assumption that the values of numeric features
    are normally distributed. Under this assumption, each conditional
    distribution is modeled with a single Gaussian distribution, the mean and
    standard deviation of which are estimated from the observed data [1].

    In this implementation, each conditional likelihood can be parameterized as
    a univariate Gaussian distribution with mean conditioned on the target
    variable and standard deviation not conditioned on the target variable.
    This model (gnb1) is known as a discrete analog to logistic regression [2].
    The standard deviation can alternatively be conditioned on the target
    variable (gnb2), permitting non-linear decision boundaries.

    It has been shown that despite the assumption of conditional independence,
    Gaussian naive Bayes can still be optimal under certain conditions [3].

    Attributes
    ----------
    class_prior_ : array, shape (n_classes,)
        Probability of each class.
    class_count_ : array, shape (n_classes,)
        Number of training samples observed in each class.
    epsilon_ : array, shape (n_classes, n_features)
        Offset to avoid evaluation of log(0.0).
    theta_ : array, shape (n_classes, n_features)
        Mean of each feature per class.
    sigma_ : array, shape (n_features,)
        Standard deviation of each feature, unconditional of the class.

    References
    ----------
    .. [1] G. H. John and P. Langley. "Estimating Continuous Distributions in
           Bayesian Classifiers." Proceedings of the Eleventh Conference on
           Uncertainty in Artificial Intelligence, pp. 338-345, 1995.

    .. [2] A. Y. Ng and M. I. Jordan. "On Discriminative vs. Generative
           Classifiers: A Comparison of Logistic Regression and Naive Bayes."
           Advances in Neural Information Processing Systems (NIPS) 2, pp. 
           841-848, 2002.

    .. [3] H. Zhang. "The Optimality of Naive Bayes." Artificial Intelligence
           1 (2): 3, 2004.
    """

    def __init__(self):
        # Offset to avoid evaluation of log(0.0).
        self.epsilon_ = 1e-5

    def fit(self, X, y):
        """Fit Gaussian naive Bayes.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_instances,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Calculate the empirical mean (theta) of each feature
        # conditioned on each class and the unconditional standard deviation
        # (sigma) of each feature. Add epsilon to each sigma value. Calculate
        # the class priors.
        # ================================================

    def _joint_log_likelihood(self, X):
        """Compute the posterior log probability of X.

        By Bayes' Theorem, the posterior probability of x (row of X) is
            P(y) * P(x|y),
        where y is the class and x is a feature vector. The logarithm of this
        equation can be calculated as
            log P(y) + log P(x|y)
        to obtain the log probability of X.

        The log prior, log P(y), can be computed directly. The log likelihood,
        log P(x|y), can be computed by assuming the features to be distributed
        according to a Gaussian/Normal distribution.

        The probability density of the Normal distribution is
            (1 / (sqrt(2 * pi) * sigma)) * e^(-(x - theta)^2 / (2 * sigma^2)),
        which governs the relative likelihood that the value of x would equal
        that sample. The logarithm of this function can be calculated as
            -1/2 log(2 * pi) - log(sigma) - 1/2 (((x - theta)^2) / sigma^2),
        which governs the relative log likelihood that the value of x would
        equal that sample, and can be used to compute the log likelihood of x.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Instances.

        Returns
        -------
        joint_log_likelihood : array, shape = [n_instances, n_classes]
            Log likelihood for each instance conditioned on each class.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Compute the log likelihoods for each row x of X,
        # assuming that the features are distributed according to a
        # Gaussian/Normal distribution. Use the thetas, sigmas, and priors
        # calculated during fitting to compute the log likelihoods.
        # ================================================

    def predict(self, X):
        """Predict target values for instances in X.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Instances.

        Returns
        -------
        y_pred : array, shape = [n_instances,]
            Predicted target value for instances.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Use the log likelihoods to decide the predicted target
        # value. Predict the target value with the maximum (log) likelihood.
        # ================================================


if __name__ == "__main__":
    with open('output.txt', 'w') as f_out:
        df = pd.read_csv('digits.csv')
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        clf = GaussianNB()
        clf.fit(X, y)
        y_pred = clf.predict(X)

        # ================ YOUR CODE HERE ================
        # Print the accuracy of the prediction results, a blank line, and the
        # corresponding confusion matrix.
        # ================================================
