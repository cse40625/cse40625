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


def label_binarize(y):
    """Binarize labels in a one-vs-all fashion using one-hot encoding.

    The output will be a matrix where each column corresponds to one possible
    value of the input array, with the number of columns equal to the number
    of unique values in the input array.

    Parameters
    ----------
    y : array, shape = [n_instances,]
        Sequence of integer labels to encode.

    Returns
    -------
    y_bin : array, shape = [n_instances, n_classes]
        Binarized array.
    """
    n_instances = len(y)
    classes_ = np.unique(y)

    y_bin = np.zeros((n_instances, len(classes_)))
    for y_i in classes_:
        i = classes_.searchsorted(y_i)
        idx = np.where(y == y_i)
        y_bin[idx, i] = 1

    return y_bin


def softmax(X):
    """Compute the K-way softmax function inplace.

    Parameters
    ----------
    X : array, shape = [n_instances, n_features]
        The input data.

    Returns
    -------
    X_new : array, shape = [n_instances, n_features]
        The transformed data.
    """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


class SLNNClassifier(object):
    """Single-layer neural network classifier.

    A neural network is a system of interconnected "neurons" that can compute
    values from inputs by feeding information through the network. A single
    neuron can be modeled as a perceptron [1] or "logistic unit" through which
    a series of features and associated weights are passed. In a single-layer
    neural network, the output of the network is computed by applying an output
    transformation to the inputs.

    The classifier trains iteratively. At each iteration, the partial
    derivatives of the loss function with respect to the model parameters are
    computed to update the parameters.

    This implementation uses a softmax output transformation, and optimizes
    the multinomial logistic loss (also known as the cross-entropy loss) using
    (minibatch) stochastic gradient descent. This is equivalent to a neural
    network-based implementation of multinomial logistic regression [2].

    Parameters
    ----------
    batch_size : int, optional (default=100)
        Size of minibatches.
    learning_rate : float, optional (default=0.01)
        Learning rate for weight updates.
    max_iter : int, optional, optional (default=500)
        Maximum number of iterations.
    random_state : int or None, optional (default=None)
        State or seed for random number generator.

    Attributes
    ----------
    classes_ : array, shape = [n_classes,]
        Class labels for each output.
    weight_ : array, shape = [n_features, n_classes]
        Weights.
    bias_ : array, shape = [1, n_classes]
        Biases.
    n_outputs_ : int
        Number of outputs.

    References
    ----------
    .. [1] F. Rosenblatt. "The Perceptron: A Probabilistic Model for
           Information Storage and Organization in the Brain." Psychological
           Review 65 (6): 386-408, 1958.

    .. [2] C. M. Bishop. "Pattern Recognition and Machine Learning." Springer,
           2006.
    """

    def __init__(self, batch_size=100, learning_rate=0.01, max_iter=500,
                 random_state=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state

    def _init_weight(self, fan_in, fan_out):
        """Initialize weights.

        Parameters
        ----------
        fan_in : int
            Size of input.
        fan_out : int
            Size of output.

        Returns
        -------
        W : array, shape = [fan_in, fan_out]
            Initialized weights.
        """
        W = np.random.uniform(-.5, .5, (fan_in, fan_out))
        return W

    def _forward_pass(self, activations):
        """Feed forward.

        Perform a forward pass on the network by computing the values of the
        activations (neurons) in the output layer.

        The activations are initialized as input x_in. The activations of the
        output layer are computed as:
            s = W.T \dot x_in
            x_out = theta(s + bias),
        where x_in is the input activations, s is the dot product of the
        weights and the input activations, and theta() is the output
        transformation. The output hypothesis is h(x) = x_out. The dot product
        is denoted by \dot.

        Parameters
        ----------
        activations : array, shape = [batch_size, n_features]
            Initial activations.

        Returns
        -------
        activations : array, shape = [batch_size, n_features]
            Computed network activations.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Compute and return the activations at each layer.
        # ================================================

    def _compute_gradient(self, X, y, activations, batch_size):
        """Compute the gradient.

        Using the softmax activation as the output layer, our neural network
        applies h(x) = (w_k.T \dot x_n) to compute the likelihood function:
            P(Y=k |X=x_n, w) = \prod_N (exp(h(x)) / \sum_K exp(h(x)))
        where K is the number of target classes, N is the number of instances,
        w are the weights, and the dot product is denoted by \dot.

        Using the softmax activation, the neural network minimizes the
        multinomial logistic error (also known as the cross-entropy error):
            E(w) = -\sum_N ((w_k.T \dot x_n) - log(\sum_K exp(w_k.T \dot x_n)),
        which can be minimized by computing the gradient:
            grad = -\sum_N (x_n * (1 - P(Y=k |X=x_n, w)),
        where P(Y=k |X=x_n, w) is the likelihood function.

        Parameters
        ----------
        X : array, shape = [batch_size, n_features]
            Training data.
        y : array, shape = [batch_size, n_classes]
            Target values.
        activations : array, shape = [batch_size, n_features]
            Activations.
        batch_size : int
            Size of minibatches.

        Returns
        -------
        grad : array, shape = [n_weights,]
            Gradient of the error.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Compute and return the gradient.
        # ================================================

    def _update_weight(self, X, y, activations, batch_size):
        """Updates the weight vector.

        Given the learning rate and gradient of the error, the updated weight
        vector w can be computed as:
            w = w - learning_rate * grad.
        This adjusts the weight vector in the direction of negative error
        proportional to the learning rate.

        Parameters
        ----------
        Parameters
        ----------
        X : array, shape = [batch_size, n_features]
            Training data.
        y : array, shape = [batch_size, n_classes]
            Target values.
        activations : array, shape = [batch_size, n_features]
            Activations.
        batch_size : int
            Size of minibatches.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Update the weights at each layer.
        # ================================================

    def fit(self, X, y):
        """Fit a single-layer neural network.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training data.
        y : array, shape = [n_instances, n_classes]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        np.random.seed(seed=self.random_state)

        n_instances, n_features = X.shape
        batch_size = min(self.batch_size, n_instances)

        self.classes_ = np.unique(y)
        self.n_outputs_ = len(self.classes_)

        y_bin = label_binarize(y)

        # ================ YOUR CODE HERE ================
        # Instructions: Fit the single-layer neural network. Initialize the
        # weights with random samples drawn from a uniform distribution over
        # [-0.5, 0.5). Initialize the bias terms to 1. Iterate up to max_iter
        # times. Each iteration, perform the following steps with batches of
        # 100 instances at a time: feed forward the input, compute the gradient
        # of the loss function, and update the weights with the gradient. Every
        # 50 iterations, use the updated weight vector to generate predictions
        # for the target class and print the current model accuracy.
        # ================================================

    def predict(self, X):
        """Predict target values for instances in X.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Input data.

        Returns
        -------
        y_pred : array, shape = [n_instances,]
            Predicted target value for instances.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Use the output activations to decide the predicted
        # target value. Predict the target value with the highest activation.
        # ================================================


if __name__ == "__main__":
    with open('output.txt', 'w') as f_out:
        df = pd.read_csv('digits.csv')

        X = df.ix[:, :-1]
        y = df.ix[:, -1]

        clf = SLNNClassifier(random_state=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        # ================ YOUR CODE HERE ================
        # Print a blank line and the corresponding confusion matrix.
        # ================================================
