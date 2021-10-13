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
    """Compute the K-way softmax function.

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
    computed. These gradients are then used to update the model parameters.

    This implementation uses a softmax output transformation, and optimizes
    the multinomial logistic loss (also known as the cross-entropy loss) using
    (minibatch) stochastic gradient descent. The model is equivalent to a
    neural network-based implementation of multinomial logistic regression [2].

    Parameters
    ----------
    learning_rate : float, optional (default=0.01)
        Learning rate for weight updates.
    max_iter : int, optional (default=500)
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

    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def _init_parameters(self, fan_in, fan_out):
        """Initialize weights and biases.

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
        b : array, shape = [1, fan_out]
            Initialized biases.
        """
        W = np.random.uniform(0., 1., (fan_in, fan_out))
        b = np.random.uniform(0., 1., (1, fan_out))
        self.weight_ = W
        self.bias_ = b

    def _forward_pass(self, X):
        """Feed forward.

        Perform a forward pass on the network by computing the values of the
        activations (neurons) in the output layer.

        The activations are initialized as input x_in. The activations of the
        output layer are computed as:
            s = w.T \dot x_in
            x_out = theta(s + bias),
        where w is the weight vector, x_in is the input activations, s is the
        dot product of the weights and the input activations, and theta() is
        the output transformation. The output hypothesis is h(x) = x_out. The
        dot product is denoted by \dot.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Initial activations.

        Returns
        -------
        outputs : array, shape = [n_instances, n_classes]
            Computed network outputs.
        """
        activations = np.dot(X, self.weight_)
        activations += self.bias_
        activations = softmax(activations)

        return activations

    def _compute_gradient(self, X, y, activations):
        """Compute the gradients.

        Using the softmax function as the output transformation, the neural
        network applies s(x)_k = ((w_k.T \dot x) + b_k) to compute the
        likelihood function:
            P(Y=k|X=x_n, w, b) = exp(s(x_n)_k) / \sum_K exp(s(x_n)_k),
        where K is the number of target classes, N is the number of instances,
        w is the weight vector, b is the bias, and the dot product is denoted
        by \dot.

        Using the softmax activation, the neural network minimizes the
        multinomial logistic error (also known as the cross-entropy error):
            E(w_k) = -1/N \sum_N (s(x_n)_y - log(\sum_K exp(s(x_n)_k))),
        which can be minimized by computing the gradient:
            grad_k = -1/N \sum_N (x_n * (1 - P(Y=y_n|X=x_n, w, b))),
        where P(Y=k|X=x_n, w, b) is the likelihood function. The bias
        corresponds to an input value of 1, and is minimized accordingly.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training data.
        y : array, shape = [n_instances, n_classes]
            Target values.
        activations : array, shape = [n_instances, n_classes]
            Output activations.

        Returns
        -------
        w_grad : array, shape = [n_features, n_classes]
            Gradient of the error for the weights.
        b_grad : array, shape = [1, n_classes]
            Gradient of the error for the biases.
        """
        n_instances, n_features = X.shape

        w_grad = np.zeros((n_features, self.n_outputs_))
        b_grad = np.zeros((1, self.n_outputs_))

        for y_i in self.classes_:
            idx = self.classes_.searchsorted(y_i)
            X_i = X[y[:, idx] == 1]
            activations_i = activations[y[:, idx] == 1]

            z_w = np.multiply(X_i, (1 - activations_i[:, idx])[:, np.newaxis])
            z_b = (1 - activations_i[:, idx])[:, np.newaxis]
            w_grad[:, idx] = -np.sum(z_w, axis=0)
            b_grad[:, idx] = -np.sum(z_b, axis=0)

        w_grad /= n_instances
        b_grad /= n_instances

        delta = np.copy(activations)
        delta[np.arange(n_instances), self.y] -= 1
        #delta[np.arange(n_instances), ~self.y] = 0
        w_grad2 = np.dot(X.T, delta)
        b_grad2 = np.sum(delta, axis=0)
        w_grad2 /= n_instances
        b_grad2 /= n_instances

        #print('w2', w_grad2[4,:])
        #print()
        #print('w1', w_grad[4,:])
        #exit()

        #return w_grad, b_grad
        return w_grad2, b_grad2

    def _update_params(self, X, y, activations):
        """Updates the weights and bias.

        Given the learning rate and gradient of the error, the updated weight
        vector w and bias b can be computed as:
            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad.
        This adjusts the weights and bias in the direction of negative error
        proportional to the learning rate.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training data.
        y : array, shape = [n_instances, n_classes]
            Target values.
        activations : array, shape = [n_instances, n_classes]
            Output activations.
        """
        w_grad, b_grad = self._compute_gradient(X, y, activations)
        self.weight_ -= self.learning_rate * w_grad
        self.bias_   -= self.learning_rate * b_grad

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

        _, n_features = X.shape

        self.classes_ = np.unique(y)
        self.n_outputs_ = len(self.classes_)

        self.y = y
        y_bin = label_binarize(y)

        # Initialize weights and biases.
        self._init_parameters(n_features, self.n_outputs_)

        for i in range(self.max_iter):
            # Forward propagation.
            activations = self._forward_pass(X)

            # Update parameters.
            self._update_params(X, y_bin, activations)

            if __name__ == "__main__" and i % 50 == 0:
                acc = accuracy_score(y, self.predict(X))
                f_out.write("{} {:.3f}\n".format(i, acc))

        return self

    def predict(self, X):
        """Predict target values for instances in X.

        Using the softmax function as the output transformation, the output
        hypothesis is:
            h(x_n) = argmax(k, P(Y=k|X=x_n, w_k, b_k)),
        where x_n is an instance from the set of instances X, k is a class
        value from the set of classes Y, w is the weights, and b is the bias.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Input data.

        Returns
        -------
        y_pred : array, shape = [n_instances,]
            Predicted target value for instances.
        """
        y_pred = self._forward_pass(X)
        return self.classes_[np.argmax(y_pred, axis=1)]


if __name__ == "__main__":
    with open('output.txt', 'w') as f_out:
        df = pd.read_csv('digits.csv')
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        clf = SLNNClassifier(random_state=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        cm = confusion_matrix(y, y_pred)
        f_out.write("\n{}".format(cm))
