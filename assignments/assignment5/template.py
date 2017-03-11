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


def tanh(X):
    """Compute the hyperbolic tangent (tanh) function inplace.

    Parameters
    ----------
    X : array, shape = [n_instances, n_features]
        The input data.

    Returns
    -------
    X_new : array, shape = [n_instances, n_features]
        The transformed data.
    """
    return np.tanh(X, out=X)


def tanh_derivative(Z):
    """Apply the derivative of the hyperbolic tangent (tanh) function.

    Parameters
    ----------
    Z : array, shape = [n_instances, n_features]
        The data that was output from the hyperbolic tangent activation
        function during the forward pass.

    Returns
    -------
    Z_new : array, shape = [n_instances, n_features]
        The transformed data.  
    """
    return (1 - Z ** 2)


class MLNNClassifier(object):
    """Multi-layer neural network classifier.

    A multi-layer neural network is a system of interconnected "neurons" that
    can compute values from inputs by feeding information through the network.
    A single neuron can be modeled as a perceptron [1] or "logistic unit"
    through which a series of features and associated weights are passed. The
    output of neurons at earlier layers are used as the input to neurons at
    later ones. The output of the network is computed by applying an output
    transformation to the inputs of the final layer of neurons.

    The classifier trains iteratively. At each iteration, the partial
    derivatives of the loss function with respect to the model parameters are
    computed to update the parameters. These partial derivatives are then
    propagated backwards ("backpropagated") through the network [2, 3].

    This implementation uses a hyperbolic tangent activation function and
    output transformation and optimizes the squared loss function using
    (minibatch) stochastic gradient descent [4]. Weights are initialized using
    normalized initialization [5].

    Parameters
    ----------
    hidden_dim : list, optional (default=(100,))
        Number of units per hidden layer.
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
    weight_ : list, length n_layers - 1
        Weights for each layer. The ith element in the list represents the
        weight matrix corresponding to layer i.
    bias_ : list, length n_layers - 1
        Biases for each layer. The ith element in the list represents the bias
        vector corresponding to layer i + 1.
    n_layers_ : int
        Number of layers.
    n_outputs_ : int
        Number of outputs.

    References
    ----------
    .. [1] F. Rosenblatt. "The Perceptron: A Probabilistic Model for
           Information Storage and Organization in the Brain." Psychological
           Review 65 (6): 386-408, 1958.

    .. [2] P. Werbos. "Beyond Regression: New Tools for Prediction and Analysis 
           in the Behavioral Sciences." PhD Dissertation, Harvard University,
           Cambridge, 1975.

    .. [3] G. E. Hinton. "Connectionist Learning Procedures." Artificial
           Intelligence 40 (1-3): 185-234, 1989.

    .. [4] Y. S. Abu-Mostafa, M. Magdon-Ismail, and H-T Lin. "Learning from
           Data." AMLBook, 2012.

    .. [5] X. Glorot and Y. Bengio. "Understanding the Difficulty of Training
           Deep Feedforward Neural Networks." Proceedings of the 13th
           International Conference on Artificial Intelligence and Statistics
           (AISTATS), 2010.
    """

    def __init__(self, hidden_dim=(100,), batch_size=100, learning_rate=0.01,
                 max_iter=500, random_state=None):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state

    def _initialize(self, layer_dim):
        """Initialize parameters.

        Parameters
        ----------
        layer_dim : list, length = n_layers
            Number of dimensions for each layer. The ith element of the list
            holds the dimensions of the ith layer.
        """
        self.n_layers_ = len(layer_dim)

        # Initialize weights and biases.
        self.weight_ = []
        self.bias_ = []
        for i in range(self.n_layers_ - 1):
            W = self._init_weight(layer_dim[i], layer_dim[i + 1])
            self.weight_.append(W)
            self.bias_.append(np.ones(layer_dim[i + 1]))

    def _init_weight(self, fan_in, fan_out):
        """Initialize weights.

        This implementation uses normalized initialization, where weights W
        are initialized by randomly sampling from a uniform distribution over
            [-sqrt(6) / sqrt(n_j + n_{j+1}), sqrt(6) / sqrt(n_j + n_{j+1})]
        where n_j and n_{j+1} are the fan in and fan out, respectively.

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
        init_bound = np.sqrt(6. / (fan_in + fan_out))
        W = np.random.uniform(-init_bound, init_bound,
                                  (fan_in, fan_out))
        return W

    def _forward_pass(self, activations):
        """Feed forward.

        Perform a forward pass on the network by computing the values of the
        activations (neurons) in the hidden layers and the output layer.

        The activations are initialized as input x. The activations from
        layers l = 1 to L are computed as:
            s_l = (W_l).T \dot x_{l-1}
            x_l = theta(s_l + bias),
        where x_l is the activations at layer l, s_l is the dot product of the
        weights at layer l and the activations at layer l-1, and theta() is
        the output transformation. The output hypothesis is h(x) = x_L. The dot
        product is denoted by \dot.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            Activations for each layer. The ith element of the list holds the
            values of the ith layer.

        Returns
        -------
        activations : list, length = n_layers - 1
            Computed network activations.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Compute and return the activations at each layer.
        # ================================================

    def _backprop(self, y, activations, deltas):
        """Backpropagation to compute sensitivities at each layer.

        The sensitivities are computed from the loss function and its
        corresponding derivatives with respect to the weight and bias vectors.
        This implementation uses the hyperbolic tangent (tanh) function as the
        output transformation and computes the derivatives accordingly.

        The sensitivities at the final layer, L, are computed as:
            delta_L = 2 * (x_L - y) * theta'(s_L),
        where x_L is the activations at layer L, y is the target values, s_L is
        the dot product of the weights at layer L and the activations at layer
        L-1, and theta'() is the derivative of the output transformation
        applied to s_L.

        The sensitivities from layers l = L-1 to 1 are backpropagated as:
            delta_l = 2 * theta'(s_l) \mat [W_{l+1} \dot delta_{l+1}],
        where s_l is the dot product of the weights at layer l and activations
        at layer l-1, W_{l+1} is the weights at layer l+1, and theta'(s_l) is
        the derivative of the output transformation applied to s_l. Matrix
        multiplication is denoted by \mat and the dot product is denoted by
        \dot.

        Parameters
        ----------
        y : array, shape = [n_instances, n_targets]
            Target values.
        activations : list, length = n_layers - 1
            Activations for each layer. The ith element of the list holds the
            values of the ith layer.
        deltas : list, length = n_layers - 1
            Sensitivities for each layer. The ith element of the list holds the
            difference between the activations of the i + 1 layer and the
            backpropagated error. The sensitivities are gradients of loss with 
            respect to z in each layer, where z = wx + b is the value of a
            particular layer before passing through the activation function.

        Returns
        -------
        deltas : array, shape = [n_weights,]
            Gradient of the error.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Compute and return the sensitivities at each layer.
        # ================================================

    def _compute_gradient(self, activations, deltas, batch_size):
        """Compute the gradient.

        For each instance x_n (in the batch), the gradient is computed as
            G_l(x_n) = [x_{l-1} \dot (delta_l).T]
            G_l = G_l + 1/N * G_l(x_n),
        where x_l is the activations at layer l, delta_l is the sensitivities
        at layer l, N is the number of instances (in the batch), and the dot
        product is denoted by \dot.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            Activations for each layer. The ith element of the list holds the
            values of the ith layer.
        deltas : list, length = n_layers - 1
            Sensitivities for each layer. The ith element of the list holds the
            difference between the activations of the i + 1 layer and the
            backpropagated error. The sensitivities are gradients of loss with 
            respect to z in each layer, where z = wx + b is the value of a
            particular layer before passing through the activation function.
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

    def _update_weight(self, activations, deltas, batch_size):
        """Updates the weight vector.

        Given the learning rate and gradient of the error, the updated weight
        vector w can be computed as:
            w_{i+1} = w_i - learning_rate * grad,
        where i is the ith layer. This adjusts the weight vector in the
        direction of negative error proportional to the learning rate.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            Activations for each layer. The ith element of the list holds the
            values of the ith layer.
        deltas : list, length = n_layers - 1
            Sensitivities for each layer. The ith element of the list holds the
            difference between the activations of the i + 1 layer and the
            backpropagated error. The sensitivities are gradients of loss with 
            respect to z in each layer, where z = wx + b is the value of a
            particular layer before passing through the activation function.
        batch_size : int
            Size of minibatches.
        """
        # ================ YOUR CODE HERE ================
        # Instructions: Update the weights at each layer.
        # ================================================

    def fit(self, X, y):
        """Fit a multi-layer neural network.

        Parameters
        ----------
        X : array, shape = [n_instances, n_features]
            Training data.
        y : array, shape = [n_instances, n_targets]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        np.random.seed(seed=self.random_state)

        n_instances, n_features = X.shape
        hidden_dim = list(self.hidden_dim)

        self.classes_ = np.unique(y)
        self.n_outputs_ = len(self.classes_)

        y_bin = label_binarize(y)

        batch_size = min(self.batch_size, n_instances)

        layer_dim = ([n_features] + hidden_dim + [self.n_outputs_])
        self._initialize(layer_dim)

        # ================ YOUR CODE HERE ================
        # Instructions: Fit the multi-layer neural network. Iterate up to
        # max_iter times. Each iteration, feed forward the input, backpropagate
        # the partial derivatives computed with respect to the loss function,
        # and update the weights. Using the updated weight vector, generate
        # predictions for the target class. Every fifty iterations, print the
        # current model accuracy.
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
        # Instructions: Use the final activations to decide the predicted
        # target value. Predict the target value with the highest activation.
        # ================================================


if __name__ == "__main__":
    with open('output.txt', 'w') as f_out:
        df = pd.read_csv('digits.csv')

        X = df.ix[:, :-1]
        y = df.ix[:, -1]

        clf = MLNNClassifier(random_state=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        # ================ YOUR CODE HERE ================
        # Print a blank line and the corresponding confusion matrix.
        # ================================================
