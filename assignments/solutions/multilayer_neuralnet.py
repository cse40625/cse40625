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
    """Compute the hyperbolic tangent (tanh) function.

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


class MLNNClassifier(object):
    """Multi-layer neural network classifier.

    A multi-layer neural network is a system of interconnected "neurons" that
    can compute values from inputs by feeding information through the network.
    A single neuron can be modeled as a perceptron [1] or "activation unit"
    through which a series of features and associated weights are passed. The
    output of neurons at earlier layers are used as the input to neurons at
    later ones. The output of the network is computed by applying an output
    transformation to the inputs of the final layer of neurons.

    The classifier trains iteratively. At each iteration, the partial
    derivatives of the loss function with respect to the model parameters are
    computed to update the parameters. These partial derivatives are then
    propagated backwards ("backpropagated") through the network [2, 3].

    This implementation uses a hyperbolic tangent activation function and
    softmax output transformation, and optimizes the cross-entropy loss
    function using (minibatch) stochastic gradient descent. Weights and biases
    are initialized using normalized initialization [4]. The output activation,
    z, of each hidden layer is computed as z = theta(wx + b), where x is the
    input activation, w is the weight vector, b is the bias term, and theta is
    the activation function of the hidden layers [5].

    Parameters
    ----------
    hidden_dim : list, optional (default=(100,))
        Number of units per hidden layer.
    batch_size : int, optional (default=100)
        Size of minibatches.
    learning_rate : float, optional (default=0.01)
        Learning rate for weight updates.
    max_iter : int, optional (default=500)
        Maximum number of iterations.
    reg_lambda : float, optional (default=1.0)
        L2 regularization term on weights.
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

    .. [4] X. Glorot and Y. Bengio. "Understanding the Difficulty of Training
           Deep Feedforward Neural Networks." Proceedings of the 13th
           International Conference on Artificial Intelligence and Statistics
           (AISTATS), 2010.

    .. [5] Y. S. Abu-Mostafa, M. Magdon-Ismail, and H-T Lin. "Learning from
           Data." AMLBook, 2012.
    """

    def __init__(self, hidden_dim=(100,), batch_size=100, learning_rate=0.01,
                 max_iter=500, reg_lambda=1.0, random_state=None):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
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
            W, b = self._init_normalized(layer_dim[i], layer_dim[i + 1])
            self.weight_.append(W)
            self.bias_.append(b)

    def _init_normalized(self, fan_in, fan_out):
        """Initialize weights and biases using normalized initialization.

        Normalized initialization performs initialization by randomly
        sampling from a uniform distribution over:
            [-sqrt(6) / sqrt(n_j + n_{j+1}), sqrt(6) / sqrt(n_j + n_{j+1})],
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
        b : array, shape = [1, fan_out]
            Initialized biases.
        """
        init_bound = np.sqrt(6. / (fan_in + fan_out))
        W = np.random.uniform(-init_bound, init_bound,
                              (fan_in, fan_out))
        b = np.random.uniform(-init_bound, init_bound,
                              (1, fan_out))
        return W, b

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
        activations : list, length = n_layers
            Activations for each layer. The ith element of the list holds the
            values of the ith layer.

        Returns
        -------
        activations : list, length = n_layers
            Computed network activations.
        """
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = np.dot(activations[i], self.weight_[i])

            # Add bias term.
            activations[i + 1] += self.bias_[i]

            # Compute activations of the hidden layers.
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = tanh(activations[i + 1])

        # Compute activations of the last layer.
        activations[i + 1] = softmax(activations[i + 1])

        return activations

    def _backprop(self, y, activations, deltas):
        """Backpropagation to compute sensitivities at each layer.

        The sensitivities are computed from the loss function and its
        corresponding derivatives with respect to the weight and bias vectors.
        This implementation uses the hyperbolic tangent (tanh) function as the
        hidden transformation and the softmax function as the output
        transformation, and computes the derivatives accordingly.

        Using a softmax output activation and cross-entropy error, the
        sensitivities at the final layer, L, are computed as:
            delta_L = x_L - y,
        where x_L is the activations at layer L and y is the target values.

        The sensitivities from layers l = L-1 to 1 are backpropagated as:
            delta_l = theta'(s_l) \times [W_{l+1} \dot delta_{l+1}],
        where s_l is the input to layer l, W_{l+1} is the weights at layer l+1,
        and theta'(s_l) is the derivative of the hidden transformation applied
        to s_l. Element-wise multiplication is denoted by \times and the dot
        product is denoted by \dot.

        Parameters
        ----------
        y : array, shape = [n_instances, n_classes]
            Target values.
        activations : list, length = n_layers
            Activations for each layer. The ith element of the list holds the
            values of the ith layer.
        deltas : list, length = n_layers - 1
            Sensitivities for each layer. The ith element of the list holds the
            difference between the activations of the i + 1 layer and the
            backpropagated error. The sensitivities are gradients of loss with 
            respect to z in each layer, where z = theta(wx + b), where theta is
            the activation function and z is the output of a particular layer.

        Returns
        -------
        deltas : array, shape = [n_weights,]
            Gradient of the error.
        """
        last = self.n_layers_ - 2

        # Compute the deltas for the last layer.
        deltas[last] = activations[-1] - y

        # Iterate over the hidden layers.
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = np.multiply(tanh_derivative(activations[i]),
                                        np.dot(deltas[i], self.weight_[i].T))

        return deltas

    def _compute_gradient(self, activations, deltas, batch_size):
        """Compute the gradients.

        For each instance x_n (in the batch), the gradient is computed as:
            G_l(x_n) = [x_{l-1} \dot (delta_l).T]
            G_l = G_l + 1/N * G_l(x_n),
        where x_l is the activations at layer l, delta_l is the sensitivities
        at layer l, N is the number of instances (in the batch), and the dot
        product is denoted by \dot.

        Parameters
        ----------
        activations : array, shape = [batch_size, layer_dim]
            Activations for layer l-1.
        deltas : array, shape = [batch_size, layer_dim]
            Sensitivities for layer l.
        batch_size : int
            Size of minibatches.

        Returns
        -------
        W_grad : array, shape = [batch_size, layer_dim]
            Gradient of the error for the weights.
        b_grad : array, shape = [batch_size, layer_dim]
            Gradient of the error for the biases.
        """
        w_grad = np.dot(activations.T, deltas)
        w_grad /= batch_size

        b_grad = np.mean(deltas, axis=0)[np.newaxis, :]

        return w_grad, b_grad

    def _update_params(self, activations, deltas, batch_size):
        """Updates the weights and biases.

        Given the learning rate and gradient of the error, the updated weight
        vector w and bias b can be computed as:
            w_i = w_i - learning_rate * w_grad
            b_i = b_i - learning_rate * b_grad,
        where i is the ith layer. This adjusts the weights and biases in the
        direction of negative error proportional to the learning rate.

        L2 regularization is performed by adding a term to the gradient that is
        proportional to (and in the negative direction of) the parameters:
            w_grad = w_grad + ((2 * lambda) / N) * w_i
            b_grad = b_grad + ((2 * lambda) / N) * b_i,
        where lambda is a regularization parameter.

        Parameters
        ----------
        activations : list, length = n_layers
            Activations for each layer. The ith element of the list holds the
            values of the ith layer.
        deltas : list, length = n_layers - 1
            Sensitivities for each layer. The ith element of the list holds the
            difference between the activations of the i + 1 layer and the
            backpropagated error. The sensitivities are gradients of loss with 
            respect to z in each layer, where z = theta(wx + b), where theta is
            the activation function and z is the output of a particular layer.
        batch_size : int
            Size of minibatches.
        """
        for i in range(self.n_layers_ - 1):
            w_grad, b_grad = self._compute_gradient(activations[i], deltas[i],
                                                    batch_size)

            # Apply L2 regularization (weight decay).
            w_grad += ((2 * self.reg_lambda) / batch_size) * self.weight_[i]
            b_grad += ((2 * self.reg_lambda) / batch_size) * self.bias_[i]

            # Apply weight elimination.
            #W_grad += ((2 * self.reg_lambda) / batch_size) * \
            #          (self.weight_[i] / (1 + (self.weight_[i] ** 2)) ** 2)
            #W_grad += ((2 * self.reg_lambda) / batch_size) * \
            #          (self.bias_[i] / (1 + (self.bias_[i] ** 2)) ** 2)

            self.weight_[i] -= self.learning_rate * w_grad
            self.bias_[i]   -= self.learning_rate * b_grad

    def fit(self, X, y):
        """Fit a multi-layer neural network.

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

        layer_dim = ([n_features] + list(self.hidden_dim) + [self.n_outputs_])
        self._initialize(layer_dim)

        activations = [X]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_dim[1:])
        deltas = [np.empty_like(a_layer) for a_layer in activations[:-1]]

        for i in range(self.max_iter):
            # Epoch of training.
            start_idxs = range(0, len(X) - batch_size + 1, batch_size)
            for start_idx in start_idxs:
                excerpt = slice(start_idx, start_idx + batch_size)
                batch_X = np.array(X[excerpt])
                batch_y = np.array(y_bin[excerpt])

                # Forward propagation.
                activations[0] = batch_X
                activations = self._forward_pass(activations)

                # Backpropagation.
                deltas = self._backprop(batch_y, activations, deltas)

                # Update parameters.
                self._update_params(activations, deltas, batch_size)

            if __name__ == "__main__" and i % 50 == 0:
                acc = accuracy_score(y, self.predict(X))
                f_out.write("{} {:.3f}\n".format(i, acc))

        return self

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
        _, n_features = X.shape
        hidden_dim = list(self.hidden_dim)

        layer_dim = ([n_features] + hidden_dim + [self.n_outputs_])

        # Initialize activations.
        activations = [X]
        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_dim[i + 1])))

        # Forward pass.
        activations = self._forward_pass(activations)

        y_pred = activations[-1]
        return self.classes_[np.argmax(y_pred, axis=1)]


if __name__ == "__main__":
    with open('output.txt', 'w') as f_out:
        df = pd.read_csv('digits.csv')
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        clf = MLNNClassifier(random_state=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        cm = confusion_matrix(y, y_pred)
        f_out.write("\n{}".format(cm))
