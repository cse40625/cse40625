import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multilayer_neuralnet import MLNNClassifier


def digit_recognition(maxacts):
    """Visualize learned maximum activations (maxacts).

    Each maxact is a vector corresponding to one class (digit). The vector
    represents the features that produced the maximum activation of the output
    unit in the neural network that corresponds to the class.

    Parameters
    ----------
    maxacts : array, shape = [n_classes, n_features]
    """
    N = maxacts.shape[0]
    for n in range(N):
        plt.subplot(1, N, n+1)
        plt.imshow(maxacts[n].reshape(8, 8), cmap='gray', interpolation='none')
        plt.axis('off')
    plt.savefig('output_digit_maxacts.png')


if __name__ == "__main__":
    max_iter = 1000
    max_iter_img = 100000
    reg_lambda = 1.0

    df = pd.read_csv('digits.csv')
    X = df.ix[:, :-1]
    y = df.ix[:, -1]

    clf = MLNNClassifier(max_iter=max_iter)
    clf.fit(X, y)

    #input_img_data = np.random.random((1, 64)) * 16  # gray image with noise
    input_img_data = np.mean(X, axis=0).values.reshape((1, 64))  # mean of data

    maxacts = np.zeros((len(clf.classes_), X.shape[1]))

    # Iterate over each class.
    for c in range(len(clf.classes_)):
        digit = np.zeros((batch_size, len(clf.classes_)))
        digit[:, c] = 1
        layer_dim = ([X.shape[1]] + list(clf.hidden_dim) + [clf.n_outputs_])

        activations = [np.repeat(input_img_data, batch_size, axis=0)]
        for i in range(clf.n_layers_ - 1):
            activations.append(np.empty((batch_size, layer_dim[i + 1])))
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        # Run gradient descent.
        for i in range(max_iter_img):
            activations = clf._forward_pass(activations)
            deltas = clf._backprop(digit, activations, deltas)

            # Compute the sensitivities and gradients for the image layer.
            deltas_x = np.dot(deltas[0], clf.weight_[0].T)
            grad = activations[0][0] * deltas_x[0]
            #grad += ((2 * reg_lambda) / batch_size) * activations[0][0]
            grad += ((2 * reg_lambda) / batch_size) * \
                    ((activations[0][0]) / (1 + activations[0][0] ** 2) ** 2)

            # Adjust image.
            activations[0][0] -= clf.learning_rate * grad

        # Append the image to the maximum activations.
        maxacts[c] = activations[0][0]

    digit_recognition(maxacts)
