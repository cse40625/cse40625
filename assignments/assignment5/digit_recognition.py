import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multilayer_neuralnet import MLNNClassifier


def digit_recognition(maxacts, i=None):
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
    if not i:
        plt.savefig('output_digit_maxacts.png')
    else:
        if not os.path.exists('output_digit_frames'):
            os.makedirs('output_digit_frames')
        plt.savefig(os.path.join('output_digit_frames',
                                 'output_digit_maxacts_{:05d}.png'.format(i)))


def run_model(clf, max_iter_img, max_iter_step):
    # Iterate over each class.
    for c in range(len(clf.classes_)):
        digit = np.zeros((clf.batch_size, len(clf.classes_)))
        digit[:, c] = 1

        layer_dim = ([X.shape[1]] + list(clf.hidden_dim) + [clf.n_outputs_])

        # Initialize the activations and deltas for the image layer.
        activations = [np.repeat(input_img_data, clf.batch_size, axis=0)]
        for i in range(clf.n_layers_ - 1):
            activations.append(np.empty((clf.batch_size, layer_dim[i + 1])))
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        # Run gradient descent.
        n = 0
        for i in range(max_iter_img):
            activations = clf._forward_pass(activations)
            deltas = clf._backprop(digit, activations, deltas)

            # Compute the sensitivities and gradients for the image layer.
            deltas_x = np.dot(deltas[0], clf.weight_[0].T)
            grad = activations[0][0] * deltas_x[0]

            # Apply L2 regularization (weight decay).
            #grad += ((2 * reg_lambda) / batch_size) * activations[0][0]

            # Apply weight elimination.
            grad += ((2 * reg_lambda) / clf.batch_size) * \
                    ((activations[0][0]) / (1 + activations[0][0] ** 2) ** 2)

            # Adjust image.
            activations[0][0] -= clf.learning_rate * grad

            if i % max_iter_step == 0:
                # Append the image to the maximum activations for the class.
                if n == 0:
                    maxacts_c = activations[0][0]
                else:
                    maxacts_c = np.dstack((maxacts_c, activations[0][0]))
                n += 1

        # Append to the maximum activations for each iteration.
        if c == 0:
            maxacts = maxacts_c
        else:
            maxacts = np.vstack((maxacts, maxacts_c))

    return maxacts


if __name__ == "__main__":
    max_iter = 1000
    max_iter_imgs = 10001
    max_iter_step = 10000
    make_gif = False
    reg_lambda = 1.0

    df = pd.read_csv('digits.csv')
    X = df.ix[:, :-1]
    y = df.ix[:, -1]

    clf = MLNNClassifier(max_iter=max_iter)
    clf.fit(X, y)

    #input_img_data = np.random.random((1, 64)) * 16 # gray image with noise
    input_img_data = np.mean(X, axis=0).values.reshape((1, 64)) # mean of data

    maxacts = run_model(clf, max_iter_imgs, max_iter_step)
    if make_gif:
        for n in range(maxacts.shape[2]):
            digit_recognition(maxacts[:, :, n], i=max_iter_step * n + 1)
    else:
        digit_recognition(maxacts[:, :, -1])
