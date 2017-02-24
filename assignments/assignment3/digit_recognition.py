import matplotlib.pyplot as plt
import numpy as np


def digit_recognition(thetas):
    """Visualize learned means (thetas).

    Each theta is a vector corresponding to one class (digit). The vector
    represents the mean values of each pixel conditioned on the class.

    Parameters
    ----------
    thetas : array, shape = [n_classes, n_features]
    """
    N = thetas.shape[0]
    for n in range(N):
        plt.subplot(1, N, n+1)
        plt.imshow(thetas[n].reshape(8, 8))
        plt.axis('off')
    plt.savefig('output_digit_means.png')
