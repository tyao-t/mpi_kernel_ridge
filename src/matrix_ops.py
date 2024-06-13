# Matrix operations and mathematics, including Gaussian Kernel
import numpy as np
from data_utils import auto_gc

@auto_gc()
def gaussian_kernel_matrix(X1, X2, gamma=np.float64(1)):
    """
    Compute the Gaussian kernel matrix for the given feature matrices X1 and X2
    :param X1: Feature matrix X1 (n_samples1, n_features)
    :param X2: Feature matrix X2 (n_samples2, n_features)
    :param sigma: Standard deviation for the Gaussian kernel
    :return: Gaussian kernel matrix (n_samples1, n_samples2)
    """

    # Reshape X1 and X2 to enable broadcasting
    X1_reshaped = X1.reshape(X1.shape[0], 1, X1.shape[1])
    X2_reshaped = X2.reshape(1, X2.shape[0], X2.shape[1])
    
    sq_dists = np.sum((X1_reshaped - X2_reshaped)**2, axis=2)

    K = np.exp(-sq_dists / gamma)
    # K = np.exp(-sq_dists / (2 * (s**2)))
    return K

def test():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 10])

    sigma = 1.0
    kernel_matrix = gaussian_kernel_matrix(x, y, sigma)

    print("Gaussian kernel matrix:")
    print(kernel_matrix)

if __name__ == "__main__":
    test()