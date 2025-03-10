#!/usr/bin/env python3

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

# Compute linear kernel
def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

# Implement some other kernel functions
def polynomial_kernel(X1, X2, degree=3):
    """
    Compute the polynomial kernel between two datasets X1 and X2.
    ]
    Args:
        X1: (m, n) array where m is the number of samples and n is the number of features.
        X2: (p, n) array where p is the number of samples and n is the number of features.
        degree: The degree of the polynomial.
    
    Returns:
        (m, p) kernel matrix
    """
    return (np.dot(X1, X2.T) + 1) ** degree

# Train SVM using quadratic programming
def train_svm(X, y, C=1.0):
    m, n = X.shape
    Kernel = linear_kernel(X, X)

    P = matrix(np.outer(y, y) * Kernel)
    q = matrix(-np.ones(m))
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(y.astype(float), (1, m), 'd')
    b = matrix(0.0)

    sol = solvers.qp(P, q, G, h, A, b)

    alphas = np.ravel(sol['x'])
    w = np.sum(alphas[:, None] * y[:, None] * X, axis=0)
    support_vectors = (alphas > 1e-5)

    b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))

    return w, b

# Predict function
def predict_svm(X, w, b):
    return np.sign(np.dot(X, w) + b)