#!/usr/bin/env python3

import numpy as np

# Compute the gradient of Mean Squared Error (MSE) loss function
def compute_gradient(X, y, weights,  bias):
    y_pred = X.dot(weights) + bias
    error = y_pred - y
    grad_w = (2 / len(y)) * X.T.dot(error)
    grad_b = (2 / len(y)) * np.sum(error)
    return grad_w, grad_b

# Stochastic Gradient Descent (SGD) implementation
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100, batch_size=1):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1)) # Initialize weights
    bias = 0 # Intialize bias

    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples) # Shuffle dataset
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            grad_w, grad_b = compute_gradient(X_batch, y_batch, weights, bias)

            weights -= learning_rate * grad_w # Update weights
            bias -= learning_rate * grad_b # Update bias
        
        loss = np.mean((X.dot(weights) + bias - y) ** 2)
        loss_history.append(loss)

        if(epoch % 10 == 0):
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
            
    return weights, bias, loss_history