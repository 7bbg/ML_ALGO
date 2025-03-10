#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Mean Squared Error (MSE) cost function
def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X @ w + b  # y_hat = wX + b
    cost = (1 / m) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent algorithm
def gradient_descent(X, y, w, b, learning_rate, epochs):
    m = len(y)
    cost_history = []  # Store cost over iterations
    
    for i in range(epochs):
        predictions = X @ w + b
        dw = (-2 / m) * np.sum(X * (y - predictions))  # Partial derivative w.r.t. w
        db = (-2 / m) * np.sum(y - predictions)  # Partial derivative w.r.t. b
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Store the cost for analysis
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        
        # Print every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b, cost_history

if __name__ == "__main__":
    # Generate synthetic data (y = 3x + 7 + noise)
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 100 samples, feature range [0, 10]
    y = 3 * X + 7 + np.random.randn(100, 1) * 2  # Add noise
    
    # Division by Zero in Normalization
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    # Avoid division by zero
    X_std[X_std == 0] = 1

    X = (X - X_mean) / X_std


    # Initialize parameters (random values)
    w = np.random.randn(1)
    b = np.random.randn(1)

    # Hyperparameters
    learning_rate = 0.01
    epochs = 1000

    # Train the model
    w, b, cost_history = gradient_descent(X, y, w, b, learning_rate, epochs)

    # Predictions
    y_pred = X @ w + b[0]

    # Plot results
    plt.scatter(X, y, label="Actual Data", color="blue", alpha=0.5)
    plt.plot(X, y_pred, label="Linear Regression Fit", color="red")
    plt.xlabel("X (Feature)")
    plt.ylabel("y (Target)")
    plt.legend()
    plt.show()

    # Print final parameters
    print(f"Final weight: {w[0]:.4f}, Final bias: {b[0]:.4f}")
