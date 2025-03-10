#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

from sgd import stochastic_gradient_descent


# Generate synthetic data for linear regression (y = 3x + 4 + noise)
def generate_data(n_samples=100):
    X = np.random.rand(n_samples, 1) * 10 # Random values between 0 and 10
    y = 3 * X + 4 + np.random.randn(n_samples, 1) *2 # Adding noise
    return X, y

if __name__ == "__main__":
    # Main execution
    X, y = generate_data(n_samples=200) # Generate dataset
    weights, bias, loss_history = stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100, batch_size=10) # Train model
    print(f"Learned weights: {weights.flatten()[0]:.4f}, Bias: {bias:.4f}")

    plt.plot(loss_history, color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Reduction Over Time with SGD")
    plt.savefig("sgd.png")
    plt.show()

    # Predict a new data
    new_X = np.array([[5], [7], [9]])  # Example input sizes
    predictions = new_X.dot(weights) + bias
    print("Predictions for new data points:", predictions.flatten())