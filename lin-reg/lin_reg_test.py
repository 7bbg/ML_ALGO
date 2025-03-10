#!/usr/bin/env python3

# Predict house prices based on multiple input features.
import numpy as np
import matplotlib.pyplot as plt
from lin_reg import gradient_descent

if __name__ == "__main__":
    # Generate synthetic house pricing data
    np.random.seed(42)
    m = 100 # Number of houses


    # Features: Size (1000-3000 sq ft), Bedrooms(1-5), Age (0-50 years)
    X = np.c_[np.random.randint(1000, 3000, size=m),
            np.random.randint(1, 6, size=m),
            np.random.randint(0, 50, size=m)] # Shape (100,3)

    # True weights (chose by preference): Price = 150 * size + 10000* bedrooms - 300 * age + noise
    true_weights = np.array([150, 10000, -300])
    bias = 50000 # Base Price
    y = X @ true_weights + bias + np.random.randn(m) * 10000 # Adding noise

    # Normalize features for better convergence
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Initialize parameters
    w = np.random.randn(3) # 3 features
    b = np.random.randn()

    # Hyperparameters
    learning_rate = 0.01
    epochs = 1000

    # Train the model
    w, b, cost_history = gradient_descent(X, y, w, b, learning_rate, epochs)

    # predict house prices
    y_pred = X @ w + b

    # Plot Cost Reduction
    plt.plot(range(epochs), cost_history, label = "Cost Reduction")
    plt.xlabel("Epochs")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Function Convergence")
    plt.legend()
    plt.savefig("lin-reg.png")
    plt.show()


    # Print final model parameters
    print(f"Final Weights: {w}")
    print(f"Final Bias: {b:.2f}")
