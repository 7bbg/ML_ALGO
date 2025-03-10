#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from rf import RandomForest

if __name__ == "__main__":
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forst Model
    rf = RandomForest(n_trees=10, max_depth=5)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Compute accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f'Random Forest Accuracy: {accuracy * 100:.2f}%')