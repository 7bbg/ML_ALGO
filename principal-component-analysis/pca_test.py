#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from pca import compute_pca, reconstruct

def load_and_preprocess_mnist():
    """
    Load the MNIST dataset and standardize it.
    Returns the original data, standardized data, labels, and the scaler object.
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False) # Ensure data is not a DataFrame
    X = mnist.data.astype(np.float64)  # Convert data to float64
    y = mnist.target.astype(int)  # Convert labels to integers
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize data to zero mean and unit variance
    
    return X, X_scaled, y, scaler

def visualize_images(X, X_reconstructed):
    """
    Display original and reconstructed images side by side for comparison.
    Args:
        X: Original dataset
        X_reconstructed: Reconstructed dataset after PCA transformation
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(5):
        # Convert row array to numpy array annd reshape
        original_image = np.array(X[i], dtype=np.float64).reshape(28, 28)
        reconstructed_image = np.array(X_reconstructed[i], dtype=np.float64).reshape(28, 28)

        # Display original image
        axes[0, i].imshow(original_image, cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Display reconstructed image
        axes[1, i].imshow(reconstructed_image, cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    
    plt.suptitle("PCA on MNIST - Image Compression")
    plt.savefig("pca.png")
    plt.show()
   
if __name__ == "__main__":
    # Main Execution
    X, X_scaled, y, scaler = load_and_preprocess_mnist()  # Load and preprocess MNIST dataset
    X_pca, eigenvectors_subset = compute_pca(X_scaled, k=50)  # Compute PCA with 50 components
    X_reconstructed = reconstruct(X_pca, eigenvectors_subset, scaler)  # Reconstruct images
    visualize_images(X, X_reconstructed)  # Display results