#!/usr/bin/env python3

import numpy as np

def compute_pca(X_scaled, k=50):
    """
    Compute Principal Component Analysis (PCA) on standardized data.
    Args:
        X_scaled: Standardized feature matrix
        k: Number of principal components to retain
    Returns:
        X_pca: Transformed data in the reduced dimension space
        eigenvectors_subset: The top k eigenvectors used for transformation
    """
    cov_matrix = np.cov(X_scaled, rowvar=False)  # Compute covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # Compute eigenvalues and eigenvectors
    
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    eigenvectors_subset = eigenvectors[:, :k]  # Select top k eigenvectors
    X_pca = np.dot(X_scaled, eigenvectors_subset)  # Project data onto principal components
    
    return X_pca, eigenvectors_subset

def reconstruct(X_pca, eigenvectors_subset, scaler):
    """
    Reconstruct original from PCA-reduced data.
    Args:
        X_pca: PCA-transformed data
        eigenvectors_subset: The eigenvectors used for transformation
        scaler: The scaler object used for standardization
    Returns:
        X_reconstructed: Reconstructed images in the original feature space
    """
    X_reconstructed = np.dot(X_pca, eigenvectors_subset.T)  # Reverse projection to original space
    X_reconstructed = scaler.inverse_transform(X_reconstructed)  # Reverse standardization
    return X_reconstructed