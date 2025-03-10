# Machine Learning Algorithms from Scratch

## Overview
This project implements some machine learning algorithms from scratch, including Stochastic Gradient Descent (SGD), Support Vector Machines (SVM), Principal Component Analysis (PCA), Random Forests, Genetic Algorithms, and more.

## Features
- **SGD for Linear Regression:** Implementation of Stochastic Gradient Descent to optimize model parameters.
- **SVM for Classification:** Support Vector Machine classifier with polynomial kernel.
- **PCA for Dimensionality Reduction:** Eigen decomposition for feature extraction.
- **Random Forests:** Ensemble learning with decision trees.
- **Genetic Algorithm for Optimization:** Approximately Solving the Travelling Salesman Problem (TSP) a with evolutionary strategies.
- **Recurrent Neural Networks (RNNs):** Implementing memory-based learning for sequence data.

## How It Works
Each algorithm follows these general steps:

1. **Data Preparation:** Load or generate a dataset.
2. **Model Initialization:** Define parameters or architecture.
3. **Training:** Optimize using the respective method (example gradient descent, tree splitting, genetic evolution).
4. **Evaluation:** Compute accuracy, loss, or visualization of results.
5. **Predictions:** Apply the trained model to new data.

## Installation
Ensure you have Python installed and required dependencies:
```sh
pip install numpy pandas matplotlib scikit-learn
```

## Usage
Run individual scripts to train models and visualize results:
```sh
python sgd_test.py  # Stochastic Gradient Descent
python svm_test.py  # Support Vector Machine
python pca_test.py  # Principal Component Analysis
python rf_test.py      # Random Forest
python genetic_algo_test_TSP.py  # Genetic Algorithm
python rnn_test.py   # Recurrent Neural Networks

```
To execute algorithm module: `python3 -m [algorithm_folder].[algorithm_main_file] # Execute [algorithm_main_file]`

## Example Outputs
- Loss reduction graphs over epochs.
- Visualized decision boundaries for classifiers.
- Reconstructed images using PCA.
- Optimized solutions using Genetic Algorithms.

## Future Enhancements
- Implement more Algorithms


## License
This project is open-source and available for modification and distribution.

