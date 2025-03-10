#!/usr/bin/env python3

import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        """
        Train the decision tree by building a tree recursively.
        """
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree
        """
        num_samples, num_features = X.shape

        # Stop if max depth is reached or too few samples remain
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]
        
        # Find the bets split
        feature_idx, threshold = self._best_split(X, y)
        left_idxs, right_idxs = self._split(X[:, feature_idx], threshold)


        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return {'feature': feature_idx, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split the data
        """
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X[:,feature_idx], threshold)
                gini = self._gini(y[left_idxs], y[right_idxs])
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature_idx, threshold
        
        return best_feature, best_threshold
    def _split(self, feature_column, threshold):
        """
        Split data into left and right groups based on threshold.
        """
        left_idxs = np.where(feature_column <= threshold)[0]
        right_idxs = np.where(feature_column > threshold)[0]
        return left_idxs, right_idxs

    def _gini(self, left_y, right_y):
        """
        Calculate Gini impurity for a split.
        """
        def gini_impurity(y):
            m = len(y)
            if m == 0:
                return 0
            probs = [np.sum(y == c) / m for c in np.unique(y)]
            print(probs)
            return 1 - sum(p ** 2 for p in  probs)
        m_left, m_right = len(left_y), len(right_y)
        m_total = m_left + m_right
        return (m_left / m_total) * gini_impurity(left_y) + (m_right / m_total) * gini_impurity(right_y)

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction
        """

        if isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                return self._traverse_tree(x, node['left'])
            else:
                return self._traverse_tree(x, node['right'])
        else:
            return node
    def predict(self, X):
        """
        Make predictions for each sample in X.
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    
    def _bootstrap_sample(self, X, y):
        """
        Generate a random sample with replacement (bootstrap).
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        """
        Train multiple descision trees on bootstrap samples.
        """
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        """
        Make predictions using majority voting from all trees.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
