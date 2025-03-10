#!/usr/bin/env python3
import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize RNN parameters
        input_size: Number of features input
        hidden_size: Number of hidden units
        output_size: Number of output classes
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weight matrices
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01 # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01

        # Bais terms
        self.bh = np.zeros((hidden_size , 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        Forward pass through time.
        inputs: List of input vectors (one per time step)
        """
        h_prev = np.zeros((self.hidden_size, 1)) # Initialize hidden state
        hs, ys = {}, {} # Store hidden states and outputs
        hs[-1] = h_prev

        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1) # Convert input to column vector
            hs[t] = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hs[t - 1] + self.bh))
            ys[t] = np.dot(self.Why, hs[t]) + self.by # Output
        return ys, hs

    def backward(self, inputs, targets, hs, ys):
        """
        Backpropagation Through Time (BPTT) to computer gradients
        """
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy = ys[t] - targets[t] # Compute output error
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            dh = np.dot(self.Why.T, dy) + dh_next # Backprop to hidden state
            dh_raw = (1- hs[t] ** 2) * dh # Derivative of tanh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, inputs[t].reshape(1, -1))
            dWhh += np.dot(dh_raw, hs[t - 1].T)

            dh_next = np.dot(self.Whh.T, dh_raw) # Backprop through time
        return dWxh, dWhh, dWhy, dbh, dby
    
    def update_weights(self, dWxh, dWhh, dWhy, dbh, dby):
        """
        Update weights using gradient descent.
        """
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby
    
    def train(self, inputs, targets, epochs=100):
        """
        Train the RNN on given inputs and targets.
        """
        for epoch in range(epochs):
            ys, hs = self.forward(inputs)
            dWxh, dWhh, dWhy, dbh, dby = self.backward(inputs, targets, hs, ys)
            self.update_weights(dWxh, dWhh, dWhy, dbh, dby)
            if epoch % 10 == 0:
                loss = sum((ys[t] - targets[t]) ** 2 for t in range(len(inputs)))/ len(inputs)
                print(f"Epoch {epoch}, Loss: {loss}")
