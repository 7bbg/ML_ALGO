#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from rnn import RNN

# Generate sine wave dataset for time series production
def generate_sine_wave(seq_length = 50):
    """
    Generate a sine wave dataset for time series prediction
    """
    x = np.linspace(0, 100, seq_length)
    y = np.sin(x)
    return y

# Predict future values
def predict(rnn, seed_values, steps=50):
    """
    Generate prediction for the next `steps` time steps using the trained RNN.
    """
    predictions = []
    input_val = np.array([seed_values])
    for _ in range(steps):
        output, _ = rnn.forward([input_val])
        pred = output[0].item()
        predictions.append(pred)
        input_val = np.array([pred]) # Feed prediction as next input
    return predictions

if __name__ == "__main__":
    # Prepare dataset
    time_series = generate_sine_wave()
    X_train = [np.array([time_series[i]]) for i in range(len(time_series)-1)]
    y_train = [np.array([time_series[i + 1]]) for i in range(len(time_series)-1)]

    # Intialize and train RNN
    rnn = RNN(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)
    rnn.train(X_train, y_train, epochs=200)
    
    # Predict future sine wave functions
    future_predictions = predict(rnn, seed_values=time_series[-1], steps=50)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, label="Actual")
    plt.plot(range(len(time_series), len(time_series) + 50), future_predictions, label="Predicted", linestyle='dashed')
    plt.legend()
    plt.title("Sine Wave Prediction using RNN")
    plt.savefig("rnn.png")
    plt.show()