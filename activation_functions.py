#21070122036

import matplotlib.pyplot as plt
import numpy as np

# Define the activation functions
def binary_unipolar(x):
    return np.where(x > 0, 1, 0)

def binary_bipolar(x):
    return np.where(x > 0, 1, -1)

def bipolar_continuous(x, lambda_value=1): # tanh
    return (2 / (1 + np.exp(-lambda_value * x))) - 1
    
def unipolar_continuous(x, lambda_value=1): # sigmoid
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Create a range of input values
x = np.linspace(-10, 10, 400)

# Dictionary of activation functions for plotting
activation_functions = {
    "Binary Unipolar": binary_unipolar,
    "Binary Bipolar": binary_bipolar,
    "Bipolar Continuous (tanh)": bipolar_continuous,
    "Unipolar Continuous (sigmoid)": unipolar_continuous,
    "ReLU": relu,
    "Leaky ReLU": leaky_relu
}

# Set up subplots with specific colors
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
fig, axs = plt.subplots(3, 2, figsize=(14, 20))
axs = axs.flatten()  # Flatten the 2D array of axes

# Plot each activation function with colors and labels
for i, (name, func) in enumerate(activation_functions.items()):
    axs[i].plot(x, func(x), color=colors[i], linewidth=2)
    axs[i].set_title(name, fontsize=14, fontweight='bold')
    axs[i].set_xlabel('Input', fontsize=12)
    axs[i].set_ylabel('Output', fontsize=12)
    axs[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust the top margin
plt.suptitle('Activation Functions', fontsize=16, fontweight='bold')
plt.show()

