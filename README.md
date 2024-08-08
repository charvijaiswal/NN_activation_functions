# NN_activation_functions

Activation Functions are mathematical functions that determine whether a neuron should be activated or not. The output of an activation function is used as the input to the next layer in the network, allowing the network to learn and make decisions.

1. Binary_unipolar
    return np.where(x > 0, 1, 0)

2. Binary_bipolar
    return np.where(x > 0, 1, -1)

3. Bipolar continuous (lambda_value=1)
    return (2 / (1 + np.exp(-lambda_value * x))) - 1
    
4. Unipolar_continuous (lambda_value=1): # sigmoid
    return 1 / (1 + np.exp(-x))

5. Relu
    return np.maximum(0, x)

6. Leaky_relu (alpha=0.01)
    return np.where(x > 0, x, alpha * x)
