# NN_activation_functions

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
