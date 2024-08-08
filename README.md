# NN_activation_functions

Activation Functions are mathematical functions that determine whether a neuron should be activated or not. The output of an activation function is used as the input to the next layer in the network, allowing the network to learn and make decisions.

1. Binary_unipolar
    return np.where(x > 0, 1, 0)

![image](https://github.com/user-attachments/assets/2c557a0d-6b3c-43c8-93a6-d2a24212efc9)


2. Binary_bipolar
    return np.where(x > 0, 1, -1)

![image](https://github.com/user-attachments/assets/5009305c-9a2a-4902-b5b1-b9f826e84ab9)


3. Bipolar continuous (lambda_value=1)
    return (2 / (1 + np.exp(-lambda_value * x))) - 1

![image](https://github.com/user-attachments/assets/8dccf996-dde5-4d0d-be4f-b51d077bec77)

    
4. Unipolar_continuous (lambda_value=1): # sigmoid
    return 1 / (1 + np.exp(-x))

![image](https://github.com/user-attachments/assets/f615f510-f1bf-4677-9695-6b5d8acd978f)


5. Relu
    return np.maximum(0, x)
   
![image](https://github.com/user-attachments/assets/15473f8f-ba83-424a-a7df-6e237526058f)


6. Leaky_relu (alpha=0.01)
    return np.where(x > 0, x, alpha * x)
    
![image](https://github.com/user-attachments/assets/a445a1ae-9398-4cb1-b293-32d3f0e6460d)

