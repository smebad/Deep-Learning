# Deep Learning Journey - Day 8: Backpropagation

## What is Backpropagation?

Backpropagation is a supervised learning algorithm used for training artificial neural networks. It works by propagating the error backward through the network to update the weights using gradient descent. This process helps minimize the loss function and improves the accuracy of the model.

### Steps in Backpropagation:
1. **Forward Propagation:** Compute the predicted output by passing inputs through the network.
2. **Loss Calculation:** Measure the difference between the predicted and actual outputs using a loss function (e.g., Mean Squared Error).
3. **Backward Propagation:** Compute the gradient of the loss function with respect to each weight using the chain rule.
4. **Weight Update:** Adjust the weights using gradient descent to reduce the loss.
5. **Repeat:** Continue the process for multiple epochs until the model converges.

## Implementations

In this notebook, backpropagation is implemented in two ways:
1. **From Scratch:** Manually implementing forward and backward propagation using NumPy.
2. **Using Keras:** Utilizing the built-in optimization functions in TensorFlow/Keras.

### Backpropagation from Scratch

1. Created a small dataset:
   ```python
   import numpy as np
   import pandas as pd
   df = pd.DataFrame([[8,8,4], [7,9,5], [6,10,6], [5,15,7]], columns=['cgpa', 'profile_score', 'lpa'])
   ```
2. Initialized parameters (weights and biases) for a deep neural network.
3. Implemented forward propagation to compute activations.
4. Defined the loss function using Mean Squared Error (MSE).
5. Implemented backpropagation using gradient descent to update weights and biases.
6. Trained the model for 75 epochs, updating parameters in each iteration.

### Backpropagation using Keras

1. Created a sequential model in Keras:
   ```python
   from tensorflow import keras
   from keras import Sequential
   from keras.layers import Dense
   model = Sequential()
   model.add(Dense(2, activation='linear', input_dim=2))
   model.add(Dense(1, activation='linear'))
   ```
2. Set initial weights manually for better control over updates.
3. Compiled the model using Stochastic Gradient Descent (SGD) and Mean Squared Error loss.
4. Trained the model for 75 epochs with batch size 1.
5. Observed the weight updates and loss reduction over time.

## Key Takeaways

- Implementing backpropagation from scratch provided a deeper understanding of how neural networks learn.
- Using Keras made the process more efficient and scalable.
- Gradient descent helps minimize loss by updating weights iteratively.
- Understanding both implementations is crucial for mastering deep learning concepts.