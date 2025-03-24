# Deep Learning Journey - Day 17: Batch Normalization in Deep Learning

## What is Batch Normalization?
Batch Normalization is a technique used in deep learning to improve the training of neural networks by normalizing the inputs to each layer. It helps stabilize and accelerate training, making models more robust and efficient.

### Why Use Batch Normalization?
- **Improves Training Speed:** By normalizing the input distributions of each layer, it allows the network to converge faster.
- **Reduces Internal Covariate Shift:** Prevents drastic changes in the distribution of activations, making learning more stable.
- **Acts as Regularization:** Helps in reducing overfitting, similar to dropout.
- **Allows Higher Learning Rates:** Since activations remain controlled, models can be trained with higher learning rates without diverging.

## Experiment Overview
In this notebook, I compared training a neural network **without batch normalization** and **with batch normalization** to observe the impact on performance.

### Step 1: Training Without Batch Normalization
- A basic neural network was created using the **ReLU activation function**.
- The network architecture:
  - **Input layer:** 2 neurons
  - **Hidden layers:** 2 layers with 2 neurons each (ReLU activation)
  - **Output layer:** 1 neuron with sigmoid activation for binary classification.
- The model was trained for **200 epochs** with **Adam optimizer** and **binary cross-entropy loss**.
- The validation accuracy was recorded for comparison.

### Step 2: Training With Batch Normalization
- The same architecture was used but with **BatchNormalization layers** added after each dense layer.
- This helped stabilize the updates in the network and improved the training performance.
- The model was again trained for **200 epochs** with the same optimizer and loss function.
- Validation accuracy was compared with the previous model.

## Observations and Comparison
- **Without Batch Normalization:** The training was slower, and the model showed fluctuations in accuracy.
- **With Batch Normalization:** Training was more stable, and the model achieved higher validation accuracy faster.
- A plot was generated comparing validation accuracy over epochs for both models:
  - **Black Line:** Model without batch normalization.
  - **Green Line:** Model with batch normalization.

## Conclusion
Batch normalization significantly improved model training, making it faster and more stable. By reducing internal covariate shifts and allowing higher learning rates, it enhances performance, especially in deep networks. It is an essential technique in modern deep learning architectures.
