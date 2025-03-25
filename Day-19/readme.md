# Deep Learning Journey - Day 19: Keras Tuner | Hyperparameter Tuning a Neural Network

## What is Keras Tuner?
Keras Tuner is an open-source library that helps automate the process of hyperparameter tuning in deep learning models. Hyperparameters, such as the number of layers, neurons per layer, learning rate, and optimizer choice, significantly impact a model's performance. Instead of manually selecting these values, Keras Tuner efficiently finds the best combination of hyperparameters for optimal model performance.

## Why Use Keras Tuner?
Hyperparameter tuning is essential for improving a deep learning model's accuracy and generalization capabilities. Keras Tuner automates this process and allows for:
- Systematic exploration of different hyperparameters.
- Optimization of model performance.
- Saving time compared to manual tuning.
- Finding the best architecture and training settings for a given dataset.

## What Was Implemented in This Notebook?

### 1. **Training a Neural Network Without Hyperparameter Tuning**
- The dataset used in this project is the **Diabetes Dataset**.
- Data was preprocessed using **StandardScaler** to normalize the feature values.
- A simple **feedforward neural network** was built using Keras Sequential API.
- The model architecture consisted of:
  - One hidden layer with **32 neurons** and ReLU activation.
  - An output layer with **1 neuron** and a sigmoid activation for binary classification.
- The model was trained using **Adam optimizer** for 100 epochs.

### 2. **Using Keras Tuner for Hyperparameter Tuning**

#### a. **Tuning Optimizer Selection**
- Defined a function `build_model(hp)` where:
  - The optimizer was chosen dynamically from **Adam, SGD, or RMSprop**.
  - The model was compiled with the selected optimizer.
- Used **RandomSearch tuner** to search for the best optimizer.
- The search was performed over **5 trials**, and the best optimizer was selected based on validation accuracy.

#### b. **Tuning the Number of Neurons in the Hidden Layer**
- Defined another function to tune the **number of neurons** in the first hidden layer.
- Used **hp.Int()** to define a range for neuron selection (**8 to 128 neurons**, in steps of 8).
- The RandomSearch tuner was again used to find the best configuration for the hidden layer.

## Results and Observations
- Training without hyperparameter tuning led to decent accuracy, but performance could vary depending on the manually chosen parameters.
- With Keras Tuner, the best optimizer and neuron configuration were selected, resulting in improved model performance.
- The tuned model had a higher validation accuracy compared to the manually defined model.

## What More Can Be Done with Keras Tuner?
- **Tuning Learning Rate**: Different learning rates can be explored to find the best convergence speed.
- **Tuning Batch Size**: Finding the optimal batch size can help in efficient training.
- **Tuning Activation Functions**: Exploring ReLU, LeakyReLU, and tanh activations.
- **Tuning Multiple Layers**: Finding the best combination of hidden layers and neurons.

Keras Tuner provides a powerful and automated way to optimize deep learning models. This project demonstrated how to use it for selecting the best hyperparameters for a diabetes classification task.

