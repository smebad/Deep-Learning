# Deep Learning Journey - Day 16: Weights Initialization using Xavier/Glorot and He

## Introduction
Weight initialization is a crucial step in training deep neural networks. Poorly initialized weights can lead to slow convergence or even prevent the network from learning effectively. Two widely used initialization techniques in deep learning are **Xavier (Glorot) Initialization** and **He Initialization**. These methods help stabilize the training process and improve performance.

## Xavier (Glorot) Initialization
Xavier (also known as Glorot) initialization is designed to maintain the variance of activations throughout the network. This helps prevent the issue of vanishing or exploding gradients, which can hinder learning. This initialization method is particularly suited for activation functions that are symmetric around zero, such as **tanh**.

## He Initialization
He initialization is optimized for activation functions like **ReLU**, which introduce non-linearity and are not symmetric around zero. Since ReLU activation can cause neurons to become inactive (outputs zero for negative inputs), He initialization compensates for this by scaling the initial weights appropriately. This helps maintain stable gradients during training.

## Purpose of Using These Techniques
- Prevent vanishing and exploding gradient problems.
- Ensure that activations propagate efficiently through the network.
- Improve convergence speed and overall model performance.
- Adapt to different activation functions for optimal weight scaling.

## Experiments in This Notebook

### Experiment 1: Xavier Initialization with Tanh Activation
1. **Dataset Used**: The `ushape.csv` dataset was loaded and visualized.
2. **Model Architecture**:
   - Input layer with 2 features.
   - Hidden layers with **tanh** activation.
   - Xavier (Glorot) initialization applied to the weights.
   - Output layer with **sigmoid** activation.
3. **Training & Evaluation**:
   - Model compiled using binary cross-entropy loss and Adam optimizer.
   - Trained for 100 epochs with validation split.
   - Decision boundary visualized using `plot_decision_regions`.

### Experiment 2: He Initialization with ReLU Activation
1. **Dataset Used**: The same `ushape.csv` dataset.
2. **Model Architecture**:
   - Input layer with 2 features.
   - Hidden layers with **ReLU** activation.
   - He initialization applied to the weights.
   - Output layer with **sigmoid** activation.
3. **Training & Evaluation**:
   - Model compiled using binary cross-entropy loss and Adam optimizer.
   - Trained for 100 epochs with validation split.
   - Decision boundary visualized using `plot_decision_regions`.

## Why I Used Xavier with Tanh and He with ReLU
- **Xavier Initialization** works well with **tanh** because it maintains variance across layers, preventing vanishing gradients.
- **He Initialization** is better suited for **ReLU** because it accounts for the activation function’s property of setting negative values to zero.

## Observations & Comparisons
- **Tanh with Xavier Initialization** led to smooth convergence but required careful tuning to avoid saturation issues.
- **ReLU with He Initialization** trained faster and achieved better stability since ReLU prevents saturation in hidden layers.
- **Decision Boundaries**: The decision regions plotted for both models showed distinct patterns, highlighting how weight initialization impacts learning.

## Conclusion
Proper weight initialization is essential for stable and efficient training in deep learning. By choosing Xavier for **tanh** and He for **ReLU**, we ensured effective gradient flow and improved model convergence. These experiments demonstrate the importance of aligning initialization techniques with activation functions to maximize network performance.

