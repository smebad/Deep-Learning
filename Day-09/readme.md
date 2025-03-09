# Deep Learning Journey - Day 9: Gradient Descent (Batch vs Stochastic vs Mini-Batch)

## What is Gradient Descent in Neural Networks?
Gradient descent is an optimization algorithm used to train neural networks by updating the model's parameters iteratively to minimize the loss function. It calculates the gradient (partial derivative) of the loss function with respect to the weights and biases, adjusting them step by step in the opposite direction of the gradient to find the optimal values.

### Importance of Gradient Descent
- It enables neural networks to learn from data by minimizing the loss function.
- It helps find the best weights and biases to improve model accuracy.
- It is crucial for deep learning models to efficiently train on large datasets.

## Types of Gradient Descent
There are three main types of gradient descent:

### 1. **Batch Gradient Descent**
- Uses the entire dataset to compute the gradient before updating the model parameters.
- More stable but computationally expensive for large datasets.
- Slower convergence but guarantees a more optimal solution.

### 2. **Stochastic Gradient Descent (SGD)**
- Updates the model parameters after computing the gradient for each individual data point.
- Faster and requires less memory but can be noisy and unstable.
- Often used in online learning or real-time applications.

### 3. **Mini-Batch Gradient Descent**
- Divides the dataset into small batches and updates the model parameters after computing the gradient for each batch.
- A balance between batch and stochastic gradient descent.
- Faster than batch gradient descent while reducing the noise of stochastic gradient descent.

## What Was Implemented in This Notebook?
In this notebook, I experimented with different types of gradient descent:

1. **Stochastic Gradient Descent (SGD):**
   - Implemented with a batch size of `1`, meaning weights are updated after each data point.
   - Trained the model for `500` epochs.
   - Observed loss fluctuations due to higher variance in updates.
   
2. **Batch Gradient Descent:**
   - Implemented with a batch size equal to the entire dataset (`250` in this case).
   - Trained the model for `10` epochs.
   - More stable loss curve but required more computation.
   
3. **Comparison of Loss:**
   - Plotted the loss curves for both approaches.
   - Observed differences in training time, stability, and convergence.

## Key Takeaways
- **Batch Gradient Descent** is more stable but slow for large datasets.
- **Stochastic Gradient Descent (SGD)** is fast but can be noisy.
- **Mini-Batch Gradient Descent** offers a trade-off between stability and speed.

This experiment helped solidify my understanding of how gradient descent works and how different batch sizes impact training efficiency.

