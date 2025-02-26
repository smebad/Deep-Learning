# Deep Learning Journey - Day 4: Problem with Perceptron

## Understanding the Problem with Perceptron
The **Perceptron** is a fundamental building block in neural networks. It is a simple linear classifier that can separate data points using a single straight line (decision boundary). However, it has a major limitation: it can only classify **linearly separable** data.

### What is Non-Linear Data?
Non-linear data is a type of dataset where the classes **cannot be separated by a single straight line** in a 2D space. Instead, they require a more complex decision boundary, such as a curve or multiple lines.

For example:
- The **AND gate** and **OR gate** can be separated by a straight line (linear separability).
- The **XOR gate**, however, is **not linearly separable** and cannot be classified correctly using a single-layer perceptron.

This limitation is why we need a **Multi-Layer Perceptron (MLP)**, which introduces hidden layers to solve non-linear problems.

---
## What I Did in My Notebook
### 1. Created Logical Gate Data (AND, OR, XOR)
- Defined datasets for **AND**, **OR**, and **XOR gates**.
- Plotted the datasets using `seaborn.scatterplot()` to visualize their structure.

### 2. Trained Perceptrons on AND, OR, and XOR Datasets
- Used `sklearn.linear_model.Perceptron` to train a perceptron on each dataset.
- Extracted weights and bias to visualize decision boundaries.

### 3. Plotted Decision Boundaries
- For **AND** and **OR**, I plotted the learned decision boundaries, showing that they are linearly separable.
- For **XOR**, I attempted to fit a perceptron but observed that it **failed** to classify the points correctly.

### 4. Visualized the XOR Decision Boundary Failure
- Used `mlxtend.plotting.plot_decision_regions()` to show that a single perceptron fails on XOR.

---
## Key Takeaways
- **Single-layer perceptrons can only classify linearly separable data.**
- **XOR data is non-linearly separable**, meaning it cannot be solved using a single perceptron.
- **Multi-Layer Perceptrons (MLPs)** introduce hidden layers and activation functions (like ReLU, sigmoid) to solve non-linear problems.
- This experiment highlights why deep learning techniques, such as neural networks with multiple layers, are essential for solving complex classification tasks.

---
## Next Steps
Moving forward, I will explore:
1. **Multi-Layer Perceptrons (MLP)** to solve the XOR problem.
2. Implementing **activation functions** to introduce non-linearity.
3. Understanding **backpropagation** and weight updates in deep learning models.

