# Deep Learning Journey - Day 2: Perceptron Trick

## Introduction
On Day 2 of my deep learning journey, I explored the **Perceptron Trick**, an essential concept in neural networks and machine learning. The perceptron trick is a simple algorithm used to adjust the weights of a perceptron to correctly classify linearly separable data. This method iteratively updates the weights based on the difference between predicted and actual labels.

## What is the Perceptron Trick?
The **Perceptron Trick** is a weight adjustment technique used in the perceptron learning algorithm. The perceptron is a basic neural network model that classifies input data into two categories. If the data is linearly separable, the perceptron can learn a decision boundary by adjusting the weights iteratively.

The perceptron update rule follows this formula:
\[ w = w + \eta (y - \hat{y}) x \]
where:
- \( w \) are the weights,
- \( \eta \) is the learning rate,
- \( y \) is the actual label,
- \( \hat{y} \) is the predicted label,
- \( x \) is the input feature vector.

## Implementation
### 1. Generating Synthetic Data
I generated a binary classification dataset using `make_classification()` from `sklearn.datasets`. This dataset consists of 100 samples with two informative features, ensuring that the classes are linearly separable.

```python
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
```

### 2. Implementing the Perceptron Trick
I implemented the perceptron algorithm with random weight updates for misclassified points. The perceptron function initializes the weights to 1 and iteratively updates them.

```python
def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)  # Insert a bias term
    weights = np.ones(X.shape[1])   # Initialize weights
    learning_rate = 0.1

    for i in range(1000):
        j = np.random.randint(0, 100)  # Pick a random sample
        y_hat = step(np.dot(X[j], weights))  # Make a prediction
        weights = weights + learning_rate * (y[j] - y_hat) * X[j]  # Update weights

    return weights[0], weights[1:]

def step(x):
    return 1 if x > 0 else 0

intercept_, coef_ = perceptron(X, y)  # Train the perceptron
```

### 3. Visualizing the Decision Boundary
After training the perceptron, I calculated the decision boundary using the learned weights and plotted it alongside the data points.

```python
m = -(coef_[0] / coef_[1])  # Slope
b = -(intercept_ / coef_[1])  # Intercept
x_input = np.linspace(-3, 3, 100)  # Generate x values
y_input = m * x_input + b  # Calculate y values

plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color='red', linewidth=2)  # Plot decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
plt.ylim(-3, 2)
plt.show()
```

## Key Takeaways
- The **Perceptron Trick** is a simple yet effective way to adjust weights in a perceptron model.
- Using `make_classification()`, I generated a linearly separable dataset.
- Implementing the perceptron algorithm involved initializing weights, making predictions, and updating weights iteratively.
- The decision boundary was successfully visualized, demonstrating how the perceptron correctly classifies data.





