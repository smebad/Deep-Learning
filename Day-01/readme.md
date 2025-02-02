# Day 1: Perceptron

## Introduction
This project marks the beginning of my deep learning journey. In this notebook, I explored the **Perceptron** model, which is one of the simplest forms of neural networks. I applied it to a tiny dataset to understand how the model learns to classify data using a linear decision boundary. This notebook serves as the foundation for more advanced deep learning concepts.

## What is a Perceptron?
A perceptron is a basic binary linear classifier that makes decisions by computing a weighted sum of the inputs and then applying a threshold (or activation function). Mathematically, the perceptron is represented as:
  
  \[
  y = f(\mathbf{w} \cdot \mathbf{x} + b)
  \]
  
where:
- \(\mathbf{w}\) is the weight vector,
- \(\mathbf{x}\) is the input vector,
- \(b\) is the bias, and
- \(f\) is the activation function (often a step function).

## Concepts Learned
- **Data Loading and Visualization:**  
  I loaded a dataset using Pandas and visualized the data with scatter plots (using Seaborn) to understand the relationship between features such as `cgpa` and `resume_score` and the target variable `placed`.
  
- **Preprocessing:**  
  I extracted features and labels from the dataset and prepared them for training.
  
- **Perceptron Model:**  
  I implemented the perceptron model using scikit-learn's `Perceptron` class. This involved:
  - Fitting the model to the data.
  - Learning the weight coefficients and bias.
  
- **Visualization of Decision Boundary:**  
  Using `mlxtend.plotting.plot_decision_regions`, I visualized how the perceptron separates the classes with a decision boundary.

## Code Overview

### 1. Data Loading and Visualization
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('placement.csv')
print(df.head())

# Create a scatter plot to visualize data distribution
sns.scatterplot(x='cgpa', y='resume_score', hue='placed', data=df)
plt.xlabel('CGPA')
plt.ylabel('Resume Score')
plt.show()
```
## 2. Data Preparation

```python
# Extract features and target
X = df.iloc[:, 0:2].values  # Independent variables: cgpa, resume_score
y = df.iloc[:, 2].values    # Dependent variable: placed

# Using the Perceptron model from scikit-learn
from sklearn.linear_model import Perceptron
model = Perceptron()
model.fit(X, y)

# Display learned coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```
## 3. Visualizing the Decision Boundary

```python
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, y, clf=model, legend=2)
plt.xlabel('CGPA')
plt.ylabel('Resume Score')
plt.title('Decision Boundary of the Perceptron Model')
plt.show()
```
## Conclusion

In this notebook, I learned the basics of the Perceptron:

- **How it works:** The perceptron computes a linear combination of inputs (weighted sum plus bias) and then applies an activation function to produce a binary output.
- **Data Preprocessing:** I practiced loading, visualizing, and preparing data from a CSV file.
- **Model Training:** I applied the Perceptron to a small dataset to understand how the model learns and adjusts its weights during training.
- **Visualization:** By plotting the decision regions, I was able to visually inspect how the model separates the two classes.

This exercise laid the foundation for my deep learning studies, as understanding the perceptron is critical before moving on to more complex neural network architectures. I look forward to exploring more advanced topics in deep learning as I progress through this learning series.
