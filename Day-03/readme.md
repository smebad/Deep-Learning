# Deep Learning Journey - Day 3: Perceptron Loss Function

## What is the Perceptron Loss Function?
The perceptron loss function is used to update the weights of a perceptron model during training. It ensures that misclassified points contribute to the adjustment of the decision boundary. The perceptron loss function can be mathematically expressed as:

\[ L(w) = - y_i (w \cdot x_i + b) \]

where:
- \( y_i \) is the actual class label (+1 or -1),
- \( w \) is the weight vector,
- \( x_i \) is the input vector,
- \( b \) is the bias term.

If a point is correctly classified (i.e., \( y_i (w \cdot x_i + b) > 0 \)), no loss is incurred. However, if a point is misclassified, the weights are updated accordingly.

## What I Did in My Notebook
1. **Generated a Synthetic Dataset**
   - Used `make_classification` from `sklearn.datasets` to create a 2D dataset with two separable classes.
   - Visualized the dataset using `matplotlib.pyplot.scatter`.

2. **Implemented the Perceptron Algorithm**
   - Initialized weights \( w_1, w_2 \) and bias \( b \) to 1.
   - Defined a learning rate of 0.1.
   - Iterated over the dataset and updated the weights whenever a misclassification occurred using the perceptron loss function.

3. **Trained the Perceptron**
   - Ran 1000 iterations to adjust the decision boundary iteratively.
   - Computed the final weights and bias.

4. **Plotted the Decision Boundary**
   - Derived the slope and intercept for the decision boundary using:
     \[ m = - \frac{w_1}{w_2} \]
     \[ c = - \frac{b}{w_2} \]
   - Plotted the decision boundary alongside the dataset.

## Key Insights from the Implementation
- The perceptron loss function effectively updates weights to minimize misclassification.
- The decision boundary is adjusted iteratively through weight updates.
- A properly tuned learning rate is crucial for convergence.

## Final Thoughts
This experiment provided a hands-on understanding of the perceptron loss function and its role in training a simple linear classifier. The perceptron algorithm remains a fundamental concept in deep learning and machine learning, forming the basis for more advanced models like Support Vector Machines and Neural Networks.


