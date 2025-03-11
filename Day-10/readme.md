# Deep Learning Journey - Day 10: Vanishing & Exploding Gradient Problem in ANN

## What is the Vanishing & Exploding Gradient Problem?
In artificial neural networks (ANNs), the vanishing and exploding gradient problem occurs during backpropagation when updating weights. This problem is common in deep networks and can hinder training.

- **Vanishing Gradient**: When gradients become very small, making weight updates insignificant, which slows down learning or completely halts it.
- **Exploding Gradient**: When gradients become excessively large, causing weight updates to grow uncontrollably, leading to instability in training.

### Why Does This Problem Occur?
1. **Activation Functions**: Sigmoid and Tanh functions can cause gradients to shrink, leading to vanishing gradients.
2. **Deep Networks**: As the number of layers increases, repeated multiplications of small gradients cause them to vanish, while large values cause them to explode.
3. **Weight Initialization**: Poor initialization can amplify the problem by making gradients too large or too small from the start.

---

## What Was Done in This Notebook?
1. **Dataset Creation**
   - Used `make_moons()` to generate a dataset for binary classification.
   - Visualized the dataset using Matplotlib.
2. **Implementation with Sigmoid Activation**
   - Built a neural network with multiple layers using the `sigmoid` activation function.
   - Measured the change in weights after training to observe vanishing gradients.
3. **Implementation with ReLU Activation**
   - Built a deeper neural network with `ReLU` activation.
   - Compared weight updates before and after training.
   - Observed that ReLU helps mitigate the vanishing gradient problem.
4. **Comparison of Gradients**
   - Calculated the percentage change in weights before and after training.
   - Demonstrated how deep networks are affected by vanishing/exploding gradients.

---

## Key Takeaways
- **Sigmoid Activation**: Prone to vanishing gradients due to its bounded range.
- **ReLU Activation**: Helps mitigate the vanishing gradient problem by preventing small gradient updates.
- **Deep Networks**: More layers increase the risk of vanishing/exploding gradients.
- **Gradient Tracking**: Calculating weight changes helps in diagnosing training issues.

### Possible Solutions to Vanishing & Exploding Gradients
- Use `ReLU` or `Leaky ReLU` instead of `sigmoid` or `tanh`.
- Apply `Batch Normalization` to stabilize activations.
- Use proper weight initialization techniques (`Xavier` or `He` initialization).
- Implement `Gradient Clipping` to prevent excessively large weight updates.

This experiment provided hands-on experience in understanding the vanishing/exploding gradient problem and methods to counter it in deep learning models.

