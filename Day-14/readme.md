# Deep Learning Journey - Day 14: Regularization in Deep Learning

## What is Regularization in Deep Learning?
Regularization in deep learning is a set of techniques used to prevent overfitting, which occurs when a model learns the noise in the training data instead of the actual pattern. Overfitting results in poor generalization, meaning the model performs well on training data but fails on unseen data.

Regularization methods help improve the model's ability to generalize by adding constraints to the learning process. Some common regularization techniques include:

- **L1 and L2 Regularization (Weight Decay):** Adds a penalty to the loss function to prevent large weights.
- **Dropout:** Randomly drops units during training to force the network to learn robust features.
- **Early Stopping:** Stops training when validation loss stops improving.
- **Batch Normalization:** Stabilizes learning and reduces dependence on initialization.

## Why Use Regularization?
Deep learning models have millions of parameters and can easily memorize training data instead of generalizing from it. Regularization ensures that the model:

- Avoids overfitting by preventing excessive reliance on certain neurons or weights.
- Improves generalization to new, unseen data.
- Helps maintain smooth decision boundaries instead of overly complex ones.

## What Was Done in This Notebook?
In this notebook, I explored the impact of **regularization in deep learning** using a **binary classification task** on a synthetic dataset. The steps taken were:

1. **Dataset Creation & Visualization:**
   - Used `make_moons` from `sklearn.datasets` to generate a non-linearly separable dataset with noise.
   - Visualized the dataset using `matplotlib`.

2. **Training a Baseline Model (Without Regularization):**
   - Built a neural network using `Sequential` API from TensorFlow/Keras with two hidden layers.
   - Used **ReLU activation** for hidden layers and **sigmoid activation** for the output layer.
   - Compiled the model using the Adam optimizer with a learning rate of 0.01.
   - Trained the model for 500 epochs and plotted decision boundaries & loss curves.

3. **Implementing L1 Regularization:**
   - Modified the model by adding **L1 regularization (L1 penalty = 0.001)** to both hidden layers.
   - Trained the model with the same settings as the baseline.
   - Plotted decision boundaries & loss curves to compare with the baseline model.

4. **Comparing the Weight Distributions:**
   - Extracted the learned weights from the first layer of both models.
   - Used `seaborn` to visualize the weight distributions through **box plots** and **histograms**.
   - Compared the minimum weight values of both models.

## Observations & Insights
- The baseline model, without regularization, overfit the training data, leading to more complex decision boundaries.
- The L1-regularized model produced a **simpler decision boundary**, indicating that regularization helped in reducing overfitting.
- The weight distribution analysis showed that **L1 regularization pushes many weights towards zero**, enforcing sparsity.
- The regularized model had **more stable loss curves**, indicating better generalization.

## Conclusion
Regularization is an essential technique in deep learning to **control overfitting and improve model generalization**. This experiment demonstrated that **L1 regularization leads to sparser weight distributions and more robust decision boundaries**. Future explorations can include **L2 regularization, Dropout, and Batch Normalization** to further compare their effectiveness.

