# Deep Learning Journey - Day 13: Dropout Layers in ANN

## What are Dropout Layers in an Artificial Neural Network (ANN)?

Dropout is a regularization technique used in artificial neural networks (ANNs) to prevent overfitting. It works by randomly setting a fraction of the input units to zero at each update during training time, which helps to break reliance on specific neurons and encourages the network to learn more robust features.

## Why Should We Use Dropout?

When training deep neural networks, it's common for the model to overfit the training data, meaning it performs well on training but poorly on unseen data. Dropout helps in:
- Reducing overfitting
- Improving model generalization
- Preventing the network from relying too much on specific neurons

## What Was Implemented in This Notebook?

1. **Without Dropout:**
   - A simple artificial neural network (ANN) was trained without using dropout layers.
   - The model was evaluated on training and test data, showing potential overfitting.

2. **With Dropout:**
   - Dropout layers were introduced into the ANN.
   - The network was retrained with dropout applied to certain layers.
   - The model was evaluated again to compare performance and check if overfitting was reduced.

## Can Dropout Be Used for Regression?

Yes! Dropout can be used in regression models as well. The key difference is:
- In regression, the output layer typically does not have an activation function like `softmax` or `sigmoid` but uses a linear activation (`None` in Keras).
- The dropout layers can still be applied to the hidden layers to improve generalization.

### Example Code for Dropout in Regression:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Dropout applied to prevent overfitting
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')  # Linear activation for regression
])
```

## Comparison and Observations

- The model without dropout overfits, showing lower training loss but higher test loss.
- The model with dropout generalizes better, reducing overfitting while maintaining good performance on unseen data.
- Dropout is an effective way to improve ANN robustness, especially for large datasets or deep networks.

---
### Conclusion
Dropout is a simple yet powerful technique to regularize ANNs. It prevents the model from over-relying on specific neurons and enhances generalization, making it a useful tool for both classification and regression problems.
