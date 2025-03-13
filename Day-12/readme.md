# Deep Learning Journey - Day 12: Data Scaling in Neural Networks

## What is Data Scaling in Neural Networks?
Data scaling is the process of transforming input features to a common scale, typically within a range such as [-1,1] or [0,1]. In neural networks, data scaling ensures that input features contribute equally to the learning process, preventing issues where features with larger numerical values dominate those with smaller values. This leads to improved convergence and model performance.

## Why Should We Use Data Scaling?
Neural networks rely on gradient-based optimization techniques, such as stochastic gradient descent (SGD). If the input features are on different scales, the gradients may be uneven, causing the network to converge slowly or get stuck in local minima. Data scaling helps:
- Improve convergence speed
- Reduce the risk of vanishing or exploding gradients
- Enhance model performance and accuracy

## What I Did in This Notebook
In this notebook, I explored the impact of data scaling on neural network training by:
1. Loading and preprocessing the `Social_Network_Ads.csv` dataset
2. Visualizing the raw feature distributions using scatter plots
3. Training a neural network on unscaled data and observing its performance
4. Applying `StandardScaler` to scale the features
5. Retraining the model with scaled data and comparing the results

## Implementations and Comparisons
### 1. Training Without Scaling
- A simple feedforward neural network with two dense layers was implemented.
- The model was trained on unscaled data for 100 epochs.
- Validation accuracy was plotted to observe performance.

### 2. Applying Standard Scaling
- Used `StandardScaler` from `sklearn.preprocessing` to scale features.
- Transformed both training and testing sets before feeding them into the model.
- Trained the same neural network architecture on the scaled dataset.
- Compared validation accuracy before and after scaling.

## Observations and Results
- The model trained on unscaled data showed slower convergence and lower validation accuracy.
- After applying data scaling, the model trained more efficiently and achieved better performance.
- The validation accuracy curve improved significantly when using scaled data.

## Conclusion
Scaling input features is a crucial preprocessing step in training neural networks. It ensures that the optimization process is smooth and prevents numerical instability. This experiment demonstrated the clear benefits of using data scaling for training deep learning models efficiently.

