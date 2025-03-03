# Deep Learning Journey - Day 6: Handwritten Digit Classification using ANN

## Overview
This project is part of my Deep Learning journey (Day 6). In this notebook, I implemented an Artificial Neural Network (ANN) to classify handwritten digits using the MNIST dataset. The goal is to build a deep learning model capable of recognizing digits from 0 to 9 based on pixel values.

## What is the MNIST Dataset?
The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits (0-9), with 60,000 images for training and 10,000 for testing. Each image is a 28x28 pixel grayscale image, making it a standard benchmark dataset for computer vision and deep learning tasks.

I used this dataset because it is widely used for learning and experimenting with neural networks. It helps in understanding how deep learning models process image data, how to optimize ANN architectures, and how to improve classification accuracy.

## Steps in the Notebook
### 1. Importing Libraries
I started by importing necessary libraries such as TensorFlow, Keras, NumPy, Matplotlib, and scikit-learn.

### 2. Loading the MNIST Dataset
- Loaded the dataset using `keras.datasets.mnist.load_data()`.
- The dataset was split into training and testing sets.

### 3. Exploring the Data
- Checked the shape of training and test images.
- Displayed a sample image using `matplotlib.pyplot.imshow()`.

### 4. Data Preprocessing
- Normalized the pixel values by dividing by 255 to scale them between 0 and 1. This helps in faster training and better performance.
- Converted the images into a format suitable for the neural network.

### 5. Building the ANN Model
- Defined a Sequential model using Keras.
- Added a `Flatten` layer to convert the 2D image into a 1D array.
- Added two hidden layers with 128 and 32 neurons, using the ReLU activation function.
- Added an output layer with 10 neurons (one for each digit) using the Softmax activation function.
- Displayed the model summary.

### 6. Compiling and Training the Model
- Used the Adam optimizer for efficient training.
- Used Sparse Categorical Crossentropy as the loss function since this is a multi-class classification problem.
- Set accuracy as the evaluation metric.
- Trained the model for 20 epochs with a validation split of 20%.

### 7. Evaluating the Model
- Predicted probabilities for the test set.
- Converted probabilities into class labels.
- Calculated accuracy using `accuracy_score()` from scikit-learn.

### 8. Visualizing Results
- Plotted training and validation loss over epochs.
- Plotted training and validation accuracy over epochs.

### 9. Testing Model Predictions
- Used `model.predict()` on single images to check predictions.
- Displayed the images along with their predicted labels.

## Conclusion
This project helped me understand how to build and train an Artificial Neural Network for image classification tasks. The MNIST dataset served as a great learning tool for experimenting with deep learning models. By fine-tuning the model, adding more layers, or using convolutional neural networks (CNNs), I can further improve the accuracy of digit classification.


