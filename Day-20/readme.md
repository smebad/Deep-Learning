# Deep Learning Journey - Day 20: Padding and Strides in CNN

## Introduction to CNN
A **Convolutional Neural Network (CNN)** is a deep learning model primarily used for analyzing visual data. CNNs are inspired by the human **visual cortex**, which processes images by detecting edges, patterns, and complex structures. Unlike traditional neural networks, CNNs use convolutional layers to efficiently extract features from images, making them ideal for tasks like image classification, object detection, and segmentation.

### CNN vs. Visual Cortex
The human visual cortex processes images through a hierarchy of layers, each detecting more abstract features. Similarly, CNNs use multiple convolutional layers to extract features from low-level edges to high-level structures.

## Convolutional Operations
### What is Padding in CNN?
Padding refers to adding extra pixels around an input image before applying a convolution operation. It helps preserve spatial dimensions and avoid losing information at the edges.
- **Valid Padding**: No extra pixels are added (reduces output size).
- **Same Padding**: Extra pixels are added to ensure the output size matches the input size.

### What are Strides in CNN?
Strides determine how much the convolution filter moves across the image. A stride of **(1,1)** moves pixel by pixel, while **(2,2)** skips one pixel, reducing the output size and increasing computational efficiency.

## Purpose of Padding and Strides
- **Padding** helps retain more spatial information and prevents loss of features at the borders.
- **Strides** help control the dimensionality reduction and computational efficiency of the network.

## What I Did in This Notebook
### Training Without Strides
1. Loaded the **MNIST dataset**, which consists of handwritten digit images.
2. Created a CNN model with three convolutional layers, each using **valid padding**.
3. Used **ReLU activation** in convolutional layers to introduce non-linearity.
4. Flattened the output and added **dense layers** for classification.
5. Summarized the model structure.

### Training With Strides
1. Modified the CNN architecture to use **same padding** and **strides (2,2)**.
2. Used the same dataset and training setup.
3. Observed how the output shape changed due to strides, reducing spatial dimensions while maintaining key features.
4. Compared both models in terms of computational efficiency and feature extraction.

## Key Observations
- The model with **valid padding** maintained the full feature map but required more computation.
- The model with **same padding and strides** reduced the feature map size, making it computationally more efficient.
- **Padding and strides** can be adjusted based on model requirements (accuracy vs. efficiency).

### Conclusion
This experiment highlighted the importance of **padding and strides** in CNNs. By tuning these parameters, we can balance model complexity, efficiency, and performance based on the problem at hand.

