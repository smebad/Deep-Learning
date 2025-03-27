# Deep Learning Journey - Day 22: First CNN Architecture - LeNet

## Introduction
LeNet-5 is one of the first Convolutional Neural Network (CNN) architectures, introduced by Yann LeCun and his colleagues in 1989. It was primarily designed for handwritten digit recognition and played a crucial role in the advancement of deep learning. LeNet-5 laid the foundation for modern CNN architectures that are widely used today for computer vision tasks.

## Purpose of LeNet-5
When LeNet-5 was developed, it was used for:
- Handwritten digit recognition, specifically for postal code recognition in banking systems.
- Early applications of neural networks in computer vision.
- Demonstrating the effectiveness of CNNs in feature extraction and hierarchical learning.

## What I Did in This Notebook
In this notebook, I implemented the LeNet-5 architecture using TensorFlow and Keras on the MNIST dataset. The key steps included:
1. **Data Preprocessing**
   - Loaded the MNIST dataset.
   - Normalized pixel values to the range [0,1].
   - Resized images from (28,28) to (32,32) to match the original LeNet-5 input size.
   
2. **Building the LeNet-5 Model**
   - Used **Conv2D** layers for feature extraction.
   - Applied **AveragePooling2D** layers to reduce spatial dimensions.
   - Flattened the output and added **Dense** layers for classification.
   - Used the **tanh** activation function as per the original LeNet-5 architecture.
   
3. **Training the Model**
   - Compiled the model using Adam optimizer and sparse categorical cross-entropy loss.
   - Trained the model for 10 epochs.
   - Evaluated its accuracy on the test dataset.

## Modern CNN Architectures
After LeNet-5, several more advanced CNN architectures have been introduced, including:
- **AlexNet (2012)** - Introduced deep CNNs to large-scale image classification.
- **VGGNet (2014)** - Popular for its simplicity and uniform architecture.
- **GoogLeNet (2014)** - Introduced the Inception module.
- **ResNet (2015)** - Introduced residual connections to solve the vanishing gradient problem.
- **DenseNet (2017)** - Improved feature reuse through dense connections.
- **EfficientNet (2019)** - Optimized for accuracy and efficiency.

## Next Steps
I will continue exploring more advanced CNN architectures and their applications. These modern architectures have greatly improved accuracy and efficiency in image classification, object detection, and other computer vision tasks.

