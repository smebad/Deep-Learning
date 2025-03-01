# Deep Learning Journey - Day 5: Customer Churn Prediction using ANN

## Introduction
In this notebook, I implemented an Artificial Neural Network (ANN) to predict customer churn. Churn prediction is a crucial task for businesses to understand why customers leave and to take proactive measures to retain them.

## What is ANN?
Artificial Neural Networks (ANNs) are a subset of machine learning models inspired by the structure and function of the human brain. ANNs consist of layers of interconnected neurons that process input data and learn patterns to make predictions. They are particularly effective for handling complex relationships in data, making them suitable for tasks like classification, regression, and pattern recognition.

## Why This Dataset?
The dataset used in this project is the **Churn_Modelling.csv** file, which contains information about bank customers. The target variable **Exited** indicates whether a customer left the bank (1) or remained (0). The goal is to train an ANN model to predict customer churn based on features like credit score, geography, gender, age, balance, and more.

## Steps Followed in the Notebook
### 1. Data Preprocessing
- Loaded the dataset using Pandas.
- Checked for missing values and duplicate entries.
- Dropped unnecessary columns (**RowNumber, CustomerId, Surname**) since they do not contribute to prediction.
- Used one-hot encoding to convert categorical variables (**Geography, Gender**) into numerical form.
- Split the data into independent variables (**X**) and dependent variable (**y**).
- Further split the dataset into training and test sets (80% training, 20% testing).
- Applied **StandardScaler** to normalize the features for better model performance.

### 2. Building the ANN Model
- Created a **Sequential** model using TensorFlow/Keras.
- Added layers:
  - **Input layer** with 11 neurons (equal to the number of features) and **ReLU** activation.
  - **Hidden layer** with 11 neurons and **ReLU** activation.
  - **Output layer** with 1 neuron and **Sigmoid** activation (for binary classification).
- Compiled the model using **Adam optimizer** and **binary cross-entropy loss**.

### 3. Training the Model
- Trained the ANN for **100 epochs** using the training data.
- Used **validation_split=0.2** to monitor model performance on unseen validation data during training.

### 4. Evaluating the Model
- Predicted churn probabilities on the test set.
- Converted probabilities to binary labels (1 or 0) using a threshold of 0.5.
- Calculated the accuracy score of the model.

### 5. Visualizing Model Performance
- Plotted **training vs validation loss** to check for overfitting.
- Plotted **training vs validation accuracy** to analyze model improvement over epochs.

## Key Takeaways
- ANNs can effectively learn patterns in customer churn data.
- Feature scaling is crucial for ANN performance.
- Monitoring loss and accuracy helps in optimizing model training.
- The model can be fine-tuned further by adjusting the number of layers, neurons, learning rate, or trying different activation functions.

