# Deep Learning Journey - Day 7: Admission Prediction using ANN

## Overview
This project focuses on predicting the likelihood of a student's admission into a university based on various factors using an Artificial Neural Network (ANN). The dataset contains multiple features such as GRE score, TOEFL score, university rating, and other relevant factors. The goal is to train a deep learning model to predict the chance of admission accurately.

## What is an Artificial Neural Network (ANN)?
Artificial Neural Networks (ANNs) are computing systems inspired by the biological neural networks of the human brain. ANNs consist of multiple layers, including an input layer, hidden layers, and an output layer. They are widely used for tasks such as classification and regression due to their ability to learn complex patterns in data.

## Why This Dataset?
The admission prediction dataset is ideal for learning and implementing ANN because:
- It is a regression problem, allowing us to practice using neural networks for continuous value prediction.
- It includes multiple independent variables, making it a great dataset for feature engineering and deep learning.
- It provides a real-world application of AI in education and decision-making.

## Steps Followed in This Notebook

### 1. Importing Libraries
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
```

### 2. Loading and Exploring the Dataset
```python
df = pd.read_csv('Admission_Predict_Ver1.1.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
df.duplicated().sum()
```
- Checked for missing values and duplicates.
- Dropped the unnecessary column `Serial No.`.

### 3. Defining Features and Target Variable
```python
X = df.iloc[:, 0:-1]  # Independent variables
y = df.iloc[:, -1]    # Dependent variable
```

### 4. Splitting the Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### 5. Normalizing the Data
```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 6. Building the Neural Network Model
```python
model = Sequential()

# Input Layer
model.add(Dense(7, activation='relu', input_dim=7))

# Hidden Layer
model.add(Dense(7, activation='relu'))

# Output Layer
model.add(Dense(1, activation='linear'))

model.summary()
```

### 7. Compiling and Training the Model
```python
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train_scaled, y_train, epochs=200, validation_split=0.2)
```

### 8. Predicting and Evaluating the Model
```python
y_pred = model.predict(X_test_scaled)
r2_score(y_test, y_pred)
```

### 9. Visualizing the Training Progress
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

## Key Takeaways
- Used an Artificial Neural Network to predict admission chances.
- Normalized the dataset to improve model performance.
- Achieved a trained ANN model with two hidden layers.
- Evaluated the model using the R² score to measure prediction accuracy.
- Visualized the loss curve to analyze model training performance.

## Conclusion
This project demonstrated the use of Artificial Neural Networks for regression tasks. By tuning the model and experimenting with different architectures, the prediction accuracy can be further improved. This hands-on approach helped reinforce the fundamentals of deep learning applied to real-world problems.

