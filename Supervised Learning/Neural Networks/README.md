# Neural Networks

Neural Networks are a class of **supervised machine learning models** inspired by the structure and functioning of the human brain. They consist of interconnected layers of nodes (neurons) that learn complex patterns in data through weighted connections.

<img src = "https://substackcdn.com/image/fetch/$s_!fSMF!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc41b2249-b92e-4dde-aced-c578ccc04ba1_983x473.png" width = "500">

---

## 1. Overview

A neural network is composed of **layers of interconnected neurons**, where each neuron performs:

[
z = w \cdot x + b
]

followed by an **activation function** such as ReLU, Sigmoid, or Tanh.

Neural networks excel at learning **non-linear patterns**, making them suitable for image recognition, NLP, time-series forecasting, and many high-dimensional problems.

---

## 2. Architecture of a Neural Network

A standard neural network has three types of layers:

### **• Input Layer**

Takes the raw features/data.

### **• Hidden Layers**

Perform transformations using **weights**, **biases**, and **activation functions**.

### **• Output Layer**

Produces the final prediction (class label or regression value).

<img src = "https://pub.mdpi-res.com/computation/computation-11-00052/article_deploy/html/images/computation-11-00052-g001-550.jpg?1678088436" width = "500">

---

## 3. How Neural Networks Learn (Forward + Backprop)

### **Step-by-step:**

1. Input passes through each layer (**forward propagation**)
2. Model produces a prediction
3. Loss function calculates the error
4. **Backpropagation** computes gradients
5. **Gradient Descent** updates weights and biases
6. Repeat for multiple epochs until convergence

<img src = "https://miro.medium.com/1*vTPtIjnwBqjqTpYfvcurKw.png" width = "500">

---

## 4. Activation Functions

Activation functions introduce **non-linearity**. Common ones include:

### **• ReLU (Rectified Linear Unit)**

Fast and widely used in deep learning.

### **• Sigmoid**

Used for binary classification.

### **• Tanh**

Zero-centered alternative to Sigmoid.

### **• Softmax**

Used for multi-class classification.

<img src = "https://i.ytimg.com/vi/aywf1vAIc6Y/sddefault.jpg?v=67ebad85" width = "500">

---

## 5. Loss Functions

Neural networks optimize a loss function that measures prediction error.

### **For Classification**

* Binary Cross-Entropy
* Categorical Cross-Entropy

### **For Regression**

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)

<img src = "https://miro.medium.com/v2/resize:fit:1400/0*ayO78n7DEJAk198B" width = "500">

---

## 6. Common Challenges in Neural Networks

### **• Overfitting**

Solution: dropout, regularization, early stopping.

### **• Vanishing/Exploding Gradients**

Solution: ReLU activation, batch normalization, better initialization.

### **• Computational Cost**

Solution: GPUs, mini-batch training.

<img src = "https://pengfeinie.github.io/images/overfitting_21.png" width = "500">

---

## 7. Training a Neural Network Using Scikit-Learn (MLPClassifier)

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Define the neural network
model = MLPClassifier(hidden_layer_sizes=(64, 32),
                      activation='relu',
                      max_iter=500)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

---

## 8. Applications of Neural Networks

Neural networks are used in:

* Computer vision
* Natural language processing
* Speech recognition
* Recommendation systems
* Fraud detection
* Time-series forecasting

<img src = "https://ars.els-cdn.com/content/image/1-s2.0-S1568494615006122-fx1.jpg" width = "500">

---

## References

1. Rumelhart, Hinton & Williams — *Learning Representations by Back-propagating Errors*
2. Goodfellow, Bengio & Courville — *Deep Learning*
3. Scikit-learn MLP Documentation
