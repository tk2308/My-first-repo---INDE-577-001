# Multilayer Perceptron (MLP)

A **Multilayer Perceptron (MLP)** is a type of **feedforward artificial neural network** used for **classification** and **regression** tasks. It learns complex, non-linear decision boundaries by applying layers of neurons connected through learned weights.

<img src="https://aiml.com/wp-content/uploads/2022/06/Multilayer-perceptron-MLP.png" width="500">

---

## 1. Overview

An MLP consists of:

* **Input Layer** – receives features
* **Hidden Layers** – perform learned transformations
* **Output Layer** – produces final predictions

Each neuron applies a weighted sum and an activation function:

[
z = w_1x_1 + w_2x_2 + \cdots + b
]

[
a = f(z)
]

Where:

* (w) = weights
* (b) = bias
* (f) = activation function

MLPs can model **non-linear patterns** that algorithms like Linear Regression cannot.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200518233718/model_visulai.jpg" width="500">

---

## 2. How MLP Works (Step-by-Step)

1. Input features are fed into the first layer
2. Weighted sums + biases are computed
3. Activation functions introduce non-linearity
4. Forward propagation produces predictions
5. Errors are computed using a loss function
6. Backpropagation adjusts weights to reduce error

<img src="khttps://miro.medium.com/v2/resize:fit:1200/1*T4ARzySpEQvEnr_9pc78pg.jpeg" width="500">

---

## 3. Activation Functions

Activation functions allow MLPs to learn complex patterns.

### Common Activations:

* **ReLU** – fast convergence, widely used
* **Sigmoid** – used in binary classification
* **Tanh** – centered version of sigmoid
* **Softmax** – used in multi-class classification

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRBlV0lcL2jVoX49tBlJyGf9WdA2VjK-QfDOw&s" width="500">

---

## 4. Loss Functions

Depending on the problem type:

### **Classification**

* Binary Cross-Entropy
* Categorical Cross-Entropy

### **Regression**

* Mean Squared Error
* Mean Absolute Error

<img src="https://miro.medium.com/v2/resize:fit:1400/0*FOIMf5cDkBznAT14" width="500">

---

## 5. Training MLP Using Backpropagation

MLP learns by minimizing loss through **Gradient Descent**:

[
w := w - \alpha \frac{\partial L}{\partial w}
]

[
b := b - \alpha \frac{\partial L}{\partial b}
]

Where:

* (L) = loss
* (\alpha) = learning rate

Backpropagation computes gradients layer-by-layer using the chain rule.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20250701163824448467/Backpropagation-in-Neural-Network-1.webp" width="500">

---

## 6. Advantages of MLP

* Learns **non-linear** and **complex** relationships
* Works for both classification and regression
* Supports multiple layers and large feature sets
* Can approximate any function (Universal Approximation Theorem)

---

## 7. Limitations of MLP

* Requires large datasets
* Computationally expensive
* Risk of overfitting
* Needs proper hyperparameter tuning

<img src="https://cdn.prod.website-files.com/614c82ed388d53640613982e/6360ef26a44bba38e5a48fb2_good-fitting-model-vs-overfitting-model-1.png" width="500">

---

## 8. Data Preprocessing for MLP

MLPs require well-processed data.

### Recommended Steps:

* Normalize or standardize features
* Encode categorical variables
* Remove outliers
* Balance classes (if classification)

<img src="https://miro.medium.com/1*BMHsnpLo_Crnsw0gCMZ0OQ.jpeg" width="500">

---

## 9. Model Training in Scikit-Learn

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# MLP model
mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',
                    max_iter=500)

mlp.fit(X_train, y_train)

# Predictions
y_pred = mlp.predict(X_test)
```

---

## 10. Real-World Applications

* Image and handwriting recognition
* Fraud detection
* Medical classification
* Natural language processing
* Customer churn prediction

<img src="https://lh7-rt.googleusercontent.com/docsz/AD_4nXfDB9R9V6ozZ7SM_IzPSp-H2qptLS_ApCGopYIGzTAlBPputQ-flXIA0R37Ubm5a1y4BEvA-MwmstaDE2y4pK14JUIRTnpGSnJiCLM2lpVBWpue7mDhYOtjzqto1ltPCe1286yL-3XeKQhR-tUQRAMxsFE?key=t44fzT5QFvN1UaxQYnN6Ew" width="500">

---

## 11. References

1. Rumelhart, Hinton & Williams — *Learning Representations by Back-Propagating Errors*
2. Scikit-Learn Documentation — MLPClassifier & MLPRegressor
3. “Deep Learning” by Ian Goodfellow

