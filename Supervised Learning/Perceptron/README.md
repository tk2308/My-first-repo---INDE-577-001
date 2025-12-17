# Perceptron Algorithm

---

## 1. Introduction

The **Perceptron** is one of the earliest supervised learning algorithms used for **binary classification**.
It mimics a biological neuron by computing a weighted sum of inputs followed by an activation function.

The Perceptron is a **linear classifier**, meaning it can only learn linearly separable decision boundaries.

<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20251209120638608023/bhu.webp" width="700">
</p>

---

## 2. Problem Type

1. Learning Type: Supervised Learning
2. Task: Binary Classification
3. Model Type: Linear Classifier

---

## 3. Perceptron Architecture

The perceptron consists of:

1. Input features
2. Weights associated with each feature
3. A bias term
4. An activation function

---

## 4. Mathematical Formulation

### 4.1 Linear Combination

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$


Where:

1. ( x_i ) are input features
2. ( w_i ) are weights
3. ( b ) is the bias

---

### 4.2 Activation Function

$$
\hat{y} =
\begin{cases}
1, & \text{if } z \ge 0 \\
0, & \text{otherwise}
\end{cases}
$$

---

## 5. Learning Rule (Weight Update)

$$
w_i \leftarrow w_i + \eta (y - \hat{y}) x_i
$$

$$
b \leftarrow b + \eta (y - \hat{y})
$$

---

## 6. Training Workflow

1. Initialize weights and bias
2. Iterate over training samples
3. Compute weighted sum
4. Apply activation function
5. Update weights if misclassified
6. Repeat for multiple epochs

<p align="center">
  <img src="https://i.sstatic.net/HX8Pn.png" width="700">
</p>

---

## 7. Hyperparameters

1. Learning Rate (( \eta ))
2. Number of Epochs
3. Random State

---

## 8. Assumptions

1. Data is linearly separable
2. Output labels are binary
3. Features are numeric
4. Feature scaling improves convergence

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8VbLHmMCUA0-PdUjJykfTwEaE3i1eICf9KA&s" width="700">
</p>

---

## 9. Advantages

1. Simple and fast to implement
2. Computationally efficient
3. Easy to interpret
4. Foundation of neural networks

---

## 10. Limitations

1. Cannot handle non-linear decision boundaries
2. Fails on XOR-type problems
3. No probabilistic outputs
4. Sensitive to feature scaling

---

## 11. Relation to Neural Networks

1. Single perceptron → linear model
2. Multiple perceptrons → Multi-Layer Perceptron
3. Non-linear activations → Deep learning

---

## 12. Summary

The Perceptron is a foundational machine learning algorithm that introduced weight-based learning and formed the basis of modern neural networks.

---

## 13. References

1. Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*
2. [https://scikit-learn.org/stable/modules/linear_model.html#perceptron](https://scikit-learn.org/stable/modules/linear_model.html#perceptron)
3. [https://en.wikipedia.org/wiki/Perceptron](https://en.wikipedia.org/wiki/Perceptron)


