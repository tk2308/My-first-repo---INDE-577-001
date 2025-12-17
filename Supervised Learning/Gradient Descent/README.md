# Gradient Descent Optimization

Gradient Descent is a foundational optimization algorithm used in machine learning to minimize a loss function by iteratively updating model parameters in the direction of the steepest descent. 
It is widely used in regression models, neural networks, logistic regression, support vector machines, and deep learning.

<p align="center">
<img src = "https://cdn.analyticsvidhya.com/wp-content/uploads/2024/09/631731_P7z2BKhd0R-9uyn9ThDasA.webp" width = "500">
</p>

---

## 1. Overview

Gradient Descent is an iterative optimization algorithm used to find the set of parameters (weights) that minimize a cost or loss function.

The idea:

- Compute how much each parameter contributes to the error
- Adjust the parameters in the direction that reduces error
- Repeat until convergence

It works for convex and non-convex problems (like deep neural networks), but behavior differs based on the function shape.

---

## 2. The Core Idea

At each iteration:

\[
\theta := \theta - \alpha \cdot \nabla J(\theta)
\]

Where:  
- \( \theta \) → model parameters  
- \( \alpha \) → learning rate  
- \( \nabla J(\theta) \) → gradient of loss function  

The gradient tells us the direction of **maximum increase**, so we move in the *opposite direction* to reduce loss.

<p align="center">
<img src = "https://miro.medium.com/v2/resize:fit:1400/0*IYeBGx90QcOfvX0w.png" width = "500">
</p>

---

## 3. Learning Rate (α)

The learning rate determines how big each step should be:

- Too **small** → slow training  
- Too **big** → divergence or oscillation  
- Optimal → fast and stable convergence  

<p align="center">
<img src = "https://www.bdhammel.com/assets/learning-rate/lr-types.png" width = "500">
</p>

---

## 4. Types of Gradient Descent

### 4.1 Batch Gradient Descent
Uses **all training examples** at every step.

**Pros:** stable, converges smoothly  
**Cons:** slow for large datasets  

<p align="center">
<img src = "https://i0.wp.com/spotintelligence.com/wp-content/uploads/2024/05/batch-gradient-descent.jpg?fit=1200%2C675&ssl=1" width = "500">
</p>

---

### 4.2 Stochastic Gradient Descent (SGD)

Updates parameters **after each training example**.

**Pros:** fast, allows escape from local minima  
**Cons:** noisy updates  

<p align="center">
<img src = "https://spotintelligence.com/wp-content/uploads/2024/03/stochastic-gradient-descent-ml.jpg" width = "500">
</p>

---

### 4.3 Mini-Batch Gradient Descent  
Most commonly used. Splits data into batches (e.g., 32, 64, 128).

**Pros:** balance between speed and stability  
**Cons:** batch size needs tuning  

<p align="center">
<img src = "https://substackcdn.com/image/fetch/$s_!Kuy_!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd5aa826d-4783-4821-89cc-d65148196cf4_887x356.png" width = "500">
</p>

---

## 5. Loss Functions Used

Gradient Descent requires a differentiable loss function.  
Examples:

### 5.1 Regression Loss  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  

### 5.2 Classification Loss  
- Logistic Loss (Binary Cross Entropy)  
- Softmax Cross Entropy  

---

## 6. Cost Function Surface

Gradient Descent can be visualized on a 3D surface showing how parameters affect loss.

<p align="center">
<img src = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSu-BqQWSA25ucvesg4chYCZS0kz21z4iZBOg&s" width = "500">
</p>

---

## 7. Convergence Criteria

Gradient Descent stops when:

- Loss improvement becomes minimal  
- Gradient becomes near zero  
- Max iterations reached  


---

## 8. Challenges & Solutions

### 8.1 Local Minima / Saddle Points  
SGD and variants help escape.

### 8.2 Choosing Learning Rate  
Use learning rate schedulers or adaptive optimizers.

### 8.3 Slow Convergence  
Feature scaling (standardization) improves performance.

<p align="center">
<img src = "https://i.ytimg.com/vi/HcAGThqN7UE/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLA1XSvrkcLBBxWI6jt6syZMSiM3qg" width = "500">
</p>

---

## 9. Advanced Gradient Descent Variants

### 9.1 Momentum  
Reduces oscillations by smoothing updates.

<p align="center">
<img src = "https://i.ytimg.com/vi/Q_sHSpRBbtw/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDS1NOdF6gFreVE0XkGnsH7ggV-Bw" width = "500">
</p>

### 9.2 RMSProp  
Adapts learning rate by tracking squared gradients.

<p align="center">
<img src = "https://images.deepai.org/glossary-terms/2b2ca6a239fc4b2c89368b46abfa64df/rms.png" width = "500">
</p>

### 9.3 Adam Optimizer  
Combines Momentum + RMSProp  
Most widely used in deep learning.

<p align="center">
<img src = "https://www.xenonstack.com/hubfs/adam-optimization.png" width = "500">
</p>

---

## 10. Applications

Gradient Descent is used across nearly all machine learning models:

- Linear Regression  
- Logistic Regression  
- Neural Networks  
- Deep Learning  
- Support Vector Machines  
- Reinforcement Learning  


---

## 11. References

1. Andrew Ng – Machine Learning Course Notes  
2. Deep Learning Book – Goodfellow, Bengio, Courville  
3. Scikit-learn Documentation  
4. Stanford CS229 Lecture Notes on Gradient Descent
