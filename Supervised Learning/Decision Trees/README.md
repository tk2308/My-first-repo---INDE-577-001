# Decision Tree Classification

Decision Trees are supervised machine learning models used for **classification** and **regression**. They predict outcomes by recursively splitting data based on feature values, forming a tree-like structure that is intuitive and easy to interpret.

<img src = "https://i.ytimg.com/vi/ZVR2Way4nwQ/maxresdefault.jpg" width = "500">

---

## Overview

A decision tree learns a set of decision rules from input features to predict a target variable. Each internal node represents a feature-based condition, each branch represents an outcome of that condition, and each leaf node represents a final prediction.

Decision trees are commonly used due to their interpretability and ability to model non-linear relationships. They also serve as the foundation for ensemble models such as **Random Forests**.

---

## Tree Structure

A decision tree consists of:

- **Root Node** – Represents the full dataset  
- **Decision Nodes** – Apply conditions on features  
- **Branches** – Outcomes of decision rules  
- **Leaf Nodes** – Final predicted class or value  

<img src = "[https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQRBaG0IhnP0oXHWTlo6qDNW0iOABz2rRKFSw&s](https://techeasyblog.com/wp-content/uploads/2024/06/decision-tree.png)" width="400">

---

## How Decision Trees Learn

Decision trees follow a **greedy, top-down** approach. At each node, the algorithm selects the feature and threshold that result in the purest child nodes. This recursive process continues until a stopping condition is met.

Common stopping conditions include:
- Maximum tree depth
- Minimum samples required to split
- Pure or near-pure leaf nodes

---

## Impurity Measures

Decision trees use impurity metrics to decide how data should be split.

### Gini Impurity

Gini impurity measures how often a randomly chosen element would be incorrectly classified.

- A Gini score of 0 indicates a pure node
- Lower values correspond to better splits

<img src = "https://storage.googleapis.com/lds-media/images/gini-impurity-diagram.width-1200.png" width ="400">

---

### Entropy (Information Gain)

Entropy measures the level of uncertainty in a node.

Entropy is used to calculate **Information Gain**, which quantifies the reduction in uncertainty after a split.

<img src = "https://i.ytimg.com/vi/Xfgq8zh8sDQ/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDaF-n7Mqt_PFaUu3z9DLSELPcl9g" width = "400">

---

## CART Algorithm

Scikit-learn implements decision trees using the **Classification and Regression Tree (CART)** algorithm. CART always builds **binary trees** by minimizing impurity at each split.

Where:
- `k` is the selected feature  
- `t` is the threshold  
- `m` is the total number of samples  

---

## Prediction Process

To generate a prediction:

1. A feature vector enters the root node  
2. Decision rules are evaluated at each node  
3. The input follows a branch based on conditions  
4. The leaf node outputs the prediction  

<img src = "https://cdn.prod.website-files.com/6634a8f8dd9b2a63c9e6be83/669a6b70c6ea02632c34a53f_421637.image1.jpeg" width = "400">

---

## Overfitting and Pruning

Decision trees are prone to overfitting when allowed to grow too deep.

### Pre-Pruning Techniques
- Limit maximum depth
- Set minimum samples per split
- Set minimum samples per leaf

### Post-Pruning
- Train a full tree
- Remove branches that do not improve performance

<img src ="https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Before_after_pruning.png/1200px-Before_after_pruning.png" width = "400">

---

## Model Evaluation (Classification)

Decision tree classifiers are evaluated using the following metrics:

### Confusion Matrix
Displays predicted versus actual class labels.

<img src = "https://images.prismic.io/encord/edfa849b-03fb-43d2-aba5-1f53a8884e6f_image5.png?auto=compress,format" width ="400">

### Precision

### Recall

### F1 Score

These metrics are particularly useful when working with imbalanced datasets.

---

## Advantages

- Highly interpretable and easy to explain
- Requires minimal data preprocessing
- Can model non-linear relationships
- Works with numerical and categorical data

---

## Limitations

- Sensitive to small changes in data
- Prone to overfitting
- Greedy splitting may not find optimal trees
- Can become computationally expensive

---

## Applications

Decision trees are widely used in:
- Customer behavior prediction
- Risk and fraud detection
- Medical decision support
- Feature importance analysis

<img src ="https://jaro-website.s3.ap-south-1.amazonaws.com/2024/11/Decision-Tree-Applications.webp" width = "400">

---

## References

1. Breiman, L. et al. *Classification and Regression Trees*, 1984  
2. Quinlan, J. R. *Induction of Decision Trees*, 1986  
3. Scikit-learn Documentation – Decision Trees  
   https://scikit-learn.org/stable/modules/tree.html









