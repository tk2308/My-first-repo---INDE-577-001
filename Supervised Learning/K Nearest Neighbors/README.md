# K-Nearest Neighbors (KNN) Classification

K-Nearest Neighbors (KNN) is a **supervised machine learning algorithm** used for **classification** and **regression**. It is a simple, intuitive, and non-parametric method that predicts outcomes based on the similarity between data points.

<img src="https://media.licdn.com/dms/image/v2/D4D12AQEaIeOIleYxQw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1711781577058?e=2147483647&v=beta&t=F4KOa5-9KrMTNzkxJ9FIRxNFHdvn3nd_Xu00CI4I1Lo" width="500">

---

## 1. Overview

KNN classifies a new data point by checking the **‘K’ most similar points (neighbors)** from the training dataset and assigning the majority class among those neighbors.

It relies entirely on **distance metrics** to measure similarity.

---

## 2. How KNN Works (Step-by-Step)

1. Choose a value of **K**  
2. Compute distance between new data point and all training points  
3. Sort distances  
4. Take the top K nearest neighbors  
5. Perform **majority voting**  
6. Assign the predicted class  

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_ary_9vclfUOn810jZil7WIHn9JoU9nYn7w&s" width="500">

---

## 3. Distance Metrics in KNN

KNN heavily depends on how distance is measured. Common choices:

### • Euclidean Distance
Most common for continuous numeric data.

### • Manhattan Distance
Useful when movement is grid-like (e.g., city block paths).

### • Minkowski Distance
Generalization of Euclidean and Manhattan.

<img src="https://cdn.botpenguin.com/assets/website/Euclidean_Distance_1_59a98c213f.png" width="500">

---

## 4. Choosing the Right K

- **Small K → Overfitting** (model becomes too sensitive)  
- **Large K → Underfitting** (model becomes too smooth)  

A good practice is to use **odd values** of K when classes are even to avoid ties.

<img src="https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/1_elbow-method.jpeg" width="500">

---

## 5. Data Preprocessing for KNN

KNN is **sensitive to feature scales** because distance-based methods require numerical comparability.

### Preprocessing steps:
- Standardize or normalize the data  
- Remove outliers  
- Handle missing values  
- Encode categorical variables  

<img src="https://miro.medium.com/1*BMHsnpLo_Crnsw0gCMZ0OQ.jpeg" width="500">

---

## 6. Model Training in Scikit-Learn

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
