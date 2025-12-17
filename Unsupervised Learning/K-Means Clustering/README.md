# K-Means Clustering

K-Means is an **unsupervised machine learning algorithm** used to **group data into clusters** based on similarity. It works by partitioning data into *K distinct clusters* such that each point belongs to the cluster with the nearest centroid.

<p align="center">
  <img src="https://www.ejable.com/wp-content/uploads/2023/11/Unlabeled-data-vs.-labeled-clusters-2.webp" width="500">
</p>


---

## 1. Overview

K-Means aims to divide data into **K clusters**, where each cluster is represented by a **centroid**.

The algorithm minimizes the **within-cluster sum of squares (WCSS)** by iteratively updating cluster assignments.

$$
\text{WCSS} = \sum_{k=1}^{K} \sum_{i \in C_k} \lVert x_i - \mu_k \rVert^2
$$


---

## 2. How K-Means Works (Step-by-Step)

1. Choose the number of clusters **K**
2. Randomly initialize **K centroids**
3. Assign each data point to the nearest centroid
4. Update centroids as the mean of all points in the cluster
5. Repeat until centroids stabilize (**convergence**)

<img src = "https://lh4.googleusercontent.com/-LLC8RDjcP3RhLEP67nDHhN8n90U8Z_42uG32Z9TxArZbKkM0jD3NaNz2CmmfqdUd7iEcChroOVrOmO7YR4yMk8PHZvuDXFVoHaCiBDXPX_XQbyIsdR4DYrh8jDUmrEkWQmpxwQqmGrdeZ9sVg" width = "500">

---

## 3. Choosing the Right K (Elbow Method)

K-Means requires selecting the correct number of clusters.

* Too few clusters → oversimplified
* Too many clusters → overfitting

The **Elbow Method** helps identify the optimal K by plotting WCSS vs. K.

<p align="center">
  <img src="https://miro.medium.com/0*aY163H0kOrBO46S-.png" width="500">
</p>

---

## 4. Distance Metric in K-Means

K-Means typically uses **Euclidean distance**:

$$
d = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$


<p align="center">
  <img src="https://cdn.prod.website-files.com/5ef788f07804fb7d78a4127a/623c68cda43982d04d80a752_Engati-Euclidean-distance%20(1).jpg" width="500">
</p>


---

## 5. Data Preprocessing for K-Means

Because K-Means is distance-based, **feature scaling is essential**.

### Recommended preprocessing:

* Standardize numerical features
* Remove outliers
* Encode categorical variables (if any)
* Check for feature imbalance

<p align="center">
  <img src="https://miro.medium.com/1*BMHsnpLo_Crnsw0gCMZ0OQ.jpeg" width="500">
</p>


---

## 6. Visualizing K-Means Clusters

K-Means creates **circular, equally sized clusters** in ideal datasets.
Visualization often includes:

* Cluster scatter plots
* Centroid markers
* PCA for 2D/3D projection

<p align="center">
  <img src="https://miro.medium.com/0*uJMVSV1wM5ZtOZhg.png" width="500">
</p>


---

## 7. Model Training in Scikit-Learn

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Running K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Centroids
centroids = kmeans.cluster_centers_

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

---

## 8. Advantages

* Simple and fast
* Works well on large datasets
* Easy to interpret
* Efficient with well-separated clusters

---

## 9. Limitations

* Must choose K manually
* Sensitive to outliers
* Assumes spherical clusters
* Not suitable for complex cluster shapes

<p align="center">
  <img src="https://fastercapital.com/i/K-means-clustering--How-to-Use-a-Simple-and-Fast-Algorithm-for-Marketing-Segmentation-and-Grouping--Challenges-and-Limitations-of-K-means.webp" width="500">
</p>


---

## 10. Applications

K-Means is widely used in:

* Customer segmentation
* Image compression
* Anomaly detection
* Document clustering
* Market segmentation

<p align="center">
  <img src="https://cdn-ileecnj.nitrocdn.com/JHsXwyfxJOYTadtVKgrLqQCwYuZZjQpq/assets/images/optimized/rev-72cea9a/www.lyzr.ai/wp-content/uploads/2024/11/napkin-selection-1-2-1024x781.png" width="500">
</p>


---

## References

1. MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations*
2. Scikit-learn KMeans Documentation
3. Stanford CS231n Notes – Clustering
Would you like all of them in the same markdown format as this one?

