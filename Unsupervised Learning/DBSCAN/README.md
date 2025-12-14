# **DBSCAN Clustering**

DBSCAN (**Density-Based Spatial Clustering of Applications with Noise**) is an **unsupervised machine learning algorithm** used to identify clusters of **arbitrary shape** and detect **noise/outliers** based on data density.

It is especially powerful when clusters are irregular, overlapping, or when the dataset contains significant noise.

<img src="https://media.springernature.com/m685/springer-static/image/art%3A10.1038%2Fs41598-023-45190-4/MediaObjects/41598_2023_45190_Fig1_HTML.png" width="500">

---

## **1. Overview**

DBSCAN groups points that are closely packed together while marking points in low-density regions as **noise**.

DBSCAN requires two parameters:

* **eps** → Maximum distance between two points to be considered neighbors
* **min_samples** → Minimum number of points required to form a dense region

It does **not** require specifying the number of clusters beforehand.

---

## **2. Key Concepts**

### **• Core Point**

A point with at least *min_samples* neighbors within *eps*

### **• Border Point**

A point within *eps* of a core point but with fewer than *min_samples* neighbors

### **• Noise (Outlier)**

A point that is neither core nor border

<img src = "https://www.researchgate.net/publication/315326812/figure/fig2/AS:473095908663297@1489806262333/A-cluster-consists-of-core-points-red-and-border-points-green-Core-points-have-at.png" width = "500">

---

## **3. How DBSCAN Works (Step-by-Step)**

1. Pick an unvisited point
2. Retrieve all neighboring points within **eps**
3. * If neighbors < **min_samples** → mark point as **noise**
   * If neighbors ≥ **min_samples** → start a new cluster
4. Expand the cluster by recursively adding density-reachable points
5. Continue until all points are processed


---

## **4. Advantages of DBSCAN**

* Detects **arbitrarily shaped clusters**
* Automatically identifies **noise and outliers**
* Works well when clusters have variable density
* No need to preselect number of clusters (`k`) like K-Means

---

## **5. Limitations of DBSCAN**

* Struggles with **varying density** in the same dataset
* Performance decreases in **high-dimensional** spaces
* Sensitive to choice of **eps**

---

## **6. Choosing eps and min_samples**

### **• eps**

Typically chosen using a **k-distance graph**, also called an **elbow plot**.

### **• min_samples**

Rule of thumb:

```
min_samples = 2 × number_of_features
```

---

## **7. DBSCAN in Scikit-Learn**

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Running DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Adding labels to DataFrame
df['dbscan_cluster'] = dbscan_labels

# Counting clusters and noise
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = np.sum(dbscan_labels == -1)

print("Clusters found:", n_clusters)
print("Noise points:", n_noise)
```

---

## **8. Visualizing DBSCAN Clusters**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=dbscan_labels, s=40)
plt.title("DBSCAN Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

## **9. Applications of DBSCAN**

* Fraud detection
* Geospatial data analysis
* Noise filtering for sensor data
* Image segmentation
* Customer segmentation with outlier detection

---

## **10. References**

1. Ester, M. et al. *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases*
2. Scikit-Learn Documentation – DBSCAN
3. Research papers on density-based clustering
