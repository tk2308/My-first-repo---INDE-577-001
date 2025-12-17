# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is an **unsupervised dimensionality reduction technique** used to transform high-dimensional data into a lower-dimensional space while preserving as much variance as possible.

It is widely used in **machine learning preprocessing**, **visualization**, and **noise reduction**.

<img src="https://teamraft.com/wp-content/uploads/jan_04_2019.png" width="500">

---

## 1. Overview

PCA identifies the **principal components**, which are new axes capturing the maximum variance in the dataset.
These components are **orthogonal** and sorted by the amount of variance they explain.

PCA helps:

* Reduce computational cost
* Remove multicollinearity
* Improve model performance
* Visualize high-dimensional data

---

## 2. How PCA Works (Step-by-Step)

1. Standardize the dataset
2. Compute the covariance matrix
3. Calculate eigenvalues & eigenvectors
4. Sort eigenvectors by descending eigenvalues
5. Select top components
6. Transform the original dataset

<img src="https://b1879915.smushcdn.com/1879915/wp-content/uploads/2024/12/Process-of-PCA.jpg?lossy=2&strip=1&webp=1" width="500">

---

## 3. Covariance Matrix

PCA begins by constructing a **covariance matrix**, showing how two variables vary together.

$$
\mathrm{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
$$


High covariance → variables change together
Low covariance → variables behave independently

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20251118162513340431/covariance_matrix_2.webp" width="500">

---

## 4. Eigenvalues and Eigenvectors

* **Eigenvalues** measure how much variance a principal component captures.
* **Eigenvectors** represent the directions (axes) of the new feature space.

Largest eigenvalue → most important component

<img src="https://miro.medium.com/1*qG4PEnoSWQRLoEL_P1ruhw.jpeg" width="500">

---

## 5. Explained Variance Ratio

This shows how much of the dataset’s total variance each principal component explains.

$$
\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}
$$


Often visualized using a **scree plot**:

<img src="https://statisticsglobe.com/wp-content/uploads/2022/12/screeplot_pca_mtcars.png" width="500">

---

## 6. PCA for Visualization

PCA can reduce data to **2 or 3 dimensions** to help visualize clusters or patterns.

Common uses:

* K-Means clustering visualization
* Outlier detection
* Exploratory data analysis

---

## 7. Data Preprocessing for PCA

PCA is **affected by scale**, so preprocessing is important.

Required steps:

* Standardize numerical features
* Remove outliers
* Encode categorical variables if necessary
* Check for missing values

---

## 8. Implementing PCA in Scikit-Learn

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

---

## 9. PCA vs Other Techniques

| Technique | Purpose                             | Key Idea                     |
| --------- | ----------------------------------- | ---------------------------- |
| PCA       | Dimensionality reduction            | Uses variance & eigenvectors |
| t-SNE     | Visualization                       | Preserves local structure    |
| LDA       | Supervised dimensionality reduction | Maximizes class separability |

---

## 10. Applications of PCA

* Image compression
* Noise filtering
* Face recognition
* Clustering (preprocessing)
* Reducing multicollinearity

<img src="https://media.licdn.com/dms/image/v2/D4D12AQHq3Cw6bYwFOA/article-inline_image-shrink_1000_1488/B4DZmBCcd8GwAU-/0/1758806528070?e=2147483647&v=beta&t=1JBjQicpsRs_zvBO1jQ5123i29SgOlUisNCWC6_knxc" width="500">

---

## References

1. Jolliffe, I. T., *Principal Component Analysis*
2. Scikit-learn PCA Documentation — [https://scikit-learn.org/stable/modules/decomposition.html](https://scikit-learn.org/stable/modules/decomposition.html)
3. Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space*
