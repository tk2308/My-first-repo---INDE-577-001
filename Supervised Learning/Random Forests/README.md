# Random Forest (Classification & Regression)

Random Forest is a **supervised machine learning algorithm** that uses an **ensemble of decision trees** to perform **classification** or **regression**. It improves accuracy, reduces overfitting, and handles high-dimensional data effectively.

<p align="center">
<img src="https://substackcdn.com/image/fetch/$s_!Uxm0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F959001a5-9bfa-4d0c-9a19-e48bca541cdd_1010x716.png" width="500">
</p>

---

## 1. Overview

Random Forest builds **multiple independent decision trees** and combines their predictions through:

* **Majority voting** → Classification
* **Averaging** → Regression

This ensemble approach creates a **robust**, **accurate**, and **generalizable** model.

<p align="center">
<img src="https://serokell.io/files/vz/vz1f8191.Ensemble-of-decision-trees.png" width="500">
</p>

---

## 2. How Random Forest Works (Step-by-Step)

1. Select random samples from the dataset (**bootstrap sampling**)
2. Build many decision trees using random subsets of features
3. Make predictions using each tree
4. Combine all predictions using:

   * Majority vote (classification)
   * Mean value (regression)

<p align="center">
<img src="https://datasciencedojo.com/wp-content/uploads/2-7.png" width="500">
</p>

---

## 3. Key Concepts

### **• Bootstrap Aggregation (Bagging)**

Each tree sees a slightly different dataset, increasing diversity.

### **• Random Feature Selection**

Each split considers only a random subset of features — reducing correlation between trees.

### **• Ensemble Prediction**

More trees → more stable, reliable predictions.

---

## 4. Advantages of Random Forest

* **High accuracy**
* **Resistant to overfitting**
* Handles **non-linear relationships**
* Works well with large datasets
* Measures **feature importance**
* Handles both **numerical and categorical** data

---

## 5. Limitations

* Can be **computationally expensive** with many trees
* Harder to interpret than a single decision tree
* Slower predictions for real-time systems

---

## 6. Feature Importance

Random Forest provides a ranking of features based on how much they reduce impurity across all trees.

---

## 7. Hyperparameters to Tune

### **Main parameters:**

* `n_estimators` → number of trees
* `max_depth` → maximum depth of each tree
* `min_samples_split` → minimum samples to split
* `max_features` → number of features to consider per split
* `bootstrap=True/False`


<p align="center">
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/03/Screenshot-2020-03-04-at-15.11.56.png" width="500">
</p>


---

## 8. Random Forest in Scikit-Learn

### **Classification Example**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
```

### **Regression Example**

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, max_depth=None)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 9. When to Use Random Forest

Use Random Forest when:

* Dataset has **many features**
* You want **high accuracy without heavy preprocessing**
* You need a model resistant to **noise and overfitting**
* You want to know **which features matter most**

---

## 10. References

1. Breiman, L. *Random Forests*, 2001
2. Scikit-Learn Documentation – Random Forest
3. Ensemble Learning Theory and Bagging Methodsrate the GitHub-ready Markdown.
