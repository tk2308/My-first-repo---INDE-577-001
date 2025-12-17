# Linear Regression

Linear Regression is a **supervised machine learning algorithm** used for **predicting continuous numerical values**. It models the relationship between input features and a target variable by fitting a straight line that minimizes prediction error.

<p align="center">
<img src="https://statistics.laerd.com/spss-tutorials/img/lr/linear-nonlinear-relationships.png" width="500">
</p>

---

## 1. Overview

Linear Regression attempts to model the relationship between variables using a **straight line**:

[
y = mx + b
]

It assumes that the change in the input feature produces a **proportional and linear** change in the output.

---

## 2. How Linear Regression Works (Step-by-Step)

1. Take the input features (X) and target (y)
2. Fit a straight line that minimizes prediction error
3. Compute slope (m) and intercept (b)
4. Predict values using the line equation
5. Measure error and adjust the line

<p align="center">
<img src="https://miro.medium.com/1*WCcaObzvvVzcrg8CBi6iCQ.jpeg" width="500">
</p>

---

## 3. Cost Function — Mean Squared Error (MSE)

Linear Regression uses **Mean Squared Error (MSE)** to quantify how far predictions are from actual values:

[
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
]

A lower MSE indicates a better fit.

<p align="center">
<img src="https://vitalflux.com/wp-content/uploads/2020/09/Regression-terminologies-Page-3.png" width="500">
</p>

---

## 4. Gradient Descent in Linear Regression

Gradient Descent iteratively adjusts the parameters to reduce MSE.

Parameter updates:

[
m := m - \alpha \frac{\partial}{\partial m}MSE
]

[
b := b - \alpha \frac{\partial}{\partial b}MSE
]

Where:

* (m) → slope
* (b) → intercept
* (\alpha) → learning rate

---

## 5. Assumptions of Linear Regression

Linear Regression relies on several statistical assumptions:

* **Linearity** — (X) and (y) have a linear relationship
* **Homoscedasticity** — equal variance of residuals
* **Independence** — errors are not correlated
* **Normality** — residuals follow a normal distribution

<p align="center">
<img src="https://quantifyinghealth.com/wp-content/uploads/2022/12/checking-the-linearity-assumption.png" width="500">
</p>

---

## 6. Data Preprocessing for Linear Regression

To build a reliable model, ensure data quality.

### Recommended preprocessing:

* Handle missing values
* Remove outliers
* Reduce multicollinearity
* Standardize features (optional)

---

## 7. Model Training in Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

---

## 8. Applications

Linear Regression is used in:

* Predicting housing prices
* Sales forecasting
* Stock market trend estimation
* Demand prediction
* Medical and scientific modeling

<p align="center">
<img src = "https://dataaspirant.com/wp-content/uploads/2023/10/3-8.png" width="500">
</p>

---

## 9. References

1. *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
2. Scikit-Learn Documentation — Linear Regression
3. Andrew Ng — Machine Learning Notes
