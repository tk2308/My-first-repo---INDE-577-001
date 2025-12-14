# Linear Regression

Linear Regression is a **supervised machine learning algorithm** used for **predicting continuous numerical values**. It models the relationship between input features and a target variable by fitting a straight line that minimizes prediction error.

<img src="https://statistics.laerd.com/spss-tutorials/img/lr/linear-nonlinear-relationships.png" width="500">

---

## 1. Overview

Linear Regression attempts to model the relationship between variables using a **straight line**:

\[
y = mx + b
\]

It assumes that the change in the input feature produces a **proportional and linear** change in the output.

---

## 2. How Linear Regression Works (Step-by-Step)

1. Take the input features \(X\) and target \(y\)  
2. Fit a straight line that minimizes error  
3. Compute slope and intercept  
4. Predict values using the line equation  
5. Evaluate error between predicted and actual values  

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpk6Odnbf8OMNgaov5v9wLG5PViK2RlIjiSw&s" width="500">

---

## 3. Cost Function (Mean Squared Error)

Linear Regression minimizes **MSE (Mean Squared Error)**:

\[
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
\]

Lower values indicate a better fit.

<img src="https://vitalflux.com/wp-content/uploads/2020/09/Regression-terminologies-Page-3.png" width="500">

---

## 4. Gradient Descent in Linear Regression

Gradient descent updates the slope and intercept to reduce error:

\[
m := m - \alpha \frac{\partial}{\partial m} MSE
\]
\[
b := b - \alpha \frac{\partial}{\partial b} MSE
\]

Where:  
- \(m\) → slope  
- \(b\) → intercept  
- \(\alpha\) → learning rate  

<img src="https://i.sstatic.net/WTTor.jpg" width="500">

---

## 5. Assumptions of Linear Regression

Linear regression relies on several key assumptions:

- **Linearity** → relationship between X and y is linear  
- **Homoscedasticity** → constant variance of residuals  
- **Independence** → errors are not correlated  
- **Normality of residuals**  

<img src="https://www.qualtrics.com/m/assets/support/wp-content/uploads/2017/07/Screen-Shot-2017-07-19-at-9.46.11-AM.png" width="500">

---

## 6. Data Preprocessing for Linear Regression

Although simple, Linear Regression requires clean data.

### Recommended steps:
- Handle missing values  
- Remove multicollinearity  
- Normalize/standardize features (optional)  
- Detect outliers  


---

## 7. Model Training in Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creating the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
