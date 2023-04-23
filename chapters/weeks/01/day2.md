# Linear Regression and Gradient Descent

## Learning resources

[Linear regression by StatQuest](https://youtu.be/7ArmBVF2dCs)

[Gradient descent by StatQuest](https://youtu.be/sDv4f4s2SB8)

[Gradient Descent, Step-by-Step by Welch Labs](https://youtu.be/5u0jaA3qAGk)

[Linear Regression - Fun and Easy Machine Learning by Siraj Raval](https://www.youtube.com/live/uwwWVAgJBcM?feature=share)

## Introduction
Linear regression is a widely used statistical method that seeks to model the relationship between a dependent variable (also known as the target or output variable) and one or more independent variables (also known as predictors or input variables). The goal of linear regression is to find the best-fitting line (in simple linear regression) or hyperplane (in multiple linear regression) that describes this relationship. Gradient descent is an optimization algorithm that helps us find the best parameters for the linear regression model.

## Simple Linear Regression
### Equation of a Straight Line
In simple linear regression, we have one dependent variable (y) and one independent variable (x). The relationship between them can be represented as a straight line with the equation: y = mx + b, where m is the slope and b is the y-intercept.

### Least Squares Method
To find the best-fitting line, we use the least squares method, which seeks to minimize the sum of the squared differences between the observed values and the values predicted by the model.

## Multiple Linear Regression
Multiple linear regression extends simple linear regression to cases where we have multiple independent variables. The equation for multiple linear regression is: y = β0 + β1x1 + β2x2 + ... + βnxn + ε, where β0 is the y-intercept, βi are the coefficients for each independent variable, xi are the independent variables, and ε is the error term.

## Gradient Descent
Gradient descent is an iterative optimization algorithm that helps us find the best parameters for our linear regression model by minimizing the cost function.

### Cost Function
The cost function, also known as the loss function or objective function, measures the difference between the predicted values and the actual values. For linear regression, we typically use the mean squared error (MSE) as the cost function.

### Gradient Descent Algorithm
The gradient descent algorithm works by iteratively updating the model parameters (e.g., β0, β1, ... , βn) in the direction of the steepest decrease in the cost function. The algorithm involves the following steps:

1. Initialize the model parameters with random values.
2. Compute the gradient of the cost function with respect to each parameter.
3. Update the parameters by subtracting a fraction of the gradient, called the learning rate, from the current parameter values.
4. Repeat steps 2 and 3 until the cost function converges to a minimum value or a maximum number of iterations is reached.

## Regularization Techniques
Regularization techniques, such as L1 (Lasso) and L2 (Ridge) regularization, can be used to prevent overfitting in linear regression models. These techniques add a penalty term to the cost function, which encourages the model to use fewer or smaller parameters, leading to a simpler and more generalizable model.

## Implementation in Python
In this section, we will demonstrate how to implement linear regression and gradient descent using Python and popular libraries such as NumPy and scikit-learn.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Visualize the data
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Generated Data")
plt.show()

# Add x0 = 1 to each instance
X_b = np.c_[np.ones((100, 1)), X]

# Gradient descent settings
eta = 0.1  # Learning rate
n_iterations = 1000
m = 100  # Number of instances

# Initialize random weights
theta = np.random.randn(2, 1)

# Gradient descent
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print("Theta from gradient descent:", theta)

# Make predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)

# Visualize the fitted line
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression with Gradient Descent")
plt.show()

# Compare with scikit-learn's LinearRegression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("Theta from scikit-learn:", lin_reg.intercept_, lin_reg.coef_)

y_predict_sklearn = lin_reg.predict(X_new)

# Verify if the predictions are the same
assert np.allclose(y_predict, y_predict_sklearn), "Predictions are different"
```

In this example, we first generate sample data with a linear relationship between the input variable X and the output variable y. Then, we visualize the data using matplotlib.

Next, we implement gradient descent to learn the parameters of the linear regression model. We start by initializing the parameters (theta) randomly and set up a learning rate (eta) and a maximum number of iterations. Inside the loop, we calculate the gradients of the cost function with respect to the parameters and update the parameters accordingly.

After the gradient descent algorithm has finished, we print out the learned parameters and use them to make predictions for new data points. We then visualize the fitted line together with the original data.

Finally, we compare the results with scikit-learn's LinearRegression implementation to verify that our gradient descent implementation yields similar results.


## Conclusion
Linear regression is a powerful and widely used method for modeling relationships between variables. Gradient descent is an optimization algorithm that helps us find the best parameters for the linear regression model by minimizing the cost function. By understanding these concepts and their implementation, you can build effective models for a wide range of applications.
