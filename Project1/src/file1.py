import numpy as np
from math import sin
from utils import load_data, save_sklearn_model, evaluate_predictions
from sklearn.linear_model import LinearRegression


# Load the data
data_path = '../data/data.npz'
x, y = load_data(data_path)

# Initialize the regressor
regressor = LinearRegression(fit_intercept = True)

# Number of data points
n = len(x)

# Initialize array
x3 = np.empty((0, n))

# Append x3
for x1, x2 in x:
    temp = sin(x1) * x2
    x3 = np.append(x3, temp)


# Build the X matrix
X = np.insert(x, 2, x3, axis = 1)

# Fit the model
regressor.fit(X, y)

# Get the parameters
theta0 = regressor.intercept_
theta1, theta2, theta3 = regressor.coef_
print("\nThe parameter values are: theta0 = {}, theta1 = {}, theta2 = {}, theta3 {}.".format(theta0, theta1, theta2, theta3))

# Make the predictions of the model
y_pred = regressor.predict(X)

# Print the prediction
MSE = evaluate_predictions(y_pred, y)
print("Task 1 Linear Rregression Model MSE: {}\n".format(MSE))


# Save the model
save_sklearn_model(regressor, '../deliverable/linear_regression.pickle')
