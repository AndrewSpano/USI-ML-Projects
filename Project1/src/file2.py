# import useful modules
import numpy as np
from math import sin

# import useful keras modules
from tensorflow.keras import Sequential, optimizers, utils, models, layers, metrics
# import module to help with cross-validation
from sklearn.model_selection import train_test_split
# get the load_data() function
from utils import load_data, evaluate_predictions, save_keras_model, load_sklearn_model, calculate_variance



# load the data from the data file using it's relative path
x, y = load_data("../data/data.npz")

# get the training set, the validation set and the test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Build the regressor
regressor = models.Sequential()
# Add dense layers which use the ReLU activation function
regressor.add(layers.Dense(units = 64, activation = 'relu', input_shape = x_train.shape[1:]))
regressor.add(layers.Dense(units = 64, activation = 'relu'))
# Add the output layer
regressor.add(layers.Dense(units = 1))

# Compile the model
regressor.compile(loss = 'mse', optimizer = optimizers.RMSprop(lr = 0.05), )

# Train the network
regressor.fit(x_train, y_train, epochs = 200, verbose = 0, batch_size = 32, validation_split = 0.2)


# Evaluate the results
evaluate_results_train = regressor.evaluate(x_train, y_train, verbose = 0)
evaluate_results = regressor.evaluate(x_test, y_test, verbose = 0)

# Print the Errors
print('\nError in Training: {} ', evaluate_results_train)
print('Error in Test: {} ', evaluate_results)


# Save the model in a picke file
save_keras_model(regressor, '../deliverable/Neural_Network.pickle')

# For the comparison, load the Linear Regressor from Task 1
linear_regressor = load_sklearn_model("../deliverable/Linear_Regression.pickle")



# ---------------------------------------------------------- START COMPARISON ----------------------------------------------------------

# Explain the procedure
print("\n")
print("Let e_lr, e_nn denote denote the accuracies in the test set of the models from Task 1 and Task 2 respectively.")
print("First we must see if the Null Hypothesis holds: H_0: E[e_1] = E[e_2].")
print("We reject H_0 if the T student statistic is outside the 95% confidence interval: (-1.96, 1.96).")
print("If it doesn't hold, then the model with the smaller variable is preferable.\n")


# Model from Task 1

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

# Make the predictions of the model
y1_pred = linear_regressor.predict(X)
MSE = evaluate_predictions(y1_pred, y)
e_lr = MSE * n

# calculate variance squared
s_lr = calculate_variance(y, y1_pred, e_lr)



# Model from Task 2

# Make the prediction
y2_pred = regressor.predict(x_test)
y2_pred = y2_pred.reshape(y_test.shape)
MSE = evaluate_predictions(y2_pred, y_test)
e_nn = MSE * len(x_test)

# calculate variance squared
s_nn = calculate_variance(y_test, y2_pred, e_nn)




# Test statistics: T
l_lr = len(x)
l_nn = len(x_test)
T = (e_lr - e_nn) / np.sqrt(s_lr / l_lr + s_nn / l_nn)



# Print statistics and finish
print("Task 1 model: Mean: {} -- Variance Squared: {}".format(e_lr, s_lr))
print("Task 2 model: Mean: {} -- Variance Squared: {}".format(e_nn, s_nn))
print("\nT = {}\n".format(T))

# Decide Null Hypothesis
if T > -1.96 and T < 1.96:
    print("Null Hypothesis H_0 accepted because: -1.96 < T = {} < 1.96".format(T))
    print("Therefore, the accuracy levels can be considered as stastically similar\n")
else:
    print("Null Hypothesis H_0 rejected because T = {} is not in the 95% confidence interval: (-1.96, 1.96)")
    print("Therefore, the model with the smallest variance is statistically better, which is:\n")
    if s_lr < s_nn:
        print("Model from Task 1")
    else:
        print("Model from Task 2")
    print("And hence, the accuracy levels can be considered as statistically different.\n")