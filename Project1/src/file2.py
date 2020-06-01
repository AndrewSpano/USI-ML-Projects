# import numpy
import numpy as np

# import useful keras modules
from tensorflow.keras import Sequential, optimizers, utils, models, layers, metrics
# import module to help with cross-validation
from sklearn.model_selection import train_test_split
# get the load_data() function
from utils import load_data, evaluate_predictions, save_keras_model



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
print('Error in Training: {} ', evaluate_results_train)
print('Error in Test: {} ', evaluate_results)


# Save the model in a picke file
save_keras_model(regressor, '../deliverable/Neural_Network.pickle')