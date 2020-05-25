# For creating the model
from tensorflow.keras import utils, Sequential, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# For some calculation at the end
import numpy as np

# Get the load_data() and save_keras_model functions
from utils import load_cifar10, save_keras_model, load_keras_model, plot_history



# Load the test CIFAR-10 data
(x_train, y_train), (x_test, y_test) = load_cifar10()

# Normalize to 0-1 range
x_train = x_train / 255.
x_test = x_test / 255.
# Pre-process targets
n_classes = 3
y_train = utils.to_categorical(y_train, n_classes)
y_test = utils.to_categorical(y_test, n_classes)


# Create a dictionary that will contain all the 2x2 = 4 models we create, and another one for their scores
neural_networks = {}
scores = {}

# Perform a grid search for the following hyper-parameters
learning_rates = [0.0001, 0.01]
neurons = [8, 64]


# Implementation of early stopping
my_callback = EarlyStopping(monitor = 'val_acc', patience = 10, restore_best_weights = True)


# For each possible combination
for learning_rate in learning_rates:
    for neuron_number in neurons:

        # Build the model
        neural_networks[(learning_rate, neuron_number)] = Sequential()

        ''' # ------ THIS PART IS WITH NORMAL ReLU --------
        neural_networks[(learning_rate, neuron_number)].add(Conv2D(8, (5, 5), strides = (1, 1), activation = 'relu', input_shape = (32, 32, 3)))
        '''
        # Add the first convolutional layer with 8 filters of size 5x5, stride of 1x1 and LeakeReLU activation (for bonus)
        neural_networks[(learning_rate, neuron_number)].add(Conv2D(8, (5, 5), strides = (1, 1), activation = 'linear', input_shape = (32, 32, 3)))
        # LeakyReLU activation has to be added as an extra layer, according to https://github.com/tensorflow/tensorflow/issues/27142
        neural_networks[(learning_rate, neuron_number)].add(LeakyReLU(alpha = 0.15))

        # Add a Max pooling layer of size 2x2
        neural_networks[(learning_rate, neuron_number)].add(MaxPooling2D(pool_size = (2, 2)))

        ''' # ------ THIS PART IS WITH NORMAL ReLU --------
        neural_networks[(learning_rate, neuron_number)].add(Conv2D(16, (3, 3), strides = (2, 2), activation = 'relu', input_shape = (32, 32, 3)))
        '''
        # Add the second convolutional layer with 16 filters of size 3x3, stride of 2x2 and LeakeReLU activation (for bonus)
        neural_networks[(learning_rate, neuron_number)].add(Conv2D(16, (3, 3), strides = (2, 2), activation = 'linear', input_shape = (32, 32, 3)))
        # LeakyReLU activation has to be added as an extra layer, according to https://github.com/tensorflow/tensorflow/issues/27142
        neural_networks[(learning_rate, neuron_number)].add(LeakyReLU(alpha = 0.15))

        # Add an Average pooling layer of size 2x2
        neural_networks[(learning_rate, neuron_number)].add(AveragePooling2D(pool_size = (2, 2)))

        # Add a layer to convert the 2D feature maps to flat vectors
        neural_networks[(learning_rate, neuron_number)].add(Flatten())

        # Add a dropout layer with dropout probability of 0.3 before each Dense layer (bonus part)
        neural_networks[(learning_rate, neuron_number)].add(Dropout(0.3))

        # Add a dense layer with (neuron_number) neurons and tanh activation. Also add L2 regularization with factor 0.005 (bonus part)
        neural_networks[(learning_rate, neuron_number)].add(Dense(neuron_number, activation = 'tanh', kernel_regularizer = l2(0.005)))

        # Add a dropout layer with dropout probability of 0.3 before each Dense layer (bonus part)
        neural_networks[(learning_rate, neuron_number)].add(Dropout(0.3))

        # Add a dense output layer with softmax activation. Also add L2 regularization with factor 0.005 (bonus part)
        neural_networks[(learning_rate, neuron_number)].add(Dense(n_classes, activation = 'softmax', kernel_regularizer = l2(0.005)))

        # Compile the model with the according learning rate
        neural_networks[(learning_rate, neuron_number)].compile(optimizer = optimizers.RMSprop(lr = learning_rate),
                                                                                                loss = 'categorical_crossentropy',                   
                                                                                                metrics = ['accuracy'],
                                                                                                )
        neural_networks[(learning_rate, neuron_number)].summary()

        # Train the model
        batch_size = 128
        epochs = 500
        history = neural_networks[(learning_rate, neuron_number)].fit(x_train,
                                                                        y_train,
                                                                        batch_size = batch_size,
                                                                        epochs = epochs,
                                                                        callbacks = [my_callback],
                                                                        validation_split = 0.2)

        # Evaluate the model
        score = neural_networks[(learning_rate, neuron_number)].evaluate(x_test, y_test)
        # Get only the accuracy
        scores[(learning_rate, neuron_number)] = score[1]
        print('Test loss: {} - Accuracy: {}'.format(*score))

# See which configuration (model) gives the highest accuracy
max_configuration = max(scores, key = scores.get)
# Pick the corresponding model to be the "most promising" one
best_model = neural_networks[max_configuration]

# Save it
save_keras_model(best_model, "../deliverable/nn_task2.h5")

# Print some information about it for the 3rd bullet of Task 2
print("\n\nThe Hyper-parameters which give the \"most promising\" model are (learning_rate, number_of_neurons) = {}".format(max_configuration))
print("The accuracy of this model is {}".format(scores[max_configuration]))


# Now for the 4th bullet we have to compare our best model with the model from Task 1. We start by loading the model from Task 1
task1_model = load_keras_model("../deliverable/nn_task1.h5")

# Print information about the two models that we are going to compare
print("\n\nThe summary of the model from Task 1 is:")
task1_model.summary()
print("\n\nThe summary of the best model found from the grid search (Task 2) is:")
best_model.summary()


# ---------------------------------------------------------- START COMPARISON ----------------------------------------------------------

# Explain the procedure
print("\n\n")
print("Let e_1, e_2 denote denote the Classification Accuracies in the test set of the models from Task 1 and Task 2 respectively")
print("First we must see if the Null Hypothesis holds: H_0: E[e_1] = E[e_2]")
print("We reject H_0 if T (formula in the slides \"4_Model_performance\" page 21, on iCorsi) is outside the 95% confidence interval: (-1.96, 1.96)")
print("If it doesn't hold, then the model with the smaller variable is preferable\n")


# Model from Task 1

# Make the prediction
y1_pred = (task1_model.predict(x_test) > .5).astype(int)
# Calculate the accuracy for each instance
e_1 = (y_test == y1_pred).astype(int)
# Calculate the mean accuracy
mean_e_1 = e_1.mean()
# Calculate the variance squared
s2_1 = mean_e_1 * (1 - mean_e_1)



# Model from Task 2

# Make the prediction
y2_pred = (best_model.predict(x_test) > .5).astype(int)
# Calculate the accuracy for each instance
e_2 = (y_test == y2_pred).astype(int)
# Calculate the mean accuracy
mean_e_2 = e_2.mean()
# Calculate the variance squared
s2_2 = mean_e_2 * (1 - mean_e_2)




# Test statistics: T = (mean_e_1 - mean_e_2) / sqrt(s2_1/l + s2_2/l), where l is the number of test_points
l = len(x_test)
T = (mean_e_1 - mean_e_2) / np.sqrt(s2_1 / l + s2_2 / l)



# Print statistics and finish
print("Task 1 model: Mean: {} -- Variance Squared: {}".format(mean_e_1, s2_1))
print("Task 2 model: Mean: {} -- Variance Squared: {}".format(mean_e_2, s2_2))
print("\nT = {}\n".format(T))

# Decide Null Hypothesis
if T > -1.96 and T < 1.96:
    print("Null Hypothesis H_0 accepted because: -1.96 < T = {} < 1.96".format(T))
    print("Therefore, the accuracy levels can be considered as stastically similar\n")
else:
    print("Null Hypothesis H_0 rejected because T = {} is not in the 95% confidence interval: (-1.96, 1.96)")
    print("Therefore, the model with the smallest variance is preferable, which is:\n")
    if s2_1 < s2_2:
        print("Model from Task 1")
    else:
        print("Model from Task 2")
    print("And hence, the accuracy levels can be considered as stastically different\n")