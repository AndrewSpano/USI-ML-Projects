# For creating the model
from tensorflow.keras import utils, Sequential, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


# Get the load_data() and save_keras_model functions
from utils import load_cifar10, save_keras_model, plot_history



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
        # get only the accuracy
        scores[(learning_rate, neuron_number)] = score[1]
        print('Test loss: {} - Accuracy: {}'.format(*score))

max_configuration = max(scores, key = scores.get)
print(max_configuration)
print(scores[max_configuration])

