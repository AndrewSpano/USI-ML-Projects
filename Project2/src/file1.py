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


# Build model
model = Sequential()

''' # ------ THIS PART IS WITH NORMAL ReLU --------
model.add(Conv2D(8, (5, 5), strides = (1, 1), activation = 'relu', input_shape = (32, 32, 3)))
'''
# Add the first convolutional layer with 8 filters of size 5x5, stride of 1x1 and LeakeReLU activation (for bonus)
model.add(Conv2D(8, (5, 5), strides = (1, 1), activation = 'linear', input_shape = (32, 32, 3)))
# LeakyReLU activation has to be added as an extra layer, according to https://github.com/tensorflow/tensorflow/issues/27142
model.add(LeakyReLU(alpha = 0.15))

# Add a Max pooling layer of size 2x2
model.add(MaxPooling2D(pool_size = (2, 2)))

''' # ------ THIS PART IS WITH NORMAL ReLU --------
model.add(Conv2D(16, (3, 3), strides = (2, 2), activation = 'relu', input_shape = (32, 32, 3)))
'''
# Add the second convolutional layer with 16 filters of size 3x3, stride of 2x2 and LeakeReLU activation (for bonus)
model.add(Conv2D(16, (3, 3), strides = (2, 2), activation = 'linear', input_shape = (32, 32, 3)))
# LeakyReLU activation has to be added as an extra layer, according to https://github.com/tensorflow/tensorflow/issues/27142
model.add(LeakyReLU(alpha = 0.15))

# Add an Average pooling layer of size 2x2
model.add(AveragePooling2D(pool_size = (2, 2)))

# Add a layer to convert the 2D feature maps to flat vectors
model.add(Flatten())

# Add a dropout layer with dropout probability of 0.3 before each Dense layer (bonus part)
model.add(Dropout(0.3))

# Add a dense layer with 8 neurons and tanh activation. Also add L2 regularization with factor 0.005 (bonus part)
model.add(Dense(8, activation = 'tanh', kernel_regularizer = l2(0.005)))

# Add a dropout layer with dropout probability of 0.3 before each Dense layer (bonus part)
model.add(Dropout(0.3))

# Add a dense output layer with softmax activation. Also add L2 regularization with factor 0.005 (bonus part)
model.add(Dense(n_classes, activation = 'softmax', kernel_regularizer = l2(0.005)))


# Compile the model
model.compile(optimizer = optimizers.RMSprop(lr = 0.003),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'],
                )
model.summary()

# Implementation of early stopping
my_callback = EarlyStopping(monitor = 'val_accuracy', patience = 10, restore_best_weights = True)

# Train model
batch_size = 128
epochs = 500
history = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    callbacks = [my_callback],
                    validation_split = 0.2)


# Plotting
plot_history(history)

# Evaluate model
scores = model.evaluate(x_test, y_test)
print('Test loss: {} - Accuracy: {}'.format(*scores))

# Save the model
save_keras_model(model, "../deliverable/nn_task1.h5")
