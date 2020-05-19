from tensorflow.keras.models import load_model
from tensorflow.keras import utils, Sequential, optimizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# get the load_data() function
from utils import load_cifar10


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
model.add(Conv2D(8, (5, 5), strides = (1, 1), activation = 'relu', input_shape = (32, 32, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(16, (3, 3), strides = (2, 2), activation = 'relu', input_shape = (32, 32, 3)))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(8, activation = 'tanh'))
model.add(Dense(n_classes, activation = 'softmax'))


# Compile the model
model.compile(optimizer = optimizers.RMSprop(lr = 0.003),
                   loss = 'categorical_crossentropy',                   
                   metrics = ['accuracy'],
                  )
model.summary()

# implementation of early stopping
my_callback = EarlyStopping(monitor = 'val_acc', patience = 10, restore_best_weights = True)

# Train model
batch_size = 128
epochs = 30
model.fit(x_train,
          y_train,
          batch_size = batch_size,
          epochs = epochs,
          callbacks = [my_callback],
          validation_split = 0.2)

# Evaluate model
scores = model.evaluate(x_test, y_test)
print('Test loss: {} - Accuracy: {}'.format(*scores))