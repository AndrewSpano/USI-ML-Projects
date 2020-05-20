# for creating the model
from tensorflow.keras import utils, Sequential, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# for the plotting
import matplotlib.pyplot as plt

# get the load_data() and save_keras_model functions
from utils import load_cifar10, save_keras_model



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

''' ------ THIS PART IS WITH NORMAL ReLU --------
model.add(Conv2D(8, (5, 5), strides = (1, 1), activation = 'relu', input_shape = (32, 32, 3)))
'''
model.add(Conv2D(8, (5, 5), strides = (1, 1), activation = 'linear', input_shape = (32, 32, 3)))
# LeakyReLU activation has to be added as an extra layer, according to https://github.com/tensorflow/tensorflow/issues/27142
model.add(LeakyReLU(alpha = 0.15))

model.add(MaxPooling2D(pool_size = (2, 2)))

''' ------ THIS PART IS WITH NORMAL ReLU --------
model.add(Conv2D(16, (3, 3), strides = (2, 2), activation = 'relu', input_shape = (32, 32, 3)))
'''
model.add(Conv2D(16, (3, 3), strides = (2, 2), activation = 'linear', input_shape = (32, 32, 3)))
# LeakyReLU activation has to be added as an extra layer, according to https://github.com/tensorflow/tensorflow/issues/27142
model.add(LeakyReLU(alpha = 0.15))


model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(8, activation = 'tanh', kernel_regularizer = l2(0.005)))
model.add(Dropout(0.3))
model.add(Dense(n_classes, activation = 'softmax', kernel_regularizer = l2(0.005)))


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
epochs = 500
history = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    callbacks = [my_callback],
                    validation_split = 0.2)


# Plotting
plt.figure(figsize = (10, 5))
plt.subplot(121)
# Plot training & validation accuracy values
plt.plot(history.history['acc'], label = 'Train_Accuracy')
plt.plot(history.history['val_acc'], label = 'Val_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Evaluate model
scores = model.evaluate(x_test, y_test)
print('Test loss: {} - Accuracy: {}'.format(*scores))

# save the model
save_keras_model(model, "../deliverable/nn_task1.h5")