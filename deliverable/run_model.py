from tensorflow.keras.models import load_model
from tensorflow.keras import utils

import sys
sys.path.insert(1, '../src')

# get the load_data() function
from utils import load_cifar10

if __name__ == '__main__':

    # Load the test CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    # ...


    # Preprocessing
    # Normalize to 0-1 range
    x_train = x_train / 255.
    x_test = x_test / 255.
    # Pre-process targets
    n_classes = 3
    y_train = utils.to_categorical(y_train, n_classes)
    y_test = utils.to_categorical(y_test, n_classes)
    # ...


    # Load the trained models
    #for example
    model_task1 = load_model('./nn_task1.h5')
    model_task2 = load_model('./nn_task2.h5')


    # Predict on the given samples
    #for example
    y_pred_task1 = model_task1.predict(x_test)
    y_pred_task2 = model_task2.predict(x_test)


    # Evaluate the missclassification error on the test set
    #for example
    assert y_test.shape == y_pred_task1.shape
    assert y_test.shape == y_pred_task2.shape
    acc1 = (y_test == y_pred_task1).mean()
    acc2 = (y_test == y_pred_task2).mean()
    print("Accuracy model task 1:", acc1)
    print("Accuracy model task 2:", acc2)