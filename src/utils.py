from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.datasets import cifar10

def load_cifar10(num_classes=3):
    """
    Downloads CIFAR-10 dataset, which already contains a training and test set,
    and return the first `num_classes` classes.
    Example of usage:
    >>> (x_train, y_train), (x_test, y_test) = load_cifat10()
    :param num_classes: int, default is 3 as required by the assignment.
    :return: the filtered data.
    """
    (x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()

    fil_train = y_train_all[:, 0] < num_classes
    fil_test = y_test_all[:, 0] < num_classes

    y_train = y_train_all[fil_train]
    y_test = y_test_all[fil_test]

    x_train = x_train_all[fil_train]
    x_test = x_test_all[fil_test]

    return (x_train, y_train), (x_test, y_test)

def save_keras_model(model, filename):
    """
    Saves a Keras model to disk.
    Example of usage:
    >>> model = Sequential()
    >>> model.add(Dense(...))
    >>> model.compile(...)
    >>> model.fit(...)
    >>> save_keras_model(model, 'my_model.h5')
    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    save_model(model, filename)

def load_keras_model(filename):
    """
    Loads a compiled Keras model saved with models.save_model.
    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = models.load_model(filename)
    return model