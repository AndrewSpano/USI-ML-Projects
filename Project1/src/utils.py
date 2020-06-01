import joblib
from tensorflow.keras import models
import numpy as np


def save_sklearn_model(model, filename):
    """
    Saves a Scikit-learn model to disk.
    Example of usage:
    
    >>> reg = sklearn.linear_models.LinearRegression()
    >>> reg.fit(x_train, y_train)
    >>> save_sklearn_model(reg, 'my_model.pickle')    

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    joblib.dump(model, filename)


def load_sklearn_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


def save_keras_model(model, filename):
    """
    Saves a Keras model to disk.
    Example of usage:

    >>> model = Sequential()
    >>> model.add(Dense(...))
    >>> model.compile(...)
    >>> model.fit(...)
    >>> save_keras_model(model, 'my_model')

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    models.save_model(model, filename)


def load_keras_model(filename):
    """
    Loads a compiled Keras model saved with models.save_model.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = models.load_model(filename)

    return model




def load_data(filename):
    """
    Loads the data from a saved .npz file.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param filename: string, path to the .npz file storing the data.
    :return: two numpy arrays:
        - x, a Numpy array of shape (n_samples, n_features) with the inputs;
        - y, a Numpy array of shape (n_samples, ) with the targets.
    """
    data = np.load(filename)
    x = data['x']
    y = data['y']

    return x, y



def evaluate_predictions(y_true, y_pred):
    """
    Evaluates the mean squared error between the values in y_true and the values
    in y_pred.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param y_true: Numpy array, the true target values from the test set;
    :param y_pred: Numpy array, the values predicted by your model.
    :return: float, the the mean squared error between the two arrays.
    """
    assert y_true.shape == y_pred.shape
    return ((y_true - y_pred) ** 2).mean()

