import joblib
import numpy as np
from math import sin
from sklearn.model_selection import train_test_split

# used to load linear regression model
import sys
sys.path.insert(1, "../src")
from utils import load_data, load_sklearn_model,load_keras_model, evaluate_predictions, calculate_variance


def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


if __name__ == '__main__':
    # Load the data
    # This will be replaced with the test data when grading the assignment
    data_path = '../data/data.npz'
    x, y = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


    ############################################################################
    # EDITABLE SECTION OF THE SCRIPT: if you need to edit the script, do it here
    ############################################################################

    # Load the Linear Regression model
    linear_regressor = load_sklearn_model("./Linear_Regression.pickle")
    # Load the Neural Network model
    regressor = load_keras_model("./Neural_Network.pickle")


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

    # Print statistics and finish
    print("\nTask 1 model: Mean: {} -- Variance Squared: {}".format(e_lr, s_lr))
    print("Task 2 model: Mean: {} -- Variance Squared: {}".format(e_nn, s_nn))


    ############################################################################
    # STOP EDITABLE SECTION: do not modify anything below this point.
    ############################################################################

    # Load the trained model
    baseline_model_path = './baseline_model.pickle'
    baseline_model = load_model(baseline_model_path)

    # Predict on the given samples
    y_pred = baseline_model.predict(x)

    # Evaluate the prediction using MSE
    mse = evaluate_predictions(y_pred, y)
    print('MSE for baseline model: {}\n'.format(mse))
