# For the models
from tensorflow.keras.models import load_model
from tensorflow.keras import utils

# For some calculation at the end
import numpy as np

# To uses the read function
import sys
sys.path.insert(1, '../src')

# Get the load_data() function
from utils import load_cifar10



if __name__ == '__main__':

    # Load the test CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10()


    # Preprocessing
    # Normalize to 0-1 range
    x_train = x_train / 255.
    x_test = x_test / 255.
    # Pre-process targets
    n_classes = 3
    y_train = utils.to_categorical(y_train, n_classes)
    y_test = utils.to_categorical(y_test, n_classes)


    # Load the trained models
    model_task1 = load_model('./nn_task1.h5')
    model_task2 = load_model('./nn_task2.h5')




    # Print information about the two models that we are going to compare
    print("\n\nThe summary of the model from Task 1 is:")
    model_task1.summary()
    print("\n\nThe summary of the best model found from the grid search (Task 2) is:")
    model_task2.summary()


    # ---------------------------------------------------------- START COMPARISON ----------------------------------------------------------

    # Explain the procedure (this part has been copy-pasted from ../src/file2.py)
    print("\n\n")
    print("Let e_1, e_2 denote denote the Classification Accuracies in the test set of the models from Task 1 and Task 2 respectively")
    print("First we must see if the Null Hypothesis holds: H_0: E[e_1] = E[e_2]")
    print("We reject H_0 if T (formula in the slides \"4_Model_performance\" page 21, on iCorsi) is outside the 95% confidence interval: (-1.96, 1.96)")
    print("If it doesn't hold, then the model with the highest accuracy is preferable\n")


    # Model from Task 1

    # Make the prediction
    y1_pred = np.argmax(model_task1.predict(x_test), 1)
    # Calculate the accuracy for each instance
    e_1 = (np.argmax(y_test, 1) == y1_pred).astype(int)
    # Calculate the mean accuracy
    mean_e_1 = e_1.mean()
    # Calculate the variance squared
    s2_1 = mean_e_1 * (1 - mean_e_1)



    # Model from Task 2

    # Make the prediction
    y2_pred = np.argmax(model_task2.predict(x_test), 1)
    # Calculate the accuracy for each instance
    e_2 = (np.argmax(y_test, 1) == y2_pred).astype(int)
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
        print("Therefore, the model with the highest accuracy is preferable, which is:\n")
        if mean_e_1 > mean_e_2:
            print("Model from Task 1")
        else:
            print("Model from Task 2")
        print("And hence, the accuracy levels can be considered as stastically different\n")
