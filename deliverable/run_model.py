from tensorflow.keras.models import load_model

if __name__ == '__main__':

    # Load the test CIFAR-10 data

    # ...


    # Preprocessing

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
    mce1 = (y_test != y_pred_task1).mean()
    mce2 = (y_test != y_pred_task2).mean()
    print("MCE model task 1:", mse1)
    print("MCE model task 2:", mse2)
