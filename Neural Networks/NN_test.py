import pandas as pd
import numpy as np
from NN3Layer import ThreeLayerNN
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

def test_accuracy(model, X_test, y_test):
    i = 0
    for idx, xi in enumerate(X_test):
        if model.predict(xi) != y_test[idx]:
            i+=1
    return i / len(y_test)

if __name__ == '__main__':
    attributes = ['variance','skewness','curtosis','entropy','genuine']

    df_train = pd.read_csv('./data/bank-note/train.csv', names=attributes)
    df_test = pd.read_csv('./data/bank-note/test.csv', names=attributes)

    df_train['genuine'].iloc[df_train['genuine'] == 0] = -1
    df_test['genuine'].iloc[df_test['genuine'] == 0] = -1

    X_train = df_train.loc[:, df_train.columns != 'genuine'].to_numpy(dtype='float64')
    y_train = df_train['genuine'].to_numpy(dtype='float64')

    X_test = df_test.loc[:, df_test.columns != 'genuine'].to_numpy(dtype='float64')
    y_test = df_test['genuine'].to_numpy(dtype='float64')

    # Calculate gradients for one example for problem 2a
    NN = ThreeLayerNN(X_train.shape[1], 5, X_train, y_train)
    sample_example = X_train[0]
    y_hat = NN.forward(sample_example)
    dW1, dW2, dW3 = NN.backpropogate(y_train[0], y_hat, 0.1)

    print('Results for problem 2a:')
    print('Test example:')
    print(X_train[0])
    print('First layer edge weight gradients:')
    print(dW1)
    print('Second layer edge weight gradients:')
    print(dW2)
    print('Third layer edge weight gradients:')
    print(dW3)

    #Run stochastic gradient descent with the NN using Gaussian inititialization
    widths = [5, 10, 25, 50, 100]
    models = []
    for width in widths:
        NN = ThreeLayerNN(X_train.shape[1], width, X_train, y_train)
        epochs = 100
        lr = 1 / width
        NN.train(epochs, lr, 2)
        models.append(NN)
        print('Training error of NN with a width of %d : %f' % (width, test_accuracy(NN, X_train, y_train)))
        print('Test error of NN with a width of %d : %f' % (width, test_accuracy(NN, X_test, y_test)))
        print('')

    #Run stochastic gradient descent with the NN using Gaussian inititialization
    widths = [5, 10, 25, 50, 100]
    models = []
    for width in widths:
        NN = ThreeLayerNN(X_train.shape[1], width, X_train, y_train, True)
        epochs = width * 2
        lr = 1 / (width * 10)
        NN.train(epochs, lr, 2)
        models.append(NN)
        print('Training error of NN with a width of %d : %f' % (width, test_accuracy(NN, X_train, y_train)))
        print('Test error of NN with a width of %d : %f' % (width, test_accuracy(NN, X_test, y_test)))
        print('')