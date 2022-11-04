import numpy as np
import pandas as pd
from Perceptron import Perceptron

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

    perceptron_standard = Perceptron(X_train, y_train)
    perceptron_standard.train_standard(1)

    error_standard = test_accuracy(perceptron_standard, X_test, y_test)
    print('Average test error of standard Perceptron: %f' % error_standard)
    print('Standard Perceptron weight vector')
    print(perceptron_standard.w)
    print('')

    perceptron_voted = Perceptron(X_train, y_train)
    perceptron_voted.train_voted(1)

    error_voted = test_accuracy(perceptron_voted, X_test, y_test)
    print('Average test error of voted Perceptron: %f' % error_voted)
    print('Voted Perceptron weight vectors and votes (first and last 5 vectors):')
    print(perceptron_voted.w_arr[0:5])
    print('...')
    print(perceptron_voted.w_arr[-5:])
    print('')

    perceptron_avg = Perceptron(X_train, y_train)
    perceptron_avg.train_average(1)

    error_avg = test_accuracy(perceptron_avg, X_test, y_test)
    print('Average test error of average Perceptron: %f' % error_avg)
    print('Average Perceptron weight vector')
    print(perceptron_avg.a)
    print('')
