import numpy as np
import pandas as pd
from SVM import SVMClassifier

def test_accuracy(model, X_test, y_test, gamma=None):
    i = 0
    for idx, xi in enumerate(X_test):
        if gamma is None:
            if model.predict(xi) != y_test[idx]:
                i+=1
        else:
            if model.predict(xi, gamma) != y_test[idx]:
                i+=1
    return i / len(y_test)

def find_same_supports(arr1, arr2):
    ctr = 0
    for i, val in enumerate(arr1):
        if val and arr2[i]:
            ctr += 1
    return ctr

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

    C = [(100/873), (500/873), (700/873)]
    alpha = [0.1, 0.5, 1, 2, 5, 10]
    lr0 = [0.1, 0.5, 1, 2, 5, 10]

    classifier = SVMClassifier(df_train)

    #The following code finds the alpha and l0 parameters, takes a while to run
    # best_error = 100
    # best_params = {}
    # for a in alpha:
    #     for l in lr0:
    #         errors = []
    #         for i in range(10):
    #             classifier.train_primal_ssgd(C[0], l, a)
    #             errors.append(test_accuracy(classifier, X_test, y_test))
    #         mean_err = sum(errors) / len(errors)
    #         if mean_err < best_error:
    #             best_error = mean_err
    #             best_params['alpha'] = a
    #             best_params['lr0'] = l
    # best_error = 100
    # for l in lr0:
    #     errors = []
    #     for i in range(10):
    #         classifier.train_primal_ssgd(C[0], l, alpha=None)
    #         errors.append(test_accuracy(classifier, X_test, y_test))
    #     mean_err = sum(errors) / len(errors)
    #     if mean_err < best_error:
    #         best_error = mean_err
    #         best_param = l

    best_params = {'alpha': 0.1, 'lr0': 10}
    best_param = 0.1
    C_str = ['100/873', '500/873', '700/873']

    #Test SVM classifiers trained by minimizing primal form
    #Test using first learning rate scheduler
    for i in range(3):
        classifier.train_primal_ssgd(C[i], 10, alpha=0.1)
        print('Train Error when C = %s: %f' % (C_str[i], test_accuracy(classifier, X_train, y_train)))
        print('Test Error when C = %s: %f' % (C_str[i], test_accuracy(classifier, X_test, y_test)))
        print('Weights:')
        print(classifier.w)
    #Test using second learning rate scheduler
    for i in range(3):
        classifier.train_primal_ssgd(C[i], 0.1, alpha=None)
        print('Train Error when C = %s: %f' % (C_str[i], test_accuracy(classifier, X_train, y_train)))
        print('Test Error when C = %s: %f' % (C_str[i], test_accuracy(classifier, X_test, y_test)))
        print('Weights:')
        print(classifier.w)

    #Test SVM classifiers trained by minimizing dual form
    print('Training SVM Classifier by minimizing dual form with C = 100/873 ~45 seconds')
    classifier_0 = SVMClassifier(df_train)
    classifier_0.train_dual(C[0])
    print('Weights:')
    print(classifier_0.w0)
    print('Bias:')
    print(classifier_0.b)
    print('Train Error when C = 100/873: %f' % (test_accuracy(classifier_0, X_train, y_train)))
    print('Test Error when C = 100/873: %f' % (test_accuracy(classifier_0, X_test, y_test)))

    print('Training SVM Classifier by minimizing dual form with C = 500/873 ~45 seconds')
    classifier_1 = SVMClassifier(df_train)
    classifier_1.train_dual(C[1])
    print('Weights:')
    print(classifier_1.w0)
    print('Bias:')
    print(classifier_1.b)
    print('Train Error when C = 500/873: %f' % (test_accuracy(classifier_1, X_train, y_train)))
    print('Test Error when C = 500/873: %f' % (test_accuracy(classifier_1, X_test, y_test)))

    print('Training SVM Classifier by minimizing dual form with C = 700/873 ~45 seconds')
    classifier_2 = SVMClassifier(df_train)
    classifier_2.train_dual(C[2])
    print('Weights:')
    print(classifier_2.w0)
    print('Bias:')
    print(classifier_2.b)
    print('Train Error when C = 700/873: %f' % (test_accuracy(classifier_2, X_train, y_train)))
    print('Test Error when C = 700/873: %f' % (test_accuracy(classifier_2, X_test, y_test)))

    #Test SVM using the kernel trick
    gammas = [0.1, 0.5, 1, 5, 100]
    for c in C:
        for gamma in gammas:
            classifier_kernel = SVMClassifier(df_train)
            classifier_kernel.train_dual(c, gamma)
            test_err = test_accuracy(classifier_kernel, X_test, y_test)
            train_err = test_accuracy(classifier_kernel, X_train, y_train)
            print('Training error where C = %f and gammma = %f : %f' % (c, gamma, train_err))   
            print('Testing error where C = %f and gammma = %f : %f' % (c, gamma, test_err))
            print('Number of support vectors %d' % (len(classifier_kernel.nonzero_alphas)))
            
    #Testing for shares support vectors with different values of gamma
    classifier_kernel1 = SVMClassifier(df_train)
    classifier_kernel1.train_dual(C[1], 0.01)
    truth_arr1 = classifier_kernel1.alpha > 0.001

    classifier_kernel2 = SVMClassifier(df_train)
    classifier_kernel2.train_dual(C[1], 0.1)
    truth_arr2 = classifier_kernel2.alpha > 0.001

    classifier_kernel3 = SVMClassifier(df_train)
    classifier_kernel3.train_dual(C[1], 0.5)
    truth_arr3 = classifier_kernel3.alpha > 0.001

    classifier_kernel4 = SVMClassifier(df_train)
    classifier_kernel4.train_dual(C[1], 1)
    truth_arr4 = classifier_kernel4.alpha > 0.001

    classifier_kernel5 = SVMClassifier(df_train)
    classifier_kernel5.train_dual(C[1], 5)
    truth_arr5 = classifier_kernel5.alpha > 0.001

    classifier_kernel6 = SVMClassifier(df_train)
    classifier_kernel6.train_dual(C[1], 100)
    truth_arr6 = classifier_kernel6.alpha > 0.001

    print('Support vectors shared when gamma = 0.01 and when gamma = 0.1: %d' % find_same_supports(truth_arr1, truth_arr2))
    print('Support vectors shared when gamma = 0.1 and when gamma = 0.5: %d' % find_same_supports(truth_arr2, truth_arr3))
    print('Support vectors shared when gamma = 0.5 and when gamma = 1: %d' % find_same_supports(truth_arr3, truth_arr4))
    print('Support vectors shared when gamma = 1 and when gamma = 5: %d' % find_same_supports(truth_arr4, truth_arr5))
    print('Support vectors shared when gamma = 5 and when gamma = 100: %d' % find_same_supports(truth_arr5, truth_arr6))

