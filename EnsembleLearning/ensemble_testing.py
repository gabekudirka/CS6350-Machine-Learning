import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from decision_tree import DecisionTree as DecisionTreeAdaBoost
from DecisionTree.decision_tree import DecisionTree as DecisionTreeOG
from AdaBoost import AdaBoostTree
from BaggedTrees import BaggedTrees, RandomForest
import time
import json
import matplotlib.pyplot as plt

def test_tree_accuracy(decision_tree, test_data):
    preds = test_data.apply(lambda row : decision_tree.predict(row), axis=1)
    diff = preds == test_data['label']
    if (diff == True).all():
        return 0
    else:
        error_count = diff.value_counts()[False]
        return error_count / len(test_data)

def process_data(df, attributes, replace_unknown=False, map_labels=True):
    #If specified, replace all 'uknown' values with column majority
    if replace_unknown:
        for attribute in attributes:
            if df[attribute].dtype.kind not in 'iufc':
                most_common = 'unknown'
                counts = df[attribute].value_counts()
                if counts[[0]].index[0] == 'unknown' and len(counts) > 1:
                    most_common = counts[[1]].index[0]
                else:
                    most_common = counts[[0]].index[0]
                df[attribute][df[attribute] == 'unknown'] = most_common
    
    #Replace numerical columns with boolean values based on median threshold
    for attribute in attributes:
        if df[attribute].dtype.kind in 'iufc':
            median = df[attribute].median()
            binary_col = df[attribute] > median
            df[attribute] = binary_col

    if map_labels:
        df.label[df.label == 'yes'] = 1
        df.label[df.label == 'no'] = -1
            
    return df

if __name__ == '__main__':
    attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
    'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    df_train = pd.read_csv('./data/bank/train.csv', names=attributes + ['label'])
    df_test = pd.read_csv('./data/bank/test.csv', names=attributes + ['label'])

    df_train = process_data(df_train, attributes, replace_unknown=False)
    df_test = process_data(df_test, attributes, replace_unknown=False)

    #Test AdaBoost
    training_errors_ab = []
    test_errors_ab = []
    T = [1, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500]
    adaboost_predictors = []

    for t in T:
        adaboost = AdaBoostTree(df_train, attributes)
        adaboost.build_model(t)
        training_errors_ab.append(test_tree_accuracy(adaboost, df_train))
        test_errors_ab.append(test_tree_accuracy(adaboost, df_test))
        adaboost_predictors.append(adaboost)

    print(test_tree_accuracy(adaboost_predictors[-1], df_train))
    print(test_tree_accuracy(adaboost_predictors[-1], df_test)) 

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(8)
    plt.plot(T, training_errors_ab, label='Training Error')
    plt.plot(T, test_errors_ab, label='Testing Error')
    plt.title('Adaboost Test/Training Error by Iteration')
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    training_errors_stumps = []
    testing_errors_stumps = []
    for i, predictor in enumerate(adaboost_predictors):
        training_errors_stumps_iter = []
        testing_errors_stumps_iter = []
        for stump in predictor.classifiers:
            training_errors_stumps_iter.append(test_tree_accuracy(stump, df_train))
            testing_errors_stumps_iter.append(test_tree_accuracy(stump, df_test))
        training_errors_stumps.append(training_errors_stumps_iter)
        testing_errors_stumps.append(testing_errors_stumps_iter)

    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(16)

    for i, iter_errors in enumerate(training_errors_stumps):
        for error in iter_errors:
            plt.scatter(T[i], error, c='r', s=2)
    for i, iter_errors in enumerate(testing_errors_stumps):
        for error in iter_errors:
            plt.scatter(T[i], error,  c='b', s=2)

    plt.title('Stump Errors by Iteration')
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.show()

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(8)

    for i, iter_errors in enumerate(training_errors_stumps):
        mean = sum(iter_errors) / len(iter_errors)
        plt.scatter(T[i], mean, label='Training Error', c='r')
    for i, iter_errors in enumerate(testing_errors_stumps):
        mean = sum(iter_errors) / len(iter_errors)
        plt.scatter(T[i], mean, label='Testing Error', c='b')

    plt.title('Mean Stump Error by Iteration')
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.show()

    #Test Bagged Trees
    bagged_trees = BaggedTrees(df_train, attributes)
    training_errors_bt = []
    test_errors_bt = []
    #T = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    T = [1, 2, 2, 5, 5, 5, 10, 20, 25, 25, 50, 50, 100, 100, 100]

    start = time.time()
    for t in T:
        iter_start = time.time()
        bagged_trees.build_trees(t)
        training_errors_bt.append(test_tree_accuracy(bagged_trees, df_train))
        test_errors_bt.append(test_tree_accuracy(bagged_trees, df_test)) 
        iter_end = time.time()
        print (iter_end - iter_start)
        print(len(bagged_trees.trees))

    end = time.time()
    print( end - start)
    textfile = open("training_errs_bt.txt", "w")
    for element in training_errors_bt:
        textfile.write(str(element) + ", ")
    textfile.close()

    textfile = open("test_errs_bt.txt", "w")
    for element in test_errors_bt:
        textfile.write(str(element) + ", ")
    textfile.close()

    f = plt.figure()
    f.set_figwidth(8)
    f.set_figheight(10)
    plt.plot(T, training_errors_bt, label='Training Error')
    plt.plot(T, test_errors_bt, label='Testing Error')
    plt.title('Bagged Trees Test/Training Error by Iteration')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    #Bagged Trees bias/variance decomposition
    print('Building Trees')
    num_trees = 500
    num_samples = 1000
    bagged_predictors = []
    
    iter_ctr = 0
    start = time.time()
    for i in range(100):
        iter_start = time.time()
        bagged_trees = BaggedTrees(df_train, attributes, df_train)
        bagged_trees.build_trees(num_trees, num_samples)
        bagged_predictors.append(bagged_trees)
        iter_ctr += 1
        iter_end = time.time()
        print(iter_ctr)
        print(iter_end - iter_start)

    end = time.time()
    print(end - start)

    single_trees = [predictor.trees[0] for predictor in bagged_predictors]
    single_tree_biases = []
    single_tree_variances = []
    ctr = 0
    for idx, row in df_test.iterrows():
        preds = []
        for tree in single_trees:
            try:
                pred = tree.predict(row)
            except:
                ctr+=1
                continue
            preds.append(pred)
        preds = np.asarray(preds)
        avg_pred = np.mean(preds)
        bias = (avg_pred - row['label'])**2
        single_tree_biases.append(bias)
        var = np.var(preds)
        single_tree_variances.append(var)

    single_tree_biases = np.asarray(single_tree_biases)
    single_tree_variances = np.asarray(single_tree_variances)

    single_tree_biases = single_tree_biases[~np.isnan(single_tree_biases)]
    single_tree_variances = single_tree_variances[~np.isnan(single_tree_variances)]

    single_tree_bias = sum(single_tree_biases) / len(single_tree_biases)
    single_tree_var = sum(single_tree_variances) / len(single_tree_variances)
    single_tree_squared_err = single_tree_bias + single_tree_var
    print('Errors during runtime: %d' % ctr)

    bagged_tree_biases = []
    bagged_tree_variances = []
    ctr = 0
    for idx, row in df_test.iterrows():
        preds = []
        for predictor in bagged_predictors:
            try:
                pred = predictor.predict(row)
            except:
                ctr+=1
                continue
            preds.append(pred)
        preds = np.asarray(preds)
        avg_pred = np.mean(preds)
        bias = (avg_pred - row['label'])**2
        bagged_tree_biases.append(bias)
        var = np.var(preds)
        bagged_tree_variances.append(var)
    #Just in case an error occurred - ensure viable results
    bagged_tree_biases = np.asarray(bagged_tree_biases)
    bagged_tree_variances = np.asarray(bagged_tree_variances)

    bagged_tree_biases = bagged_tree_biases[~np.isnan(bagged_tree_biases)]
    bagged_tree_variances = bagged_tree_variances[~np.isnan(bagged_tree_variances)]

    bagged_trees_bias = sum(bagged_tree_biases) / len(bagged_tree_biases)
    bagged_trees_var = sum(bagged_tree_variances) / len(bagged_tree_variances)
    bagged_trees_squared_err = bagged_trees_bias + bagged_trees_var
    print('Errors during runtime: %d' % ctr)

    #Print and save results
    print('Single Trees Bias: %f' % single_tree_bias)
    print('Single Trees Variance: %f' % single_tree_var)
    print('Single Trees Estimated Squared Error: %f' % single_tree_squared_err)

    print('Bagged Trees Bias: %f' % bagged_trees_bias)
    print('Bagged Trees Variance: %f' % bagged_trees_var)
    print('Bagged Trees Estimated Squared Error: %f' % bagged_trees_squared_err)

    textfile = open("var_bias_results.txt", "w")
    textfile.write('Single Trees Bias:' + "\n")
    textfile.write(str(single_tree_bias) + "\n")

    textfile.write('Single Trees Variance:' + "\n")
    textfile.write(str(single_tree_var) + "\n")

    textfile.write('Single Trees Estimated Squared Error:' + "\n")
    textfile.write(str(single_tree_squared_err) + "\n")

    textfile.write('Bagged Trees Bias:' + "\n")
    textfile.write(str(bagged_trees_bias) + "\n")

    textfile.write('Bagged Trees Variance:' + "\n")
    textfile.write(str(bagged_trees_var) + "\n")

    textfile.write('Bagged Trees Estimated Squared Error:' + "\n")
    textfile.write(str(bagged_trees_squared_err) + "\n")
    textfile.close()
    

    #Test Random Forest
    training_errors_rf = {}
    test_errors_rf = {}
    subset_sizes = [2, 4, 6]
    T = [1, 2, 2, 5, 5, 5, 10, 20, 25, 25, 50, 50, 100, 100, 100]

    start = time.time()
    for subset_size in subset_sizes:
        random_forest = RandomForest(df_train, attributes)
        training_errors_rf[subset_size] = []
        test_errors_rf[subset_size] = []
        for t in T:
            iter_start = time.time()
            random_forest.build_trees(t, subset_size)
            training_errors_rf[subset_size].append(test_tree_accuracy(random_forest, df_train))
            test_errors_rf[subset_size].append(test_tree_accuracy(random_forest, df_test))
            iter_end = time.time()
            print (iter_end - iter_start)
            print(len(random_forest.trees))
    end = time.time()
    print( end - start)

    with open('training_errs_rf.txt', 'w') as convert_file:
        convert_file.write(json.dumps(training_errors_rf))

    with open('test_errors_rf.txt', 'w') as convert_file:
        convert_file.write(json.dumps(test_errors_rf))

    f = plt.figure()
    f.set_figwidth(8)
    f.set_figheight(10)
    plt.plot(T, training_errors_rf[2], label='Training Error - Subset Size 2')
    plt.plot(T, test_errors_rf[2], label='Testing Error - Subset Size 4')
    plt.plot(T, training_errors_rf[4], label='Training Error - Subset Size 6')
    plt.plot(T, test_errors_rf[4], label='Testing Error - Subset Size 2')
    plt.plot(T, training_errors_rf[6], label='Training Error - Subset Size 4')
    plt.plot(T, test_errors_rf[6], label='Testing Error - Subset Size 6')
    plt.title('Random Forest Test/Training Error by Iteration')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    #Random Forest bias/variance decomposition
    print('Building Trees')
    num_trees = 500
    num_samples = 1000
    rf_predictors = []
    
    iter_ctr = 0
    start = time.time()
    for i in range(100):
        iter_start = time.time()
        random_forest = RandomForest(df_train, attributes, df_train)
        random_forest.build_trees(num_trees, 2, num_samples)
        rf_predictors.append(random_forest)
        iter_ctr += 1
        iter_end = time.time()
        print(iter_ctr)
        print(iter_end - iter_start)

    end = time.time()
    print(end - start)

    single_trees = [predictor.trees[0] for predictor in rf_predictors]
    single_tree_biases = []
    single_tree_variances = []
    ctr = 0
    for idx, row in df_test.iterrows():
        preds = []
        for tree in single_trees:
            try:
                pred = tree.predict(row)
            except:
                ctr+=1
                continue
            preds.append(pred)
        preds = np.asarray(preds)
        avg_pred = np.mean(preds)
        bias = (avg_pred - row['label'])**2
        single_tree_biases.append(bias)
        var = np.var(preds)
        single_tree_variances.append(var)

    single_tree_biases = np.asarray(single_tree_biases)
    single_tree_variances = np.asarray(single_tree_variances)

    single_tree_biases = single_tree_biases[~np.isnan(single_tree_biases)]
    single_tree_variances = single_tree_variances[~np.isnan(single_tree_variances)]

    single_tree_bias = sum(single_tree_biases) / len(single_tree_biases)
    single_tree_var = sum(single_tree_variances) / len(single_tree_variances)
    single_tree_squared_err = single_tree_bias + single_tree_var
    print('Errors during runtime: %d' % ctr)

    random_forest_biases = []
    random_forest_variances = []
    ctr = 0
    for idx, row in df_test.iterrows():
        preds = []
        for predictor in rf_predictors:
            try:
                pred = predictor.predict(row)
            except:
                ctr+=1
                continue
            preds.append(pred)
        preds = np.asarray(preds)
        avg_pred = np.mean(preds)
        bias = (avg_pred - row['label'])**2
        random_forest_biases.append(bias)
        var = np.var(preds)
        random_forest_variances.append(var)
    #Just in case an error occurred - ensure viable results
    random_forest_biases = np.asarray(random_forest_biases)
    random_forest_variances = np.asarray(random_forest_variances)

    random_forest_biases = random_forest_biases[~np.isnan(random_forest_biases)]
    random_forest_variances = random_forest_variances[~np.isnan(random_forest_variances)]

    random_forest_bias = sum(random_forest_biases) / len(random_forest_biases)
    random_forest_var = sum(random_forest_variances) / len(random_forest_variances)
    random_forest_squared_err = random_forest_bias + random_forest_var
    print('Errors during runtime: %d' % ctr)

    #Print and save results
    print('Single Trees Bias: %f' % single_tree_bias)
    print('Single Trees Variance: %f' % single_tree_var)
    print('Single Trees Estimated Squared Error: %f' % single_tree_squared_err)

    print('Random Forest Bias: %f' % random_forest_bias)
    print('Random Forest Variance: %f' % random_forest_var)
    print('Random Forest Estimated Squared Error: %f' % random_forest_squared_err)

    textfile = open("var_bias_results_rf.txt", "w")
    textfile.write('Single Trees Bias:' + "\n")
    textfile.write(str(single_tree_bias) + "\n")

    textfile.write('Single Trees Variance:' + "\n")
    textfile.write(str(single_tree_var) + "\n")

    textfile.write('Single Trees Estimated Squared Error:' + "\n")
    textfile.write(str(single_tree_squared_err) + "\n")

    textfile.write('Random Forest Bias:' + "\n")
    textfile.write(str(random_forest_bias) + "\n")

    textfile.write('Random Forest Variance:' + "\n")
    textfile.write(str(random_forest_var) + "\n")

    textfile.write('Random Forest Estimated Squared Error:' + "\n")
    textfile.write(str(random_forest_squared_err) + "\n")
    textfile.close()