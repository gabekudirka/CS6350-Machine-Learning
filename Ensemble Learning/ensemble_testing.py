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

    df_train = pd.read_csv('../data/bank/train.csv', names=attributes + ['label'])
    df_test = pd.read_csv('../data/bank/test.csv', names=attributes + ['label'])

    df_train = process_data(df_train, attributes, replace_unknown=False)
    df_test = process_data(df_test, attributes, replace_unknown=False)

    #Test Bagged Trees
    # bagged_trees = BaggedTrees(df_train, attributes)
    # training_errors_bt = []
    # test_errors_bt = []
    # #T = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    # T = [1, 2, 2, 5, 5, 5, 10, 20, 25, 25, 50, 50, 100, 100, 100]

    # start = time.time()
    # for t in T:
    #     iter_start = time.time()
    #     bagged_trees.build_trees(t)
    #     training_errors_bt.append(test_tree_accuracy(bagged_trees, df_train))
    #     test_errors_bt.append(test_tree_accuracy(bagged_trees, df_test)) 
    #     iter_end = time.time()
    #     print (iter_end - iter_start)
    #     print(len(bagged_trees.trees))

    # end = time.time()
    # print( end - start)
    # textfile = open("training_errs_bt.txt", "w")
    # for element in training_errors_bt:
    #     textfile.write(str(element) + ", ")
    # textfile.close()

    # textfile = open("test_errs_bt.txt", "w")
    # for element in test_errors_bt:
    #     textfile.write(str(element) + ", ")
    # textfile.close()

    #Bagged Trees bias/variance decomposition
    num_trees = 500
    num_samples = 1000
    bagged_predictors = []

    for i in range(100):
        bagged_trees = BaggedTrees(df_train, attributes)
        bagged_trees.build_trees(num_trees, num_samples)
        bagged_predictors.append(bagged_trees)

    single_trees = [predictor.trees[0] for predictor in bagged_predictors]
    single_tree_biases = []
    single_tree_variances = []
    for idx, row in df_test.iterrows():
        preds = np.asarray([tree.predict(row) for tree in single_trees])
        avg_pred = np.mean(preds)
        bias = (avg_pred - row['label'])**2
        single_tree_biases.append(bias)
        var = np.var(preds)
        single_tree_variances.append(var)
    single_tree_bias = sum(single_tree_biases) / len(single_tree_biases)
    single_tree_var = sum(single_tree_variances) / len(single_tree_variances)
    single_tree_squared_err = single_tree_bias + single_tree_var

    bagged_tree_biases = []
    bagged_tree_variances = []
    for idx, row in df_test.iterrows():
        preds = np.asarray([predictor.predict(row) for predictor in bagged_predictors])
        avg_pred = np.mean(preds)
        bias = (avg_pred - row['label'])**2
        bagged_tree_biases.append(bias)
        var = np.var(preds)
        bagged_tree_variances.append(var)
    bagged_trees_bias = sum(bagged_tree_biases) / len(bagged_tree_biases)
    bagged_trees_var = sum(bagged_tree_variances) / len(bagged_tree_variances)
    bagged_trees_squared_err = bagged_trees_bias + bagged_trees_var

    # textfile = open("test_errs_bt.txt", "w")
    # for element in test_errors_bt:
    #     textfile.write(str(element) + ", ")
    # textfile.close()
    

    #Test Random Forest
    # training_errors_rf = {}
    # test_errors_rf = {}
    # subset_sizes = [2, 4, 6]
    # T = [1, 2, 2, 5, 5, 5, 10, 20, 25, 25, 50, 50, 100, 100, 100]

    # start = time.time()
    # for subset_size in subset_sizes:
    #     random_forest = RandomForest(df_train, attributes)
    #     training_errors_rf[subset_size] = []
    #     test_errors_rf[subset_size] = []
    #     for t in T:
    #         iter_start = time.time()
    #         random_forest.build_trees(t, subset_size)
    #         training_errors_rf[subset_size].append(test_tree_accuracy(random_forest, df_train))
    #         test_errors_rf[subset_size].append(test_tree_accuracy(random_forest, df_test))
    #         iter_end = time.time()
    #         print (iter_end - iter_start)
    #         print(len(random_forest.trees))
    # end = time.time()
    # print( end - start)

    # with open('training_errs_rf.txt', 'w') as convert_file:
    #     convert_file.write(json.dumps(training_errors_rf))

    # with open('test_errors_rf.txt', 'w') as convert_file:
    #     convert_file.write(json.dumps(test_errors_rf))