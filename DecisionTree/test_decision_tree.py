from decision_tree import DecisionTree
import pandas as pd

def process_data(df, attributes, replace_unknown=False):
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
            
    return df

def test_tree_accuracy(decision_tree, test_data):
    preds = test_data.apply(lambda row : decision_tree.predict(row), axis=1)
    diff = preds == test_data['label']
    if (diff == True).all():
        return 1.0
    else:
        error_count = diff.value_counts()[False]
        return error_count / len(test_data)

def test_decision_tree(df_train, df_test, attributes, max_max_depth):
    purity_functions = ['entropy', 'gini', 'me']
    for max_depth in range(1, max_max_depth+1):
        for purity_function in purity_functions:
            tree = DecisionTree(df_train, attributes).build_tree(purity_type=purity_function, max_depth=max_depth)
            training_error = test_tree_accuracy(tree, df_train)
            testing_error = test_tree_accuracy(tree, df_test)
            print('Max Depth: %d | Purity Function: %s | Test Set: Training data | Error: %.3f' % (max_depth, purity_function, training_error))
            print('Max Depth: %d | Purity Function: %s | Test Set: Testing data | Error: %.3f' % (max_depth, purity_function, testing_error))

if __name__ == '__main__':
    #TESTING FOR PROBLEM 2B
    with open ( './data/cars/data-desc.txt' , 'r' ) as f:
        desc_lines = f.readlines()

    attributes1 = desc_lines[-1].strip().split(',')
    attributes1 = attributes1[:-1]

    df_train1 = pd.read_csv('./data/cars/train.csv', names=attributes1 + ['label'])
    df_test1 = pd.read_csv('./data/cars/test.csv', names=attributes1 + ['label'])

    test_decision_tree(df_train1, df_test1, attributes1, 6)

    #TESTING FOR PROBLEM 3A
    # attributes2 = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
    # 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    # df_train2 = pd.read_csv('./data/bank/train.csv', names=attributes2 + ['label'])
    # df_test2 = pd.read_csv('./data/bank/test.csv', names=attributes2 + ['label'])

    # df_train2 = process_data(df_train2, attributes2, replace_unknown=False)
    # df_test2 = process_data(df_test2, attributes2, replace_unknown=False)

    # test_decision_tree(df_train2, df_test2, attributes2, 16)

    #TESTING FOR PROBLEM 3B

    attributes3 = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
    'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    df_train3 = pd.read_csv('./data/bank/train.csv', names=attributes3 + ['label'])
    df_test3 = pd.read_csv('./data/bank/test.csv', names=attributes3 + ['label'])

    df_train3 = process_data(df_train3, attributes3, replace_unknown=True)
    df_test3 = process_data(df_test3, attributes3, replace_unknown=True)

    test_decision_tree(df_train3, df_test3, attributes3, 16)

