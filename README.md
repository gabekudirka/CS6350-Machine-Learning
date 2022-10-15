This is a machine learning library developed by Gabrielius Kudirka for CS5350/6350 in University of Utah.

### Decision Tree Tutorial
From the decision_tree.py file in the DecisionTree folder you can import the DecisionTree class. To initialize a decision tree object using this class, you need to pass in the training data for the decision tree as a pandas dataframe and a list of the attributes in the data. Then to build a tree the decision tree object has a function called build_tree which takes in the type of information gain as a string ('entropy', 'gini', or 'me) and the maximum depth of the tree as an integer. The default information gain is entropy and the default maximum depth is 16. After creating the decision tree object and calling build_tree, you can use the decision tree to make predictions using the object's predict function. This function takes in a single example and returns the prediction from the tree.
Example:
```
tree = DecisionTree(df, attributes)
tree.build_tree(purity_type='gini', max_depth=8)
prediction = tree.predict(test_example)
```