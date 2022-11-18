This is a machine learning library developed by Gabrielius Kudirka for CS5350/6350 in University of Utah.

### Decision Tree Tutorial
From the decision_tree.py file in the DecisionTree folder you can import the DecisionTree class. To initialize a decision tree object using this class, you need to pass in the training data for the decision tree as a pandas dataframe and a list of the attributes in the data. Then to build a tree the decision tree object has a function called build_tree which takes in the type of information gain as a string ('entropy', 'gini', or 'me) and the maximum depth of the tree as an integer. The default information gain is entropy and the default maximum depth is 16. After creating the decision tree object and calling build_tree, you can use the decision tree to make predictions using the object's predict function. This function takes in a single example and returns the prediction from the tree.
Example:
```
tree = DecisionTree(df, attributes)
tree.build_tree(purity_type='gini', max_depth=8)
prediction = tree.predict(test_example)
```

### AdaBoost Tutorial
From the Adaboost.py file in the Ensemble Learning folder you can import the AdaBoostTree class. To initialize an AdaBoostTree you need to pass in the training data as a pandas dataframe and a list of the attributes in the data. Then to train the adaboost algorithm you need to call the build_model function from the AdaBoost object, passing in the number of weak learners you want to build and the purity type for each stump ('entropy', 'gini', or 'me'). After training the model, you can make predictions by calling the predict function of the AdaBoost object and passing in an example that you want to predict.
Example:
```
adaboost = AdaBoostTree(df, attributes)
adaboost.build_model(50, purity_type='gini')
prediction = adaboost.predict(test_example)
```

### Bagged Trees tutorial
From the BaggedTrees.py file in the Ensemble Learning folder you can import the BaggedTrees class. To initialize a BaggedTrees object you need to pass in the training data as a pandas dataframe and a list of attributes in the data. Then to train the model you can call the build_trees function of the BaggedTrees object, passing in the number of trees you want to build. If you only want to train on a sample of the data passed in, you can also specify the number of samples you want to train on however by default the BaggedTrees algorithm will train on all of the data. After training the model, you can make predictions by calling the predict function of the BaggedTrees object and passing in an example that you want to predict.
Example:
```
bagged_trees = BaggedTrees(df, attributes)
bagged_trees.build_trees(50)
prediction = bagged_trees.predict(test_example)
```

### Random Forest tutorial
From the BaggedTrees.py file in the Ensemble Learning folder you can import the RandomForest class. To initialize a RandomForest object you need to pass in the training data as a pandas dataframe and a list of attributes in the data. Then to train the model you can call the build_trees function of the RandomForest object, passing in the number of trees you want to build and the size of the subset of attributes that will be used when training the trees. If you only want to train on a sample of the data passed in, you can also specify the number of samples you want to train on however by default the RandomForest algorithm will train on all of the data. After training the model, you can make predictions by calling the predict function of the RandomForest object and passing in an example that you want to predict.
Example:
```
random_forest = RandomForest(df, attributes)
random_forest.build_trees(50)
prediction = random_forest.predict(test_example)
```

### LMS tutorial
From the LinearRegression.py file in the Linear Regression folder you can import the LinearRegressor class. To initialize a LinearRegressor object you need to pass in the X and y data seperately as numpy arrays. Then to train the model using batch gradient descent you can call the gradient_descent function of the LinearRegressor object, passing in the learning rate and the tolerance for testing for convergence. By default the tolerance is 1e-6. If you want to train the model using stochastic gradient descent you can call the stochastic_gradient_descent function specifying the learning rate and the number of iterations you would like to train for. After training the model, you can make predictions by calling the predict function of the LinearRegressor object and passing in an example that you want to predict. You can access the learned weights through the w attribute of the LinearRegressor object.
Example:
```
regressor = LinearRegressor(X, y)
r = 0.01
regressor.gradient_descent(r)
prediction = regressor.predict(test_example)
regressor.stochastic_gradient_descent(r, 100)
prediction = regressor.predict(test_example)
weights = regressor.w
```

### LMS tutorial
From the Perceptron.py file in the Linear Regression folder you can import the Perceptron class. To initialize you need to pass
in the X and y training data seperately as numpy arrays. Then to train the perceptron using the standard perceptron algorithm you can call the train_standard function of the Perceptron object, passing in the learning rate and number of epochs. To train the Perceptron using the voted perceptron algorithm you can call the train_voted function of the Perceptron object, passing in the learning rate and number of epochs. To train the Perceptron using the average perceptron algorithm, you can call the train_average function of the perceptron object, passing in the learning rate and number of epochs. For all algorithms, the default number of epochs is 10. After training with any of these algorithms you can call the predict function of the Perceptron object, passing in the example you wish to make a prediction for. If you train using the standard algorithm, you can access the learned weights by from the w attribute of the Perceptron object. If training with the voted perceptron algorithm, you can access the array of learned weight vectors from the w_arr attribute of the Perceptron object. If training with the average perceptron algorithm you can access the cumulated average weight vector by accessing the a attribute of the Perceptron object.
Example:
```
perceptron = Perceptron(X, y)
r = 0.01
epochs = 15
perceptron.train_standard(r, epochs)
prediction = perceptron.predict(test_example)
weights = perceptron.w

perceptron.train_voted(r, epochs)
prediction = perceptron.predict(test_example)
weight_arrs = perceptron.w_arr

perceptron.train_average(r, epochs)
prediction = perceptron.predict(test_example)
avg_weights = perceptron.a
```