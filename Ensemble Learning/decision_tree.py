import pandas as pd
pd.options.mode.chained_assignment = None
import math
import random

#This class defines a node in the decision tree
class Node:
    def __init__(
        self, 
        parent, 
        examples,
        attributes,
        depth
    ):
        self.parent = parent
        self.examples = examples
        self.attributes = attributes
        self.depth = depth
        self.branches = {}
        self.split_attribute = None
        self.leaf = False

    def set_split_attribute(self, attribute):
        self.split_attribute = attribute

    #This function is called if a node is determined to be a leaf node
    def make_leaf(self, label):
        self.leaf = True
        self.label = label

#The decision tree class defines an object which stores a decision tree based on passed in data
#Using this object a user can create a decision tree with data and use the tree to make predictions
class DecisionTree:
    #The Decision tree object is initialized with the data
    def __init__(
        self,
        data,
        attributes,
        weighted = False,
        full_dataset = None
    ):
        self.attributes = attributes
        if not weighted:
            data['weights'] = 1 / data.shape[0]
        self.examples = data
        self.labels = data['label']
        if full_dataset is None:
            self.full_dataset = self.examples
        else:
            self.full_dataset = full_dataset
        self.unique_vals = {}
        for attribute in attributes:
            self.unique_vals[attribute] = pd.unique(self.full_dataset[attribute])

    #This is a driver function for the id3 algorithm. This function is called by the user and sets
    #the user selected entropy and purity function and then calls the id3 recursive algorithm to build the decision tree
    def build_tree(self, purity_type='entropy', max_depth=16, feature_subset_size=None):
        self.max_depth = max_depth
        self.feature_subset_size = feature_subset_size
        self.root_node = Node(None, self.examples, self.attributes, 0)
        if purity_type == 'entropy':
            self.purity_function = self.calculate_entropy
        elif purity_type == 'gini':
            self.purity_function = self.calculate_gini_index
        elif purity_type == 'me':
            self.purity_function = self.calculate_majority_error
        
        self.total_purity = self.purity_function(self.examples, self.examples.weights.sum())

        self.id3(self.examples, self.attributes, self.root_node, 0)
        
        return self

    # Use the sum of weights to get the most common label for assigning to a leaf node
    def get_most_common(self, examples):
        max = -1
        for label in examples.label.unique():
            label_weights = examples[examples['label'] == label]['weights'].sum()
            if label_weights > max:
                max = label_weights
                most_common = label

        return most_common

    #This is the recursive function that runs the id3 algorithm to create a decision tree
    def id3(self, examples, attributes, node, depth):
        #This is the first base case. When the set of examples passed in to the id3 algorithm is empty
        #it creates a leaf node with the most common label of the parent as the label
        if examples.empty:
            most_common_label = self.get_most_common(node.parent.examples)
            node.make_leaf(most_common_label)
            return node
        #This is the second and third base case. When the max depth is reached or if there are no attributes
        #left, a leaf node is created with the most common label in the examples as the label
        elif depth == self.max_depth or len(attributes) == 0:
            most_common_label = self.get_most_common(examples)
            node.make_leaf(most_common_label)
            return node
        #This is the third base case. If all the labels in the example are the same, a leaf node is created
        #With that most common label as the label
        elif (examples['label'] == examples['label'].iloc[0]).all():
            node.make_leaf(examples['label'].iloc[0])
            return node
            
        #For the random forest algorithm, if a subset size is specified, choose the best attribute from a random subset of the attributes
        if self.feature_subset_size is None or len(attributes) < self.feature_subset_size:
            attribute_subset = attributes
        else:
            attribute_subset = random.sample(attributes, self.feature_subset_size)
        split_attribute = self.select_split_attribute(attribute_subset, examples)
        node.set_split_attribute(split_attribute)
        possible_values = self.unique_vals[split_attribute]

        #Iterates through all of the possible values of the split attribute and on each iteration
        #creates a new node, sets it as a branch of the parent, and calls the id3 function with that
        #node, the examples where the split attribute equals the current value, and a list of the attributes
        #without the split attribute
        for value in possible_values:
            new_attributes = attributes.copy()
            new_attributes.remove(split_attribute)
            new_examples = examples[examples[split_attribute] == value]
            new_node = Node(node, new_examples, new_attributes, depth+1)
            node.branches[value] = new_node
            self.id3(new_examples, new_attributes, new_node, depth+1)

    #This function gets the information gain of each of the attributes and returns the attribute with the highest gain
    def select_split_attribute(self, attributes, examples):
        gains = {}
        for attribute in attributes:
            gains[attribute] = self.calculate_info_gain(examples, attribute)
        return max(gains, key=gains.get)

    #This function calculates the information gain with the user selected purity function
    def calculate_info_gain(self, examples, attribute):
        unique_vals = pd.unique(examples[attribute])
        total = examples.weights.sum()
        gain = 0
        for val in unique_vals:
            examples_with_val = examples[examples[attribute] == val]
            examples_weight_sum = examples_with_val.weights.sum()
            purity = self.purity_function(examples_with_val, examples_weight_sum)
            weighted_purity = (examples_weight_sum/total)*purity
                
            gain += weighted_purity

        return self.total_purity - gain

    #Calculates the entropy using weights
    def calculate_entropy(self, examples, total):
        labels = examples['label'].unique()
        entropy = 0
        for label in labels:
            examples_label = examples[examples['label'] == label]
            p = examples_label.weights.sum() / total
            entropy -= p*math.log(p, 2)

        return entropy

    #Calculates the Gini coefficient using weights
    def calculate_gini_index(self, examples, total):
        labels = examples['label'].unique()
        gini_index = 0
        for label in labels:
            examples_label = examples[examples['label'] == label]
            p = examples_label.weights.sum() / total
            gini_index += p**2

        return 1 - gini_index

    #This function is used after the tree has been built to make a prediction for a given example
    #The tree starts at the root node and this function goes down the tree based on the attribute
    #values in the example until it reaches a leaf node and then returns the leaf node's label
    def predict(self, example):
        node = self.root_node
        while not node.leaf:
            split_attribute = node.split_attribute
            node = node.branches[example[split_attribute]]
        return node.label