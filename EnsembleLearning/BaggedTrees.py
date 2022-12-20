import sys
sys.path.append('../')
from .decision_tree import DecisionTree
import numpy as np

# This class implements the bagged trees algorithm
class BaggedTrees:
    def __init__(
        self,
        data,
        attributes,
        full_dataset = None
    ):
        self.examples = data
        self.attributes = attributes
        self.trees = []
        # If using less than the full training set, ensure that all possible attribute values are accounted for by also 
        # inputting the full dataset
        if full_dataset is None:
            self.full_dataset = self.examples
        else:
            self.full_dataset = full_dataset

    def build_trees(self, num_trees, num_samples=None):
        #If a number of samples is specified, randomly sample and use that many samples from the full dataset
        if num_samples is None:
            examples_subset = self.examples
        else:
            examples_subset = self.examples.sample(num_samples)

        # Creates as many trees as specified by num_trees
        for t in range(num_trees):
            #Create a mask to uniformly sample from the training data with replacement
            mask = np.random.randint(0,examples_subset.shape[0], examples_subset.shape[0])
            sampled_examples = examples_subset.iloc[mask]
            #Create a decision tree with unlimited depth
            tree = DecisionTree(sampled_examples, self.attributes, full_dataset=self.full_dataset, weighted=True).build_tree(purity_type='entropy', max_depth=float('inf'))
            self.trees.append(tree)

    def empty_trees(self):
        self.trees = []

    def predict(self, example):
        pred = 0
        for tree in self.trees:
            pred += tree.predict(example)
        if pred >= 0:
            return 1
        else:
            return -1

# This class implements the random forest algorithm, very similar to bagged trees
class RandomForest:
    def __init__(
        self,
        data,
        attributes,
        full_dataset = None
    ):
        self.examples = data
        self.attributes = attributes
        self.trees = []
        if full_dataset is None:
            self.full_dataset = self.examples
        else:
            self.full_dataset = full_dataset

    def build_trees(self, num_iterations, feature_subset_size, num_samples=None):
        if num_samples is None:
            examples_subset = self.examples
        else:
            examples_subset = self.examples.sample(num_samples)
            
        for t in range(num_iterations):
            mask = np.random.randint(0,examples_subset.shape[0], examples_subset.shape[0])
            sampled_examples = examples_subset.iloc[mask]
            #The only difference from the bagged trees algorithm is that a feature subset size is specified here
            tree = DecisionTree(sampled_examples, self.attributes).build_tree(purity_type='entropy', max_depth=float('inf'), feature_subset_size=feature_subset_size)
            self.trees.append(tree)

    def empty_trees(self):
        self.trees = []

    def predict(self, example):
        pred = 0
        for tree in self.trees:
            pred += tree.predict(example)
        if pred >= 0:
            return 1
        else:
            return -1

