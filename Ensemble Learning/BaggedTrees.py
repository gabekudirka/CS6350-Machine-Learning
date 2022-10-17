import sys
sys.path.append('../')
from decision_tree import DecisionTree
import numpy as np

class BaggedTrees:
    def __init__(
        self,
        data,
        attributes,
    ):
        self.examples = data
        self.attributes = attributes
        self.trees = []

    def build_trees(self, num_trees, num_samples=None):
        if num_samples is None:
            examples_subset = self.examples
        else:
            examples_subset = self.examples.sample(num_samples)

        for t in range(num_trees):
            mask = np.random.randint(0,examples_subset.shape[0], examples_subset.shape[0])
            sampled_examples = examples_subset.iloc[mask]
            tree = DecisionTree(sampled_examples, self.attributes).build_tree(purity_type='entropy', max_depth=float('inf'))
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

class RandomForest:
    def __init__(
        self,
        data,
        attributes,
    ):
        self.examples = data
        self.attributes = attributes
        self.trees = []

    def build_trees(self, num_iterations, feature_subset_size, num_samples=None):
        if num_samples is None:
            examples_subset = self.examples
        else:
            examples_subset = self.examples.sample(num_samples)
            
        for t in range(num_iterations):
            mask = np.random.randint(0,examples_subset.shape[0], examples_subset.shape[0])
            sampled_examples = examples_subset.iloc[mask]
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

