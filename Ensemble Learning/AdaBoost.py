from torch import full
from decision_tree import DecisionTree
import numpy as np

class Stump:
    #The Decision tree object is initialized with the data
    def __init__(
        self,
        data,
        attributes,
        purity_type,
        full_dataset
    ):
        self.stump = DecisionTree(data, attributes, weighted=True, full_dataset=full_dataset).build_tree(purity_type=purity_type, max_depth=1)
        self.vote = self.compute_vote()

    def compute_vote(self):
        self.weighted_error = self.compute_error()
        vote = 0.5 * np.log( (1-self.weighted_error) / self.weighted_error )
        return vote

    def compute_error(self):
        self.preds = self.stump.examples.apply(lambda row : self.stump.predict(row), axis=1)
        diff = self.preds == self.stump.labels
        if (diff == True).all():
            return 0
        else:
            error_count = diff.value_counts()[False]
            return error_count / len(self.stump.examples)

    def predict(self, example):
        return self.stump.predict(example)

class AdaBoostTree:
    #The AdaBoost tree object is initialized with the data
    def __init__(
        self,
        data,
        attributes,
        full_dataset = None
    ):
        self.examples = data
        self.attributes = attributes
        if full_dataset is None:
            self.full_dataset = self.examples
        else:
            self.full_dataset = full_dataset

    def build_model(self, num_iterations, purity_type = 'entropy'):
        self.classifiers = []
        self.weights = np.ones(self.examples.shape[0]) * (1 / self.examples.shape[0])
        self.examples['weights'] = self.weights

        for t in range(num_iterations):
            stump = Stump(self.examples, self.attributes, purity_type)
            correct_preds = self.examples.label * stump.preds
            self.weights = self.weights * np.exp( -1 * stump.vote * (correct_preds.to_numpy(dtype='float64')) )
            self.examples['weights'] = self.weights
            self.classifiers.append(stump)

    def predict(self, example):
        pred = 0
        for classifier in self.classifiers:
            pred += (classifier.vote * classifier.predict(example))
        if pred >= 0:
            return 1
        else:
            return -1

