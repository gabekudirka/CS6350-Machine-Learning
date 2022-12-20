from torch import full
from .decision_tree import DecisionTree
import numpy as np

#This class is for decision stumps used by the AdaBoost algorithm
class Stump:
    def __init__(
        self,
        data,
        attributes,
        purity_type,
        full_dataset
    ):
        self.stump = DecisionTree(data, attributes, weighted=True, full_dataset=full_dataset).build_tree(purity_type=purity_type, max_depth=1)
        self.vote = self.compute_vote()

    #This function computes the vote after getting the weighted error
    def compute_vote(self):
        self.weighted_error = self.compute_error()
        vote = 0.5 * np.log( (1-self.weighted_error) / self.weighted_error )
        return vote

    #This function computes the weighted error which is the sum of weights where the prediction and label were incorrect
    def compute_error(self):
        self.preds = self.stump.examples.apply(lambda row : self.stump.predict(row), axis=1)
        self.stump.examples['diff'] = self.preds == self.stump.labels
        if (self.stump.examples['diff'] == True).all():
            return 0
        else:
            #Compute the sum of the weights of the error here
            #Make sure to normalize weights every time
            errors = self.stump.examples[self.stump.examples['diff'] == False]
            return errors['weights'].sum()

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

    #This function builds the specified number of decision tree stumps and then calculates the weights based on their predictions and vote
    def build_model(self, num_iterations, purity_type = 'entropy'):
        self.classifiers = []
        #Initialize the weights to be 1/n where n is the number of samples
        self.weights = np.ones(self.examples.shape[0]) * (1 / self.examples.shape[0])
        self.examples['weights'] = self.weights

        #Create a stump for every specified iteration
        for t in range(num_iterations):
            #Create the stump with the current weights
            stump = Stump(self.examples, self.attributes, purity_type, self.full_dataset)
            #Calculate yi*hi(x)
            correct_preds = self.examples.label * stump.preds
            #Update the weights
            self.weights = self.weights * np.exp( -1 * stump.vote * (correct_preds.to_numpy(dtype='float64')) )
            #normalize the weights
            self.weights = self.weights / self.weights.sum()
            self.examples['weights'] = self.weights
            self.classifiers.append(stump)

    #Take votes from every classifier to make a prediction
    def predict(self, example):
        pred = 0
        for classifier in self.classifiers:
            pred += (classifier.vote * classifier.predict(example))
        if pred >= 0:
            return 1
        else:
            return -1

