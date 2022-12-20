from matplotlib.pyplot import xscale
import numpy as np
import pandas as pd

class Perceptron:
    def __init__(
        self,
        X,
        y
    ):
        bias_col = np.ones(X.shape[0])
        X = np.c_[X, bias_col]
        self.X = X
        self.y = y
        self.type = None

    def train_standard(self, r, epochs=10):
        self.type = 'standard'
        #Shuffle examples? 
        self.w = np.random.rand(self.X.shape[1])
        for i in range(epochs):
            for idx, xi in enumerate(self.X):
                yi = self.y[idx]
                y_hat = 1 if self.w.T @ xi >= 0 else -1
                if y_hat != yi:
                    self.w += r * (yi * xi)

    def train_voted(self, r, epochs=10):
        self.type = 'voted'
        w = np.zeros(self.X.shape[1])
        self.w_arr = []
        c = 1
        for i in range(epochs):
            for idx, xi in enumerate(self.X):
                yi = self.y[idx]
                y_hat = 1 if w.T @ xi >= 0 else -1
                if y_hat != yi:
                    w_old = w.copy()
                    self.w_arr.append((w_old, c))
                    w += r * (yi * xi)
                    c = 1
                else:
                    c += 1

    def train_average(self, r, epochs=10):
        self.type = 'average'
        w = np.zeros(self.X.shape[1])
        self.a = np.zeros(self.X.shape[1])
        for i in range(epochs):
            for idx, xi in enumerate(self.X):
                yi = self.y[idx]
                y_hat = 1 if w.T @ xi >= 0 else -1
                if y_hat != yi:
                    w += r * (yi * xi)
                self.a = self.a + w

    def predict(self, example):
        if self.type == 'standard':
            pred = 1 if self.w.T @ np.append(example, 1.0) >= 0 else -1
        elif self.type == 'voted':
            weighted_sum = 0
            x = np.append(example, 1)
            for w, c in self.w_arr:
                iter_pred = 1 if w.T @ x >= 0 else -1
                weighted_sum += (c * iter_pred)
            pred = 1 if weighted_sum >= 0 else -1
        elif self.type == 'average':
            pred = 1 if self.a.T @ np.append(example, 1.0) >= 0 else -1
        else:
            pred = None
            print('No perceptron has been learned')
        return pred