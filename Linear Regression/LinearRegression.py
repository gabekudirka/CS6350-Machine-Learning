import numpy as np

class LinearRegressor:
    #The Decision tree object is initialized with the data
    def __init__(
        self,
        X,
        y,
        
    ):
        bias_col = np.ones(X.shape[0])
        X = np.c_[X, bias_col]
        self.X = X
        self.y = y

    def gradient_descent(self, learning_rate, tolerance=1e-6):
        np.seterr(all='raise')
        self.w = np.zeros(self.X.shape[1])
        self.costs = []
        diff = tolerance + 1

        while diff > tolerance:
            dw = []
            for j in range(self.w.shape[0]):
                dwj = 0
                for i in range(self.X.shape[0]):
                    try:
                        dwj += (self.y[i] - np.dot(self.w.T, self.X[i]))*self.X[i,j]
                    except:
                        print('Lower the learning rate')
                        return
                dwj = dwj * -1
                dw.append(dwj)
            new_weights = self.w - (learning_rate * np.asarray(dw))
            diff = np.linalg.norm(new_weights - self.w)
            self.w = new_weights
            cost = [(self.y[idx] - (self.w.T @ self.X[idx]))**2 for idx in range(self.X.shape[0])]
            cost = 0.5 * sum(cost)
            self.costs.append(cost)

    def stochastic_gradient_descent(self, learning_rate, num_iterations = 1000):
        np.seterr(all='raise')
        self.w = np.zeros(self.X.shape[1])
        self.costs = []

        for ctr in range(num_iterations):
            for i in range(self.X.shape[0]):
                for j in range(self.w.shape[0]):
                    try:
                        grad = (self.y[i] - np.dot(self.w.T, self.X[i]))*self.X[i,j]
                    except:
                        print('Lower the learning rate')
                        return
                    self.w[j] = self.w[j] + learning_rate*grad
                cost = [(self.y[idx] - (self.w.T @ self.X[idx]))**2 for idx in range(self.X.shape[0])]
                cost = 0.5 * sum(cost)
                self.costs.append(cost)

    def predict(self, example):
        return np.dot(self.w.T, np.append(example, 1))