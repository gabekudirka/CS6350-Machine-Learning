import numpy as np

#This class contains code to run gradient descent and stochastic gradient descent on a dataset
class LinearRegressor:
    def __init__(
        self,
        X,
        y,
        
    ):
        bias_col = np.ones(X.shape[0])
        X = np.c_[X, bias_col]
        self.X = X
        self.y = y

    #This function performs gradient descent on the datasset
    def gradient_descent(self, learning_rate, tolerance=1e-6):
        np.seterr(all='raise')
        #Initialize the weight vector to 0
        self.w = np.zeros(self.X.shape[1])
        self.costs = []
        diff = tolerance + 1

        #Iterate until convergence
        while diff > tolerance:
            dw = []
            for j in range(self.w.shape[0]):
                dwj = 0
                for i in range(self.X.shape[0]):
                    #If the learning rate is too large the gradients will explode causing errors, this is caught
                    try:
                        dwj += (self.y[i] - np.dot(self.w.T, self.X[i]))*self.X[i,j]
                    except:
                        print('Lower the learning rate')
                        return
                dwj = dwj * -1
                dw.append(dwj)
            #Update the weights
            new_weights = self.w - (learning_rate * np.asarray(dw))
            diff = np.linalg.norm(new_weights - self.w)
            self.w = new_weights
            #Calculate the cost for this iteration
            cost = [(self.y[idx] - (self.w.T @ self.X[idx]))**2 for idx in range(self.X.shape[0])]
            cost = 0.5 * sum(cost)
            self.costs.append(cost)

    #This function perfrorms stochastic gradient descent on the dataset
    def stochastic_gradient_descent(self, learning_rate, num_iterations = 1000):
        np.seterr(all='raise')
        self.w = np.zeros(self.X.shape[1])
        self.costs = []

        #Loop for a set number of iterations
        for ctr in range(num_iterations):
            for i in range(self.X.shape[0]):
                for j in range(self.w.shape[0]):
                    #calculate the gradient
                    try:
                        grad = (self.y[i] - np.dot(self.w.T, self.X[i]))*self.X[i,j]
                    except:
                        print('Lower the learning rate')
                        return
                    #Perform the weight update after each sample
                    self.w[j] = self.w[j] + learning_rate*grad
                #Calculate the costs
                cost = [(self.y[idx] - (self.w.T @ self.X[idx]))**2 for idx in range(self.X.shape[0])]
                cost = 0.5 * sum(cost)
                self.costs.append(cost)

    #Apply the weight vector to make a prediction
    def predict(self, example):
        return np.dot(self.w.T, np.append(example, 1))