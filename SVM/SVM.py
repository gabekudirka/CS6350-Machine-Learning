import numpy as np
from scipy.optimize import minimize


class SVMClassifier:
    def __init__(
        self,
        data
    ):
        self.data = data
        self.kernel = False
        # self.X = X
        # self.y = y
        #self.w = np.zeros(X.shape[0]+1)

    def train_primal_ssgd(self, C, lr0, alpha=None, epochs=100):
        # Initialize weight vector
        w = np.zeros(self.data.shape[1])
        N = self.data.shape[0]
        bias_col = np.ones(N)

        for t in range(1, epochs+1):
            #Shuffle data
            shuffled_data = self.data.sample(frac=1)
            X = shuffled_data.iloc[:,:-1].to_numpy(dtype='float64')
            X = np.c_[X, bias_col]
            y = shuffled_data.iloc[:,-1].to_numpy(dtype='float64')

            #Update learning rate
            if alpha is not None:
                lr = lr0 / (1 + (lr0 * t / alpha))
            else:
                lr = lr0 / (1 + t)

            #Perform calculate sub gradient and perform weight update
            for i, xi in enumerate(X):
                yi = y[i]
                w0 = w[:-1]
                w0b = np.append(w0, 0)
                if yi * (w.T @ xi) <= 1:
                    w -= lr * w0b
                    w += lr * C * N * yi * xi
                else:
                    w[:-1] = (1 - lr) * w0

        self.w = w

    def train_dual(self, C, gamma=None):
        if gamma is not None:
            self.kernel = True
            self.gamma = gamma
        #Get data in correct form
        X = self.data.iloc[:,:-1].to_numpy(dtype='float64')
        y = self.data.iloc[:,-1].to_numpy(dtype='float64')
        self.X = X
        self.y = y
        N, M = X.shape

        #Compute the gram matrix of X, multiplied by y and y transpose
        #This is for the sum_i sum_j y_i y_j x_i^T x_j part of the minimization function
        gram_y = np.zeros((N, N))
        for m in range(N):
            for n in range(N):
                if self.kernel:
                    Xy = self.gaussian_kernel(X[m,:], X[n,:], gamma)
                else:
                    Xy = X[m,:] @ X[n,:]
                Xy = Xy * y[n] * y[m]
                gram_y[m,n] = Xy

        # This computes the minimization function using the modified gram matrix computed above
        fun = lambda alpha: 0.5 * (alpha.T @ (gram_y @ alpha)) - alpha.sum()
        # This is the derivative of the minimization function - necessary for the minimize function
        jac = lambda alpha: (alpha.T @ gram_y) - np.ones(alpha.shape[0])
        # This is the constraint that sum_i alphai y_i = 0
        constraints = ({'type': 'eq', 'fun': lambda alpha: alpha @ y, 'jac': lambda alpha: y})
        # This enforces the bounds that for alpha where 0 <= alpha <= C
        bounds = [(0,C)]*N
        # Use an array of ones for the intial guess for alpha
        x0 = np.ones(N)

        # Use the above functions and constraints in the minimize function
        min_res = minimize(fun=fun, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints)
        self.alpha = min_res.x
        mask = self.alpha > 0.001
        self.nonzero_alphas = self.alpha[mask]
        #Compute the optimal weights and biases using the optimal alphas computed above
        self.w0 = np.sum(self.alpha * y * X.T, axis = 1)
        self.b = np.mean(y - (self.w0 @ X.T))
        self.w = np.append(self.w0, self.b)

    def gaussian_kernel(self, xi, xj, gamma):
        return np.exp(-np.linalg.norm(xi-xj)**2 / gamma)

    def predict(self, example):
        if not self.kernel:
            return np.sign(self.w.T @ np.append(example, 1.0))
        else:
            # gaussian_kernel = lambda xi, xj: np.exp(-1 * (np.linalg.norm(xi - xj)**2 / gamma))
            # pred = 0
            # for i, xi in enumerate(self.X):
            #     pred += self.alpha[i] * self.y[i] * gaussian_kernel(xi, example)
            # pred += self.b
            kernel_res = np.asarray([self.gaussian_kernel(xi, example, self.gamma) for xi in self.X])
            return np.sign(np.sum(self.alpha * self.y * kernel_res, axis=None))
            

