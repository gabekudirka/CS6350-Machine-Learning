import numpy as np

class Layer:
    def __init__(
            self, 
            input_size, 
            output_size, 
            zero_init,
            final_layer=False, 
    ):
        self.input_size = input_size + 1
        if final_layer:
            self.output_size = output_size
        else:
            self.output_size = output_size + 1
        
        if zero_init:
            self.w = np.zeros((self.input_size, self.output_size))
        else:
            self.w = np.random.normal(0, 1, (self.input_size, self.output_size))

        self.final_layer = final_layer
    
    def forward(self, x):
        if self.final_layer:
            return np.dot(x, self.w)
        else:
            return self.sigmoid(np.dot(x, self.w))
    
    def backpropogate(self, nodes, grad):
        dW = np.dot(grad, self.w.T)
        if not self.final_layer:
            return dW * self.d_sigmoid(nodes)
        else:
            return dW
    
    def update(self, grad, lr):
        self.w -= lr * grad

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def d_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

class ThreeLayerNN:
    def __init__(
        self, 
        input_size, 
        hidden_size,
        X,
        y,
        zero_init = False
    ):
        self.X = X
        self.y = y
        self.N, self.D = X.shape
        self.nodes = []

        self.hidden1 = Layer(input_size, hidden_size, zero_init)
        self.hidden2 = Layer(hidden_size, hidden_size, zero_init)
        self.output_layer = Layer(hidden_size, 1, zero_init, final_layer=True)

    def train(self, epochs, lr_0, d):
        self.avg_losses = []
        shuffle_mask = np.arange(self.N)

        for t in range(epochs):
            losses = []
            #Shuffle the data
            np.random.shuffle(shuffle_mask)
            X = self.X[shuffle_mask, :]
            y = self.y[shuffle_mask]

            for i in range(self.N):
                y_hat = self.forward(X[i])
                losses.append(self.loss(y_hat, y[i]))
                lr = lr_0 / (1 + (lr_0 / d) * t)
                self.backpropogate(y[i], y_hat, lr)

            self.avg_losses.append(sum(losses) / len(losses))

    def loss(self, y, y_hat):
        return 0.5 * (y - y_hat)**2

    def forward(self, x):
        #Clear nodes and add x as first layer with a bias term
        self.nodes.clear()
        x = np.append(1, x)
        self.nodes.append(np.asarray([x]))

        #Complete first pass
        layer_res1 = self.hidden1.forward(self.nodes[0])
        self.nodes.append(layer_res1)

        #Complete second pass
        layer_res2 = self.hidden2.forward(self.nodes[1])
        self.nodes.append(layer_res2)

        #Complete third pass
        layer_res3 = self.output_layer.forward(self.nodes[2])
        self.nodes.append(layer_res3)

        return float(layer_res3)

    def backpropogate(self, y, y_hat, lr):
        #Calculate node gradients
        dL = y_hat - y
        dLdZ3 = self.output_layer.backpropogate(self.nodes[2], dL)
        dLdZ2 = self.hidden2.backpropogate(self.nodes[1], dLdZ3)

        #Calculate weight gradients and update weights
        dZ2dW1 = np.dot(self.nodes[0].T, dLdZ2)
        self.hidden1.update(dZ2dW1, lr)
        dZ3dW2 = np.dot(self.nodes[1].T, dLdZ3)
        self.hidden2.update(dZ3dW2, lr)
        dLdW3 = np.dot(self.nodes[2].T, dL)
        self.output_layer.update(dLdW3, lr)

        return dZ2dW1, dZ3dW2, dLdW3

    def predict(self, example):
        return np.sign(self.forward(example))
    