import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ReLuNN(nn.Module):
    def __init__(self, input_size, width, depth):
        super(ReLuNN, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth - 2
        
        self.input_layer = nn.Linear(self.input_size, self.width)
        self.hidden_layer = nn.Linear(self.width, self.width)
        self.output_layer = nn.Linear(self.width, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for i in range(self.depth):
            x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

class TanHNN(nn.Module):
    def __init__(self, input_size, width, depth):
        super(TanHNN, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth - 2
        
        self.input_layer = nn.Linear(self.input_size, self.width)
        self.hidden_layer = nn.Linear(self.width, self.width)
        self.output_layer = nn.Linear(self.width, 1)

    def forward(self, x):
        x = F.tanh(self.input_layer(x))
        for i in range(self.depth):
            x = F.tanh(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

class BankNoteDataset(Dataset):
    def __init__(self, filepath, attributes, label_col):
        self.df = pd.read_csv(filepath, names=attributes)
        self.df[label_col].iloc[self.df[label_col] == 0] = -1
        self.X = self.df.loc[:, self.df.columns != label_col].to_numpy(dtype='float64')
        self.y = self.df[label_col].to_numpy(dtype='float64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32, device=DEVICE)
        y = torch.tensor(self.y[idx], dtype=torch.float32, device=DEVICE)
        return X, y

    def get_dims(self):
        N, D = self.X.shape
        return N, D

def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def he(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train(dataloader, model, optimizer, epochs=10):
    avg_losses = []
    loss_function = nn.MSELoss()
    model = model.to(device=DEVICE)
    for t in range(epochs):
        losses = []
        for i, (x, y) in enumerate(dataloader):
            model.train()
            pred = model(x)
            loss = loss_function(torch.reshape(pred, y.shape), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        losses = np.asarray(losses).flatten()
        avg_losses.append(losses.mean())
    return avg_losses

def test(dataloader, model):
    loss_function = nn.MSELoss()
    model = model.to(device=DEVICE)
    incorrect_ctr = 0
    avg_loss = 0
    for i, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            pred = model(x)
            if torch.sign(pred) != y:
                incorrect_ctr += 1
            
            loss = loss_function(torch.reshape(pred, y.shape), y)
            avg_loss += loss
            
    return incorrect_ctr / len(dataloader), avg_loss / len(dataloader)

if __name__ == '__main__':
    attributes = ['variance','skewness','curtosis','entropy','genuine']
    train_dataset = BankNoteDataset('./data/bank-note/train.csv', attributes, 'genuine')
    test_dataset = BankNoteDataset('./data/bank-note/test.csv', attributes, 'genuine')

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    train_dataloader_test = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    LEARNING_RATE = 0.001
    EPOCHS = 20
    N, D = train_dataset.get_dims()

    widths = [5, 10, 25, 50, 100]
    depths = [3, 5, 9]
    losses = {}
    for depth in depths:
        losses[depth] = {}
        for width in widths:
            model = ReLuNN(D, width, depth)
            model.apply(he)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            losses[depth][width] = train(train_dataloader, model, optimizer, EPOCHS)

            train_error, train_loss = test(train_dataloader_test, model)
            test_error, test_loss = test(test_dataloader, model)

            print('Model with depth = %d and width = %d:' % (depth, width))
            print('Training error: %f - Test error %f' % (train_error, test_error))
            print('Training loss: %f - Test loss %f' % (train_loss, test_loss))
            print('')

    widths = [5, 10, 25, 50, 100]
    depths = [3, 5, 9]
    losses = {}
    for depth in depths:
        losses[depth] = {}
        for width in widths:
            model = TanHNN(D, width, depth)
            model.apply(xavier)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            losses[depth][width] = train(train_dataloader, model, optimizer, EPOCHS)
            
            train_error, train_loss = test(train_dataloader_test, model)
            test_error, test_loss = test(test_dataloader, model)

            print('Model with depth = %d and width = %d:' % (depth, width))
            print('Training error: %f - Test error %f' % (train_error, test_error))
            print('Training loss: %f - Test loss %f' % (train_loss, test_loss))
            print('')