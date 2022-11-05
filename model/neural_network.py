import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, input_length, loss_fn_name='mse'):
        super(NeuralNetwork, self).__init__()
        loss_fn = {
            'mse': nn.MSELoss(),
        }
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_length, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32, bias=True),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32, 16, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.loss_fn = loss_fn[loss_fn_name]
        self.optimizer = None
        self.scheduler = None
        self.dataset = None
        self.dataloader = None
        self.history = None

    def set_optimizer(self, optimizer_name, learning_rate):
        optimizer = {
            'adam': torch.optim.Adam(self.parameters(), lr=learning_rate),
            'sgd': torch.optim.SGD(self.parameters(), lr=learning_rate),
        }
        self.optimizer = optimizer[optimizer_name]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 ** epoch)

    def set_scheduler(self, scheduler_name, gamma):
        scheduler = {
            'lambda': torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: gamma ** epoch),
        }
        self.scheduler = scheduler[scheduler_name]

    def set_dataset(self, train_input, train_target):
        self.dataset = TensorDataset(train_input, train_target)

    def set_dataloader(self, batch_size, shuffle=True, drop_last=False):
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def init_weights(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.01)

    def init_model(self):
        self.linear_relu_stack.apply(self.init_weights)
        self.train()

    def forward(self, x):
        return self.linear_relu_stack(x)

    def run(self, epochs):
        self.history = np.zeros(epochs)
        for t in range(epochs):
            for batch_idx, samples in enumerate(self.dataloader):
                x_train, y_train = samples
                logits = self(x_train)
                loss = self.loss_fn(logits.squeeze(), y_train)
                self.history[t] = loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            if t % 100 == 0:
                print('Epoch %d, Loss %f' % (t, loss.item()))

    def draw_history(self):
        plt.plot(self.history)
        plt.show()
