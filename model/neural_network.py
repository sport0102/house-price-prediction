from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_length):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_length, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
