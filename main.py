import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(150, 64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(64, 32, bias=True),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


if __name__ == '__main__':
    # 데이터 로드
    test_csv = pd.read_csv('input/test.csv')
    train_csv = pd.read_csv('input/train.csv')
    y = train_csv['SalePrice']

    # 데이터 전처리
    divider_index = len(train_csv)
    total_csv = pd.concat([train_csv, test_csv])
    total_data = total_csv.dropna(axis=1)
    total_data = pd.get_dummies(total_data)
    train_data = total_data[:divider_index]
    test_data = total_data[divider_index:]

    # 데이터 정규화
    ss = StandardScaler()
    ss.fit(train_data)
    train_scaled = ss.transform(train_data)
    train_scaled = torch.tensor(train_scaled).float()
    train_target = torch.tensor(y).float()

    ss.fit(test_data)
    test_scaled = ss.transform(test_data)
    test_scaled = torch.tensor(test_scaled).float()

    model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.15)

    epochs = 50000
    hist = np.zeros(epochs)
    for t in range(epochs):
        logits = model(train_scaled)
        loss = loss_fn(logits.squeeze(), train_target)
        if t % 100 == 99:
            print(t, loss.item())
        hist[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    result = model.forward(test_scaled)
    result = result.detach().numpy()
    result = result.reshape(-1)
    result = pd.DataFrame(result)
    result = result.rename(columns={0: 'SalePrice'})
    id = test_csv['Id']
    result['Id'] = id
    result.to_csv('result.csv', index=False)
