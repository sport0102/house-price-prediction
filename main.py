import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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


def root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.square(torch.log(y_pred) - torch.log(y_true))))


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

    # 데이터 분할 (테스트 데이터, 검증 데이터)
    train_input, val_input, train_target, val_target = train_test_split(train_data, y, test_size=0.2)
    print(train_input.shape, val_input.shape, train_target.shape, val_target.shape)
    print(type(train_input), type(val_input), type(train_target), type(val_target))

    # 데이터 정규화
    ss = StandardScaler()
    # 학습 데이터
    ss.fit(train_input)
    train_scaled = ss.transform(train_input)
    train_scaled = torch.tensor(train_scaled).float()
    train_target = torch.tensor(train_target.to_numpy()).float()
    # 검증 데이터
    ss.fit(val_input)
    val_scaled = ss.transform(val_input)
    val_scaled = torch.tensor(val_scaled).float()
    val_target = torch.tensor(val_target.to_numpy()).float()
    # 테스트 데이터
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

    # 검증 데이터로 검증
    result = model.forward(val_scaled)
    score = root_mean_squared_error(val_target, result.squeeze())
    print("score:", score.item())

    # 테스트 데이터 예측
    # result = model.forward(test_scaled)
    # result = result.detach().numpy()
    # result = result.reshape(-1)
    # result = pd.DataFrame(result)
    # result = result.rename(columns={0: 'SalePrice'})
    # id = test_csv['Id']
    # result['Id'] = id
    # result.to_csv('result.csv', index=False)
