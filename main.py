import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from model.neural_network import NeuralNetwork
from util.loss import root_mean_squared_error

if __name__ == '__main__':
    # 데이터 로드
    test_csv = pd.read_csv('input/test.csv')
    train_csv = pd.read_csv('input/train.csv')
    y = train_csv['SalePrice']
    # train_csv = train_csv[['MSSubClass', 'OverallQual', 'OverallCond', 'LotArea']]

    # 데이터 전처리
    divider_index = len(train_csv)
    total_csv = pd.concat([train_csv, test_csv])
    total_data = total_csv.dropna(axis=1)
    total_data = pd.get_dummies(total_data)
    train_data = total_data[:divider_index]
    test_data = total_data[divider_index:]

    # 데이터 분할 (테스트 데이터, 검증 데이터)
    train_input, val_input, train_target, val_target = train_test_split(train_data, y, test_size=0.2)

    # 데이터 정규화
    ss_train_scaled = StandardScaler()
    # 학습 데이터
    ss_train_scaled.fit(train_input)
    train_scaled = ss_train_scaled.transform(train_input)
    train_scaled = torch.tensor(train_scaled).float()
    train_target = np.expand_dims(train_target.to_numpy(), 1).astype(np.float32)
    ss_train_target = StandardScaler()
    ss_train_target.fit(train_target)
    train_target = ss_train_target.transform(train_target)
    train_target = torch.tensor(train_target).float()
    train_target = train_target.squeeze()

    # 검증 데이터
    ss_train_scaled.fit(val_input)
    val_scaled = ss_train_scaled.transform(val_input)
    val_scaled = torch.tensor(val_scaled).float()
    val_target = np.expand_dims(val_target.to_numpy(), 1).astype(np.float32)
    ss_train_target.fit(val_target)
    val_target = ss_train_target.transform(val_target)
    val_target = torch.tensor(val_target).float()
    val_target = val_target.squeeze()

    # 테스트 데이터
    ss_train_scaled.fit(test_data)
    test_scaled = ss_train_scaled.transform(test_data)
    test_scaled = torch.tensor(test_scaled).float()

    # 모델 생성
    model = NeuralNetwork(input_length=train_scaled.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 배치 지정
    dataset = TensorDataset(train_scaled, train_target)
    dataloader = DataLoader(dataset, batch_size=2 ** 4, shuffle=True, drop_last=False)

    # 에포크 지정
    epochs = 200
    hist = np.zeros(epochs)
    for t in range(epochs):
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            logits = model(x_train)
            loss = loss_fn(logits.squeeze(), y_train)
            hist[t] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # logits = model(train_scaled)
        # loss.py = loss_fn(logits.squeeze(), train_target)
        # hist[t] = loss.py.item()
        # optimizer.zero_grad()
        # loss.py.backward()
        # optimizer.step()
        if t % 100 == 0:
            print('Epoch %d, Loss %f' % (t, loss.item()))

    # 검증 데이터로 검증
    model.eval()
    result = model.forward(val_scaled)
    score = root_mean_squared_error(val_target, result.squeeze(), ss_train_target)
    print("score:", score.item())

    # 테스트 데이터 예측
    result = model.forward(test_scaled)
    result = result.detach().numpy()
    result = result.reshape(-1)
    result = ss_train_target.inverse_transform(np.expand_dims(result, 1))
    result = pd.DataFrame(result)
    result = result.rename(columns={0: 'SalePrice'})
    id = test_csv['Id']
    result['Id'] = id
    result.to_csv('result.csv', index=False)
