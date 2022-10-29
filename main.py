import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from data.data_manager import DataManager
from model.neural_network import NeuralNetwork
from util.loss import root_mean_squared_error

if __name__ == '__main__':
    # 데이터 로드
    train_data_file_path = 'input/train.csv'
    test_data_file_path = 'input/test.csv'
    target_column = 'SalePrice'
    data_manager = DataManager(train_data_file_path=train_data_file_path,
                               test_data_file_path=test_data_file_path,
                               target_column=target_column)

    # 데이터 전처리
    data_manager.preprocess()

    # 검증 데이터 분할
    data_manager.split_validation_data()

    # 데이터 정규화
    data_manager.normalization(scaler='standard')

    train_input, train_target, valid_input, valid_target, test_input = data_manager.get_data()

    # 모델 생성
    model = NeuralNetwork(input_length=train_input.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 배치 지정
    dataset = TensorDataset(train_input, train_target)
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
        if t % 100 == 0:
            print('Epoch %d, Loss %f' % (t, loss.item()))

    # 검증 데이터로 검증
    model.eval()
    result = model.forward(valid_input)
    valid_target = data_manager.get_inverse_transform_data(valid_target, scaler_type='target')
    valid_result = data_manager.get_inverse_transform_data(result.squeeze(), scaler_type='target')
    score = root_mean_squared_error(valid_target, valid_result)
    print("score:", score.item())

    # 테스트 데이터 예측
    result = model.forward(test_input)
    result = data_manager.get_inverse_transform_data(result.squeeze(), scaler_type='target')

    # csv 포맷으로 저장
    export_file_path = 'result.csv'
    data_manager.export_csv(export_file_path, result)
