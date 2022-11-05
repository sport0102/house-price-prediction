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
    # data_manager.set_use_column(
    #     ['MSSubClass', 'MSZoning', 'LotShape', 'LotArea', 'LandContour', 'Condition1', 'BldgType', 'HouseStyle',
    #      'OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtFinType1', 'BsmtFullBath', 'FullBath',
    #      'TotRmsAbvGrd', 'Functional', 'GarageCars', 'GarageQual', 'PoolQC'])
    data_manager.set_use_column(
        ['LandContour', 'Condition1', 'BldgType', 'HouseStyle',
         'OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtFinType1', 'BsmtFullBath',
         'Functional', 'GarageCars', 'GarageQual', 'PoolQC'])


    # 데이터 전처리
    data_manager.preprocess()

    # 검증 데이터 분할
    data_manager.split_validation_data()

    # 데이터 정규화
    data_manager.normalization(scaler='standard')

    train_input, train_target, valid_input, valid_target, test_input = data_manager.get_data()

    # 모델 생성
    batch_size = 2 ** 4
    epoch = 1000
    model = NeuralNetwork(input_length=train_input.shape[1], loss_fn_name='mse')
    model.train()
    model.set_optimizer(optimizer_name='adam', learning_rate=0.001)
    model.set_scheduler(scheduler_name='lambda', gamma=0.99)
    model.set_dataset(train_input, train_target)
    model.set_dataloader(batch_size=batch_size)
    model.run(epochs=epoch)
    model.draw_history()

    # 검증 데이터로 검증
    model.eval()
    result = model.forward(valid_input)
    valid_target = data_manager.get_inverse_transform_data(valid_target, scaler_type='target')
    valid_result = data_manager.get_inverse_transform_data(result.squeeze(), scaler_type='target')
    score = root_mean_squared_error(valid_target, valid_result)
    print("score:", score.item())

    # 검증 데이터 합쳐서 학습
    train_input, train_target = data_manager.merge_train_valid_data()
    epoch = 50
    model.set_optimizer(optimizer_name='adam', learning_rate=0.00001)
    model.set_dataset(train_input, train_target)
    model.set_dataloader(batch_size=batch_size)
    model.run(epochs=epoch)
    model.draw_history()

    # 테스트 데이터 예측
    model.eval()
    result = model.forward(test_input)
    result = data_manager.get_inverse_transform_data(result.squeeze(), scaler_type='target')

    # csv 포맷으로 저장
    export_file_path = 'result.csv'
    data_manager.export_csv(export_file_path, result)
