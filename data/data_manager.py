import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data.scaler.standard_scaler import StandardScaler


class DataManager:
    def __init__(self, train_data_file_path, test_data_file_path, target_column):
        self.train_df = pd.read_csv(train_data_file_path)
        self.test_df = pd.read_csv(test_data_file_path)
        self.train_df_target = self.train_df[target_column]
        self.train_input = None
        self.train_target = None
        self.valid_input = None
        self.valid_target = None
        self.test_input = None
        self.train_scaler = None
        self.target_scaler = None

    def set_use_column(self, use_column):
        self.train_df = self.train_df[use_column]
        self.test_df = self.test_df[use_column]

    def preprocess(self):
        divider_index = len(self.train_df)
        total_csv = pd.concat([self.train_df, self.test_df])
        total_data = total_csv.dropna(axis=1)
        total_data = pd.get_dummies(total_data)
        self.train_input = torch.tensor(total_data[:divider_index].values)
        self.train_target = torch.tensor(self.train_df_target)
        self.test_input = torch.tensor(total_data[divider_index:].values)

    def split_validation_data(self, test_size: float = 0.2):
        train_input, val_input, train_target, val_target = train_test_split(self.train_input, self.train_target,
                                                                            test_size=test_size)
        self.train_input = train_input
        self.train_target = train_target
        self.valid_input = val_input
        self.valid_target = val_target

    def normalization(self, scaler):
        scaler_map = {
            'standard': self.normalization_with_standard_scaler
        }
        scaler_map[scaler]()

    def normalization_with_standard_scaler(self):
        self.train_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.train_input = self.train_scaler.fit_transform(self.train_input)
        self.valid_input = self.train_scaler.fit_transform(self.valid_input)
        self.test_input = self.train_scaler.fit_transform(self.test_input)

        self.train_target = self.target_scaler.fit_transform(self.train_target)
        self.valid_target = self.target_scaler.fit_transform(self.valid_target)

    def get_inverse_transform_data(self, data, scaler_type):
        scaler_type_map = {
            'train': self.train_scaler,
            'target': self.target_scaler
        }
        return scaler_type_map[scaler_type].inverse_transform(data)

    def get_data(self):
        return self.train_input, self.train_target, self.valid_input, self.valid_target, self.test_input

    def merge_train_valid_data(self):
        self.train_input = torch.cat([self.train_input, self.valid_input])
        self.train_target = torch.cat([self.train_target, self.valid_target])
        return self.train_input, self.train_target

    def export_csv(self, export_file_path, result):
        result = pd.DataFrame(result.detach().numpy())
        result = result.rename(columns={0: 'SalePrice'})
        result['Id'] = self.test_df['Id']
        result.to_csv(export_file_path, index=False)
