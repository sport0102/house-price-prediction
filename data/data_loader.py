import pandas as pd
import torch
from sklearn.model_selection import train_test_split


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

