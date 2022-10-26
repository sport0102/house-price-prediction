import pandas as pd


class DataLoader:
    def __init__(self, train_file_path, test_file_path):
        pd.read_csv(file_path)
        self.train_input = None

    def read_csv(self, file_path):
        df = pd.read_csv(file_path)
        return df
