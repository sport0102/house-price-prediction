import torch


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor):
        self.mean = data.mean(0, keepdim=True)
        self.std = data.std(0, unbiased=False, keepdim=True)

    def transform(self, data: torch.Tensor):
        data -= self.mean
        data /= (self.std + 1e-7)
        return data

    def fit_transform(self, data: torch.Tensor):
        self.fit(data.float())
        return self.transform(data.float())

    def inverse_transform(self, data: torch.Tensor):
        return data * self.std + self.mean
