import numpy as np
import torch


def root_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor):
    return np.sqrt(np.mean(np.square(np.log(y_true.detach().numpy()) - np.log(y_pred.detach().numpy()))))
