import numpy as np


def root_mean_squared_error(y_true, y_pred, ss):
    return np.sqrt(np.mean(np.square(np.log(ss.inverse_transform(np.expand_dims(y_pred.detach().numpy(), 1))) - np.log(
        ss.inverse_transform(np.expand_dims(y_true.detach().numpy(), 1))))))
