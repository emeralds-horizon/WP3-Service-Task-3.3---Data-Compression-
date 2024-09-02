import numpy as np
import torch


def masked_mape_np(y_true, y_est, mask):  # MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask.cpu().detach().numpy().astype('float32')
        mask /= np.mean(mask)

        # detach tensor to create numpy
        y_true_detached = y_true.cpu().detach().numpy()
        y_est_detached = y_est.cpu().detach().numpy()

        mape = np.abs(np.divide(np.subtract(y_est_detached, y_true_detached).astype('float32'), y_true_detached))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100.


def mean_absolute_error(y_true, y_est, mask):  # MAE
    """
    mean absolute error

    Parameters
    ----------
    y_true, y_est: torch.Tensor, shape is (batch_size, num_of_vertices)

    Returns
    ----------
    torch.float64
    """
    mae = torch.abs(y_true - y_est)
    mae = mae * mask
    return mae


def mean_squared_error(y_true, y_est, mask):  # RMSE
    """
    Mean Squared Error

    Parameters
    ----------
    y_true, y_est: torch.Tensor, shape is (batch_size, num_of_features)

    Returns
    ----------
    torch.float64
    """

    # rmse = torch.mean((y_true - y_est) ** 2)
    # rmse = rmse * mask

    rmse = (y_true - y_est) ** 2
    rmse = (rmse * mask).sum() / mask.sum()
    return rmse
