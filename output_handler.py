import pandas as pd
import metrics
import torch


def mask_replacement(est, real, mask):
    """
    Replaces values in the `est` tensor with values from the `real` tensor wherever the `mask` tensor has positive values.

    :param est: list of tensors, estimated values for each road segment
    :param real: list of tensors, real values for each road segment
    :param mask: list of tensors, a mask indicating where replacements should occur (positive values).
        a mask with values 0, 0.5, and 1.
        0 for originally missing values.
        0.5 for the selected artificial missing values.
        1 for the remaining observed values.

    Returns:
    - torch.Tensor: list of tensors with values from `real` replacing those in `est` where `mask` is positive.

    """
    est_real_mix_lst = []

    for e, r, m in zip(est, real, mask):
        # where mask > 0, replace est with real
        # where mask == 1 
        masked_est = torch.where(m > 0, r, e)
        est_real_mix_lst.append(masked_est)

    return est_real_mix_lst


def segments_error(est, real, mask, file_name):
    """
    :param est: estimated values for each road segment, list of tensors
    :param real: real value of road segments, list of tensors
    :param mask : list of masks. a mask with values 0, 0.5, and 1.
        0 for originally missing values.
        0.5 for the selected artificial missing values.
        1 for the remaining observed values.
    :param file_name: file name and location for saving, string
    :return:
    csv file, containing error value of each road segment per data sample
    """

    masked_est = mask_replacement(est, real, mask)
    yhat = torch.stack(masked_est, dim=1).numpy()
    ytrue = torch.stack(real, dim=1).numpy()

    # mape = metrics.masked_mape_np(output, t_real, 0.0).item()
    # rmse = metrics.mean_squared_error(output, t_real).item()

    # Flatten the tensors for DataFrame creation
    yhat_flat = yhat.reshape(-1, yhat.shape[-1])
    ytrue_flat = ytrue.reshape(-1, ytrue.shape[-1])

    # Create DataFrames directly from flattened arrays
    df_est = pd.DataFrame(data=yhat_flat, columns=[f'S_{i + 1}' for i in range(yhat_flat.shape[1])])
    df_real = pd.DataFrame(data=ytrue_flat, columns=[f'S_{i + 1}' for i in range(ytrue_flat.shape[1])])

    # Round values in df_est to two decimal points
    df_est = df_est.round(2)

    # Save as Excel with separate sheets for predicted and real values
    with pd.ExcelWriter(file_name) as writer:
        df_est.to_excel(writer, index=False, sheet_name='Pred Sheet')
        df_real.to_excel(writer, index=False, sheet_name='Real Sheet')
