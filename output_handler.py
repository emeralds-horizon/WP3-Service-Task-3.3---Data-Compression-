import pandas as pd
import metrics
import torch


def segments_error(pred, real, file_name):
    """
    :param pred: predicted values for each road segment, list of tensors
    :param real: real value of road segments, list of tensors
    :param file_name: file name and location for saving, string
    :return:
    csv file, containing error value of each road segment per data sample
    """

    yhat = torch.stack(pred, dim=1).numpy()
    ytrue = torch.stack(real, dim=1).numpy()

    # mape = metrics.masked_mape_np(output, t_real, 0.0).item()
    # rmse = metrics.mean_squared_error(output, t_real).item()

    # Flatten the tensors for DataFrame creation
    yhat_flat = yhat.reshape(-1, yhat.shape[-1])
    ytrue_flat = ytrue.reshape(-1, ytrue.shape[-1])

    # Create DataFrames directly from flattened arrays
    df_pred = pd.DataFrame(data=yhat_flat, columns=[f'S_{i + 1}' for i in range(yhat_flat.shape[1])])
    df_real = pd.DataFrame(data=ytrue_flat, columns=[f'S_{i + 1}' for i in range(ytrue_flat.shape[1])])

    # Round values in df_pred to two decimal points
    df_pred = df_pred.round(2)

    # Save as Excel with separate sheets for predicted and real values
    with pd.ExcelWriter(file_name) as writer:
        df_pred.to_excel(writer, index=False, sheet_name='Pred Sheet')
        df_real.to_excel(writer, index=False, sheet_name='Real Sheet')
