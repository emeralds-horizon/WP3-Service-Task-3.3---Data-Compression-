import csv
import numpy as np
from scipy.sparse.linalg import eigs
from metrics import mean_absolute_error, mean_squared_error, masked_mape_np


def compute_val_loss(net, val_loader, loss_function, sw, epoch):
    '''
    compute mean loss on validation set

    Parameters
    ----------
    net: model

    val_loader: gluon.data.DataLoader

    loss_function: func

    sw: mxboard.SummaryWriter

    epoch: int, current epoch

    '''
    val_loader_length = len(val_loader)
    tmp = []
    for index, (val_w, val_d, val_r, val_t) in enumerate(val_loader):
        output = net([val_w, val_d, val_r])
        l = loss_function(output, val_t)
        tmp.extend(l.asnumpy().tolist())
        print('validation batch %s / %s, loss: %.2f' % (
            index + 1, val_loader_length, l.mean().asscalar()))

    validation_loss = sum(tmp) / len(tmp)
    sw.add_scalar(tag='validation_loss',
                  value=validation_loss,
                  global_step=epoch)
    print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))


def predict(net, test_loader):
    '''
    predict

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    Returns
    ----------
    prediction: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''

    test_loader_length = len(test_loader)
    prediction = []
    for index, (test_w, test_d, test_r, _) in enumerate(test_loader):
        prediction.append(net([test_w, test_d, test_r]).asnumpy())
        print('predicting testing set batch %s / %s' % (index + 1,
                                                        test_loader_length))
    prediction = np.concatenate(prediction, 0)
    return prediction


def evaluate(net, test_loader, true_value, num_of_vertices, sw, epoch):
    '''
    compute MAE, RMSE, MAPE scores of the prediction
    for 3, 6, 12 points on testing set

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    true_value: np.ndarray, all ground truth of testing set
                shape is (num_of_samples, num_for_predict, num_of_vertices)

    num_of_vertices: int, number of vertices

    sw: mxboard.SummaryWriter

    epoch: int, current epoch

    '''
    prediction = predict(net, test_loader)
    prediction = (prediction.transpose((0, 2, 1))
                  .reshape(prediction.shape[0], -1))
    for i in [3, 6, 12]:
        print('current epoch: %s, predict %s points' % (epoch, i))

        mae = mean_absolute_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices])
        rmse = mean_squared_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices]) ** 0.5
        mape = masked_mape_np(true_value[:, : i * num_of_vertices],
                              prediction[:, : i * num_of_vertices], 0)

        print('MAE: %.2f' % (mae))
        print('RMSE: %.2f' % (rmse))
        print('MAPE: %.2f' % (mape))
        print()
        sw.add_scalar(tag='MAE_%s_points' % (i),
                      value=mae,
                      global_step=epoch)
        sw.add_scalar(tag='RMSE_%s_points' % (i),
                      value=rmse,
                      global_step=epoch)
        sw.add_scalar(tag='MAPE_%s_points' % (i),
                      value=mape,
                      global_step=epoch)


def weighted_kl(y_pred, y_true, weight, epsilon=1e-3):  # Kullback-Leibler (KL)
    N, M, B = y_pred.size()  # N: batches |  M: nodes  | B: speed buckets
    w_N, w_M = weight.size()  # w_N: batches  |  w_M: nodes
    assert w_N == N, w_M == M
    log_pred = torch.log10(y_pred + epsilon)
    log_true = torch.log10(y_true + epsilon)
    log_sub = torch.subtract(log_pred, log_true)
    mul_op = torch.multiply(y_pred, log_sub)
    sum_hist = torch.sum(mul_op, dim=2)
    if weight is not None:
        sum_hist = torch.multiply(weight, sum_hist)
    weight_avg_kl_div = torch.sum(sum_hist)
    avg_kl_div = weight_avg_kl_div / torch.sum(weight)

    return avg_kl_div
