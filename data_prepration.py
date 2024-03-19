import numpy as np
import csv
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse.linalg import eigs


def search_data(sequence_length, num_of_batches, label_start_idx, units, points_per_hour):
    '''
        Parameters
        ----------
        sequence_length: int, length of historical data

        num_of_batches: int, the number of batches will be used for training (ex. number of weeks)

        label_start_idx: int, the index of predicting target

        units: int, week: 7 * 24, day: 24, recent(hour): 1

        points_per_hour: int, number of points per hour, depends on data

        Returns
        ----------
        list[idx]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx > sequence_length:
        return None

    idxes = []
    for i in range(1, num_of_batches + 1):
        idx = label_start_idx - points_per_hour * units * i

        if idx >= 0:
            idxes.append(idx)
        else:
            return None

    if len(idxes) != num_of_batches:
        return None

    return idxes[::-1]


def normalization(train, val, test):
    '''
        Parameters
        ----------
        train, val, test: np.ndarray

        Returns
        ----------
        stats: dict, two keys: mean and std

        train_norm, val_norm, test_norm: np.ndarray, shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm


def get_sample_indices(data, num_of_weeks, num_of_days, num_of_hours, label_start_idx, points_per_hour):

    '''
    Parameters
    ----------
    data: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the index of predicting target

    points_per_hour: int, default 12, number of points per hour

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_of_vertices, num_of_features)
    '''

    temporal_len = data.shape[0]

    week_indices = search_data(temporal_len, num_of_weeks, label_start_idx, 7 * 24, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(temporal_len, num_of_days, label_start_idx, 24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(temporal_len, num_of_hours, label_start_idx, 1, points_per_hour)
    if not hour_indices:
        return None

    week_sample = np.array([data[i] for i in week_indices])  # [unit size][spatial size][feature size]
    day_sample = np.array([data[i] for i in day_indices])  # [unit size][spatial size][feature size]
    hour_sample = np.array([data[i] for i in hour_indices])  # [unit size][spatial size][feature size]

    target = data[label_start_idx]

    return week_sample, day_sample, hour_sample, target


def read_dataset(filename, num_of_weeks, num_of_days, num_of_hours, points_per_hour=12):

    # read data:
    data = np.load(filename)['data']  # [temporal][spatial][features]

    # generate samples:
    all_samples = []
    for idx in range(data.shape[0]):
        sample = get_sample_indices(data, num_of_weeks, num_of_days, num_of_hours, idx, points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample

        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 1, 2))[:, :, 0]
        ))

    split_line1 = int(len(all_samples) * 0.6)  # 60% of the length of all_samples
    split_line2 = int(len(all_samples) * 0.8)  # 80% of the length of all_samples

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]  # 60%
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]  # 20%
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]  # 20%

    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set  # testing_set : [dataset][samples][road]

    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape, train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    (week_stats, train_week_norm, val_week_norm, test_week_norm) = normalization(train_week, val_week, test_week)
    (day_stats, train_day_norm, val_day_norm, test_day_norm) = normalization(train_day, val_day, test_day)
    (recent_stats, train_recent_norm, val_recent_norm, test_recent_norm) = normalization(train_hour, val_hour, test_hour)

    data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    # data = {
    #     'train': {
    #         'week': train_week,
    #         'day': train_day,
    #         'recent': train_hour,
    #         'target': train_target,
    #     },
    #     'val': {
    #         'week': val_week,
    #         'day': val_day,
    #         'recent': val_hour,
    #         'target': val_target
    #     },
    #     'test': {
    #         'week': test_week,
    #         'day': test_day,
    #         'recent': test_hour,
    #         'target': test_target
    #     },
    #     'stats': {
    #         'week': week_stats,
    #         'day': day_stats,
    #         'recent': recent_stats
    #     }
    # }

    return data


def data_loader_creator(device, dataset_name, data, batch_size, shuffle):
    '''

    Parameters
    ----------
    device: string -> cpu or gpu
    dataset_name: string -> 'train' or 'val' or 'test'
    data: "data" numpy array generated in main
    batch_size: int -> number of batches
    shuffle: bool  -> does data need to shuffle or not

    Returns
    -------

    '''

    device = torch.device(device)  # Use "cuda" if you have a GPU and want to use it

    # Convert NumPy arrays to PyTorch tensors and move them to the specified device
    week = torch.tensor(data[dataset_name]['week'], device=device)
    day = torch.tensor(data[dataset_name]['day'], device=device)
    recent = torch.tensor(data[dataset_name]['recent'], device=device)
    target = torch.tensor(data[dataset_name]['target'], device=device)

    # Create a TensorDataset from the tensors
    dataset = TensorDataset(week, day, recent, target)

    # Create a DataLoader for the training dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def adjacency_matrix(filename, num_of_vertices):
    '''
        Parameters
        ----------
        filename: str, path of the csv file contains edges information

        num_of_vertices: int, the number of vertices

        Returns
        ----------
        A: np.ndarray, adjacency matrix

    '''

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        f.__next__()  # pass header
        edges = [(int(i[0]), int(i[1])) for i in reader]

    adj = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    for i, j in edges:
        adj[i, j] = 1

    return adj


def scaled_laplacian_matrix(w):
    '''
        compute \tilde{L}

        Parameters
        ----------
        w: np.ndarray, adjacency matrix with shape of (N, N), N is the num of vertices

        Returns
        ----------
        scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert w.shape[0] == w.shape[1]

    d = np.diag(np.sum(w, axis=1))
    lap = d - w

    lambda_max = eigs(lap, k=1, which='LR')[0].real  # scaled

    return (2 * lap) / lambda_max - np.identity(w.shape[0])


def cheb_polynomial(lp_matrix, k):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    lp_matrix: scaled Laplacian, np.ndarray, shape (N, N)

    k: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    n = lp_matrix.shape[0]
    cheb_polynomials = [np.identity(n), lp_matrix.copy()]

    for i in range(2, k):
        cheb_polynomials.append(2 * lp_matrix * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials
