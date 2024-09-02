import numpy as np
import csv
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse.linalg import eigs
import pandas as pd

# for avoiding test overflow warning
# import random
# seed = 1472378598
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# for avoiding test overflow warning


def initialize_missing_values(data):
    # Create a mask for NaNs and negative values
    mask = ~(np.isnan(data) | (data < 0))
    # Replace NaNs and negative values with 0
    data_filled = np.where(mask, data, 0.0)
    return data_filled, mask

def search_data(sequence_length, num_of_batches, label_start_idx, units, points_per_hour):
    '''
        Parameters
        ----------
        sequence_length: int, length of historical data

        num_of_batches: int, the number of batches will be used for training (ex. number of weeks)

        label_start_idx: int, the index of target

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


def get_sample_indices(data, num_of_weeks, num_of_days, num_of_hours, target_idx, points_per_hour):

    """
    Parameters
    ----------
    data: np.ndarray
                   shape is (num_of_vertices, sequence_length)

    num_of_weeks, num_of_days, num_of_hours: int

    target_idx: int, the index of target (e.g. current time)

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
    """

    temporal_len = data.shape[1]
    week_indices = search_data(temporal_len, num_of_weeks, target_idx, 7 * 24, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(temporal_len, num_of_days, target_idx, 24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(temporal_len, num_of_hours, target_idx, 1, points_per_hour)
    if not hour_indices:
        return None

    data = data.T  # size: timesteps * counters

    week_sample = np.array([data[i] for i in week_indices])  # [unit size][spatial size][feature size]
    day_sample = np.array([data[i] for i in day_indices])  # [unit size][spatial size][feature size]
    hour_sample = np.array([data[i] for i in hour_indices])  # [unit size][spatial size][feature size]
    target = data[target_idx]

    return week_sample.T, day_sample.T, hour_sample.T, target, week_indices, day_indices, hour_indices, target_idx


def arti_missing_generator(sample_data, mask, arti_missing_ratio):
    """
    Generate a new sample and mask with artificially missing values.

    Parameters
    ----------
    sample_data : input sample with shape of number_of_vertices
    mask : the mask indicating observed (True) and missing (False) values.
    arti_missing_ratio : int - The percentage of observed values to mark as artificially missing.

    Returns
    -------
    new_sample : new sample with artificial missing values.
    new_mask : a new mask with values 0, 0.5, and 1.
        0 for originally missing values.
        0.5 for the selected artificial missing values.
        1 for the remaining observed values.
    """

    num_observed = mask.astype(float).sum()  # number of observed values
    observed_indices = np.where(mask == True)[0]  # indices of observed values

    arti_missing_fraction = int(num_observed * (arti_missing_ratio / 100.0))

    # randomly select indices to be marked as artificially missing
    arti_missing_indices = np.random.choice(
        observed_indices,
        size=arti_missing_fraction,
        replace=False
    )

    new_mask = mask.astype(float).flatten()
    new_mask[arti_missing_indices] = 0.5

    new_sample = sample_data.copy()
    new_sample.flatten()[arti_missing_indices] = 0  # set artificial missing values to zero

    return new_sample, new_mask



def read_dataset(filename, num_of_weeks, num_of_days, num_of_hours, data_per_hour, arti_missing_ratio):
    """
    arti_missing_ratio: int, The percentage of the observed values in each sample which take as artificial
     missing values
    """

    # read data:
    roads_names = []
    average_speeds = []
    with h5py.File(filename, 'r') as f:
        for road in f['average_speed']:
            roads_names.append(road)
            average_speeds.append(f['average_speed'][road][:])

        roads_names = np.array(roads_names)
        average_speeds = np.array(average_speeds)  # size: nodes * timesteps
        timestamps = list(f['timestamps'][:])
        timestamps = [pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S') for ts in np.concatenate(timestamps)]

        # if average_speeds.shape[0] != len(timestamps):
        #     average_speeds = average_speeds.T # size: timesteps * nodes

    # initializing misssing values
    average_speeds, mask = initialize_missing_values(average_speeds)

    # generate samples:
    all_samples = []

    for idx in range(len(timestamps)): #time
        sample = get_sample_indices(average_speeds, num_of_weeks, num_of_days, num_of_hours, idx, data_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, current, week_indices, day_indices, hour_indices, current_idx = sample

        # artificial missing values generation:
        current_rt, mask_arti = arti_missing_generator(current, mask[:, current_idx], arti_missing_ratio)

        # mask of historical data
        week_mask = mask[:, week_indices].astype(float)
        day_mask = mask[:, day_indices].astype(float)
        hour_mask = mask[:, hour_indices].astype(float)

        all_samples.append((
            week_sample,  # shape: (number_of_vertices, num_of_weeks)
            day_sample,  # shape: (number_of_vertices, num_of_days)
            hour_sample,   # shape: (number_of_vertices, num_of_hours)
            current_rt,  # shape: (number_of_vertices) including artificial missing values
            current,  # shape: (number_of_vertices)
            week_mask,  # shape: (number_of_vertices, num_of_weeks)
            day_mask,  # shape: (number_of_vertices, num_of_days)
            hour_mask,   # shape: (number_of_vertices, num_of_hours)
            mask_arti  # shape: number_of_vertices  - filled with True and False
        ))

    split_line1 = int(len(all_samples) * 0.6)  # 60% of the length of all_samples
    split_line2 = int(len(all_samples) * 0.8)  # 80% of the length of all_samples

    training_set = all_samples[:split_line1]  # 60%
    validation_set = all_samples[split_line1: split_line2]  # 20%
    testing_set = all_samples[split_line2:]  # 20%

    train_week, train_day, train_hour, train_current, train_real, train_week_mask, train_day_mask, train_hour_mask, train_mask = map(np.array, zip(*training_set))
    val_week, val_day, val_hour, val_current, val_real, val_week_mask, val_day_mask, val_hour_mask, val_mask = map(np.array, zip(*validation_set))
    test_week, test_day, test_hour, test_current, test_real, test_week_mask, test_day_mask, test_hour_mask, test_mask = map(np.array, zip(*testing_set))

    print('training data: week: {}, day: {}, recent: {}, current: {}, real:{}, mask: {}'.format(
        train_week.shape, train_day.shape, train_hour.shape, train_current.shape, train_real.shape, train_mask.shape))
    print('validation data: week: {}, day: {}, recent: {}, current: {}, real:{}, mask: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_current.shape, val_real.shape, val_mask.shape))
    print('testing data: week: {}, day: {}, recent: {}, current: {}, real:{}, mask: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_current.shape, test_real.shape, test_mask.shape))

    (week_stats, train_week_norm, val_week_norm, test_week_norm) = normalization(train_week, val_week, test_week)
    (day_stats, train_day_norm, val_day_norm, test_day_norm) = normalization(train_day, val_day, test_day)
    (recent_stats, train_recent_norm, val_recent_norm, test_recent_norm) = normalization(train_hour, val_hour, test_hour)

    data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'current': train_current,
            'real': train_real,
            'week_mask': train_week_mask,
            'day_mask': train_day_mask,
            'hour_mask': train_hour_mask,
            'mask': train_mask
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'current': val_current,
            'real': val_real,
            'week_mask': val_week_mask,
            'day_mask': val_day_mask,
            'hour_mask': val_hour_mask,
            'mask': val_mask
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'current': test_current,
            'real': test_real,
            'week_mask': test_week_mask,
            'day_mask': test_day_mask,
            'hour_mask': test_hour_mask,
            'mask': test_mask
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    return data, roads_names


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
    current = torch.tensor(data[dataset_name]['current'], device=device)
    real = torch.tensor(data[dataset_name]['real'], device=device)
    week_mask = torch.tensor(data[dataset_name]['week_mask'], device=device)
    day_mask = torch.tensor(data[dataset_name]['day_mask'], device=device)
    hour_mask = torch.tensor(data[dataset_name]['hour_mask'], device=device)
    mask = torch.tensor(data[dataset_name]['mask'], device=device)

    # Create a TensorDataset from the tensors
    dataset = TensorDataset(week, day, recent, current, real, week_mask, day_mask, hour_mask, mask)

    # Create a DataLoader for the training dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def adjacency_matrix(filename, num_of_vertices, roads_names):
    '''
        Parameters
        ----------
        filename: str, path of the xlsx file contains edges information

        num_of_vertices: int, the number of vertices

        Returns
        ----------
        A: np.ndarray, adjacency matrix

    '''

    df = pd.read_excel(filename, engine='openpyxl')
    edges = [(row[0], row[1]) for row in df.itertuples(index=False)]


    adj = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    for origin, destination in edges:
        i = np.argwhere(roads_names == origin)[0][0]
        j = np.argwhere(roads_names == destination)[0][0]
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
