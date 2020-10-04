import numpy as np


def get_window(dataset, start_index, look_back, look_forward,
               columns, shift=1, end_index=None, feature_columns=None,
               label_columns=None, single_step=False):
    """
    Function to create windowed input output pairs from timeseries dataset

    :param dataset: Dataset to be windowed
    :param start_index: Index to start windows
    :param look_back: Input sequence length
    :param look_forward: Output sequence length to be forecast
    :param shift: How many time steps to shift the windows, default 1
    :param columns: list: a list of all column names in the dataset including the label(s) column
    :param end_index: Index to stop windows, by default last time step in series
    :param feature_columns: features to include in input window sequence, create indexed variable, ie columns[:250]
    :param label_columns: features to include in output window sequence, create indexed variable, ie columns[:-1]
    :param single_step: If true, only forecast next step
    :return: Input and output windows as numpy arrays
    """

    data = []
    labels = []

    column_indices = {name: i for i, name in enumerate(columns)}

    start_index = start_index + look_back
    if end_index is None:
        end_index = len(dataset) - look_forward

    for i in range(start_index, end_index):
        indices = range(i - look_back, i, shift)
        if feature_columns is None:
            data.append(dataset[indices])
        else:
            data.append(np.array([dataset[:, column_indices[name]]
                                  for name in feature_columns]).T[indices])

        if label_columns is not None:
            targets = np.array([dataset[:, column_indices[name]]
                                for name in label_columns]).T

            if single_step:
                labels.append(targets[i + look_forward])
            else:
                labels.append(targets[i:i + look_forward])

        else:
            if single_step:
                labels.append(dataset[i + look_forward])
            else:
                labels.append(dataset[i:i + look_forward])

    return np.array(data), np.array(labels)
