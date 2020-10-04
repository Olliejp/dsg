import pandas as pd
import numpy as np
import os
from scipy import interpolate
from utils import add_NSG_columns_and_sort_columns_alphabetically


def remove_row(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """
    Remove row based on index.
    """
    where = df.index == idx
    df = df[~where]
    return df


def convert_objects_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns with object dtype to floats in order to use them in models.
    """
    indicator = df.dtypes == 'object'
    categorical_columns = df.columns[indicator].tolist()
    for col in categorical_columns:
        df[col] = df[col].astype('float')
    return df


def interpolation_reconstruction_delta_T(df: pd.DataFrame, delta_T: float = 0.01) -> pd.DataFrame:
    """
    Compute interpolated timestamp for each scenario, with kind=`slinear`.
    The interpolation is made on a regular grid of x-values spanning the same interval as during the simulatio with a delta_T specified by user.
    IMPORTANT - ASSUMES time.min() < 1.08

    Parameters
    ----------
    df : pd.DataFrame
        Simulation dataframe
    delta_T : float
        The time difference between consecutive data.
    """
    list_concat = []
    for col in df.columns[1:]:
        X = df['All calculations', 'Time in s'].values
        y = df[col].values
        f = interpolate.interp1d(X, y, kind='slinear')
        xgrid = np.arange(1.08, X.max(), delta_T)
        ygrid = f(xgrid)
        to_concat = pd.DataFrame(
            {('All calculations', 'Time in s'): xgrid, col: ygrid})
        list_concat.append(to_concat)
    dfs = [df.set_index(('All calculations', 'Time in s'))
           for df in list_concat]
    interpolate_construction = pd.concat(dfs, axis=1)
    return interpolate_construction.reset_index()


def pre_normalisation_mean_and_std(df: pd.DataFrame) -> Tuple(float, float):
    """
    find mean and std of a given df for
    normalisation purposes
    """
    x = df.values.astype(float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, std


def find_means_and_stds(
    sample_size: int,
    root_dir: str,
    simulation_files_dict: Dict[int, str]
) -> Tuple(np.array, np.array):
    """
    find the average mean and std of each column from
    a sample of size sample_size
    run this before running clean_batch_of_raw_data()
    """
    means = []
    stds = []
    for i in range(sample_size):
        df = pd.read_csv(os.path.join(root_dir,
                                      simulation_files_dict[i+1]), sep=',', header=[0, 1])
        df = convert_objects_to_float(df)
        df = add_NSG_columns_and_sort_columns_alphabetically(df)
        df = interpolation_reconstruction_delta_T(df, delta_T=0.01)
        mean, std = pre_normalisation_mean_and_std(df)
        means.append(mean)
        stds.append(std)
    return np.concatenate((means)).reshape(sample_size, 250).mean(axis=0).reshape(1, 250), np.concatenate((stds)).reshape(sample_size, 250).mean(axis=0).reshape(1, 250)


def normalise_df(
    df: pd.DataFrame,
    means: np.array,
    stds: np.array
) -> pd.DataFrame:
    """
    normalise a single dataframe column by column
    it is probably possible to vectorise this and make it faster...
    means and stds can either be saved to the workspace or can
    change clean_batch_of_raw_data() to take them as inputs
    """
    for i in range(1, 250):
        df.iloc[:, i] = df.iloc[:, i].astype(float) - means[0][i]
        if stds[0][i] == 0:  # some variables may return zero st. dev. for certain batches
            continue
        else:
            df.iloc[:, i] = df.iloc[:, i].astype(float)/stds[0][i]
    return df


def remove_disturbance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all data before the disturbance happens,
    it is said to happen at 1.075s
    """
    where = df['All calculations', 'Time in s'].astype(float) > 1.075
    return df[where]


def angle_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform angle data: theta -> (sin(theta), cos(theta))
    creates two new columns and removes the angle column
    note that this adds 10 new columns to the dataframe
    """
    generators = ["G 01", "G 02", "G 03", "G 04",
                  "G 05", "G 06", "G 07", "G 08", "G 09", "G 10"]

    for gen in generators:
        sinangle = np.sin(
            (np.pi/180)*df.loc[:, (gen, "Rotor angle with reference to reference machine angle in deg")])
        cosangle = np.cos(
            (np.pi/180)*df.loc[:, (gen, "Rotor angle with reference to reference machine angle in deg")])
        columns = df.columns
        sinangleind = columns.get_loc(
            (gen, "Rotor angle with reference to reference machine angle in deg"))
        df = df.drop(
            (gen, "Rotor angle with reference to reference machine angle in deg"), axis=1)
        df.insert(sinangleind, (gen, "sin of rotor angle"), sinangle)
        df.insert(sinangleind+1, (gen, "cos of rotor angle"), cosangle)
    return df


def remove_pre_cascade_data(df: pd.DataFrame, time_cutoff: float = 0.5) -> pd.DataFrame:
    """
    Remove all data just before cascading occurs. The 'time_cutoff' gives the time before the cascade that we stop the data  
    """
    X = df['All calculations', 'Time in s'].values
    where = df['All calculations', 'Time in s'].astype(
        float) < X.max() - time_cutoff
    return df[where]


def keep_fixed_time_data(df: pd.DataFrame, time_window: float = 1.3) -> pd.DataFrame:
    """
    Keep only data in a small time_window after the start of the data-set. The 'time_window' gives the time for which we will retain the data  
    """
    X = df['All calculations', 'Time in s'].values
    where = df['All calculations', 'Time in s'].astype(
        float) < X.min() + time_window
    return df[where]
