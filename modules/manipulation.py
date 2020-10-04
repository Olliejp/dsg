import joblib
import os
import pandas as pd
from cleaning import (
    remove_row,
    convert_objects_to_float,
    interpolation_reconstruction,
    normalise_df,
    angle_transform,
    remove_disturbance,
    keep_fixed_time_data,
    interpolation_reconstruction_delta_T
)
from utils import add_NSG_columns_and_sort_columns_alphabetically


def load(directory: str, filename: str) -> pd.DataFrame:
    """
    Load a dataframe from pickle format.
    """
    path = os.path.join(directory, filename)
    df = joblib.load(path)
    return df


def save(df: pd.DataFrame, directory: str, filename: str) -> pd.DataFrame:
    """
    Save a dataframe into pickle format.
    """
    path = os.path.join(directory, filename)
    joblib.dump(df, path)


def clean_batch_of_raw_data_time_windowed(
    df: pd.DataFrame,
    time_window: float,
    means,
    stds
) -> pd.DataFrame:
    """
    Clean a batch of raw data.

    Parameters
    ----------
    df : pd.DataFrame
        A batch of raw messages.

    Returns
    -------
    pd.DataFrame
        Batch of clean data
    """
    df = convert_objects_to_float(df)
    df = add_NSG_columns_and_sort_columns_alphabetically(df)
    df = angle_transform(df)
    df = remove_disturbance(df)
    df = keep_fixed_time_data(df, time_window)
    df = interpolation_reconstruction_delta_T(df, 0.01)
    df = normalise_df(df, means, stds)
    df = df.droplevel(1, axis=1)
    return df


def load_clean_save_raw_data_by_batch(
    root_dir: str,
    output_data_dir: str,
    sep: str,
    header: List[int],
    time_window: float,
    means,
    stds
) -> pd.DataFrame:
    """
    Load, clean and save raw data batch by batch.

    Parameters
    ----------
    root_dir : str
        Directory to probe for loading.
    output_data_dir : str
        Directory where to save cleaned data.
    sep : str
        Separator used when reading CSVs.
    cols_to_keep : List[str]
        List of columns to keep. All other columns are dropped.
    """
    for i, file in enumerate(os.listdir(root_dir)):
        if file.endswith('results_cascs.csv'):
            continue
        else:
            print(i, file)
            file_path = os.path.join(root_dir, file)
            df = pd.read_csv(file_path, sep=',', header=[0, 1])
            try:
                df = clean_batch_of_raw_data_time_windowed(
                    df, time_window, means, stds)
                save(df, output_data_dir, file.replace('.csv', '.pkl'))
            except:
                print("----- Error in processing ", file,
                      " --- Mostly due to cascading failure ")
