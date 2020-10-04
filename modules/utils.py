import pandas as pd
from typing import Dict, List
import os
from cleaning import convert_objects_to_float


def get_list_simulations(root_dir: str) -> List[str]:
    """
    Get a list with all the name of the simulation csv files.
    """
    file_list_simulations = []
    for i, file in enumerate(os.listdir(root_dir)):
        if file.endswith('results_cascs.csv'):
            continue
        else:
            file_list_simulations.append(file)
    return file_list_simulations


def get_dict_scenario_csv(file_list_simulations: List[str]) -> Dict[int, str]:
    """
    Build dictionary of scenario number and the corresponding name of the csv file
    with simulation data.
    """
    simulation_files_dict = {}
    for file_name in file_list_simulations:
        simulation_files_dict[int(file_name.split("_")[0])] = file_name
    return simulation_files_dict


def add_NSG_columns_and_sort_columns_alphabetically(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add NSG columns when missing from dataframe in order
    to get homogeneous set of columns in each dataframe.
    """
    cols_to_add = [('NSG_1', 'Active Power in MW'), ('NSG_1', 'Reactive Power in Mvar'),
                   ('NSG_2', 'Active Power in MW'), ('NSG_3', 'Active Power in MW')]
    for col in cols_to_add:
        if not col in df.columns:
            df[col] = '0'
    df = df.sort_index(axis=1)
    return df


def calculate_correlation(
    df: pd.DataFrame,
    t1: int = 1.1,
    t2: int = 2
) -> pd.DataFrame:
    """
    Compute correlation between variables in `df`. As we have time series data,
    `t1` and `t2` correspond, respectively, to the time when the disturbance is removed,
    and the time until when we want to compute the correlation.

    In the correlation matrix, NaN are filled by 1, this is due to constant values with time.
    """
    sub_data_frame = df[df[('All calculations', 'Time in s')].astype(
        float) > t1]
    sub_data_frame = sub_data_frame[sub_data_frame[(
        'All calculations', 'Time in s')].astype(float) < t2]
    df_corr = sub_data_frame.astype(float).corr(method='pearson')
    df_corr.fillna(1, inplace=True)
    return df_corr.abs()


def get_corr_from_scenario_number(
    root_dir: str,
    simulation_files_dict: Dict[int, str],
    scenario_number: int
) -> pd.DataFrame:
    """
    Get correlation between variables for a given file 
    (use the `scenario_number` and corresponding `simulation_files_dict`
    to read the csv file)
    """
    df1 = pd.read_csv(os.path.join(
        root_dir, simulation_files_dict[scenario_number]), sep=',', header=[0, 1])
    df1 = add_NSG_columns_and_sort_columns_alphabetically(df1)
    df1 = convert_objects_to_float(df1)
    df_corr_1 = calculate_correlation(df1)
    return df_corr_1
