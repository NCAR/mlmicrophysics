import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from os.path import join, exists
from dask import delayed


def load_cam_output(path, file_start="TAU_run1.cam.h1", file_end="nc"):
    """
    Load set of model output from CAM/CESM into xarray Dataset object.

    Args:
        path: Path to directory containing model output
        file_start: Shared beginning of model files
        file_end: Filetype shared by all files.

    Returns:
        xarray Dataset object containing the model output
    """
    if not exists(path):
        raise FileNotFoundError("Specified path " + path + " does not exist")
    data_files = sorted(glob(join(path, file_start + "*" + file_end)))
    if len(data_files) > 0:
        cam_dataset = xr.open_mfdataset(data_files, decode_times=False)
    else:
        raise FileNotFoundError("No matching CAM output files found in " + path)
    return cam_dataset


def get_cam_output_times(path, time_var="time", file_start="TAU_run1.cam.h1", file_end="nc"):
    if not exists(path):
        raise FileNotFoundError("Specified path " + path + " does not exist")
    data_files = sorted(glob(join(path, file_start + "*" + file_end)))
    file_time_list = []
    for data_file in data_files:
        ds = xr.open_dataset(data_file, decode_times=False, decode_cf=False)
        time_minutes = (ds[time_var].values * 24 * 60).astype(int)
        file_time_list.append(pd.DataFrame({"time": time_minutes,
                                            "filename": [data_file] * len(time_minutes)}))
        ds.close()
        del ds
    return pd.concat(file_time_list, ignore_index=True)



def unstagger_vertical(dataset, variable, vertical_dim="lev"):
    """
    Interpolate a 4D variable on a staggered vertical grid to an unstaggered vertical grid. Will not execute
    until compute() is called on the result of the function.

    Args:
        dataset: xarray Dataset object containing the variable to be interpolated
        variable: Name of the variable being interpolated
        vertical_dim: Name of the vertical coordinate dimension.

    Returns:
        xarray DataArray containing the vertically interpolated data
    """
    var_data = dataset[variable]
    unstaggered_var_data = xr.DataArray(0.5 * (var_data[:, :-1].values + var_data[:, 1:].values),
                                        coords=[var_data.time, dataset[vertical_dim], var_data.lat, var_data.lon],
                                        dims=("time", vertical_dim, "lat", "lon"),
                                        name=variable + "_" + vertical_dim)
    return unstaggered_var_data



def split_staggered_variable(dataset, variable, vertical_dim="lev"):
    """
    Split vertically staggered variable into top and bottom subsets with the unstaggered
    vertical coordinate

    Args:
        dataset: xarray Dataset object
        variable: Name of staggered variable
        vertical_dim: Unstaggered vertical dimension

    Returns:
        top_var_data, bottom_var_data: xarray DataArrays containing the unstaggered vertical data
    """
    var_data = dataset[variable]
    top_var_data = xr.DataArray(var_data[:, :-1], coords=[var_data.time,
                                                          dataset[vertical_dim],
                                                          var_data["lat"],
                                                          var_data["lon"]],
                                dims=("time", vertical_dim, "lat", "lon"),
                                name=variable + "_top")
    bottom_var_data = xr.DataArray(var_data[:, 1:], coords=[var_data.time,
                                                            dataset[vertical_dim],
                                                            var_data["lat"],
                                                            var_data["lon"]],
                                   dims=("time", vertical_dim, "lat", "lon"),
                                   name=variable + "_bottom")
    return xr.Dataset({variable + "_top": top_var_data, variable + "_bottom": bottom_var_data})


def add_index_coords(dataset, row_coord="lat", col_coord="lon", depth_coord="lev"):
    """
    Calculate the index values of the row, column, and depth coordinates in a Dataset.
    Indices range from 0 to length of coordinate - 1.

    Args:
        dataset: xarray Dataset
        row_coord: name of the row coordinate variable. Default lat.
        col_coord: name of the column coordinate variable. Default lon.
        depth_coord: name of the depth coordinate variable. Default lev.

    Returns:
        row, col, depth: DataArrays with the row, col, and depth indices
    """
    row = xr.DataArray(np.arange(dataset[row_coord].shape[0]), dims=(row_coord,), name="row")
    col = xr.DataArray(np.arange(dataset[col_coord].shape[0]), dims=(col_coord,), name="col")
    depth = xr.DataArray(np.arange(dataset[depth_coord].shape[0]), dims=(depth_coord,), name="depth")
    return xr.Dataset({"row": row, "col": col, "depth": depth})


def calc_pressure_field(dataset, pressure_var_name="pressure"):
    """
    Calculate pressure at each location based on the surface pressure and vertical coordinate
    information.

    Args:
        dataset:
        pressure_var_name:

    Returns:

    """
    pressure = xr.DataArray((dataset["hyam"] * dataset["P0"] + dataset["hybm"] * dataset["PS"]).transpose("time", "lev", "lat", "lon"))
    pressure.name = pressure_var_name
    pressure.attrs["units"] = "Pa"
    pressure.attrs["long_name"] = "atmospheric pressure"
    return pressure


def calc_temperature(dataset, density_variable="RHO_CLUBB_lev", pressure_variable="pressure"):
    """
    Calculation temperature from pressure and density. The temperature variable is added to the
    dataset object in place.

    Args:
        dataset: xarray Dataset object containing pressure and density variable
        density_variable: name of the density variable
        pressure_variable: name of the pressure variable
    """
    temperature = dataset[pressure_variable] / dataset[density_variable] / 287.0
    temperature.attrs["units"] = "K"
    temperature.attrs["long_name"] = "temperature derived from pressure and density"
    temperature.name = "temperature"
    return temperature


def convert_to_dataframe(dataset, variables, times, time_var="time",
                         subset_variable="QC_TAU_in", subset_threshold=0):
    """
    Convert 4D Dataset to flat dataframe for machine learning.

    Args:
        dataset: xarray Dataset containing all relevant variables and times.
        variables: List of variables in dataset to be included in DataFrame. All variables should have the same
            dimensions and coordinates.
        times: Iterable of times to select from dataset.
        time_var: Variable used as the time coordinate.
        subset_variable: Variable used to select a subset of grid points from file
        subset_threshold: Threshold that must be exceeded for examples to be kept.
    Returns:

    """
    data_frames = []
    for t, time in enumerate(times):
        print(t, time)
        time_df = dataset[variables].sel(**{time_var: time}).to_dataframe()
        data_frames.append(time_df.loc[time_df[subset_variable] > subset_threshold].reset_index())
        print(data_frames[-1])
        del time_df
    return pd.concat(data_frames)


def load_csv_data(csv_path, index_col="Index"):
    """
    Read pre-processed csv files into memory.

    Args:
        csv_path: Path to csv files
        index_col: Column label used as the index

    Returns:
        `pandas.DataFrame` containing data from all csv files in the csv_path directory.
    """
    csv_files = sorted(glob(join(csv_path, "*.csv")))
    all_data = []
    for csv_file in csv_files:
        all_data.append(pd.read_csv(csv_file, index_col=index_col))
    return pd.concat(all_data, axis=0)


def subset_data_files_by_date(data_path, data_end,
                              train_date_start=0, train_date_end=8000,
                              test_date_start=9000,
                              test_date_end=18000, validation_frequency=3):
    """
    For a large set of csv files, this sorts the files into training, validation and testing data.
    This way the full dataset does not have to be loaded and then broken into pieces.

    Args:
        data_path:
        data_end:
        train_date_start:
        train_date_end:
        test_date_start:
        test_date_end:
        validation_frequency:

    Returns:

    """
    if train_date_start > train_date_end:
        raise ValueError("train_date_start should not be greater than train_date_end")
    if test_date_start > test_date_end:
        raise ValueError("test_date_start should not be greater than test_date_end")
    if train_date_end > test_date_start:
        raise ValueError("train and test date periods overlap.")
    csv_files = pd.Series(sorted(glob(join(data_path, "*" + data_end))))
    file_times = csv_files.str.split("/").str[-1].str.split("_").str[-1].str.strip(data_end).astype(int).values
    print(file_times)
    train_val_ind = np.where((file_times >= train_date_start) & (file_times <= train_date_end))[0]
    test_ind = np.where((file_times >= test_date_start) & (file_times <= test_date_end))[0]
    val_ind = train_val_ind[::validation_frequency]
    train_ind = np.isin(train_val_ind, val_ind, invert=True)
    train_files = csv_files.loc[train_ind]
    val_files = csv_files.loc[val_ind]
    test_files = csv_files.loc[test_ind]
    return train_files, val_files, test_files


def subset_data_by_date(data, train_date_start=0, train_date_end=1, test_date_start=2, test_date_end=3,
                        validation_frequency=3, subset_col="time"):
    """
    Subset temporal data into training, validation, and test sets by the date column.

    Args:
        data: pandas DataFrame containing all data for training, validation, and testing.
        train_date_start: First date included in training period
        train_date_end: Last date included in training period
        test_date_start: First date included in testing period
        test_date_end: Last date included in testing period.
        validation_frequency: How often days are separated from training dataset for validation.
            Should be an integer > 1. 2 is every other day, 3 is every third day, etc.
        subset_col: Name of column being used for date evaluation.

    Returns:
        training_set, validation_set, test_set
    """
    if train_date_start > train_date_end:
        raise ValueError("train_date_start should not be greater than train_date_end")
    if test_date_start > test_date_end:
        raise ValueError("test_date_start should not be greater than test_date_end")
    if train_date_end > test_date_start:
        raise ValueError("train and test date periods overlap.")
    train_indices = (data[subset_col] >= train_date_start) & (data[subset_col] <= train_date_end)
    test_indices = (data[subset_col] >= test_date_start) & (data[subset_col] <= test_date_end)
    train_and_validation_data = data.loc[train_indices]
    test_data = data.loc[test_indices]
    train_and_validation_dates = np.unique(train_and_validation_data[subset_col].values)
    validation_dates = train_and_validation_dates[validation_frequency::validation_frequency]
    train_dates = train_and_validation_dates[np.isin(train_and_validation_dates,
                                                     validation_dates,
                                                     assume_unique=True,
                                                     invert=True)]
    train_data = data.loc[np.isin(data[subset_col].values, train_dates)]
    validation_data = data.loc[np.isin(data[subset_col].values, validation_dates)]
    return train_data, validation_data, test_data


def log10_transform(x, eps=1e-15):
    return np.log10(np.maximum(x, eps))


def neg_log10_transform(x, eps=1e-15):
    return np.log10(np.maximum(-x, eps))


def zero_transform(x, eps=1e-15):
    return np.zeros(x.shape, dtype=np.float32)

