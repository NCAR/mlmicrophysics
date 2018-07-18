import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from os.path import join, exists


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
                                        coords=[var_data.time, var_data[vertical_dim], var_data.lat, var_data.lon],
                                        dims=("time", vertical_dim, "lat", "lon"))
    return unstaggered_var_data


def convert_to_dataframe(dataset, variables, times, time_var="time", subset_var="QC_TAU_in", subset_threshold=0):
    """
    Convert 4D Dataset to flat dataframe for machine learning.

    Args:
        dataset: xarray Dataset containing all relevant variables and times.
        variables: List of variables in dataset to be included in DataFrame. All variables should have the same
            dimensions and coordinates.
        times: Iterable of times to select from dataset.
        time_var: Variable used as the time coordinate.

    Returns:

    """
    data_frames = []
    for t, time in enumerate(times):
        time_df = dataset[variables].isel(**{time_var: t}).to_dataframe()
        data_frames.append(time_df.loc[time_df[subset_var] > subset_threshold].reset_index())
        del time_df
    return pd.concat(data_frames)


