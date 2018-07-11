import numpy as np
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





