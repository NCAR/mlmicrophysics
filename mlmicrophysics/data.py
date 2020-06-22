import s3fs
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from os.path import join, exists
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from operator import lt, le, eq, ne, ge, gt

scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}

ops = {"<": lt, "<=": le, "==": eq, "!=": ne, ">=": ge, ">": gt}


def log10_transform(x, eps=1e-18):
    return np.log10(np.maximum(x, eps))


def neg_log10_transform(x, eps=1e-18):
    return np.log10(np.maximum(-x, eps))


def zero_transform(x, eps=None):
    return np.zeros(x.shape, dtype=np.float32)


def inverse_log10_transform(x):
    return 10.0 ** x


def inverse_neg_log10_transform(x):
    return -10.0 ** x


transforms = {"log10_transform": log10_transform,
              "neg_log10_transform": neg_log10_transform,
              "zero_transform": zero_transform}


inverse_transforms = {"log10_transform": inverse_log10_transform,
                      "neg_log10_transform": inverse_neg_log10_transform,
                      "zero_transform": zero_transform}


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


def convert_to_dataframe(dataset, variables, times, time_var,
                         subset_variable, subset_threshold):
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
        if type(subset_variable) == list:
            valid = np.zeros(time_df.shape[0], dtype=bool)
            for s, sv in enumerate(subset_variable):
                valid[time_df[subset_variable] >= subset_threshold[s]] = True
        else:
            valid = time_df[subset_variable] >= subset_threshold
        data_frames.append(time_df.loc[valid].reset_index())
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


def subset_data_files_by_date(data_path,
                              train_date_start=0, train_date_end=8000,
                              test_date_start=9000,
                              test_date_end=18000, validation_frequency=3):
    """
    For a large set of csv files, this sorts the files into training, validation and testing data.
    This way the full dataset does not have to be loaded and then broken into pieces.

    Args:
        data_path:
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
    
    if data_path == "ncar-aiml-data-commons/microphysics":
        fs = s3fs.S3FileSystem(anon=True)
        csv_files = pd.Series(sorted(fs.ls("ncar-aiml-data-commons/microphysics")))
        data_end = "*.parquet"
    else:
        data_end = "*.csv"
        csv_files = pd.Series(sorted(glob(join(data_path, "*" + data_end))))
    
    file_times = csv_files.str.split("/").str[-1].str.split("_").str[-1].str.strip(data_end).astype(int).values
    print("File times:\n",file_times)
    train_val_ind = np.where((file_times >= train_date_start) & (file_times <= train_date_end))[0]
    test_ind = np.where((file_times >= test_date_start) & (file_times <= test_date_end))[0]
    val_ind = train_val_ind[::validation_frequency]
    train_ind = np.isin(train_val_ind, val_ind, invert=True)
    train_files = csv_files.loc[train_val_ind[train_ind]]
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


def categorize_output_values(output_values, output_transforms, output_scalers=None):
    """
    Transform and rescale the values of a single output variable based on specified transforms and scaling functions.
    The microphysics tendencies can be either 0 or nonzero and tend to have exponential distributions. Machine
    learning models tend to perform better when the data are more Gaussian. To account for this, we perform
    log transforms on the nonzero values and then normalize them. The rain water number concentration (Nr)
    tendencies have both positive and negative exponential tendencies, so we have separate models for each side
    and separate transformations and normalizations.

    Currently, we have support for the following transforms:
    * log10_transform: applies np.log10
    * neg_log10_transform: applies np.log10 to the additive inverse of the inputs
    * zero_transform: returns an array the same shape as the input but with all 0s

    We support the following scalers:
    * StandardScaler:
    * MinMaxScaler: Rescales values to range between 0 and 1
    * MaxAbsScaler:

    Args:
        output_values: numpy.ndarray of shape (samples, 1) with the raw output values from the microphysics
        output_transforms: dictionary with keys indicating sign of tendency and values containing a 3-element list
         [comparison operator, splitting threshold, transform type].
        output_scalers:

    Returns:

    """

    labels = np.zeros(output_values.shape, dtype=int)
    transformed_outputs = np.zeros(output_values.shape)
    scaled_outputs = np.zeros(output_values.shape)
    if output_scalers is None:
        output_scalers = {}
    for label, comparison in output_transforms.items():
        class_indices = ops[comparison[0]](output_values, float(comparison[1]))
        labels[class_indices] = label
        transformed_outputs[class_indices] = transforms[comparison[2]](output_values[class_indices],
                                                                       eps=float(comparison[1]))
        if comparison[3] != "None":
            if label not in list(output_scalers.keys()):
                output_scalers[label] = scalers[comparison[3]]()
                scaled_outputs[class_indices] = output_scalers[label].fit_transform(
                    transformed_outputs[class_indices].reshape(-1, 1)).ravel()
            else:
                scaled_outputs[class_indices] = output_scalers[label].transform(
                    transformed_outputs[class_indices].reshape(-1, 1)).ravel()
        else:
            output_scalers[label] = None
    return labels, transformed_outputs, scaled_outputs, output_scalers


def open_data_file(filename):
    if "ncar-aiml-data-commons/microphysics" in filename:
        fs = s3fs.S3FileSystem(anon=True)
        fobj = fs.open(filename)
        ds = pd.read_parquet(fobj).set_index('Index')
        return ds
    else:
        ds = pd.read_csv(filename, index_col="Index")
        return ds

def assemble_data_files(files, input_cols, output_cols, input_transforms, output_transforms,
                        input_scaler, output_scalers=None, train=True, subsample=1,
                        meta_cols=("lat", "lon", "lev", "depth", "row", "col", "pressure", "temperature",
                                   "time", "qrtend_MG2", "nrtend_MG2", "nctend_MG2")):
    """
    This function loads data from a list of files

    Args:
        files: List of files being loaded
        input_cols: List of input columns for training the neural networks
        output_cols: List of output columns
        input_transforms: List of log transformations corresponding to each input
        output_transforms: List of output log transformations corresponding to each output
        input_scaler: Scaler Object used to normalize all inputs
        output_scalers: Dictionary of scaler objects for each output field
        train: Whether to fit the Scaler objects or
        subsample:
        meta_cols:
    Returns:

    """
    all_input_data = []
    all_output_data = []
    all_meta_data = []
    for i,filename in enumerate(files):
        if i%10 == 0:
            print(f"Finished loading {i}/{len(files)} files... opening file {filename}")
        data = open_data_file(filename)
        if subsample < 1:
            sample_index = int(np.round(data.shape[0] * subsample))
            sample_indices = np.sort(np.random.permutation(np.arange(data.shape[0]))[:sample_index])
        else:
            sample_indices = np.arange(data.shape[0])
        all_input_data.append(data.loc[sample_indices, input_cols])
        all_output_data.append(data.loc[sample_indices, output_cols])
        all_meta_data.append(data.loc[sample_indices, meta_cols])
        del data
    print("Combining data")
    combined_input_data = pd.concat(all_input_data, ignore_index=True)
    combined_output_data = pd.concat(all_output_data, ignore_index=True)
    combined_meta_data = pd.concat(all_meta_data, ignore_index=True)
    print("Combined Data Size", combined_input_data.shape)
    del all_input_data[:]
    del all_output_data[:]
    print("Transforming data")
    for var, transform_name in input_transforms.items():
        combined_input_data.loc[:, var] = transforms[transform_name](combined_input_data[var])
    transformed_output_data = pd.DataFrame(0,
                                           columns=combined_output_data.columns,
                                           index=combined_output_data.index,
                                           dtype=np.float32)
    scaled_output_data = pd.DataFrame(0,
                                      columns=combined_output_data.columns,
                                      index=combined_output_data.index,
                                      dtype=np.float32)
    output_labels = pd.DataFrame(0,
                                 columns=combined_output_data.columns,
                                 index=combined_output_data.index,
                                 dtype=np.int32)
    if output_scalers is None:
        output_scalers = {}
    for output_var in output_cols:
        if output_var not in output_scalers:
            output_scalers[output_var] = None
        output_labels.loc[:, output_var],\
            transformed_output_data.loc[:, output_var],\
            scaled_output_data.loc[:, output_var],\
            output_scalers[output_var] = categorize_output_values(combined_output_data.loc[:,
                                                                  output_var].values.reshape(-1, 1),
                                                                  output_transforms[output_var],
                                                                  output_scalers=output_scalers[output_var])
    if train:
        scaled_input_data = pd.DataFrame(input_scaler.fit_transform(combined_input_data),
                                         columns=combined_input_data.columns)
    else:
        scaled_input_data = pd.DataFrame(input_scaler.transform(combined_input_data),
                                         columns=combined_input_data.columns)
    return scaled_input_data, output_labels, transformed_output_data, \
           scaled_output_data, output_scalers, combined_meta_data


def uniform_stratify_data(output_labels, scaled_output_data, category_size, output_bin_dict):
    """
    Resample data to have uniform sample sizes across different labels and across different bins.

    Args:
        output_labels: DataFrame of label values for each example.
        scaled_output_data: Normalized output values for each label
        category_size: Number of samples per label in each output
        output_bin_dict: Dictionary of bins for each output type and label.

    Returns:
        Resampled indices for each output type and label.
    """
    sampling_indices = dict()
    out_index = output_labels.index.values
    for output_col in output_labels.columns:
        print(output_col)
        sampling_indices[output_col] = dict()
        label_vals = np.unique(output_labels[output_col])
        for label_val in label_vals:
            print(label_val)
            label_count = np.count_nonzero(output_labels[output_col] == label_val)
            if label_count < category_size:
                label_replace = True
            else:
                label_replace = False
            if label_val == 0:
                sampling_indices[output_col][label_val] = np.random.choice(out_index[output_labels[output_col] == label_val],
                                                                           size=category_size, replace=label_replace)
            else:
                label_bins = output_bin_dict[output_col][label_val]
                scaled_hist, _ = np.histogram(scaled_output_data[output_col][output_labels[output_col] == label_val],
                                              bins=label_bins)
                uniform_bin_size = category_size // scaled_hist.size
                sampling_indices[output_col][label_val] = np.zeros(category_size)
                s_index = 0
                for b in range(label_bins.size - 1):
                    if scaled_hist[b] < uniform_bin_size:
                        bin_replace = True
                    else:
                        bin_replace = False
                    out_sample_indices = out_index[(output_labels[output_col] == label_val) &
                                                   (scaled_output_data[output_col] >= label_bins[b]) &
                                                   (scaled_output_data[output_col] < label_bins[b + 1])]
                    print(b, label_bins[b], out_sample_indices.size)
                    sampling_indices[output_col][label_val][s_index:s_index + uniform_bin_size] = \
                        np.random.choice(out_sample_indices,
                                         size=uniform_bin_size,
                                         replace=bin_replace)
    return sampling_indices


def inverse_transform_data(data, transform_dict):
    """
    Reconstruct original data values from partially log-transformed data

    Args:
        data: pandas DataFrame containing log-transformed data
        transform_dict: dictionary mapping variable names in data to the type of transformation
    Returns:
        out_data: pandas DataFrame with data as close to original form as possible. Values below
            the eps threshold will be set to the eps threshold.
    """
    out_data = pd.DataFrame(data.values, columns=data.columns, index=data.index)
    for var, transform_name in transform_dict.items():
        if var in out_data.columns:
            out_data.loc[:, var] = inverse_transforms[transform_name](data[var])
    return out_data

def repopulate_input_scaler(scale_file, scaler_type="StandardScaler"):
    """
    Given a csv file containing scaling values, repopulate a scikit-learn
    scaling object with those same values.

    Args:
        scale_file (str): path to csv file containing scale values
        scaler_type (str):

    Returns:
        scaler_obj: Scaler Object with normalizing values specified.
    """
    scaler_obj = scalers[scaler_type]()
    scale_data = pd.read_csv(scale_file)
    for column in scale_data.columns[1:]:
        setattr(scaler_obj, column + "_", scale_data[column].values)
    return scaler_obj


def repopulate_output_scalers(scale_file, output_transforms):
    """
    From an output_scale_values.csv file, repopulate the scikit-learn
    scaling objects for each output type.

    Args:
        scale_file:
        output_transforms:

    Returns:

    """
    scale_data = pd.read_csv(scale_file, index_col="output")
    output_scalers = {}
    for out_var in output_transforms.keys():
        output_scalers[out_var] = {}
        for out_label, transform in output_transforms[out_var].items():
            if transform[2] == "None":
                output_scalers[out_var][out_label] = None
            else:
                output_scalers[out_var][out_label] = scalers[transform[2]]()
                for col in scale_data.columns:
                    setattr(output_scalers[out_var][out_label],
                            col, scale_data.loc[out_var, col])
    return output_scalers


