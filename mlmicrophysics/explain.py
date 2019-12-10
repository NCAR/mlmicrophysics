from multiprocessing import Pool
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from copy import deepcopy


def feature_importance(x, y, model, metric_function, x_columns=None, permutations=30, processes=1,
                       col_start="perm_", seed=8272):
    """
    Calculate permutation feature importance scores for an arbitrary machine learning model.

    Args:
        x: ndarray of dimension (n_examples, n_features) that contains the input data for the ML model.
        y: ndarray of dimension (n_examples, ) that contains the true target values.
        model: machine learning model object in scikit-learn format (contains fit and predict methods).
        metric_function: scoring function with the input format (y_true, y_predicted) to match scikit-learn.
        x_columns (ndarray or None): list or array of column names. If not provided, indices will be used instead.
        permutations (int): Number of times a column is randomly shuffled.
        processes (int): Number of multiprocessor processes used for parallel computation of importances
        col_start (str): Start of output columns.
        seed (int): Random seed.

    Returns:
        pandas DataFrame of dimension (n_columns, permutations) that contains the change in score
        for each column and permutation.
    """
    if x_columns is None:
        x_columns = np.arange(x.shape[1])
    if type(x_columns) == list:
        x_columns = np.array(x_columns)
    predictions = model.predict(x)
    score = metric_function(y, predictions)
    print(score)
    np.random.seed(seed=seed)
    perm_matrix = np.zeros((x_columns.shape[0], permutations))

    def update_perm_matrix(result):
        perm_matrix[result[0]] = result[1]
    if processes > 1:
        pool = Pool(processes)
        for c in range(len(x_columns)):
            pool.apply_async(feature_importance_column,
                             (x, y, c, permutations, deepcopy(model), metric_function, np.random.randint(0, 100000)),
                              callback=update_perm_matrix)
        pool.close()
        pool.join()
    else:
        for c in range(len(x_columns)):
            result = feature_importance_column(x, y, c, permutations, model,
                                               metric_function, np.random.randint(0, 100000))
            update_perm_matrix(result)
    diff_matrix = score - perm_matrix
    out_columns = col_start + pd.Series(np.arange(permutations)).astype(str)
    return pd.DataFrame(diff_matrix, index=x_columns, columns=out_columns)


def feature_importance_column(x, y, column_index, permutations, model, metric_function, seed):
    """
    Calculate the permutation feature importance score for a single input column. It is the error score on
    a given set of data after the values in one column have been shuffled among the different examples.

    Args:
        x: ndarray of dimension (n_examples, n_features) that contains the input data for the ML model.
        y: ndarray of dimension (n_examples, ) that contains the true target values.
        column_index: Index of the x column being permuted
        permutations: Number of permutations run to calculate importance score distribution
        model: machine learning model object in scikit-learn format (contains fit and predict methods).
        metric_function: scoring function with the input format (y_true, y_predicted) to match scikit-learn.
        seed (int): random seed.

    Returns:
        column_index, permutation, perm_score
    """
    try:
        rs = np.random.RandomState(seed=seed)
        perm_indices = np.arange(x.shape[0])
        perm_scores = np.zeros(permutations)
        x_perm = np.copy(x)
        for p in range(permutations):
            print(column_index, p)
            rs.shuffle(perm_indices)
            x_perm[np.arange(x.shape[0]), column_index] = x[perm_indices, column_index]
            perm_pred = model.predict(x_perm)
            perm_scores[p] = metric_function(y, perm_pred)
        return column_index, perm_scores
    except Exception as e:
        print(traceback.format_exc())
        raise e


def partial_dependence_mp(x, model_file, var_val_count, n_procs):
    """
    Perform partial dependence calculations in parallel with multiprocessing

    Args:
        x: training data array
        model_file: file name for keras model hdf5 file
        var_val_count: number of partial dependence values per variable
        n_procs: number of processes for multiprocessing

    Returns:
        pd_vals: partial dependence values, var_vals: values for each input variable paired with partial dependence
    """
    var_vals = np.zeros((x.shape[1], var_val_count), dtype=np.float32)
    for j in range(x.shape[1]):
        var_vals[j] = np.linspace(x[:, j].min(), x[:, j].max(), var_val_count)
    num_splits = n_procs
    pd_vals = np.zeros((x.shape[1], var_val_count, x.shape[0]), dtype=np.float32)
    split_points = np.linspace(0, x.shape[0], num_splits + 1).astype(int)
    pool = Pool(n_procs)

    def update_pd_vals(result):
        pd_vals[result[1], :, result[2]:result[3]] = result[0]
    for j in range(x.shape[1]):
        print(j)
        for n in range(num_splits):
            pool.apply_async(partial_dependence_1d_mp, (x[split_points[n]:split_points[n+1]], split_points[n],
                                                        split_points[n+1]), dict(var_index=j,
                                model_file=model_file, var_vals=var_vals), callback=update_pd_vals)
    pool.close()
    pool.join()
    return pd_vals, var_vals


def partial_dependence_1d_mp(x, split_start, split_end, var_index=0, model_file=None, var_vals=None):
    """
    Calculate how the mean prediction of an ML model varies if one variable's value is fixed across all input
    examples.

    Args:
        x: array of input variables
        model: scikit-learn style model object
        var_index: column index of the variable being investigated
        var_vals: values of the input variable that are fixed.

    Returns:
        Array of partial dependence values.
    """
    try:
        from keras.models import load_model
        import tensorflow as tf
        import keras.backend as K
        #sess = tf.Session(config=tf.ConfigProto(device_count={"CPU": 1}, intra_op_parallelism_threads=1,
        #                    inter_op_parallelism_threads=1))
        #K.set_session(sess)
        with tf.device("/cpu:0"):
            model = load_model(model_file)
            partial_dependence = np.zeros((var_vals.shape[1], x.shape[0]), dtype=np.float32)
            x_copy = np.copy(x)
            for v, var_val in enumerate(var_vals[var_index]):
                print(var_index, var_val)
                x_copy[:, var_index] = var_val
                partial_dependence[v] = model.predict(x_copy).ravel()
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return partial_dependence, var_index, split_start, split_end


def partial_dependence_tau_mp(x, var_val_count, n_procs):
    var_vals = np.zeros((x.shape[1], var_val_count), dtype=np.float32)
    for j in range(x.shape[1]):
        var_vals[j] = np.linspace(x[:, j].min(), x[:, j].max(), var_val_count)
    num_splits = n_procs
    tau_outputs = 4
    pd_vals = np.zeros((x.shape[1], var_val_count, tau_outputs, x.shape[0]), dtype=np.float32)
    split_points = np.linspace(0, x.shape[0], num_splits + 1).astype(int)
    pool = Pool(n_procs)

    def update_pd_vals(result):
        pd_vals[:, :, :, result[1]:result[2]] = result[0]

    for j in range(x.shape[1]):
        for n in range(num_splits):
            pool.apply_async(partial_dependence_1d_tau, (x[split_points[n]:split_points[n + 1]], split_points[n],
                                                        split_points[n + 1], var_vals),
                             callback=update_pd_vals)
    pool.close()
    pool.join()
    return pd_vals, var_vals


def partial_dependence_1d_tau(x, split_start, split_end, var_vals):
    """
    Calculate how the mean prediction of an ML model varies if one variable's value is fixed across all input
    examples.

    Args:
        x: array of input variables
        split_start: index of first example
        split_end: index of last example

    Returns:
        Array of partial dependence values.
    """
    try:
        from .call_collect import call_collect
        tau_outputs = 4
        partial_dependence = np.zeros((x.shape[1], var_vals.shape[1], tau_outputs, x.shape[0]), dtype=np.float32)
        x_copy = np.copy(x)
        for var_index in range(x.shape[1]):
            for v, var_val in enumerate(var_vals[var_index]):
                x_copy[:, var_index] = var_val
                partial_dependence[var_index, v] = np.vstack(call_collect(1, x_copy[:, 0],
                                                                x_copy[:, 1], x_copy[:, 2],
                                                                x_copy[:, 3], x_copy[:, 4], x[:, 5],
                                                                x_copy[:, 6], x_copy[:, 7]))
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return partial_dependence, split_start, split_end


def partial_dependence_1d(x, var_index=0, model=None, var_vals=None):
    """
    Calculate how the mean prediction of an ML model varies if one variable's value is fixed across all input
    examples.

    Args:
        x: array of input variables
        model: scikit-learn style model object
        var_index: column index of the variable being investigated
        var_vals: values of the input variable that are fixed.

    Returns:
        Array of partial dependence values.
    """
    partial_dependence = np.zeros((var_vals.shape[0], x.shape[0]), dtype=np.float32)
    x_copy = np.copy(x)
    for v, var_val in enumerate(var_vals):
        x_copy[:, var_index] = var_val
        partial_dependence[v] = model.predict(x_copy).ravel()
    return partial_dependence


def partial_dependence_2d(x, model, var_1_index, var_1_vals, var_2_index, var_2_vals):
    """
    Calculate the partial dependence values on a 2D grid where the columns correspond to var 1
    and the rows correspond the var 2.
    Partial dependence fixes the value of 1 or 2 variables for all examples, feeds them through
    the machine learning model, and calculates the mean of the resulting predictions.

    Args:
        x: Input training data in a numpy array
        model: scikit-learn style model object being evaluated
        var_1_index: Column index of first variable
        var_1_vals: Values of variable 1 to be evaluated for partial dependence.
            Values should be monotonic
        var_2_index: Column index of the second variable
        var_2_vals: Values of variable 2 to be evaluated for partial dependence

    Returns:
        pd_grid, array of shape (var_2_vals.size, var_1_vals.size) containing partial dependence values
    """
    pd_grid = np.zeros((var_2_vals.size, var_1_vals.size))
    x_copy = np.copy(x)
    for v2, var_2_val in enumerate(var_2_vals):
        x_copy[:, var_2_index] = var_2_val
        for v1, var_1_val in enumerate(var_1_vals):
            x_copy[:, var_1_index] = var_1_val
            pd_grid[v2, v1] = model.predict(x_copy).mean()
    return pd_grid


def conditional_input_prediction_2d(x, y_pred, var_1_index, var_2_index, var_1_bins, var_2_bins, dependence_function=np.mean):
    """
    For a given set of 2 input values, calculate a summary statistic based on all of the examples that fall within
    each binned region of the input data space. The goal is to show how the model predictions vary on average as
    a function of 2 input variables.

    Args:
        x: Array of input data values
        y_pred: ML model predictions
        var_1_index: Index of first input column
        var_2_index: Index of second input column
        var_1_bins: Bins used to segment variable 1
        var_2_bins: Bins used to segment variable 2
        dependence_function: Summary statistic function

    Returns:
        dependence_matrix: An array of partial dependence values with rows for each variable 2 bin and
            columns for each variable 1 bin
        dependence_counts: Array containing the number of examples within each bin.
    """
    dependence_matrix = np.zeros((var_2_bins.size - 1, var_1_bins.size - 1))
    dependence_counts = np.zeros(dependence_matrix.shape, dtype=int)
    for i in range(var_2_bins.size - 1):
        for j in range(var_1_bins.size - 1):
            valid_indices = np.where((x[:, var_1_index] >= var_1_bins[j]) &
                                     (x[:, var_1_index] < var_1_bins[j + 1]) &
                                     (x[:, var_2_index] >= var_2_bins[i]) &
                                     (x[:, var_2_index] < var_2_bins[i + 1])
                                     )[0]
            if valid_indices.size > 0:
                dependence_matrix[i, j] = dependence_function(y_pred[valid_indices])
                dependence_counts[i, j] = valid_indices.size
            else:
                dependence_matrix[i, j] = np.nan
    return dependence_matrix, dependence_counts


def conditional_input_prediction_1d(x, y_pred, var_index, var_bins, dependence_function=np.mean):
    """
    Calculate a partial dependence curve for a single variable.

    Args:
        x:
        y_pred:
        var_index:
        var_bins:
        dependence_function:

    Returns:

    """
    dependence_matrix = np.zeros(var_bins.size - 1)
    dependence_counts = np.zeros(dependence_matrix.shape, dtype=int)
    for i in range(var_bins.size - 1):
        valid_indices = np.where((x[:, var_index] >= var_bins[i]) & (x[:, var_index] < var_bins[i + 1]))[0]
        if valid_indices.size > 0:
            dependence_matrix[i] = dependence_function(y_pred[valid_indices])
            dependence_counts[i] = valid_indices.size
        else:
            dependence_matrix[i] = np.nan
    return dependence_matrix, dependence_counts


def partial_dependence_plot_2d(var_1_vals, var_2_vals, dependence_matrix,
                               var_1_name, var_2_name, output_file, dpi=300,
                               figsize=(8, 8), cmap="viridis", label_fontsize=14,
                               title="Partial Dependence"):
    """
    Plot 2D partial dependence field and associated frequencies.

    Args:
        var_1_vals: Array of values indexed for variable 1
        var_2_vaks: Array of values indexed for variable 2
        dependence_matrix: 2D array of partial dependence values
        var_1_name: Name of variable 1
        var_2_name: Name of variable 2
        output_file: Name of image file
        dpi: number of dots per inch (default 300)
        figsize: (width, height) of figure in inches
        cmap: Colormap used
        label_fontsize: Fontsize of x and y labels
        title: Title of figure
    """
    plt.figure(figsize=figsize)
    plt.pcolormesh(var_1_vals, var_2_vals, dependence_matrix, cmap=cmap)
    plt.xlabel(var_1_name, fontsize=label_fontsize)
    plt.ylabel(var_2_name, fontsize=label_fontsize)
    plt.title(title, fontsize=label_fontsize + 2)
    plt.colorbar()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()
