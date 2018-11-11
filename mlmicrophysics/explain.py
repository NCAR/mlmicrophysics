from multiprocessing import Pool
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from copy import deepcopy


def feature_importance(x, y, model, metric_function, x_columns=None, permutations=30, processes=1, seed=8272):
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
    return pd.DataFrame(diff_matrix, index=x_columns, columns=np.arange(permutations))


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
            x_perm[:, column_index] = x[perm_indices, column_index]
            perm_pred = model.predict(x_perm)
            perm_scores[p] = metric_function(y, perm_pred)
        return column_index, perm_scores
    except Exception as e:
        print(traceback.format_exc())
        raise e


def partial_dependence_1d(x, model, var_index, var_vals):
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
    partial_dependence = np.zeros(var_vals.shape)
    x_copy = np.copy(x)
    for v, var_val in enumerate(var_vals):
        x_copy[:, var_index] = var_val
        partial_dependence[v] = model.predict(x_copy).mean()
    return partial_dependence


def partial_dependence_2d(x, model, var_1_index, var_1_vals, var_2_index, var_2_vals):
    pd_grid = np.zeros((var_1_vals.size, var_2_vals.size))
    x_copy = np.copy(x)
    for v1, var_1_val in enumerate(var_1_vals):
        x_copy[:, var_1_index] = var_1_val
        for v2, var_2_val in enumerate(var_2_vals):
            x_copy[:, var_2_index] = var_2_val
            pd_grid[v1, v2] = model.predict(x_copy).mean()
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
                               dependence_title="Partial Dependence"):
    """
    Plot 2D partial dependence field and associated frequencies.

    Args:
        var_1_bins:
        var_2_bins:
        dependence_matrix:
        dependence_counts:
        var_1_name:
        var_2_name:
        output_file:
        dpi:
        figsize:
        cmap:
        label_fontsize:
        dependence_title:
        frequency_title:

    Returns:

    """
    plt.figure(figsize=figsize)
    plt.pcolormesh(var_1_vals, var_2_vals, dependence_matrix, cmap=cmap)
    plt.xlabel(var_1_name, fontsize=label_fontsize)
    plt.ylabel(var_2_name, fontsize=label_fontsize)
    plt.title(dependence_title, fontsize=label_fontsize + 2)
    plt.colorbar()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()
