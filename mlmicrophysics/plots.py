import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import cartopy.crs as ccrs
import logging

def timestep_input_distributions(timestep_ds,
                                 time,
                                 out_path,
                                 figsize=(8, 8),
                                 fontsize=14,
                                 dpi=200):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    variables = ["QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in"]
    bins = [(-18, -1, 0.5), (-18, -1, 0.5), (-12, 8, 0.5), (-12, 8, 0.5)]
    color = "blue"
    for a, ax in enumerate(axes.ravel()):
        flat_input_vals = timestep_ds[variables[a]].values.ravel()
        ax.hist(np.log10(flat_input_vals[flat_input_vals > 0]), bins=np.arange(*bins[a]), color=color)
        ax.set_xlabel(variables[a].split("_")[0], fontsize=fontsize)
        ax.set_yscale("log")
    fig.suptitle("Distributions Valid " + time.strftime("%Y-%m-%d %H:%M"))
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return


def timestep_input_maps(timestep_ds,
                        time,
                        out_path,
                        figsize=(8, 8),
                        fontsize=14,
                        dpi=200,
                        ):
    fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
    variables = ["QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in"]
    bins = [(-18, -1, 0.5), (-18, -1, 0.5), (-12, 8, 0.5), (-12, 8, 0.5)]
    for a, ax in enumerate(axes.ravel()):
        ax.coastlines()
        input_vals = timestep_ds[variables[a]].values.max(axis=0)
        print(input_vals.shape)
        log_input_vals = np.ma.array(np.where(input_vals > 0, np.log10(np.maximum(input_vals, 1e-18)), np.nan), mask=input_vals == 0)
        pc = ax.pcolormesh(timestep_ds["lon"], timestep_ds["lat"], log_input_vals, vmin=bins[a][0], vmax=bins[a][1])
        plt.colorbar(pc, ax=ax)
        ax.set_title(variables[a].split("_")[0], fontsize=fontsize)
    fig.suptitle("Distributions Valid " + time.strftime("%Y-%m-%d %H:%M"))
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


pretty_names = {"qrtend_TAU_1": "$\log_{10} dq_r/dt$",
                "nctend_TAU_1": "$\log_{10} dn_c/dt$",
                "nrtend_TAU_1": "$\log_{10} dn_r/dt > 0$",
                "nrtend_TAU_-1": "$\log_{10} dn_r/dt < 0$"}


def error_histogram(observations, predictions, observation_label, prediction_label,
                    out_path, num_bins=20, figure_height=4, fontsize=14, dpi=200, cmap="viridis"):
    """
    Plot error histograms for each microphysics tendency.

    Args:
        observations: dictionary of observed tendencies (plotted on y axis)
        predictions: dictionary of predicted tendencies (plotted on x axis)
        observation_label: Label for y-axis (observed axis)
        prediction_label: Label for x-axis (prediction axis)
        out_path: Path and filename of figure
        num_bins: number of bins in histogram
        figure_height: Height of the figure in inches. Width is scaled by number of tendencies
        fontsize: font size of axis labels and titles
        dpi: resolution of output figure in dots per inch
        cmap: colormap for histogram

    Returns:

    """

    output_columns = np.array(sorted(observations.keys()))
    prediction_columns = np.array(sorted(predictions.keys()))
    assert np.all(output_columns == prediction_columns), "Observations and predictions do not match"
    fig, axes = plt.subplots(1, output_columns.size, figsize=((figure_height + 0.5) * output_columns.size ,
                                                              figure_height))
    for a, ax in enumerate(axes):
        bin_values = np.linspace(np.minimum(predictions[output_columns[a]].min(),
                                            observations[output_columns[a]].min()),
                                 np.maximum(predictions[output_columns[a]].max(),
                                            observations[output_columns[a]].max()),
                                 num_bins)
        hist_grid, x_bins, y_bins, hist_obj = ax.hist2d(predictions[output_columns[a]],
                                                        observations[output_columns[a]],
                                                        cmin=1, bins=(bin_values, bin_values), norm=LogNorm(),
                                                        cmap=cmap)
        ax.plot(bin_values, bin_values, 'k--')
        plt.colorbar(hist_obj, ax=ax)
        ax.set_title(pretty_names[output_columns[a]], fontsize=fontsize)
        ax.set_xlabel(prediction_label, fontsize=fontsize)
        if a == 0:
            ax.set_ylabel(observation_label, fontsize=fontsize)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return


def distribution_histogram(distribution_dict, models, tendencies, model_colors,
                           out_path, num_bins=20, sub_width=4, fontsize=14, dpi=200, log_y=True):
    """
    Plot the histograms of each tendency distribution for multiple models. This plot allows one to compare the
    predicted and observed distributions of each model.

    Args:
        distribution_dict: Dictionary of distributions for each model and observations being evaluated
        models: List or array of model names (should be keys of distribution_dict)
        tendencies: List or array of tendency types (should be keys of each model in distribution_dict)
        model_colors: Dictionary of color strings for each model
        out_path: Path and file name for figure being generated
        num_bins: Number of bins for each histogram
        sub_width: Width of each sub plot in inches. Total figure is scaled by the subplot width
        fontsize: Font size for axis labels and titles
        dpi: resolution
        log_y: Whether to use a logarithmic scale for the y-axis of each subplot
    Returns:

    """
    num_models = len(models)
    num_tendencies = len(tendencies)
    fig, axes = plt.subplots(num_models, num_tendencies,
                             figsize=(sub_width * num_tendencies, sub_width * num_models))
    for m, model in enumerate(models):
        axes[m, 0].set_ylabel(model, fontsize=fontsize)
        for t, tendency in enumerate(tendencies):
            if m == 0:
                axes[m, t].set_title(pretty_names[tendency], fontsize=fontsize)
            axes[m, t].hist(distribution_dict[model][tendency], bins=num_bins, color=model_colors[model])
            if log_y:
                axes[m, t].set_yscale("log")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return