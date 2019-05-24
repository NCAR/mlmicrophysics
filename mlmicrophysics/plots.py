import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs


def timestep_input_distributions(timestep_ds,
                                 time,
                                 out_path,
                                 figsize=(8, 8),
                                 fontsize=14,
                                 dpi=200):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    variables = ["QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in"]
    bins = [(-18, -3, 0.5), (-18, -3, 0.5), (-12, 6, 0.5), (-12, 6, 0.5)]
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
    fig, axes = plt.subplots(2, 2, figsize=figsize, projection=ccrs.PlateCarree())
    variables = ["QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in"]
    bins = [(-18, -3, 0.5), (-18, -3, 0.5), (-12, 6, 0.5), (-12, 6, 0.5)]
    for a, ax in enumerate(axes.ravel()):
        ax.coastlines()
        input_vals = timestep_ds[variables[a]].values
        log_input_vals = np.ma.array(np.where(input_vals > 0, np.log10(input_vals), np.nan), mask=input_vals == 0)
        pc = ax.pcolormesh(timestep_ds["lon"], timestep_ds["lat"], log_input_vals, vmin=bins[a][0], vmax=bins[a][1])
        plt.colorbar(pc, ax=ax)
        ax.set_title(variables[a].split("_")[0], fontsize=fontsize)
    fig.suptitle("Distributions Valid " + time.strftime("%Y-%m-%d %H:%M"))
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()