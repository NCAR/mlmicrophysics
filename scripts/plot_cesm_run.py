
import numpy as np
from os.path import join, exists
from glob import glob
import xarray as xr
from mlmicrophysics.plots import timestep_input_distributions, timestep_input_maps
from os import makedirs

def main():
    model_path = "/glade/scratch/dgagne/TAU_run4_hist/"
    out_path = "/glade/p/cisl/aiml/dgagne/TAU_run4_plots/"
    if not exists(out_path):
        makedirs(out_path)
    h1_model_files = sorted(glob(join(model_path, "TAU_*.cam.h1.*.nc")))
    for h1_model_file in h1_model_files:
        print("Loading", h1_model_file)
        ds = xr.open_dataset(h1_model_file)
        times = ds["time"].values
        for t, time in enumerate(times):
            print(time)
            time_str = time.strftime("%Y%m%d_%H%M").strip()
            print(ds.sel({"time": time}))
            timestep_input_distributions(ds.sel({"time": time}), times[t],
                                         join(out_path, f"run4_input_dist_{time_str}.png"))
            timestep_input_maps(ds.sel({"time": time}), times[t],
                                         join(out_path, f"run4_input_maps_{time_str}.png"))
        ds.close()
    return


if __name__ == "__main__":
    main()
