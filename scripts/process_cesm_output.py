import argparse
import yaml
from os.path import exists, join
from mlmicrophysics.data import load_cam_output, unstagger_vertical, convert_to_dataframe
from mlmicrophysics.data import calc_pressure_field, calc_temperature, add_index_coords
from dask.distributed import Client, LocalCluster, wait
import traceback
from os import mkdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    if exists(args.config):
        with open(args.config) as config_file:
            config = yaml.load(config_file)
        model_ds = load_cam_output(config["model_path"])
        times = model_ds["time"].values
        model_ds.close()
        del model_ds
        print(times, len(times))
        if not exists(config["csv_path"]):
            mkdir(config["csv_path"])
        time_subsets = list(range(0, len(times), config["time_split_interval"])) + [len(times)]
        print(time_subsets)
        for t in range(len(time_subsets) - 1):
            sub_times = times[time_subsets[t]: time_subsets[t + 1]]
            print(sub_times)
            process_cesm_time_subset(sub_times,
                                      config["time_var"],
                                      config["model_path"],
                                      config["model_file_start"],
                                      config["model_file_end"],
                                      config["staggered_variables"],
                                      config["csv_variables"],
                                      config["subset_variable"],
                                      config["subset_threshold"],
                                      config["csv_path"])
    else:
        raise FileNotFoundError(args.config + " not found.")
    return


def process_cesm_time_subset(times, time_var, model_path, model_file_start, model_file_end, staggered_variables, csv_variables,
                             subset_variable, subset_threshold, csv_path):
    try:
        model_ds = load_cam_output(model_path, file_start=model_file_start, file_end=model_file_end)
        model_sub_ds = model_ds.sel(time=times)
        add_index_coords(model_sub_ds)
        for staggered_variable in staggered_variables:
            model_sub_ds[staggered_variable + "_lev"] = unstagger_vertical(model_sub_ds, staggered_variable)
        calc_pressure_field(model_sub_ds)
        calc_temperature(model_sub_ds)
        model_frame = convert_to_dataframe(model_sub_ds, csv_variables, times, time_var,
                                           subset_variable, subset_threshold)
        csv_filename = join(csv_path, model_file_start + "_{0:05d}_{1:05d}.csv".format(int(times[0] * 24),
                                                                                       int(times[-1] * 24)))
        model_frame.to_csv(csv_filename, index_label="Index")
        model_sub_ds.close()
        model_ds.close()
        del model_sub_ds
        del model_ds
        return 0
    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()