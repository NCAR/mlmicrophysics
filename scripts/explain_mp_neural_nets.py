import numpy as np
import pandas as pd
import argparse
import yaml
from os.path import exists, join
from glob import glob
import os
from mlmicrophysics.data import subset_data_files_by_date, assemble_data_files, repopulate_input_scaler, \
    repopulate_output_scalers, inverse_transform_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from mlmicrophysics.explain import partial_dependence_mp, partial_dependence_tau_mp

scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the training config file")
    parser.add_argument("-p", "--procs", type=int, default=1, help="Number of processors")
    parser.add_argument("-d", "--pdp", action="store_true", help="Calculate partial dependence")
    parser.add_argument("-v", "--vi", action="store_true", help="Calculate variable importance")
    parser.add_argument("-s", "--stat", action="store_true", help="Calc verification stats and plots")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file)
    data_path = config["data_path"]
    out_path = config["out_path"]
    input_cols = config["input_cols"]
    output_cols = config["output_cols"]
    input_transforms = config["input_transforms"]
    output_transforms = config["output_transforms"]
    np.random.seed(config["random_seed"])
    input_scaler = repopulate_input_scaler(join(out_path, "input_scale_values.csv"),
                                     config["input_scaler"])
    output_scalers = repopulate_output_scalers(join(out_path, "output_scale_values.csv"),
                                               output_transforms)
    subsample = config["subsample"]
    partial_dependence_config = config["partial_dependence"]
    train_files, val_files, test_files = subset_data_files_by_date(data_path, "*.csv", **config["subset_data"])
    print("Loading training data")
    scaled_input_train, \
    labels_train, \
    transformed_out_train, \
    scaled_out_train, \
    output_scalers, \
    meta_train = assemble_data_files(train_files, input_cols, output_cols, input_transforms,
                                     output_transforms, input_scaler,
                                     output_scalers=output_scalers,
                                     subsample=subsample, train=False)

    print("Loading testing data")
    scaled_input_test, \
    labels_test, \
    transformed_out_test, \
    scaled_out_test, \
    output_scalers_test, \
    meta_test = assemble_data_files(test_files, input_cols, output_cols, input_transforms,
                                    output_transforms, input_scaler, output_scalers=output_scalers,
                                    train=False, subsample=subsample)
    if args.pdp:
        model_files = sorted(glob(join(out_path, "*.h5")))
        pd_model_vals = {}
        pd_model_var_vals = {}
        for model_file in model_files:
            print(model_file)
            model_key = model_file.split("/")[-1][:-3]
            pd_model_vals[model_key], \
                pd_model_var_vals[model_key] = partial_dependence_mp(scaled_input_train,
                                                                     model_file,
                                                                     partial_dependence_config["var_val_count"],
                                                                     args.procs)
        transformed_input_train = input_scaler.inverse_transform(scaled_input_train)
        raw_input_train = inverse_transform_data(transformed_input_train, input_transforms)
        pd_tau_vals, pd_tau_var_vals = partial_dependence_tau_mp(raw_input_train,
                                                                 partial_dependence_config["var_val_count"],
                                                                 args.procs)


    return


if __name__ == "__main__":
    main()
