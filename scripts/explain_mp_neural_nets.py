import numpy as np
import pandas as pd
from multiprocessing import Pool
import argparse
import yaml
from os.path import exists, join
import os
from mlmicrophysics.data import subset_data_files_by_date, assemble_data_files
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from mlmicrophysics.explain import partial_dependence_mp

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
    input_scaler = scalers[config["input_scaler"]]()
    subsample = config["subsample"]
    if not exists(out_path):
        os.makedirs(out_path)
    train_files, val_files, test_files = subset_data_files_by_date(data_path, "*.csv", **config["subset_data"])
    print("Loading training data")
    scaled_input_train, \
    labels_train, \
    transformed_out_train, \
    scaled_out_train, \
    output_scalers, \
    meta_train = assemble_data_files(train_files, input_cols, output_cols, input_transforms,
                                         output_transforms, input_scaler, subsample=subsample)

    print("Loading testing data")
    scaled_input_test, \
    labels_test, \
    transformed_out_test, \
    scaled_out_test, \
    output_scalers_test, \
    meta_test = assemble_data_files(test_files, input_cols, output_cols, input_transforms,
                                              output_transforms, input_scaler, output_scalers=output_scalers,
                                              train=False, subsample=subsample)
    input_scaler_df = pd.DataFrame({"mean": input_scaler.mean_, "scale": input_scaler.scale_},
                                   index=input_cols)
    if args.pdp:
        partial_dependence_mp(scaled_input_train, )
    return


if __name__ == "__main__":
    main()