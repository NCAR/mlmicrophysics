from mlmicrophysics.models import DenseNeuralNetwork
from mlmicrophysics.data import subset_data_files_by_date, assemble_data, output_quantile_curves
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import os
import argparse
import yaml
from os.path import exists, join
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    data_path = config["data_path"]
    out_path = config["out_path"]
    input_cols = config["input_cols"]
    output_cols = config["output_cols"]
    qc_thresh = config["qc_thresh"]
    input_scaler = QuantileTransformer(n_quantiles=config["n_quantiles"])
    output_scaler = QuantileTransformer(n_quantiles=config["n_quantiles"])
    subsample = config["subsample"]
    np.random.seed(config["random_seed"])

    if not exists(out_path):
        os.makedirs(out_path)
    files = {}
    files["train"], files["val"], files["test"] = subset_data_files_by_date(data_path, **config["subset_data"])
    subsets = ["train", "val", "test"]
    input_data = {}
    output_data = {}
    meta_data = {}
    input_quant_data = {}
    output_quant_data = {}
    for subset in subsets:
        input_data[subset], output_data[subset], meta_data[subset] = assemble_data(files[subset],
                                                                                   input_cols,
                                                                                   output_cols,
                                                                                   subsample=subsample)
        if subset == "train":
            input_quant_data[subset] = input_scaler.fit_transform(input_data[subset])
            output_quant_data[subset] = output_scaler.fit_transform(output_data[subset])
        else:
            input_quant_data[subset] = input_scaler.transform(input_data[subset])
            output_quant_data[subset] = output_scaler.transform(output_data[subset])
    if "scratch_path" in config["keys"]:
        for subset in subsets:
            input_quant_data[subset].to_parquet(join(config["scratch_path"], f"mp_quant_input_{subset}.parquet"))
            output_quant_data[subset].to_parquet(join(config["scratch_path"], f"mp_quant_output_{subset}.parquet"))
            output_data[subset].to_parquet(join(config["scratch_path"], f"mp_output_{subset}.parquet"))
            meta_data[subset].to_parquet(join(config["scratch_path"], f"mp_meta_{subset}.parquet"))
    output_quantile_curves(input_scaler, input_cols, join(config["out_path"], "input_quantile_scaler.nc"))
    output_quantile_curves(output_scaler, output_cols, join(config["out_path"], "output_quantile_scaler.nc"))
    with open(join(config["out_path"], "input_quantile_transform.pkl"), "wb") as in_quant_pickle:
        pickle.dump(input_scaler, in_quant_pickle)
    with open(join(config["out_path"], "output_quantile_transform.pkl"), "wb") as out_quant_pickle:
        pickle.dump(output_scaler, out_quant_pickle)
    emulator_nn = DenseNeuralNetwork(**config["emulator_nn"])
    emulator_nn.fit(input_quant_data["train"], output_quant_data["train"],
                    xv=input_quant_data["val"], yv=output_quant_data["val"])
    emulator_nn.save_fortran_model(join(out_path, "quantile_neural_net_fortran.nc"))
    emulator_nn.model.save("quantile_neural_net_keras.h5")




