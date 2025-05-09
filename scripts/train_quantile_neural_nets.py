import tensorflow.keras.backend as K
K.set_floatx('float64')
from mlmicrophysics.models import DenseNeuralNetwork
from mlmicrophysics.data import subset_data_files_by_date, assemble_data, output_quantile_curves
from mlmicrophysics.callbacks import get_callbacks
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
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
    data_path = config["data"]["data_path"]
    out_path = config["data"]["out_path"]
    input_cols = config["data"]["input_cols"]
    output_cols = config["data"]["output_cols"]
    if "qc_thresh" not in config.keys():
        qc_thresh = 1e-6
    else:
        qc_thresh = config["data"]["qc_thresh"]
    input_scaler = QuantileTransformer(n_quantiles=config["data"]["n_quantiles"])
    output_scaler = QuantileTransformer(n_quantiles=config["data"]["n_quantiles"])
    scratch_path = config["data"]["scratch_path"]
    subsample = config["data"]["subsample"]
    np.random.seed(config["data"]["random_seed"])

    if not exists(out_path):
        os.makedirs(out_path)
    # Save config file
    with open(join(out_path, "training.yml"), 'w') as file:
        yaml.dump(config, file)
    print("Data path is ", data_path)
    files = dict()
    files["train"], files["val"], files["test"] = subset_data_files_by_date(data_path, **config["data"]["subset_data"])
    subsets = ["train", "val", "test"]
    input_data = {}
    output_data = {}
    meta_data = {}
    input_quant_data = {}
    output_quant_data = {}
    input_data_filtered = {}
    output_data_filtered = {}
    input_data_df = {}
    output_data_df = {}
    input_data_nn = {}
    output_data_nn = {}
    meta_filtered = {}
    
    print("Loading data")
    for subset in subsets:
        print(subset)
        input_data[subset], output_data[subset], meta_data[subset] = assemble_data(files[subset],
                                                                                   input_cols,
                                                                                   output_cols,
                                                                                   subsample=subsample,
                                                                                   qc_thresh=qc_thresh)
        # Initialize the filtered dataframes
        # Need to find a better way to create a dictionary of dataframes
        # instead of initializing them with another dictionary of dataframes
        input_data_filtered[subset] = input_data[subset]
        input_data_df[subset] = input_data[subset]
        output_data_df[subset] = output_data[subset]
        # Setup for changes to switch to incloud tendencies
        input_data_df[subset] = pd.DataFrame(input_data[subset], columns=input_cols)
        output_data_df[subset] = pd.DataFrame(output_data[subset], columns=output_cols)
        columns_remove = ["CLOUD", "FREQR"]
        input_cols_nn = [x for x in input_cols if x not in columns_remove]
        # Filter all data
        cloud_frac_filter = input_data_df[subset]["CLOUD"].values > 1.0e-2
        qc_filter = input_data_df[subset]["QC_TAU_in"].values >= 1.0e-6
        nrtend_filter = (output_data_df[subset]["nrtend_TAU"].values > 1.0e-10) | (output_data_df[subset]["nrtend_TAU"].values < -1.0e-10)
        if "qctend_TAU" in output_cols:
            qctend_filter = output_data_df[subset]["qctend_TAU"].values < 0
            filter = cloud_frac_filter & qc_filter & qctend_filter & nrtend_filter
        else:
            filter = cloud_frac_filter & qc_filter & nrtend_filter
        # Keep filter step separate so we can write a parquet file that includes CLOUD and FREQR
        input_data_filtered[subset] = input_data_df[subset].loc[filter]
        output_data_filtered[subset] = output_data_df[subset].loc[filter]
        meta_filtered[subset] = meta_data[subset].loc[filter]
        # Remove the CLOUD and FREQR columns for training
        input_data_nn[subset] = input_data_filtered[subset].drop(columns_remove, axis=1)
        output_data_nn[subset] = output_data_filtered[subset]
        if subset == "train":
            input_quant_data[subset] = pd.DataFrame(input_scaler.fit_transform(input_data_nn[subset]), columns=input_cols_nn)
            output_quant_data[subset] = pd.DataFrame(output_scaler.fit_transform(output_data_nn[subset]), columns=output_cols)
        else:
            input_quant_data[subset] = pd.DataFrame(input_scaler.transform(input_data_nn[subset]), columns=input_cols_nn)
            output_quant_data[subset] = pd.DataFrame(output_scaler.transform(output_data_nn[subset]), columns=output_cols)
    if "scratch_path" in config["data"].keys():
        if not exists(config["data"]["scratch_path"]):
            os.makedirs(config["data"]["scratch_path"])
        for subset in subsets:
            # Save input post-filtering, but pre-removal of CLOUD and FREQR columns
            input_data_filtered[subset].to_parquet(join(scratch_path, f"mp_input_filtered_{subset}.parquet"))
            # Save input data post-filtering and post-removal of CLOUD and FREQR
            input_data_nn[subset].to_parquet(join(scratch_path, f"mp_input_nn_{subset}.parquet"))
            # Save input data post-quantile transform
            input_quant_data[subset].to_parquet(join(scratch_path, f"mp_quant_input_{subset}.parquet"))
            # Save output data pre-inverse transform (quantile transformed)
            output_quant_data[subset].to_parquet(join(scratch_path, f"mp_quant_output_{subset}.parquet"))
            # Save output data post-inverse transform
            output_data_nn[subset].to_parquet(join(scratch_path, f"mp_output_{subset}.parquet"))
            # Save original meta data
            meta_data[subset].to_parquet(join(scratch_path, f"mp_meta_{subset}.parquet"))
            # Save filtered meta data
            meta_filtered[subset].to_parquet(join(scratch_path, f"mp_meta_filtered_{subset}.parquet"))
    output_quantile_curves(input_scaler, input_cols_nn, join(out_path, "input_quantile_scaler.nc"))
    output_quantile_curves(output_scaler, output_cols, join(out_path, "output_quantile_scaler.nc"))
    with open(join(out_path, "input_quantile_transform.pkl"), "wb") as in_quant_pickle:
        pickle.dump(input_scaler, in_quant_pickle)
    with open(join(out_path, "output_quantile_transform.pkl"), "wb") as out_quant_pickle:
        pickle.dump(output_scaler, out_quant_pickle)
    print("Training")
    emulator_nn = DenseNeuralNetwork(**config["model"])
    emulator_nn.fit(input_quant_data["train"], output_quant_data["train"],
                    xv=input_quant_data["val"], yv=output_quant_data["val"],
                    callbacks=get_callbacks(config))
    emulator_nn.save_fortran_model(join(out_path, "quantile_neural_net_fortran.nc"))
    emulator_nn.model.save(join(out_path, "quantile_neural_net_keras.h5"))
    test_quant_preds = emulator_nn.predict(input_quant_data["test"], batch_size=40000)
    if type(test_quant_preds) is np.ndarray:
        df_test_quant_preds = pd.DataFrame(test_quant_preds, columns=["qctend_TAU", "nctend_TAU", "nrtend_TAU"])
        df_test_quant_preds.to_parquet(join(scratch_path, f"mp_quant_output_predicted.parquet"))
    else:
        test_quant_preds.to_parquet(join(scratch_path, f"mp_quant_output_predicted.parquet"))

    test_preds = output_scaler.inverse_transform(test_quant_preds)
    df_test_preds = pd.DataFrame(test_preds, columns=["qctend_TAU", "nctend_TAU", "nrtend_TAU"])
    df_test_preds.to_parquet(join(scratch_path, f"mp_output_predicted.parquet"))
    r2_test_scores = np.zeros(len(output_cols))
    for o, output_col in enumerate(output_cols):
        r2_test_scores[o] = r2_score((output_quant_data["test"][output_col].values),
                                     (test_quant_preds[:, o]))
        print(output_col, r2_test_scores[o])

if __name__ == "__main__":
    main()
