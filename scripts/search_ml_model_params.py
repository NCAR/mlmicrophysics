import numpy as np
import yaml
from dask.distributed import Client, LocalCluster, as_completed
import argparse
from os.path import exists, join
from os import mkdir
from mlmicrophysics.data import subset_data_files_by_date, log10_transform, neg_log10_transform
from mlmicrophysics.models import DenseNeuralNetwork, DenseGAN, parse_model_config_params
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

model_classes = {"RandomForestRegressor": RandomForestRegressor,
                 "DenseNeuralNetwork": DenseNeuralNetwork,
                 "DenseGAN": DenseGAN}
scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}
transforms = {"log10_transform": log10_transform,
              "neg_log10_transform": neg_log10_transform}

metrics = {"mse": mean_squared_error,
           "mae": mean_absolute_error,
           "r2": r2_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config) as config_file:
        config = yaml.load(config_file)

    train_files, val_files, test_files = subset_data_files_by_date(config["data_path"],
                                                                   config["data_end"], **config["subset_data"])
    input_scaler = scalers[config["input_scaler"]]()
    output_scaler = scalers[config["output_scaler"]]()
    train_input, train_output = assemble_data_files(train_files,
                                                    config["input_cols"],
                                                    config["output_cols"],
                                                    config["input_transforms"],
                                                    config["output_transforms"],
                                                    input_scaler,
                                                    output_scaler)
    cluster = LocalCluster(n_workers=args.proc)
    client = Client(cluster)
    train_input_pointer = client.scatter(train_input)
    train_output_pointer = client.scatter(train_output)
    val_input, val_output = assemble_data_files(val_files,
                                                config["input_cols"],
                                                config["output_cols"],
                                                config["input_transforms"],
                                                config["output_transforms"],
                                                input_scaler,
                                                output_scaler,
                                                train=False)
    val_input_pointer = client.scatter(val_input)
    val_output_pointer = client.scatter(val_output)
    submissions = []
    val_results = dict()
    for model_name, model_params in config["models"]:
        model_config_generator = parse_model_config_params(model_params,
                                                           config["num_param_samples"],
                                                           np.random.RandomState(config["random_seed"]))
        val_results[model_name] = []
        for model_config in model_config_generator:
            submissions.append(client.submit(validate_model_configuration, model_name, model_config,
                               config["input_cols"], config["output_cols"], train_input_pointer,
                               train_output_pointer, val_input_pointer, val_output_pointer,
                               config["metrics"]))
    for result in as_completed(submissions):
        print(result)
        val_results[result[0]].append(result[1])
    val_frames = {}
    if not exists(config["out_path"]):
        mkdir(config["out_path"])
    for model_name, scores in val_results.items():
        val_frames[model_name] = pd.concat(scores, ignore_index=True)
        val_frames[model_name].to_csv(join(config["out_path"],
                                           "val_scores_{0}.csv".format(model_name)),
                                      index_label="Index")
    client.close()
    cluster.close()
    return


def assemble_data_files(files, input_cols, output_cols, input_transforms, output_transforms,
                        input_scaler, output_scaler, train=True):
    all_input_data = []
    all_output_data = []
    for filename in files:
        data = pd.read_csv(filename, index_col="Index")
        all_input_data.append(data[input_cols])
        all_output_data.append(data[output_cols])
        del data
    combined_input_data = pd.concat(all_input_data)
    combined_output_data = pd.concat(all_output_data)
    del all_input_data[:]
    del all_output_data[:]
    for var, transform_name in input_transforms.items():
        combined_input_data.loc[:, var] = transforms[transform_name](combined_input_data[var])
    for var, transform_name in output_transforms.items():
        combined_output_data.loc[:, var] = transforms[transform_name](combined_output_data[var])
    if train:
        scaled_input_data = input_scaler.fit_transform(combined_input_data)
        scaled_output_data = output_scaler.fit_transform(combined_output_data)
    else:
        scaled_input_data = input_scaler.transform(combined_input_data)
        scaled_output_data = output_scaler.transform(combined_output_data)
    return scaled_input_data, scaled_output_data


def validate_model_configuration(model_name, model_config,
                                 input_cols, output_cols,
                                 train_input, train_output,
                                 val_input, val_output,
                                 metric_list):
    model_obj = model_classes[model_name](**model_config)
    model_obj.fit(train_input, train_output)
    model_preds = model_obj.predict(val_input)
    out_metrics = pd.DataFrame(dtype=float, index=metric_list, columns=output_cols)
    for metric in metric_list:
        out_metrics.loc[metric] = metrics[metric](val_output, model_preds)
    metrics_series = out_metrics.stack()
    metrics_series.index = metrics_series.index.to_series.str.join("_").values
    val_entry = pd.concat([pd.Series({"name": model_name}), pd.Series(model_config), metrics_series])
    return model_name, val_entry

if __name__ == "__main__":
    main()
