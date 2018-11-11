import numpy as np
import yaml
from dask.distributed import Client, LocalCluster, as_completed
import argparse
from os.path import exists, join
from os import makedirs
from mlmicrophysics.data import subset_data_files_by_date, assemble_data_files
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from mlmicrophysics.metrics import hellinger_distance, heidke_skill_score, peirce_skill_score
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform, expon
import pandas as pd
import traceback

scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}


def sampler_generator(ps):
    for params in ps:
        yield params


def parse_model_config_params(model_params, num_settings, random_state):
    """

    Args:
        model_params:
        num_settings:
        random_state:

    Returns:

    """
    param_distributions = dict()
    dist_types = dict(randint=randint, expon=expon, uniform=uniform)
    for param, param_value in model_params.items():
        if param_value[0] in ["randint", "expon", "uniform"]:
            param_distributions[param] = dist_types[param_value[0]](*param_value[1:])
        else:
            param_distributions[param] = param_value
    return sampler_generator(ParameterSampler(param_distributions, n_iter=num_settings, random_state=random_state))


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
    train_input, \
        train_output_labels, \
        train_transformed_output, \
        train_scaled_output, \
        output_scalers = assemble_data_files(train_files,
                                             config["input_cols"],
                                             config["output_cols"],
                                             config["input_transforms"],
                                             config["output_transforms"],
                                             input_scaler,
                                             subsample=config["subsample"])
    print("Train Input Size:", train_input.shape)
    val_input, \
        val_output_labels, \
        val_transformed_output, \
        val_scaled_output, \
        output_scalers = assemble_data_files(val_files,
                                             config["input_cols"],
                                             config["output_cols"],
                                             config["input_transforms"],
                                             config["output_transforms"],
                                             input_scaler,
                                             output_scalers=output_scalers,
                                             train=False,
                                             subsample=config["subsample"])
    print("Val Input Size:", val_input.shape)
    cluster = LocalCluster(n_workers=args.proc, threads_per_worker=1)
    client = Client(cluster)
    print(client)
    train_input_link = client.scatter(train_input)
    train_labels_link = client.scatter(train_output_labels)
    train_scaled_output_link = client.scatter(train_scaled_output)
    val_input_link = client.scatter(val_input)
    val_output_labels_link = client.scatter(val_output_labels)
    val_scaled_output_link = client.scatter(val_scaled_output)
    submissions = []
    if not exists(config["out_path"]):
        makedirs(config["out_path"])
    for class_model_name, class_model_params in config["classifier_models"].items():

        for reg_model_name, reg_model_params in config["regressor_models"].items():
            rs = np.random.RandomState(config["random_seed"])
            class_model_config_generator = parse_model_config_params(class_model_params,
                                                                     config["num_param_samples"],
                                                                     rs)
            reg_model_config_generator = parse_model_config_params(reg_model_params,
                                                                   config["num_param_samples"],
                                                                   rs)
            class_model_configs = []
            reg_model_configs = []
            for s in range(config["num_param_samples"]):
                class_model_config = next(class_model_config_generator)
                reg_model_config = next(reg_model_config_generator)
                class_model_configs.append(class_model_config)
                reg_model_configs.append(reg_model_config)
                config_index = f"{class_model_name}_{reg_model_name}_{s:04}"
                submissions.append(client.submit(validate_model_configuration,
                                                 class_model_name, class_model_config,
                                                 reg_model_name, reg_model_config, config_index,
                                                 train_input_link, train_labels_link,
                                                 train_scaled_output_link,
                                                 val_input_link, val_output_labels_link,
                                                 val_scaled_output_link,
                                                 config["classifier_metrics"],
                                                 config["regressor_metrics"]))
            class_config_frame = pd.DataFrame(class_model_configs)
            reg_config_frame = pd.DataFrame(reg_model_configs)
            class_config_frame.to_csv(join(config["out_path"],
                                           f"{class_model_name}_{reg_model_name}_classifier_params.csv"),
                                      index_label="Config")
            reg_config_frame.to_csv(join(config["out_path"],
                                         f"{class_model_name}_{reg_model_name}_regressor_params.csv"))
            result_count = 0
            for out in as_completed(submissions):
                if out.status == "finished":
                    result = out.result()
                    print(result)
                    if result_count == 0:
                        result.to_frame().T.to_csv(join(config["out_path"],
                                                   f"{class_model_name}_{reg_model_name}_metrics.csv"),
                                                   index_label="Config")
                    else:
                        result.to_frame().T.to_csv(join(config["out_path"],
                                                   f"{class_model_name}_{reg_model_name}_metrics.csv"),
                                                   header=False,
                                                   mode="a")
                    result_count += 1
                else:
                    tb = out.traceback()
                    for line in traceback.format_tb(tb):
                        print(line)
            del submissions[:]
    client.close()
    cluster.close()
    return


def validate_model_configuration(classifier_model_name, classifier_model_config,
                                 regressor_model_name, regressor_model_config, config_index,
                                 train_scaled_input, train_labels, train_scaled_output,
                                 val_scaled_input, val_labels, val_scaled_output,
                                 classifier_metric_list, regressor_metric_list):
    """
    Train a single machine learning model configuration to predict each microphysical tendency.

    Args:
        classifier_model_name:
        classifier_model_config:
        regressor_model_name:
        regressor_model_config:
        config_index:
        train_scaled_input:
        train_labels:
        train_scaled_output:
        val_scaled_input:
        val_labels:
        val_scaled_output:
        classifier_metric_list:
        regressor_metric_list:

    Returns:

    """
    from mlmicrophysics.models import DenseNeuralNetwork, DenseGAN
    import keras.backend as K
    metrics = {"mse": mean_squared_error,
               "mae": mean_absolute_error,
               "r2": r2_score,
               "hellinger": hellinger_distance,
               "acc": accuracy_score,
               "hss": heidke_skill_score,
               "pss": peirce_skill_score}
    sess = K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
    K.set_session(sess)
    with sess.as_default():
        model_classes = {"RandomForestRegressor": RandomForestRegressor,
                         "RandomForestClassifier": RandomForestClassifier,
                         "DenseNeuralNetwork": DenseNeuralNetwork,
                         "DenseGAN": DenseGAN}
        classifier_models = {}
        regressor_models = {}
        output_label_preds = pd.DataFrame(0, index=val_labels.index, columns=val_labels.columns,
                                          dtype=np.int32)
        output_preds = pd.DataFrame(0, index=val_scaled_output.index, columns=val_scaled_output.columns,
                                    dtype=np.float32)
        output_regressor_preds = pd.DataFrame(0, index=val_scaled_output.index, columns=val_scaled_output.columns,
                                    dtype=np.float32)
        output_metric_columns = []
        for output_col in train_scaled_output.columns:
            for metric in classifier_metric_list:
                output_metric_columns.append(output_col + "_" + metric)
            for metric in regressor_metric_list:
                output_metric_columns.append(output_col + "_" + metric)
            unique_labels = np.unique(train_labels[output_col])
            for unique_label in unique_labels:
                for metric in regressor_metric_list:
                    output_metric_columns.append(f"{output_col}_{unique_label}_{metric}")
        output_metrics = pd.Series(index=output_metric_columns, name=config_index, dtype=np.float32)
        for output_col in train_scaled_output.columns:
            print(output_col)
            unique_labels = np.unique(train_labels[output_col])
            if unique_labels.size > 1:
                if classifier_model_name in ["DenseNeuralNetwork", "DenseGAN"]:
                    classifier_models[output_col] = model_classes[classifier_model_name](outputs=unique_labels.size,
                                                                                         classifier=True,
                                                                                         **classifier_model_config)
                else:
                    classifier_models[output_col] = model_classes[classifier_model_name](**classifier_model_config)
                classifier_models[output_col].fit(train_scaled_input, train_labels[output_col])
                output_label_preds.loc[:, output_col] = classifier_models[output_col].predict(val_scaled_input)
                for metric in classifier_metric_list:
                    output_metrics[output_col + "_" + metric] = metrics[metric](val_labels[output_col].values,
                                                                                output_label_preds[output_col].values)
            else:
                output_label_preds.loc[:, output_col] = unique_labels[0]
            regressor_models[output_col] = {}
            for label in unique_labels:
                if label != 0:
                    if regressor_model_name in ["DenseNeuralNetwork", "DenseGAN"]:
                        regressor_models[output_col][label] = model_classes[regressor_model_name](classifier=False,
                                                                                                  **regressor_model_config)
                    else:
                        regressor_models[output_col][label] = model_classes[regressor_model_name](**regressor_model_config)
                    regressor_models[output_col][label].fit(train_scaled_input.loc[train_labels[output_col] == label],
                                                            train_scaled_output.loc[train_labels[output_col] == label,
                                                                                    output_col])
                    if np.count_nonzero(output_label_preds[output_col] == label) > 0:
                        output_preds.loc[output_label_preds[output_col] == label,
                                         output_col] = regressor_models[output_col][
                            label].predict(val_scaled_input.loc[output_label_preds[output_col] == label])
                    output_regressor_preds.loc[val_labels[output_col] == label,
                                               output_col] = regressor_models[output_col][
                        label].predict(val_scaled_input.loc[val_labels[output_col] == label])
                    for metric in regressor_metric_list:
                        output_metrics[f"{output_col}_{label}_{metric}"] = metrics[metric](val_scaled_output.loc[val_labels[output_col] == label, output_col].values,
                                                                                           output_regressor_preds.loc[val_labels[output_col] == label, output_col].values)
            for metric in regressor_metric_list:
                output_metrics[output_col + "_" + metric] = metrics[metric](val_scaled_output[output_col].values,
                                                                            output_preds[output_col].values)
    return output_metrics


if __name__ == "__main__":
    main()
