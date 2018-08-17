import numpy as np
import yaml
from dask.distributed import Client, LocalCluster
import argparse
from os.path import exists, join
from mlmicrophysics.data import load_csv_data, subset_data_by_date
from mlmicrophysics.models import DenseNeuralNetwork, DenseGAN, parse_model_config_params
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
model_classes = {"RandomForestRegressor": RandomForestRegressor,
                 "DenseNeuralNetwork": DenseNeuralNetwork,
                 "DenseGAN": DenseGAN}
scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config) as config_file:
        config = yaml.load(config_file)
    for model_name, model_params in config["models"]:
        model_config_generator = parse_model_config_params(model_params,
                                                           config["num_param_samples"],
                                                           np.random.RandomState(config["random_seed"]))

    return


def validate_model_configuration(model_name, model_config, input_cols, output_cols):
    return

if __name__ == "__main__":
    main()
