from mlmicrophysics.models import DenseNeuralNetwork
from mlmicrophysics.data import subset_data_files_by_date, assemble_data_files
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error
from mlmicrophysics.metrics import heidke_skill_score, peirce_skill_score, hellinger_distance, root_mean_squared_error, r2_corr
import argparse
import yaml
from os.path import join, exists
import os
from datetime import datetime
import logging
from tensorflow.keras.losses import huber
# from memory_profiler import profile
import optuna
from aimlutils.echo.src.trial_suggest import * 
from aimlutils.echo.src.base_objective import *
import tensorflow as tf


logger = logging.getLogger(__name__)

scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}

class_metrics = {"accuracy": accuracy_score,
                 "heidke": heidke_skill_score,
                 "peirce": peirce_skill_score,
                 "confusion": confusion_matrix}

reg_metrics = {"rmse": root_mean_squared_error,
               "mae": mean_absolute_error,
               "r2": r2_corr,
               "hellinger": hellinger_distance,
               "mse": mean_squared_error,
               "huber": huber}

def leaky(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def ranked_probability_score(y_true_discrete, y_pred_discrete):   
    y_pred_cumulative = np.cumsum(y_pred_discrete)
    y_true_cumulative = np.cumsum(y_true_discrete)
    return np.mean((y_pred_cumulative - y_true_cumulative) ** 2) / float(y_pred_discrete.shape[1] - 1)

# @profile(precision=4)
def objective(trial, config):
    
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    # Get list of hyperparameters from the config
    hyperparameters = config["optuna"]["parameters"]

    # Now update some hyperparameters via custom rules
    trial_hyperparameters = {}
    for param_name in hyperparameters.keys():
        trial_hyperparameters[param_name] = trial_suggest_loader(trial, hyperparameters[param_name])
    
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
    
    start = datetime.now()
    logger.info(f"Loading training data for trial: {trial.number}")
    train_files, val_files, test_files = subset_data_files_by_date(data_path, **config["subset_data"])
    scaled_input_train, \
    labels_train, \
    transformed_out_train, \
    scaled_out_train, \
    output_scalers, \
    meta_train = assemble_data_files(train_files, input_cols, output_cols, input_transforms,
                                         output_transforms, input_scaler, subsample=subsample)

    logger.info("Loading testing data")
    scaled_input_test, \
    labels_test, \
    transformed_out_test, \
    scaled_out_test, \
    output_scalers_test, \
    meta_test = assemble_data_files(test_files, input_cols, output_cols, input_transforms,
                                    output_transforms, input_scaler, output_scalers=output_scalers,
                                    train=False, subsample=subsample)
    logger.info(f"Finished loading data took: {datetime.now() - start}")
    start = datetime.now()
    input_scaler_df = pd.DataFrame({"mean": input_scaler.mean_, "scale": input_scaler.scale_},
                                   index=input_cols)
    out_scales_list = []
    for var in output_scalers.keys():
        for out_class in output_scalers[var].keys():
            if output_scalers[var][out_class] is not None:
                out_scales_list.append(pd.DataFrame({"mean": output_scalers[var][out_class].mean_,
                                                     "scale": output_scalers[var][out_class].scale_},
                                                    index=[var + "_" + str(out_class)]))
    out_scales_df = pd.concat(out_scales_list)
    out_scales_df.to_csv(join(out_path, "output_scale_values.csv"),
                         index_label="output")
    logger.info(f"Finished scaling data took: {datetime.now() - start}")

    beginning = datetime.now()
    logger.info(f"BEGINNING model training: {beginning}")
    
    with tf.device("/CPU:0"):
        # initialize neural networks that will only be defined once and trained in epoch loop
        classifiers = dict()
        for output_col in output_cols:
            classifiers[output_col] = DenseNeuralNetwork(hidden_layers=trial_hyperparameters["class_hidden_layers"],
                                                         hidden_neurons=trial_hyperparameters["class_hidden_neurons"],
                                                         lr=trial_hyperparameters["class_lr"],
                                                         l2_weight=trial_hyperparameters["class_l2_weight"],
                                                         activation=trial_hyperparameters["class_activation"],
                                                         batch_size=trial_hyperparameters["class_batch_size"],
                                                         **config["classifier_networks"])
        regressors = dict()
        for output_col in output_cols:
            regressors[output_col] = dict()
            for label in [l for l in list(output_transforms[output_col].keys()) if l != 0]:
                regressors[output_col][label] = DenseNeuralNetwork(hidden_layers=trial_hyperparameters["reg_hidden_layers"],
                                                       hidden_neurons=trial_hyperparameters["reg_hidden_neurons"],
                                                       lr=trial_hyperparameters["reg_lr"],
                                                       l2_weight=trial_hyperparameters["reg_l2_weight"],
                                                       activation=trial_hyperparameters["reg_activation"],
                                                       batch_size=trial_hyperparameters["reg_batch_size"],
                                                       **config["regressor_networks"])
        reg_index = []
        for output_col in output_cols:
            for label in list(output_transforms[output_col].keys()):
                if label != 0:
                    reg_index.append(output_col + f"_{label:d}")
        test_prediction_values = np.zeros((scaled_out_test.shape[0], len(reg_index)))
        test_prediction_labels = np.zeros(scaled_out_test.shape)
        logger.info(f"Finished initializing models took: {datetime.now() - beginning}")

        for epoch in range(config["epochs"]):
            logger.info(f"Training epoch: {epoch}")
            start = datetime.now()
            score = 0
            for o, output_col in enumerate(output_cols):
                logger.info(f"Train {output_col} Classifer - epoch: {epoch}")
                hist = classifiers[output_col].fit(scaled_input_train,
                                                   labels_train[output_col])
                logger.info(f"Evaluate Classifier: {output_col}")
                test_prediction_labels[:, o] = classifiers[output_col].predict(scaled_input_test)
                logger.info(f"test_prediction_labels[:, o] min: {np.min(test_prediction_labels[:, o])} max: {np.max(test_prediction_labels[:, o])}")
                true = OneHotEncoder(sparse=False).fit_transform(labels_test[output_col].to_numpy().reshape(-1, 1))
                pred = OneHotEncoder(sparse=False).fit_transform(pd.DataFrame(test_prediction_labels[:, o]))
                score += ranked_probability_score(true, pred)
                logger.info(f"Finished training epoch {epoch} of classifier {output_col} in: {datetime.now() - start}")            
                for l, label in enumerate(list(output_transforms[output_col].keys())):
                    start = datetime.now()
                    if label != 0:
                        logger.info(f"Train {output_col} - {label} Regressor - epoch: {epoch}")
                        hist = regressors[output_col][label].fit(scaled_input_train.loc[labels_train[output_col] == label],
                                                                 scaled_out_train.loc[labels_train[output_col] == label, output_col])
                        if label > 0:
                            out_label = "pos"
                        else:
                            out_label = "neg"
                        test_prediction_values[:, l] = output_scalers[output_col][label].inverse_transform(regressors[output_col][label].predict(scaled_input_test))
                        score += mean_squared_error(transformed_out_test.loc[labels_test[output_col] == label, output_col],
                                                    test_prediction_values[labels_test[output_col] == label, l])
                        logger.info(f"Finished training epoch {epoch} of regressor {output_col} and label {label} in: {datetime.now() - start}")
            
            trial.report(score, step = epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    logger.info(f"Running entire model took: {datetime.now() - beginning}")
    
    return score


class Objective(BaseObjective):

    def __init__(self, config, metric = "val_loss", device = "cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):

        result = objective(trial, conf)
        results_dictionary = {
            "val_loss": result
        }
        return results_dictionary
