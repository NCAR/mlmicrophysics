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

import optuna
from aimlutils.hyper_opt.utils import trial_suggest_loader
from aimlutils.hyper_opt.base_objective import *
from aimlutils.hyper_opt.utils import KerasPruningCallback


logger = logging.getLogger(__name__)

scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}

class_metrics = {"accuracy": accuracy_score,
                 "heidke": heidke_skill_score,
                 "peirce": peirce_skill_score}

reg_metrics = {"rmse": root_mean_squared_error,
               "mae": mean_absolute_error,
               "r2": r2_corr,
               "hellinger": hellinger_distance,
               "mse": mean_squared_error}

def leaky(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def ranked_probability_score(y_true_discrete, y_pred_discrete):   
    y_pred_cumulative = np.cumsum(y_pred_discrete)
    y_true_cumulative = np.cumsum(y_true_discrete)
    return np.mean((y_pred_cumulative - y_true_cumulative) ** 2) / float(y_pred_discrete.shape[1] - 1)


def objective(trial, config):

    # Get list of hyperparameters from the config
    hyperparameters = config["optuna"]["parameters"]

    # Now update some hyperparameters via custom rules
    class_activation = trial_suggest_loader(trial, hyperparameters["class_activation"])
    class_hidden_layers = trial_suggest_loader(trial, hyperparameters["class_hidden_layers"])
    class_hidden_neurons = trial_suggest_loader(trial, hyperparameters["class_hidden_neurons"])
    class_lr = trial_suggest_loader(trial, hyperparameters["class_lr"])
    class_l2_weight = trial_suggest_loader(trial, hyperparameters["class_l2_weight"])
    reg_activation = trial_suggest_loader(trial, hyperparameters["reg_activation"])
    reg_hidden_layers = trial_suggest_loader(trial, hyperparameters["reg_hidden_layers"])
    reg_hidden_neurons = trial_suggest_loader(trial, hyperparameters["reg_hidden_neurons"])
    reg_lr = trial_suggest_loader(trial, hyperparameters["reg_lr"])
    reg_l2_weight = trial_suggest_loader(trial, hyperparameters["reg_l2_weight"])    
    
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
    train_files, val_files, test_files = subset_data_files_by_date(data_path, **config["subset_data"])
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
    out_scales_list = []
    for var in output_scalers.keys():
        for out_class in output_scalers[var].keys():
            print(var, out_class)
            if output_scalers[var][out_class] is not None:
                out_scales_list.append(pd.DataFrame({"mean": output_scalers[var][out_class].mean_,
                                                     "scale": output_scalers[var][out_class].scale_},
                                                    index=[var + "_" + str(out_class)]))
    out_scales_df = pd.concat(out_scales_list)
    out_scales_df.to_csv(join(out_path, "output_scale_values.csv"),
                         index_label="output")

    beginning = datetime.now()
    print(f"BEGINNING: {beginning}")
    classifiers = dict()
    regressors = dict()
    reg_index = []
    for output_col in output_cols:
        for label in list(output_transforms[output_col].keys()):
            if label != 0:
                reg_index.append(output_col + f"_{label:d}")
    test_prediction_values = np.zeros((scaled_out_test.shape[0], len(reg_index)))
    test_prediction_labels = np.zeros(scaled_out_test.shape)
    l = 0
    score = 0
    for o, output_col in enumerate(output_cols):
        print("Train Classifer ", output_col)
        classifiers[output_col] = DenseNeuralNetwork(hidden_layers=class_hidden_layers,
                                                     hidden_neurons=class_hidden_neurons,
                                                     lr=class_lr,
                                                     l2_weight=class_l2_weight,
                                                     activation=class_activation,
                                                     **config["classifier_networks"])
        hist = classifiers[output_col].fit(scaled_input_train,
                                           labels_train[output_col],
                                           scaled_input_test,
                                           labels_test[output_col],
                                           callbacks=[KerasPruningCallback(trial, "val_loss")])
        logger.info(f"finished with classifier {output_col}")
        regressors[output_col] = dict()
        print("Evaluate Classifier", output_col)
        test_prediction_labels[:, o] = classifiers[output_col].predict(scaled_input_test)
        true = OneHotEncoder(sparse=False).fit_transform(labels_test[output_col].to_numpy().reshape(-1, 1))
        pred = OneHotEncoder(sparse=False).fit_transform(pd.DataFrame(test_prediction_labels[:, o]))
        score += ranked_probability_score(true, pred)
        for label in list(output_transforms[output_col].keys()):
            if label != 0:
                print("Train Regressor ", output_col, label)
                regressors[output_col][label] = DenseNeuralNetwork(hidden_layers=reg_hidden_layers,
                                                                   hidden_neurons=reg_hidden_neurons,
                                                                   lr=reg_lr,
                                                                   l2_weight=reg_l2_weight,
                                                                   activation = reg_activation,
                                                                   **config["regressor_networks"])
                hist = regressors[output_col][label].fit(scaled_input_train.loc[labels_train[output_col] == label],
                                                         scaled_out_train.loc[labels_train[output_col] == label, output_col],
                                                         scaled_input_test.loc[labels_test[output_col] == label],
                                                         scaled_out_test.loc[labels_test[output_col] == label, output_col],
                                                         callbacks=[KerasPruningCallback(trial, "val_loss")])
                logger.info(f"finished with regressor {output_col}")
                
                if label > 0:
                    out_label = "pos"
                else:
                    out_label = "neg"
                print("Test Regressor", output_col, label)
                test_prediction_values[:, l] = output_scalers[output_col][label].inverse_transform(regressors[output_col][label].predict(scaled_input_test))
                score += mean_squared_error(transformed_out_test.loc[labels_test[output_col] == label, output_col],
                                            test_prediction_values[labels_test[output_col] == label, l])
                l += 1
    print(f"Running the model took: {datetime.now() - beginning}")
    
    return score

class Objective(BaseObjective):

    def __init__(self, study, config, metric = "val_loss", device = "cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, study, config, metric, device)

    def train(self, trial, conf):

        result = objective(trial, conf)

        results_dictionary = {
            "val_loss": result
        }
        return results_dictionary

if __name__ == "__main__":
    main()
    print("Starting script...")

    # parse arguments from config/yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    study = optuna.create_study(direction=config["direction"],
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, config, n_trials=config["n_trials"])
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
