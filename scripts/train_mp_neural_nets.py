from mlmicrophysics.models import DenseNeuralNetwork
from mlmicrophysics.data import subset_data_files_by_date, assemble_data_files
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from mlmicrophysics.metrics import heidke_skill_score, peirce_skill_score, hellinger_distance, root_mean_squared_error, r2_corr
import argparse
import yaml
from os.path import join, exists
import os
from datetime import datetime

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
               "hellinger": hellinger_distance}


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
    print(transformed_out_test.columns)
    print(transformed_out_test.index)
    meta_test.to_csv(join(out_path, "meta_test.csv"), index_label="index")
    input_scaler_df.to_csv(join(out_path, "input_scale_values.csv"), index_label="input")
    out_scales_list = []
    for var in output_scalers.keys():
        for out_class in output_scalers[var].keys():
            print(var, out_class)
            if output_scalers[var][out_class] is not None:
                out_scales_list.append(pd.DataFrame({"mean": output_scalers[var][out_class].mean_,
                                                     "scale": output_scalers[var][out_class].scale_},
                                                    index=[var + "_" + str(out_class)]))
    out_scales_df = pd.concat(out_scales_list)
    print(out_scales_df)
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
    classifier_scores = pd.DataFrame(0, index=output_cols, columns=["accuracy", "heidke", "peirce"])
    confusion_matrices = dict()
    reg_cols = ["rmse", "mae", "r2", "hellinger"]
    reg_scores = pd.DataFrame(0, index=reg_index, columns=reg_cols)
    l = 0
    for o, output_col in enumerate(output_cols):
        print("Train Classifer ", output_col)
        classifiers[output_col] = DenseNeuralNetwork(**config["classifier_networks"])
        classifiers[output_col].fit(scaled_input_train, labels_train[output_col],
                                    scaled_input_test, labels_test[output_col])
        classifiers[output_col].save_fortran_model(join(config["out_path"],
                                                        "dnn_{0}_class_fortran.nc".format(output_col[0:2])))
        classifiers[output_col].model.save(join(config["out_path"],"dnn_{0}_class.h5".format(output_col[0:2])))
        regressors[output_col] = dict()
        print("Evaluate Classifier", output_col)
        test_prediction_labels[:, o] = classifiers[output_col].predict(scaled_input_test)
        confusion_matrices[output_col] = confusion_matrix(labels_test[output_col],
                                                          test_prediction_labels  [:, o])
        for class_score in classifier_scores.columns:
            classifier_scores.loc[output_col, class_score] = class_metrics[class_score](labels_test[output_col],
                                                                                        test_prediction_labels[:, o])
        print(classifier_scores.loc[output_col])
        for label in list(output_transforms[output_col].keys()):
            if label != 0:
                print("Train Regressor ", output_col, label)
                regressors[output_col][label] = DenseNeuralNetwork(**config["regressor_networks"])
                regressors[output_col][label].fit(scaled_input_train.loc[labels_train[output_col] == label],
                                                     scaled_out_train.loc[labels_train[output_col] == label, output_col],
                                                     scaled_input_test.loc[labels_test[output_col] == label],
                                                     scaled_out_test.loc[labels_test[output_col] == label, output_col])

                if label > 0:
                    out_label = "pos"
                else:
                    out_label = "neg"
                regressors[output_col][label].save_fortran_model(join(config["out_path"],
                                                                      "dnn_{0}_{1}_fortran.nc".format(output_col[0:2],
                                                                                                      out_label)))
                regressors[output_col][label].model.save(join(config["out_path"],
                                                              "dnn_{0}_{1}.h5".format(output_col[0:2], out_label)))
                print("Test Regressor", output_col, label)
                test_prediction_values[:, l] = output_scalers[output_col][label].inverse_transform(regressors[output_col][label].predict(scaled_input_test))
                reg_label = output_col + f"_{label:d}"
                for reg_col in reg_cols:
                    reg_scores.loc[reg_label,
                                   reg_col] = reg_metrics[reg_col](transformed_out_test.loc[labels_test[output_col] == label,
                                                                                            output_col],
                                                                    test_prediction_values[labels_test[output_col] == label, l])
                print(reg_scores.loc[reg_label])
                l += 1
    print(f"Running the model took: {datetime.now() - beginning}")

    print("Saving data")
    classifier_scores.to_csv(join(out_path, "dnn_classifier_scores.csv"), index_label="Output")
    reg_scores.to_csv(join(out_path, "dnn_regressor_scores.csv"), index_label="Output")
    test_pred_values_df = pd.DataFrame(test_prediction_values, columns=reg_index)
    test_pred_labels_df = pd.DataFrame(test_prediction_labels, columns=output_cols)
    test_pred_values_df.to_csv(join(out_path, "test_prediction_values.csv"), index_label="index")
    test_pred_labels_df.to_csv(join(out_path, "test_prediction_labels.csv"), index_label="index")
    labels_test.to_csv(join(out_path, "test_cam_labels.csv"), index_label="index")
    transformed_out_test.to_csv(join(out_path, "test_cam_values.csv"), index_label="index")
    return

if __name__ == "__main__":
    main()
