SEED = 42
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

from datetime import datetime
import json
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
import sys
from typing import Any, Dict, List

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

import tensorflow as tf
from tensorflow import keras

# Import supporting functions
from .preprocessing.pipeline import Preprocessor

# Import CNN model
from .model.cnn import train_and_predict_cnn

# Import GRU model
from .model.gru import train_and_predict_gru

# Import Deep Convolutional LSTM model
from .model.deep_conv_lstm import train_and_predict_dcl

# Set seeds
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

###############################################################################################
###############################################################################################

def run_neural_networks(config, imu_dataset_files, LOG_DIR):
    preprocess = Preprocessor(imu_dataset_files, ratio=config.ratio, seed=SEED)
    len_train_set = preprocess.len_train()
    out = preprocess.fit_transform(window_size=config.window, window_shift=config.shift)
    X_train, y_train = out[0]
    X_test, y_test = out[1]

    y_test = keras.utils.to_categorical(y_test, config.classes)

    skf = StratifiedKFold(n_splits=config.folds)

    scores: Dict[str, Dict[str, List[Any]]] = {
        "logloss": {"train": [], "valid": [], "test": []},
        "accuracy": {"train": [], "valid": [], "test": []},
        "precision": {"train": [], "valid": [], "test": []},
        "recall": {"train": [], "valid": [], "test": []},
        "f1": {"train": [], "valid": [], "test": []},
        "cm": {"train": [], "valid": [], "test": []},
        "per_class_f1": {"train": [], "valid": [], "test": []},
    }

    valid_preds = np.zeros((X_train.shape[0], config.classes))
    test_preds = np.zeros((config.folds, X_test.shape[0], config.classes))
    models = []

    # Train model k times
    for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
        logging.info(f"Current fold: {i + 1}\n")

        # Split the data for training and validation 
        X_tr = X_train[train_index, :]
        X_val = X_train[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        y_tr = keras.utils.to_categorical(y_tr, config.classes)
        y_val = keras.utils.to_categorical(y_val, config.classes)

        logging.debug(f"{X_tr.shape=} {X_val.shape=} {X_test.shape=}")
        logging.debug(f"{y_tr.shape=} {y_val.shape=} {y_test.shape=}")

        # Train CNN
        if config.model == "cnn":
            pred_tr, pred_val, pred_test, model = train_and_predict_cnn(
                LOG_DIR, i, X_tr, X_val, X_test, y_tr, y_val, config
            )

        # Train GRU
        elif config.model == "gru":
            pred_tr, pred_val, pred_test, model = train_and_predict_gru(
                LOG_DIR, i, X_tr, X_val, X_test, y_tr, y_val, config
            )

        # Train deep convolutional LSTM
        elif config.model == "deep_conv_lstm":
            pred_tr, pred_val, pred_test, model = train_and_predict_dcl(
                LOG_DIR, i, X_tr, X_val, X_test, y_tr, y_val, config
            )

        models.append(model)
        trainable_params = int(sum(tf.size(var).numpy() for var in model.trainable_variables))
        logging.debug(f"Trainable parameters: {trainable_params}")

        valid_preds[valid_index] = pred_val
        test_preds[i] = pred_test

        for pred, X, y, mode in zip(
            [pred_tr, pred_val, pred_test], [X_tr, X_val, X_test], [y_tr, y_val, y_test], ["train", "valid", "test"]
        ):
            loss, acc = model.evaluate(X, y, verbose=0)
            pred = pred.argmax(axis=1)
            y = y.argmax(axis=1)
            scores["logloss"][mode].append(loss)
            scores["accuracy"][mode].append(acc)
            scores["precision"][mode].append(precision_score(y, pred, average="macro"))
            scores["recall"][mode].append(recall_score(y, pred, average="macro"))
            scores["f1"][mode].append(f1_score(y, pred, average="macro"))
            scores["cm"][mode].append(confusion_matrix(y, pred, normalize="true"))
            scores["per_class_f1"][mode].append(f1_score(y, pred, average=None))

    return scores, y_test, test_preds