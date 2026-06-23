SEED = 42
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

from datetime import datetime
import json
from collections import Counter
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
import sys
from typing import Any, Dict, List
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

# Import tiny transformer model
from .train.models.tiny_transformer import TinyTransformer
from .train.train_tiny_transformer import train_tiny_transformer

# Import transformer encoder model
from .train.models.transformer_encoder import IMUTransformerEncoder
from .train.train_transformer_encoder import train_transformer

# Import Data preprocessing pipeline
from .preprocessing.pipeline import Preprocessor
# from preprocessing.pipeline_copy import Preprocessor

# Import supporting functions
from .preprocessing.dataset import CustomDataset
from .evaluate.evaluate_classes import eval_classes_te
from .evaluate.evaluate_classes import eval_classes_tte

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from itertools import combinations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

###############################################################################################
###############################################################################################

def run_transformers(config, imu_dataset_files, LOG_DIR):
    """
    Train and evaluate the transformer models

    Inputs:
        config: model and training settings
        imu_dataset_files: tensors containing training data
        LOG_DIR: logging directory
    """
    # Set seeds for reproducibility 
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    preprocess = Preprocessor(imu_dataset_files, config.ratio, SEED)

    out = preprocess.fit_transform(config.window, config.shift)
    X_train, y_train = out[0]
    X_test, y_test = out[1]

    scores: Dict[str, Dict[str, List[Any]]] = {
        "logloss": {"train": [], "valid": [], "test": []},
        "accuracy": {"train": [], "valid": [], "test": []},
        "precision": {"train": [], "valid": [], "test": []},
        "recall": {"train": [], "valid": [], "test": []},
        "f1": {"train": [], "valid": [], "test": []},
        "cm": {"train": [], "valid": [], "test": []},
        "per_class_f1": {"train": [], "valid": [], "test": []},
    }

    skf = StratifiedKFold(n_splits=config.folds)

    valid_preds = np.zeros((X_train.shape[0], config.classes))
    test_preds = np.zeros((config.folds, X_test.shape[0], config.classes))

    for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
        logging.info(f"Fold {i+1}")

        X_tr, X_val = X_train[train_index], X_train[valid_index]
        y_tr, y_val = y_train[train_index], y_train[valid_index]

        logging.info(f"Unique training labels: {np.unique(y_tr)}")
        logging.info(f"Unique validation labels: {np.unique(y_val)}")

        logging.info(f"\n\
            Window size: {config.window}\n \
            Window shift: {config.shift}\n \
            Batch size: {config.batch}\n \
            Dimension: {config.dimension}\n \
            Depth: {config.depth}\n \
            Head: {config.heads}\n \
            Learning rate: {config.learning_rate}\n \
            Weight decay: {config.weight_decay}\n \
        ")

        if config.model == "tiny_transformer":
            logging.info(f"\n\
                Patch size: {config.patch}\n \
            ")

        train_loader = DataLoader(CustomDataset(X_tr, y_tr), batch_size=config.batch)
        val_loader = DataLoader(CustomDataset(X_val, y_val), batch_size=config.batch)
        test_loader = DataLoader(CustomDataset(X_test, y_test), batch_size=config.batch)

        if config.model == "tiny_transformer":
            best_state, best_val, best_epoch = train_tiny_transformer(train_loader, val_loader, config, i, LOG_DIR)
            model = TinyTransformer(in_channels=config.features, window_size=config.window, patch_size=config.patch,
                                        dim=config.dimension, depth=config.depth, heads=config.heads, mlp_ratio=config.mlp_ratio,
                                        num_classes=config.classes, drop=config.drop).to(config.device)

        elif config.model == "transformer_encoder":
            best_state, best_val, best_epoch = train_transformer(train_loader, val_loader, config, config.device, i, LOG_DIR)
            model = IMUTransformerEncoder(config).to(config.device)

        model.load_state_dict(best_state)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.debug(f'Total number of parameters: {total_params}')
        model.eval()

        if config.model == "tiny_transformer":
            pred_tr, labels_tr = eval_classes_tte(DataLoader(CustomDataset(X_tr, y_tr), batch_size=config.batch), model, config.device)
            pred_val, labels_val = eval_classes_tte(DataLoader(CustomDataset(X_val, y_val), batch_size=config.batch), model, config.device)
            pred_test, labels_test = eval_classes_tte(test_loader, model, config.device)

        elif config.model == "transformer_encoder":
            pred_tr, labels_tr = eval_classes_te(DataLoader(CustomDataset(X_tr, y_tr), batch_size=config.batch), model, config.device)
            pred_val, labels_val = eval_classes_te(DataLoader(CustomDataset(X_val, y_val), batch_size=config.batch), model, config.device)
            pred_test, labels_test = eval_classes_te(test_loader, model, config.device)

        valid_preds[valid_index] = pred_val
        test_preds[i] = pred_test

        # Populate scoress
        criterion = nn.CrossEntropyLoss()
        for pred, labels, mode in zip(
            [pred_tr, pred_val, pred_test],
            [labels_tr, labels_val, labels_test],
            ["train", "valid", "test"]
        ):
            pred_hard = pred.argmax(axis=1)
            logits_tensor = torch.tensor(np.log(pred + 1e-9)) # approx logloss via CE
            loss = criterion(logits_tensor, torch.tensor(labels)).item()

            scores["logloss"][mode].append(loss)
            scores["accuracy"][mode].append(accuracy_score(labels, pred_hard))
            scores["precision"][mode].append(precision_score(labels, pred_hard, average="macro", zero_division=np.nan))
            scores["recall"][mode].append(recall_score(labels, pred_hard, average="macro", zero_division=np.nan))
            scores["f1"][mode].append(f1_score(labels, pred_hard, average="macro", zero_division=np.nan))
            scores["cm"][mode].append(confusion_matrix(labels, pred_hard, normalize="true", labels=list(range(config.classes))))
            scores["per_class_f1"][mode].append(f1_score(labels, pred_hard, average=None, labels=list(range(config.classes)), zero_division=np.nan))

    return scores, y_test, test_preds