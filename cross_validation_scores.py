"""
MIT License

Copyright (c) 2020 takumiw

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from helpers.plot_cm import plot_confusion_matrix

import logging

def report_scores(scores, y_test, test_preds, config, LOG_DIR):
    """
    Report the results of model training as averages across folds and as an 
    ensamble prediction. Generate confusion matrices for the training, validation and 
    testing results

    Args: 
        scores: summary of the the training, validation and testing results
        y_test: labels for the testing data
        test_preds: model predictions for the testing data for each fold
        config: model settings
        LOG_DIR: directory to output confusion matrix
    """
    # Output Cross Validation Scores
    logging.debug("---Cross Validation Scores---")
    for mode in ["train", "valid", "test"]:
        logging.debug(f"---{mode}---")
        for metric in ["logloss", "accuracy", "precision", "recall", "f1"]:
            logging.debug(f"{metric}={np.mean(scores[metric][mode])}")
            logging.debug(f"{metric}={np.std(scores[metric][mode])}")

        class_f1_mat = scores["per_class_f1"][mode]
        class_f1_result = {}
        for class_id in range(config.classes):
            mean_class_f1 = np.mean([class_f1_mat[i][class_id] for i in range(config.folds)])
            class_f1_result[class_id] = mean_class_f1
        logging.debug(f"per-class f1={class_f1_result}")
    
    # Output Final Scores Averaged over Folds
    logging.debug("---Final Test Scores Averaged over Folds---")

    if config.model in ("cnn", "gru", "deep_conv_lstm"):
        y_test = y_test.argmax(axis=1)

    # Per-fold scores for std dev
    fold_accuracies = [accuracy_score(y_test, pred.argmax(axis=1)) for pred in test_preds]
    fold_precisions = [precision_score(y_test, pred.argmax(axis=1), average='macro') for pred in test_preds]
    fold_recalls = [recall_score(y_test, pred.argmax(axis=1), average='macro') for pred in test_preds]
    fold_f1s = [f1_score(y_test, pred.argmax(axis=1), average='macro') for pred in test_preds]

    # Ensemble prediction
    test_pred = np.mean(test_preds, axis=0).argmax(axis=1)

    logging.debug(f"accuracy={accuracy_score(y_test, test_pred):.4f} ± {np.std(fold_accuracies):.4f}")
    logging.debug(f"precision={precision_score(y_test, test_pred, average='macro', zero_division=np.nan):.4f} ± {np.std(fold_precisions):.4f}")
    logging.debug(f"recall={recall_score(y_test, test_pred, average='macro', zero_division=np.nan):.4f} ± {np.std(fold_recalls):.4f}")
    logging.debug(f"f1={f1_score(y_test, test_pred, average='macro', zero_division=np.nan):.4f} ± {np.std(fold_f1s):.4f}")
    logging.debug(f"per-class f1={f1_score(y_test, test_pred, average=None, zero_division=np.nan)}")

    # Plot comfusion matrix
    plot_confusion_matrix(
        cms=scores["cm"],
        labels=[
            "BRUSH_TEETH",
            "DRESSING",
            "DRINK",
            "EAT",
            "KITCHEN_BIN",
            "LAY",
            "LAY_SIT",
            "MEDICINE",
            "PREPARE_MEAL",
            "SHOWER",
            "SIT",
            "SIT_STAND",
            "STAIRS",
            "STAND",
            "USE_TOILET",
            "WALK",
            "WASH_FACE"
        ],
        path=f"{LOG_DIR}/confusion_matrix.pdf",
    )