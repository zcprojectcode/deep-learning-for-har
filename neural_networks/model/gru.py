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

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Activation, Dropout, Conv1D, GRU, 
    Reshape, BatchNormalization, GlobalAveragePooling1D,
    TimeDistributed, MaxPooling1D, Input, Bidirectional,
    SpatialDropout1D
)
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras import backend as K

from .callbacks.learning_curves import plot_learning_history, plot_model
from .callbacks.keras_callback import create_callback

SEED = 42
# Set seeds
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

def train_and_predict_gru(LOG_DIR: str, fold_id: int, X_train: np.ndarray, X_valid: np.ndarray,
                        X_test: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray, config
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Model]:
    """
    Train GRU

    Args:
        X_train, X_valid, X_test: input signals of shape (num_samples, window_size, num_channels)
        y_train, y_valid, y_test: labels 

    Returns:
        pred_train: train prediction
        pred_valid: validation prediction
        pred_test: test prediction
        model: best model
    """

    model = build_model(
        window = config.window,
        features = config.features,
        hidden_layer = config.hidden_layer,
        l2_lambda = config.l2_lambda,
        spatial_dropout = config.spatial_dropout, 
        dropout = config.dropout, 
        classes = config.classes,
        learning_rate = config.learning_rate
    )

    plot_model(model, path=f"{LOG_DIR}/model.png")

    callbacks = create_callback(
        model=model,
        path_chpt=f"{LOG_DIR}/trained_model_fold{fold_id}.keras",
        verbose=10,
        epochs=config.epochs,
    )

    fit = model.fit(
        X_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=config.verbose,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
    )

    plot_learning_history(fit=fit, path=f"{LOG_DIR}/history_fold{fold_id}.png")
    model = keras.models.load_model(f"{LOG_DIR}/trained_model_fold{fold_id}.keras")

    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    K.clear_session()
    return pred_train, pred_valid, pred_test, model


def build_model(window = 600, features = 270, hidden_layer = 256, l2_lambda = 0.001, 
                spatial_dropout = 0.3, dropout = 0.4, classes = 17, learning_rate = 0.001
            ) -> Model:

    model = Sequential([
        Input(shape=(window, features)),
        BatchNormalization(),

        # GRU Layers
        GRU(hidden_layer * 2, return_sequences=True,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(l2_lambda)),

        SpatialDropout1D(spatial_dropout),

        GRU(hidden_layer, return_sequences=False,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(l2_lambda)),

        Dropout(dropout),
        Dense(hidden_layer // 2, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        Dropout(dropout),
        Dense(classes, activation='softmax')
    ])

    model.compile(
        optimizer = optimizers.Adam(learning_rate = learning_rate, clipnorm = 1.0),
        loss = 'categorical_crossentropy',
        metrics=['accuracy']
    )

    return model