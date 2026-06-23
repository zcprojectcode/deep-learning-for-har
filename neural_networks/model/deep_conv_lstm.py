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
    Dense, Activation, Dropout, Conv1D, LSTM, 
    Reshape, BatchNormalization, GlobalAveragePooling1D,
    TimeDistributed, MaxPooling1D, Input, Bidirectional
)
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras import backend as K

from .callbacks.learning_curves import plot_learning_history, plot_model
from .callbacks.keras_callback import create_callback

SEED = 42
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
np.random.seed(SEED)

def train_and_predict_dcl(LOG_DIR: str, fold_id: int, X_train: np.ndarray, X_valid: np.ndarray, 
                        X_test: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray, config,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Model]:
    """
    Train DeepConvLSTM

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
        input_shape=X_train.shape[1:], 
        output_dim=y_train.shape[1], 
        lr=config.learning_rate,
        lstm_units = config.lstm_units,
        dropout_rate = config.dropout_rate,
        lstm_segments = config.lstm_segments
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


def build_model(input_shape = (600, 270), output_dim = 17, lr = 0.001, lstm_units = 128,
            dropout_rate = 0.5, lstm_segments = 15) -> Model:
    """
    Deep convolutional LSTM for HAR
    """
    n_timesteps, n_features = input_shape

    total_pool_factor = 4
    conv_filters = 128

    compressed_timesteps = n_timesteps // total_pool_factor

    if compressed_timesteps % lstm_segments != 0:
        raise ValueError(
            f"lstm_segments={lstm_segments} must evenly divide compressed "
            f"timesteps={compressed_timesteps} (input {n_timesteps} // pool {total_pool_factor}). "
            f"Choose from: {[s for s in range(1, compressed_timesteps+1) if compressed_timesteps % s == 0]}"
        )

    steps_per_segment = compressed_timesteps // lstm_segments

    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv1D(64, kernel_size=5, padding="same", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(64, kernel_size=5, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(128, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Dynamic reshape
    x = Reshape((lstm_segments, steps_per_segment * conv_filters))(x)

    # LSTM Layers
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(lstm_units // 2))(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(output_dim, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        metrics=["accuracy"]
    )
    return model