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
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .callbacks.learning_curves import plot_learning_history, plot_model
from .callbacks.keras_callback import create_callback

SEED = 42
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
np.random.seed(SEED)

def train_and_predict_cnn(LOG_DIR: str, fold_id: int, X_train: np.ndarray, X_valid: np.ndarray, 
                        X_test: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray, config,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Model]:
    """
    Train CNN

    Args:
        X_train, X_valid, X_test: input signals of shape (num_samples, window_size, num_channels)
        y_train, y_valid, y_test: labels 

    Returns:
        pred_train: train prediction
        pred_valid: validation prediction
        pred_test: test prediction
        model: best model
    """
    model = build_baseline(
        input_shape=X_train.shape[1:], output_dim=y_train.shape[1], lr=config.learning_rate
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


def build_baseline(input_shape=(600, 270), output_dim=17, lr=0.0003):
    model = Sequential()

    model.add(Conv1D(64, kernel_size=3, input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(GlobalAveragePooling1D()) 

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation("relu"))
    model.add(Dropout(0.4, seed=0))

    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=lr),
        metrics=["accuracy"]
    )

    return model