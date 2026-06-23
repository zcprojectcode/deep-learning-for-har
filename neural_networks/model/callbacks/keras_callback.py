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

from logging import getLogger
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model

from .utils import round

SEED = 42
np.random.seed(SEED)

logger = getLogger(__name__)

class PeriodicLogger(Callback):
    """
    Logging history every n epochs
    """
    def __init__(
        self, metric: str = "accuracy", verbose: int = 1, epochs: Optional[int] = None
    ) -> None:
        self.metric = metric
        self.verbose = verbose
        self.epochs = epochs

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        epoch += 1
        if epoch % self.verbose == 0:
            msg = " - ".join(
                [
                    f"Epoch {epoch}/{self.epochs}",
                    f"loss: {round(logs['loss'], 0.0001)}",
                    f"{self.metric}: {round(logs[self.metric], 0.0001)}",
                    f"val_loss: {round(logs['val_loss'], 0.0001)}",
                    f"val_{self.metric}: {round(logs[f'val_{self.metric}'], 0.0001)}",
                ]
            )
            logger.debug(msg)


def create_callback(model: Model, path_chpt: str, patience: int = 30, metric: str = "accuracy", 
    verbose: int = 10, epochs: Optional[int] = None) -> List[Any]:
    """
    Callback settings

    Args:
        model (Model)
        path_chpt (str): path to save checkpoint

    Returns:
        callbacks (List[Any]): List of Callback
    """
    callbacks = []
    callbacks.append(
        EarlyStopping(monitor="val_loss", min_delta=0, patience=patience, verbose=1, mode="min")
    )
    callbacks.append(ModelCheckpoint(filepath=path_chpt, save_best_only=True))
    callbacks.append(PeriodicLogger(metric=metric, verbose=verbose, epochs=epochs))

    return callbacks