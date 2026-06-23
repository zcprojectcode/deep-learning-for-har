import os
import sys
import argparse
import numpy as np

from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
from datetime import datetime

from config import HARConfig
from transformers.run_transformers import run_transformers
from neural_networks.run_neural_networks import run_neural_networks
from cross_validation_scores import report_scores

# Dataset
imu_dataset_files = # Filepath to PyTorch tensors containing data

# Current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up logging 
def config_logger():
    # CHANGE THIS to reflect the model being run
    EXEC_TIME = "HAR-Framework-" + datetime.now().strftime("%Y%m%d-%H%M%S") 

    LOG_DIR = os.path.join(CUR_DIR, f"logs/{EXEC_TIME}")
    os.makedirs(LOG_DIR, exist_ok=True)  # Create log directory

    formatter = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
    basicConfig(filename=f"{LOG_DIR}/{EXEC_TIME}.log", level=DEBUG, format=formatter)
    mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
    mpl_logger.setLevel(WARNING)
    # Handle logging to both logging and stdout.
    getLogger().addHandler(StreamHandler(sys.stdout))

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")

    return LOG_DIR

# Parse the command line - determine which model to run
def parse_args():
    p = argparse.ArgumentParser(description="HAR CLI")
    p.add_argument("--model", type=str, default="tiny_transformer",
                   choices=["tiny_transformer", "transformer_encoder", "cnn", "gru", "deep_conv_lstm"])
    return p.parse_args()

def main():
    LOG_DIR = config_logger()
    cli = parse_args()
    config = HARConfig(model = cli.model)

    if config.error == False:
        # Transformer model built with PyTorch 
        if config.model in ("tiny_transformer", "transformer_encoder"):
            scores, y_test, test_preds = run_transformers(config, imu_dataset_files, LOG_DIR)
        # CNN, GRU and DCLSTM built with Tensorflow
        elif config.model in ("cnn", "gru", "deep_conv_lstm"):
            scores, y_test, test_preds = run_neural_networks(config, imu_dataset_files, LOG_DIR)
        
        # Summarise results
        report_scores(scores, y_test, test_preds, config, LOG_DIR)

if __name__ == "__main__":
    main()