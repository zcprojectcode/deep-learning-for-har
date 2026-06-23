from dataclasses import asdict
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import logging
from collections import Counter
from .utils.class_balancing import temporal_smote

from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, imu_train_files, imu_test_files, ratio, seed):
        self.train_samples = []
        self.test_samples = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        logging.info(f"Training files: {imu_train_files}")
        logging.info(f"Testing files: {imu_test_files}")

        imu_train_file_list = imu_train_files.split(",")
        imu_test_file_list = imu_test_files.split(",")

        # Import tensors
        logging.info("Import train tensors")
        for imu_dataset_file in imu_train_file_list:
            file_path = Path(imu_dataset_file).expanduser().resolve()
            checkpoint = torch.load(file_path, weights_only = False)
            self.train_samples.extend(checkpoint["samples"])
        
        # Import tensors
        logging.info("Import test tensors")
        for imu_dataset_file in imu_test_file_list:
            file_path = Path(imu_dataset_file).expanduser().resolve()
            checkpoint = torch.load(file_path, weights_only = False)
            self.test_samples.extend(checkpoint["samples"])

        # Remove NaN
        logging.info("Remove NaN")

        EXCLUDED_LABELS = {}

        filtered_samples = []
        for sample_id, (imu, label) in enumerate(self.train_samples):
            if label in EXCLUDED_LABELS:
                continue
            imu = np.nan_to_num(imu, nan=0.0)
            filtered_samples.append((imu, label))

        self.train_samples = filtered_samples

        filtered_samples = []
        for sample_id, (imu, label) in enumerate(self.test_samples):
            if label in EXCLUDED_LABELS:
                continue
            imu = np.nan_to_num(imu, nan=0.0)
            filtered_samples.append((imu, label))

        self.test_samples = filtered_samples

        # Relabel data
        new_samples = []
        for data, label in self.train_samples:
            if label == 8:
                label = 3
            elif label > 8:
                label -= 1

            new_samples.append((data, label))

        self.train_samples = new_samples

        new_samples = []
        for data, label in self.test_samples:
            if label == 8:
                label = 3
            elif label > 8:
                label -= 1

            new_samples.append((data, label))

        self.test_samples = new_samples

        # Separate the IMU data from the labels
        for i, j in self.train_samples:
            self.X_train.append(i)
            self.y_train.append(j)

        for i, j in self.test_samples:
            self.X_test.append(i)
            self.y_test.append(j)
    
    def len_train(self):
        return len(self.X_train)

    def fit_transform(self, window_size, window_shift) -> Dict[str, Any]:
        """
        Returns dict with windowed data and labels 
        """
        
        # Perform windowing and store windows
        logging.info("Perform windowing")

        # Window test data
        test_samples = []
        test_labels = []
        for sample_id, imu in enumerate(self.X_test):
            seq_len = imu.shape[0]

            for start in range(0, seq_len - window_size + 1, window_shift):
                test_samples.append(imu[start:start + window_size])
                test_labels.append(self.y_test[sample_id])
        
        assert len(test_samples) == len(test_labels)
        
        # Window training data
        train_samples_init = []
        train_labels_init = []
        for sample_id, imu in enumerate(self.X_train):
            seq_len = imu.shape[0]

            for start in range(0, seq_len - window_size + 1, window_shift):
                train_samples_init.append(imu[start:start + window_size])
                train_labels_init.append(self.y_train[sample_id])
        
        assert len(train_samples_init) == len(train_labels_init)

        train_samples_init = np.array(train_samples_init)
        train_labels_init = np.array(train_labels_init)

        counts = Counter(train_labels_init)
        values = [counts[i] for i in sorted(counts.keys())]
        logging.info(f"Training distribution: {values}")

        counts = Counter(test_labels)
        values = [counts[i] for i in sorted(counts.keys())]
        logging.info(f"Testing distribution: {values}")

        # Apply SMOTE
        # logging.info(f"Apply SMOTE")
        # train_samples_flat = train_samples_init.reshape(len(train_samples_init), -1)
        # train_samples_smote, train_labels = temporal_smote(train_samples_flat, train_labels_init)

        # train_samples = train_samples_smote.reshape(len(train_samples_smote), window_size, 270)

        train_samples = train_samples_init
        train_labels = train_labels_init

        # # Apply random oversampling
        # logging.info(f"Apply random oversampling")
        # train_samples_flat = train_samples_init.reshape(len(train_samples_init), -1)
        # ros = RandomOverSampler(random_state=42)
        # train_samples_ros, train_labels = ros.fit_resample(train_samples_flat, train_labels_init)

        # train_samples = train_samples_ros.reshape(len(train_samples_ros), window_size, 270)

        # Log meta data
        logging.info(f"Number of test samples: {len(test_samples)}")
        logging.info(f"Number of training samples: {len(train_samples)}")

        # Calculate the feature-wise mean and std dev values
        all_test = np.vstack(test_samples)
        all_train = np.vstack(train_samples)

        logging.info("Calculate feature-wise mean and std dev")
        test_mean = all_test.mean(axis=0)
        test_std = all_test.std(axis=0)
        test_std[test_std == 0] = 1

        train_mean = all_train.mean(axis=0)
        train_std = all_train.std(axis=0)
        train_std[train_std == 0] = 1

        # Perform z-score normalisation 
        logging.info("Perform normalisation")
        for sample_id, sample in enumerate(test_samples):
            sample = (sample - test_mean) / test_std
            test_samples[sample_id]= sample
        
        for sample_id, sample in enumerate(train_samples):
            sample = (sample - train_mean) / train_std
            train_samples[sample_id] = sample

        # Store the output
        out = [(np.array(train_samples), np.array(train_labels)), 
                (np.array(test_samples), np.array(test_labels))]
        return out