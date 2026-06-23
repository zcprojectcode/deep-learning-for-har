from dataclasses import asdict
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import logging

from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.model_selection import train_test_split

"""
Split, window and normalise the training and testing datasets for
deep learning model training
"""
class Preprocessor:
    def __init__(self, imu_dataset_files, ratio, seed):
        self.samples = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        logging.info(f"{imu_dataset_files}")

        imu_dataset_file_list = imu_dataset_files.split(",")

        # Import tensors
        logging.info("Import tensors")
        for imu_dataset_file in imu_dataset_file_list:
            file_path = Path(imu_dataset_file).expanduser().resolve()
            checkpoint = torch.load(file_path, weights_only = False)
            self.samples.extend(checkpoint["samples"])

        # Remove NaN
        logging.info("Remove NaN")

        EXCLUDED_LABELS = {}

        filtered_samples = []
        for sample_id, (imu, label) in enumerate(self.samples):

            if label in EXCLUDED_LABELS:
                continue
            imu = np.nan_to_num(imu, nan=0.0)

            # if label == 8:
            #     label = 3
            # elif label > 8:
            #     label -= 1

            # cols_99_270 = np.concatenate([np.arange(99 + i*9, 99 + i*9 + 6) for i in range(19)])
            filtered_samples.append((imu, label))

        # (imu[:, np.r_[start1:end1, start2:end2, 99:108, 117:135, 144:162, 180:207, 225:261]], label)

        self.samples = filtered_samples

        # Remap labels to contiguous range
        unique_labels = sorted(set(label for _, label in self.samples))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        logging.info(f"Label mapping: {label_map}")

        self.samples = [(imu, label_map[label]) for imu, label in self.samples]

        # Separate the IMU data from the labels
        X, y = [], []
        for i, j in self.samples:
            X.append(i)
            y.append(j)

        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=ratio, random_state=seed)
    
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
        train_samples = []
        train_labels = []
        for sample_id, imu in enumerate(self.X_train):
            seq_len = imu.shape[0]

            for start in range(0, seq_len - window_size + 1, window_shift):
                train_samples.append(imu[start:start + window_size])
                train_labels.append(self.y_train[sample_id])
        
        assert len(train_samples) == len(train_labels)

        train_samples_init = np.array(train_samples)
        train_labels_init = np.array(train_labels)

        # Apply SMOTE
        # logging.info(f"Apply SMOTE")
        # train_samples_flat = train_samples_init.reshape(len(train_samples_init), -1)
        # smote = SMOTE(sampling_strategy='minority', random_state=42)
        # train_samples_smote, train_labels = smote.fit_resample(train_samples_flat, train_labels_init)

        # train_samples = train_samples_smote.reshape(len(train_samples_smote), window_size, 198)

        # Apply random oversampling
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
            test_samples[sample_id] = sample
        
        for sample_id, sample in enumerate(train_samples):
            sample = (sample - train_mean) / train_std
            train_samples[sample_id] = sample
        
        # # Convert data to binary
        # logging.info("Convert data to binary - use specific thresholds instead of 2 std devs")

        # # Define thresholds for columns 18-29
        # thresholds = {
        #     18: 2,
        #     19: 10,
        #     20: 5,
        #     21: 50,
        #     22: 5,
        #     23: 5,
        #     24: 2,
        #     25: 2,
        #     26: 2,
        #     27: 1,
        #     28: 2,
        #     29: 2,
        # }

        # for dataset in [test_samples, train_samples]:
        #     for sample in dataset:
        #         for row in sample:
        #             for col, threshold in thresholds.items():
        #                 row[col] = 0 if abs(row[col]) < threshold else row[col]

        # Store the output
        out = [(np.array(train_samples), np.array(train_labels)), 
                (np.array(test_samples), np.array(test_labels))]
        return out