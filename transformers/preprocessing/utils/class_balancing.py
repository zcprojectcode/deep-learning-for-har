from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from logging import getLogger
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

def temporal_smote(X, y, k=5, random_state=42):
    """
    SMOTE that interpolates directly in (n_features, n_timesteps) space,
    preserving temporal structure.
    X shape: (n_samples, n_features, n_timesteps)
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    X_new_list = [X]
    y_new_list = [y]

    for cls, count in zip(classes, counts):
        if count == majority_count:
            continue

        X_cls = X[y == cls]  # (n_cls, n_features, n_timesteps)
        n_to_generate = majority_count - count
        n_cls = len(X_cls)
        k_actual = min(k, n_cls - 1)

        # Flatten only to find KNN neighbours
        X_flat = X_cls.reshape(n_cls, -1)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X_flat)
        _, indices = nn.kneighbors(X_flat)
        knn_indices = indices[:, 1:]  # exclude self

        synthetic = []
        for _ in range(n_to_generate):
            idx = rng.integers(0, n_cls)
            neighbour_idx = knn_indices[idx, rng.integers(0, k_actual)]
            lam = rng.uniform(0, 1)
            # Interpolate in full (n_features, n_timesteps) space
            syn = X_cls[idx] + lam * (X_cls[neighbour_idx] - X_cls[idx])
            synthetic.append(syn)

        X_new_list.append(np.array(synthetic))
        y_new_list.append(np.full(n_to_generate, cls))

    return np.concatenate(X_new_list), np.concatenate(y_new_list)

def temporal_random_oversample(X, y):
    """
    Resample X and y to balance class distribution.
    """
    rng = np.random.default_rng(42)
    X = np.asarray(X)
    y = np.asarray(y)

    class_counts = Counter(y)
    majority_count = max(class_counts.values())

    target_counts = {cls: majority_count for cls in class_counts}

    fit_info = {
        'original_counts': dict(Counter(y)),
        'target_counts': target_counts,
    }

    X_parts = [X]
    y_parts = [y]

    for cls, target in target_counts.items():
        current_count = np.sum(y == cls)
        n_to_generate = target - current_count

        if n_to_generate <= 0:
            continue

        X_cls = X[y == cls]

        # Sample with replacement from the minority class
        indices = rng.integers(0, len(X_cls), size=n_to_generate)
        X_synthetic = X_cls[indices].copy()

        X_parts.append(X_synthetic)
        y_parts.append(np.full(n_to_generate, cls, dtype=y.dtype))

    X_resampled = np.concatenate(X_parts, axis=0)
    y_resampled = np.concatenate(y_parts, axis=0)

    # Shuffle
    shuffle_idx = rng.permutation(len(X_resampled))
    return X_resampled[shuffle_idx], y_resampled[shuffle_idx]