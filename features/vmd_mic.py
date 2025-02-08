#!/usr/bin/env python
"""
vmd_mic.py

Implements the VMD (Variational Mode Decomposition) and MIC (Mutual Information Criterion)
components of the feature engineering pipeline. This implementation uses sktime's VmdTransformer for VMD.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.feature_selection import mutual_info_regression
from sktime.transformations.series.vmd import VmdTransformer

def variational_mode_decomposition(signal, alpha, tau, K, DC=0, init=1, tol=1e-7):
    """
    Perform Variational Mode Decomposition using sktime's VmdTransformer.
    
    Parameters:
        signal (array-like or pd.Series): Input time series signal.
        alpha (float): Data-fidelity balancing parameter.
        tau (float): Time-step for the dual ascent.
        K (int): Number of modes to extract.
        DC (int, optional): If 1, the first mode is constrained to zero frequency. Default is 0.
        init (int, optional): 1 for initialization with zeros, 0 for random initialization. Default is 1.
        tol (float, optional): Tolerance for convergence. Default is 1e-7.
        
    Returns:
        imfs (np.ndarray): Array of Intrinsic Mode Functions (IMFs) with shape (K, T)
    """
    transformer = VmdTransformer(K=K, alpha=alpha, tau=tau, DC=DC, init=init, tol=tol)
    # Ensure the signal is a pandas Series (required by sktime)
    if not isinstance(signal, pd.Series):
        signal = pd.Series(signal)
    imfs_df = transformer.fit_transform(signal)
    # The resulting DataFrame has each column as an IMF; transpose it to get shape (K, T)
    imfs = imfs_df.values.T
    return imfs

def mutual_information_criterion(original_signal, reconstructed_signal):
    """
    Compute the Mutual Information Criterion (MIC) between the original and reconstructed signals.
    
    Uses sklearn.feature_selection.mutual_info_regression.
    
    Parameters:
        original_signal (array-like): The original input signal.
        reconstructed_signal (array-like): The signal reconstructed from the IMFs.
        
    Returns:
        mi (float): Mutual Information value.
    """
    # Ensure inputs are numpy arrays and reshape appropriately for mutual_info_regression
    X = np.array(original_signal).reshape(-1, 1)
    y = np.array(reconstructed_signal)
    mi = mutual_info_regression(X, y, random_state=0)
    return mi[0]

def determine_optimal_k(signal, k_range, vmd_params):
    """
    Determine the optimal number of modes (K) using vectorized operations where possible.
    """
    best_k = k_range[0]
    best_mi = -np.inf
    
    # Convert signal to numpy array once
    signal_array = np.asarray(signal)
    
    # Pre-set VMD parameters
    params = {
        "alpha": vmd_params.get("alpha", 2000),
        "tau": vmd_params.get("tau", 0),
        "DC": vmd_params.get("DC", 0),
        "init": vmd_params.get("init", 1),
        "tol": vmd_params.get("tol", 1e-7)
    }
    
    for k in k_range:
        try:
            imfs = variational_mode_decomposition(signal_array, K=k, **params)
            # Vectorized reconstruction
            reconstructed_signal = np.sum(imfs, axis=0)
            mi = mutual_information_criterion(signal_array, reconstructed_signal)
            
            if mi > best_mi:
                best_mi = mi
                best_k = k
                
            logging.debug(f"K={k}, Mutual Information={mi:.4f}")
            
        except Exception as e:
            logging.error(f"Error processing K={k}: {e}")
            continue
            
    return best_k 