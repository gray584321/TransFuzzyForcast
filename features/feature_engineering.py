#!/usr/bin/env python
"""
feature_engineering.py

Orchestrates the complete VMD-MIC+FE pipeline.
Utilizes functions from vmd_mic.py to:
  - Extract VMD features,
  - Compute Fuzzy Entropy for each IMF,
  - Create composite features,
  - Perform correlation-based feature selection,
  - Integrate features to form the final feature matrix.
"""

import numpy as np
import pandas as pd
import EntropyHub as eh
from sklearn.feature_selection import mutual_info_regression
import logging
import gc
import os
import concurrent.futures
from numba import njit, prange  # New import for Numba
from tqdm import tqdm

# Initialize logging
logging.getLogger(__name__)

# Import functions from vmd_mic.py
from features.vmd_mic import variational_mode_decomposition, determine_optimal_k, mutual_information_criterion

# Add at the top of your file, after imports
np.set_printoptions(precision=3, suppress=True)
# Limit numpy to using one thread to reduce memory usage
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def vmd_feature_extraction(dataframe, features_to_decompose, vmd_params_dict, k_range=range(2, 16)):
    """
    Applies VMD on selected features and extracts their IMFs.
    
    Parameters:
        dataframe (pd.DataFrame): Input DataFrame.
        features_to_decompose (list): List of feature names to decompose.
        vmd_params_dict (dict): Dictionary with VMD parameters for each feature.
        k_range (iterable): Range of K values to test for optimal decomposition.
    
    Returns:
        imfs_dict (dict): Dictionary where keys are feature names and values are IMF arrays (shape: (K, T)).
    """
    print("Starting VMD feature extraction.")
    imfs_dict = {}
    for feature in features_to_decompose:
        print(f"Processing feature: {feature} for VMD.")
        # Convert to float32 to reduce memory usage with minimal performance impact.
        signal = dataframe[feature].values.astype(np.float32)
        params = vmd_params_dict.get(feature, {})
        logging.debug(f"VMD parameters for {feature}: {params}")
        print(f"Determining optimal K for feature: {feature}.")
        optimal_k = determine_optimal_k(signal, k_range=range(2, 8), vmd_params=params)
        print(f"Optimal K determined for {feature}: {optimal_k}.")
        print(f"Performing VMD with K={optimal_k} for feature: {feature}.")
        imfs = variational_mode_decomposition(
            signal, 
            alpha=params.get("alpha", 2000), 
            tau=params.get("tau", 0),
            K=optimal_k, 
            DC=params.get("DC", 0),
            init=params.get("init", 1),
            tol=params.get("tol", 1e-7)
        )
        imfs_dict[feature] = imfs
        print(f"VMD completed for feature: {feature}. IMFs shape: {imfs.shape}")
        
        # Free memory from the signal once done with VMD computation.
        del signal
        gc.collect()  # force immediate garbage collection for this iteration
        
    print("VMD feature extraction completed for all features.")
    return imfs_dict

# =========================
# New: Numba-Accelerated Fuzzy Entropy Function
# =========================
@njit(parallel=True)
def _fuzzy_entropy_numba(signal, m, r):
    """
    Compute fuzzy entropy on a 1D signal using a numba-accelerated approach.
    
    Parameters:
        signal (np.ndarray): 1D array containing the signal values.
        m (int): Embedding dimension.
        r (float): Tolerance (e.g., relative to the standard deviation).
    
    Returns:
        float: The computed fuzzy entropy.
    """
    N = len(signal)
    phi_m = 0.0
    phi_m1 = 0.0
    # Loop over the signal for embedding dimension m and m+1.
    for i in prange(N - m):
        count_m = 0.0
        count_m1 = 0.0
        for j in range(N - m):
            # Compute Chebyshev distance for m-length subsequences.
            diff_m = 0.0
            for k in range(m):
                tmp = abs(signal[i + k] - signal[j + k])
                if tmp > diff_m:
                    diff_m = tmp
            # For subsequences extended by 1 for m+1.
            diff_m1 = diff_m
            tmp = abs(signal[i + m] - signal[j + m])
            if tmp > diff_m1:
                diff_m1 = tmp
            
            count_m += np.exp(-diff_m / r)
            count_m1 += np.exp(-diff_m1 / r)
        phi_m += count_m
        phi_m1 += count_m1
    # Return the fuzzy entropy value (with a safe check to avoid division by zero)
    return -np.log(phi_m1 / phi_m) if phi_m != 0 else np.inf

# =========================
# Updated: Fuzzy Entropy Feature Extraction Using Numba
# =========================
def fuzzy_entropy_feature_extraction(imfs_dict, m=2, r=0.15):
    """
    Compute fuzzy entropy for each feature in the given dictionary of IMFs.
    If the IMF for a feature is multi-dimensional (multiple sub-signals),
    compute fuzzy entropy for each IMF separately and store each in a tuple key (feature, index).
    
    Parameters:
        imfs_dict (dict): Dictionary where keys are feature names and 
                          values are numpy arrays representing the IMFs.
        m (int): Embedding dimension.
        r (float): Tolerance.
    
    Returns:
        fe_values (dict): Dictionary mapping (feature, imf_index) to fuzzy entropy values.
        processed_imfs (dict): Optionally processed IMFs (here the raw IMFs are returned).
    """
    fe_values = {}
    processed_imfs = {}
    
    print("Computing fuzzy entropy for each IMF...")
    for feature, imfs in tqdm(imfs_dict.items(), desc="Fuzzy Entropy"):
        processed_imfs[feature] = imfs
        if imfs.ndim == 1:
            imfs = np.array([imfs])
        for i in range(imfs.shape[0]):
            fe = _fuzzy_entropy_numba(imfs[i], m, r)
            fe_values[(feature, i)] = fe
    return fe_values, processed_imfs

def composite_feature_creation(fe_values_dict, imfs_dict, fe_thresholds):
    """
    Groups IMFs based on their fuzzy entropy values and creates composite features.
    """
    print("Creating composite features...")
    groups = {group: [] for group in fe_thresholds.keys()}
    
    # Group IMFs by fuzzy entropy
    print("Grouping IMFs by fuzzy entropy...")
    for key, fe_value in tqdm(fe_values_dict.items(), desc="Grouping IMFs"):
        if not isinstance(key, tuple):
            print(f"Warning: key {key} is not a tuple. Skipping.")
            continue
        feature, imf_idx = key
        try:
            fe_scalar = fe_value if not isinstance(fe_value, (list, tuple)) else np.mean(fe_value)
            for group, threshold in fe_thresholds.items():
                if float(fe_scalar) < float(threshold):
                    groups[group].append(imfs_dict[feature][imf_idx])
                    break
        except Exception as e:
            print(f"Error processing IMF {imf_idx} for feature {feature}: {str(e)}")
            continue
    
    # Create composite features
    print("Computing composite features...")
    composite_features = {}
    for group in tqdm(groups.keys(), desc="Creating composites"):
        imf_list = groups[group]
        if imf_list:
            composite_feature = np.mean(np.vstack(imf_list), axis=0)
            composite_features[group] = composite_feature
        else:
            T = next(iter(imfs_dict.values()))[0].shape[0] if imfs_dict else None
            if T is None:
                raise ValueError("No valid IMFs found to determine feature length")
            composite_features[group] = np.zeros(T)
    
    composite_features_df = pd.DataFrame(composite_features)
    gc.collect()
    return composite_features_df

def correlation_based_feature_selection(dataframe, target_feature, mic_threshold=0.5):
    """
    Selects remaining indicator features from dataframe based on their MIC with the target.
    
    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with indicator features.
        target_feature (str): Name of the target column.
        mic_threshold (float): MIC threshold for feature selection.
        
    Returns:
        selected_features (list): List of feature names that meet the MIC threshold.
    """
    print("Starting correlation-based feature selection.")
    selected_features = []
    excluded_columns = {"DateTime", "Ticker", "Processed_Close_Price", "Fuzzy_Entropy"}
    candidate_features = [col for col in dataframe.columns if col not in excluded_columns and 
                          col != target_feature and not col.startswith("VMD_Mode_")]
    
    X = dataframe[candidate_features]
    y = dataframe[target_feature]
    
    print(f"Calculating MIC for {len(candidate_features)} candidate features against target feature: {target_feature}.")
    for feature in tqdm(candidate_features, desc="MIC Calculation"):
        x_feat = X[feature].values.reshape(-1, 1)
        logging.debug(f"Calculating MIC for feature: {feature}.")
        mic = mutual_info_regression(x_feat, y.values, random_state=0)[0]
        logging.debug(f"MIC for feature {feature}: {mic:.4f}, threshold: {mic_threshold}.")
        if mic >= mic_threshold:
            selected_features.append(feature)
            print(f"Feature {feature} selected based on MIC threshold.")
    
    print(f"Correlation-based feature selection completed. Selected features: {selected_features}")
    return selected_features

def integrate_features(original_dataframe, composite_features_df, selected_indicator_features, reference_columns=None):
    """
    Combines composite features with selected indicator features to produce the final feature matrix.
    Ensures consistent feature selection across all files by using the same selected features.
    
    Parameters:
        original_dataframe (pd.DataFrame): The preprocessed original DataFrame.
        composite_features_df (pd.DataFrame): DataFrame containing composite features.
        selected_indicator_features (list): List of selected indicator feature names from first file.
        reference_columns (list): List of reference columns for reindexing.
    
    Returns:
        final_features_df (pd.DataFrame): Final feature matrix ready for modeling.
    """
    print("Starting feature integration.")
    print(f"Using selected indicator features from first file: {selected_indicator_features}")
    
    indicator_df = pd.DataFrame()
    
    print("Integrating selected features...")
    for feat in tqdm(selected_indicator_features, desc="Feature Integration"):
        if feat in original_dataframe.columns:
            indicator_df[feat] = original_dataframe[feat]
        else:
            print(f"Feature {feat} not found in current file, filling with zeros")
            indicator_df[feat] = 0.0
    
    indicator_df = indicator_df.reset_index(drop=True)
    composite_features_df = composite_features_df.reset_index(drop=True)
    
    final_features_df = pd.concat([indicator_df, composite_features_df], axis=1)
    
    if reference_columns is not None:
        final_features_df = final_features_df.reindex(columns=reference_columns, fill_value=0)
    
    print("Features integrated. Final feature matrix shape: {}".format(final_features_df.shape))
    print("Final feature columns:", final_features_df.columns.tolist())
    return final_features_df 