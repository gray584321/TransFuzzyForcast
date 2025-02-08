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
    for feature, imfs in imfs_dict.items():
        processed_imfs[feature] = imfs
        # Ensure imfs is at least 2D (if a single IMF, wrap it into an extra dimension)
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
    print("Starting composite feature creation.")
    groups = {group: [] for group in fe_thresholds.keys()}
    logging.debug(f"Initialized groups: {list(groups.keys())}")
    
    try:
        for key, fe_value in fe_values_dict.items():
            if not isinstance(key, tuple):
                print(f"Warning: key {key} is not a tuple. Skipping.")
                continue
            feature, imf_idx = key
            try:
                # Convert fe_value to a scalar if it isn't already
                fe_scalar = fe_value
                # If fe_scalar is a tuple or list, attempt to convert and average it
                if isinstance(fe_scalar, (list, tuple)):
                    try:
                        fe_scalar = np.mean(np.array(fe_scalar))
                    except Exception as e:
                        print(f"Error converting fuzzy entropy value for feature {feature}, IMF {imf_idx}: {e}")
                        continue
                elif hasattr(fe_scalar, "ndim") and fe_scalar.ndim > 0:
                    fe_scalar = np.mean(fe_scalar)

                for group, threshold in fe_thresholds.items():
                    if float(fe_scalar) < float(threshold):
                        groups[group].append(imfs_dict[feature][imf_idx])
                        break
            except Exception as e:
                print(f"Error processing IMF {imf_idx} for feature {feature}: {str(e)}")
                continue
        
        # Create composite features
        composite_features = {}
        for group, imf_list in groups.items():
            if imf_list:
                print(f"Creating composite feature for group: {group} with {len(imf_list)} IMFs.")
                composite_feature = np.mean(np.vstack(imf_list), axis=0)
                composite_features[group] = composite_feature
                print(f"Composite feature created for group: {group}. Shape: {composite_feature.shape}")
            else:
                logging.warning(f"No IMFs in group: {group}. Generating zeros for composite feature.")
                # Find first valid IMF to get the correct length
                T = None
                for feature in imfs_dict:
                    if imfs_dict[feature] is not None and len(imfs_dict[feature]) > 0:
                        T = len(imfs_dict[feature][0])
                        break
                if T is None:
                    raise ValueError("No valid IMFs found to determine feature length")
                composite_features[group] = np.zeros(T)
                print(f"Generated zero array for composite feature: {group}. Shape: {T}")
        
        composite_features_df = pd.DataFrame(composite_features)
        gc.collect()
        print("Composite feature creation completed. DataFrame shape: {}".format(composite_features_df.shape))
        return composite_features_df
        
    except Exception as e:
        print(f"Error in composite feature creation: {str(e)}")
        raise

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
    for feature in candidate_features:
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
    
    Parameters:
        original_dataframe (pd.DataFrame): The preprocessed original DataFrame.
        composite_features_df (pd.DataFrame): DataFrame containing composite features.
        selected_indicator_features (list): List of selected indicator feature names.
        reference_columns (list): List of reference columns for reindexing.
    
    Returns:
        final_features_df (pd.DataFrame): Final feature matrix ready for modeling.
    """
    print("Starting feature integration.")
    print(f"Selected indicator features: {selected_indicator_features}")
    # Ensure that all selected features exist in the dataframe and fill with zeros if missing
    missing_features = [feat for feat in selected_indicator_features if feat not in original_dataframe.columns]
    if missing_features:
        print(f"Warning: The following selected features are missing in the dataframe and will be filled with zeros: {missing_features}")
    indicator_df = original_dataframe.copy()
    for feat in selected_indicator_features:
        if feat not in indicator_df.columns:
            indicator_df[feat] = 0.0
    indicator_df = indicator_df[selected_indicator_features].reset_index(drop=True)
    
    composite_features_df = composite_features_df.reset_index(drop=True)
    final_features_df = pd.concat([indicator_df, composite_features_df], axis=1)
    # If a reference columns list is provided, reindex to ensure the same feature set (fill missing with zeros)
    if reference_columns is not None:
        final_features_df = final_features_df.reindex(columns=reference_columns, fill_value=0)
    print("Features integrated. Final feature matrix shape: {}".format(final_features_df.shape))
    return final_features_df 