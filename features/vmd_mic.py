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
from numba import njit, jit, prange
import numpy.fft as npfft

def fft_operations(signal):
    """Separate function for FFT operations that can't be compiled with nopython mode"""
    T = signal.shape[0]
    f_hat = npfft.fft(signal)
    freqs = npfft.fftfreq(T)
    return f_hat, freqs

def ifft_operation(u_hat):
    """Separate function for inverse FFT operation"""
    return npfft.ifft(u_hat).real

@njit(parallel=True, fastmath=True)
def _vmd_core(f_hat, freqs, alpha, tau, K, tol, max_iter, init):
    """
    Core VMD computation optimized for Numba with nopython mode.
    """
    T = freqs.shape[0]
    
    # Initialize arrays
    u_hat = np.zeros((K, T), dtype=np.complex128)
    omega = np.empty(K, dtype=np.float64)
    
    # Initialize frequencies (parallel)
    if init == 1:
        for k in prange(K):
            omega[k] = 0.5 * (k + 1) / (K + 1)
    else:
        for k in prange(K):
            omega[k] = 0.5 * np.random.random()
    
    # Initialize Lagrange multiplier
    lambda_hat = np.zeros(T, dtype=np.complex128)
    u_hat_prev = np.zeros((K, T), dtype=np.complex128)
    
    # Pre-allocate arrays for better performance
    sum_u = np.zeros(T, dtype=np.complex128)
    sum_u_all = np.zeros(T, dtype=np.complex128)
    
    # Main ADMM loop
    for n in range(max_iter):
        # Store previous iteration
        u_hat_prev[:] = u_hat[:]
        
        # Update modes in parallel
        for k in prange(K):
            # Compute sum of all modes except k
            sum_u.fill(0.0)
            for j in range(K):
                if j != k:
                    for t in range(T):
                        sum_u[t] += u_hat[j, t]
                    
            # Update mode k in frequency domain
            for t in range(T):
                numerator = f_hat[t] - sum_u[t] + lambda_hat[t] / 2.0
                denom = 1.0 + 2.0 * alpha * ((freqs[t] - omega[k]) ** 2)
                u_hat[k, t] = numerator / denom
            
            # Update center frequency omega[k]
            num = 0.0
            den = 0.0
            for t in range(T):
                weight = abs(u_hat[k, t]) ** 2
                num += freqs[t] * weight
                den += weight
            
            if den > 1e-10:
                omega[k] = num / den
            else:
                omega[k] = omega[k]  # Keep previous value
        
        # Update Lagrange multiplier
        sum_u_all.fill(0.0)
        for k in range(K):
            for t in range(T):
                sum_u_all[t] += u_hat[k, t]
                
        for t in range(T):
            lambda_hat[t] = lambda_hat[t] + tau * (f_hat[t] - sum_u_all[t])
        
        # Check convergence
        diff = 0.0
        for k in range(K):
            for t in range(T):
                diff += abs(u_hat[k, t] - u_hat_prev[k, t])**2
                
        if diff < tol:
            break
    
    return u_hat, omega, lambda_hat

def variational_mode_decomposition_fast(signal, alpha, tau, K, DC=0, init=1, tol=1e-7):
    """
    Perform Variational Mode Decomposition using a Numba-accelerated implementation.
    
    Parameters:
        signal (array-like or pd.Series): Input time series signal.
        alpha (float): Data-fidelity balancing parameter.
        tau (float): Time-step for the dual ascent.
        K (int): Number of modes to extract.
        DC (int, optional): If 1, constrain the first mode to zero frequency (by removing its mean). Default is 0.
        init (int, optional): 1 for uniform initialization, 0 for random initialization. Default is 1.
        tol (float, optional): Convergence tolerance. Default is 1e-7.
        
    Returns:
        imfs (np.ndarray): Array of Intrinsic Mode Functions (IMFs) with shape (K, T)
    """
    # Ensure the signal is a numpy array
    if isinstance(signal, pd.Series):
        f = signal.values.astype(np.float64)
    else:
        f = np.asarray(signal, dtype=np.float64)
    
    # Perform FFT operations outside of nopython mode
    f_hat, freqs = fft_operations(f)
    
    # Call the optimized VMD core routine
    u_hat, omega, lambda_hat = _vmd_core(f_hat, freqs, alpha, tau, K, tol, max_iter=500, init=init)
    
    # Reconstruct modes in time domain
    u = np.zeros((K, len(f)), dtype=np.float64)
    for k in range(K):
        u[k] = ifft_operation(u_hat[k])
    
    # Apply DC constraint if requested
    if DC == 1:
        u[0] = u[0] - np.mean(u[0])
    
    return u

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