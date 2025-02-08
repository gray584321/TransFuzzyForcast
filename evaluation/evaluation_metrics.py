#!/usr/bin/env python
"""
evaluation_metrics.py

Provides a suite of evaluation metric functions for the Enhanced LFTSformer model.
Metrics include:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² (Coefficient of Determination)
  - Explained Variance Score
  - Mean Absolute Percentage Error (MAPE)
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

def calculate_mae(true_values, predicted_values):
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(true_values, predicted_values)

def calculate_mse(true_values, predicted_values):
    """Calculate Mean Squared Error."""
    return mean_squared_error(true_values, predicted_values)

def calculate_rmse(true_values, predicted_values):
    """Calculate Root Mean Squared Error."""
    mse = mean_squared_error(true_values, predicted_values)
    return np.sqrt(mse)

def calculate_r2(true_values, predicted_values):
    """Calculate the R^2 score (Coefficient of Determination)."""
    return r2_score(true_values, predicted_values)

def calculate_explained_variance(true_values, predicted_values):
    """Calculate the explained variance score."""
    return explained_variance_score(true_values, predicted_values)

def calculate_mape(true_values, predicted_values):
    """Calculate Mean Absolute Percentage Error, ignoring zero values in true_values."""
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    mask = true_values != 0  # avoid division by zero
    return np.mean(np.abs((true_values[mask] - predicted_values[mask]) / true_values[mask])) * 100 