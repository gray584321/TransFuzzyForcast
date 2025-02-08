#!/usr/bin/env python
"""
Visualization Utilities

Provides:
  - plot_predictions: Generates line plots of true vs. predicted values.
  - generate_heatmap: Generates a heatmap from a data matrix.
  - generate_boxplot: Generates boxplots for error residuals.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_predictions(true_values, predicted_values, timestamps, model_name, save_path):
    """
    Generates a line plot of true and predicted values over time.
    
    Parameters:
        true_values (array-like): The ground truth values.
        predicted_values (array-like): The model's predicted values.
        timestamps (array-like): Time stamps corresponding to each value.
        model_name (str): Name of the model (for labeling).
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, true_values, label="True Values", linewidth=2)
    plt.plot(timestamps, predicted_values, label="Predicted Values", linewidth=2, alpha=0.8)
    plt.title(f"{model_name}: True vs. Predicted", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_heatmap(data_matrix, feature_names, title, save_path):
    """
    Generates and saves a heatmap visualization.
    
    Parameters:
        data_matrix (2D array-like): Data to visualize (e.g., correlation matrix).
        feature_names (list): Labels for the rows/columns.
        title (str): Title of the heatmap.
        save_path (str): File path to save the heatmap.
    """
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame(data_matrix, index=feature_names, columns=feature_names)
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_boxplot(residuals, model_names, title, save_path):
    """
    Generates boxplots of residuals (errors) for different models.
    
    Parameters:
        residuals (list of array-like): Residuals (prediction errors) for each model.
        model_names (list of str): Corresponding model names.
        title (str): Title of the plot.
        save_path (str): File path to save the boxplot.
    """
    plt.figure(figsize=(10, 6))
    plt.boxplot(residuals, labels=model_names, patch_artist=True)
    plt.title(title, fontsize=16)
    plt.ylabel("Residuals", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 