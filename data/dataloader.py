#!/usr/bin/env python
"""
dataloader.py

This script is the entry point for data processing. It handles:

  • Loading Raw Data: Reading your CSV stock data from the data/raw folder.
  • Initial Preprocessing: Calculating technical features via compute_features (from process_data.py),
    then applying the VMD-MIC+FE pipeline from features/feature_engineering.py to create a final feature matrix.
  • Data Manipulation: Performing log-differencing, standardization, and time scaling as needed.
  • Data Splitting: Splitting into training (90%) and testing (10%) sets.
  • Creating DataLoaders: Generating PyTorch DataLoader objects (with a rolling window approach) for model training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging
from concurrent.futures import ThreadPoolExecutor

# Initialize logging
logging.getLogger(__name__)

# Import compute_features from process_data.py to perform initial technical feature calculations.
from .processdata import compute_features

def load_raw_data(csv_filepath):
    """
    Reads the CSV file from the given path into a DataFrame.
    Parses the DateTime column into datetime objects.
    """
    print(f"Loading raw data from {csv_filepath}")
    df = pd.read_csv(csv_filepath)
    # Check for 'DateTime' (case-insensitive)
    if "DateTime" not in df.columns:
        for col in df.columns:
            if col.lower() == "datetime":
                df.rename(columns={col: "DateTime"}, inplace=True)
                print(f"Renamed column '{col}' to 'DateTime'.")
                break
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
        print("Parsed 'DateTime' column to datetime objects.")
    print(f"Raw data loaded successfully from {csv_filepath}. Shape: {df.shape}")
    return df

def preprocess_data(raw_dataframe, features_to_use, target_feature):
    """
    Preprocesses the raw_dataframe by:
      • Calculating technical features using compute_features.
      • Applying the VMD-MIC+FE pipeline from features/feature_engineering.
      • Scaling time and standardizing features.
      • Splitting into training and testing sets.
      
    Parameters:
       raw_dataframe: Raw DataFrame loaded from CSV.
       features_to_use: List of indicator feature names to consider.
       target_feature: Name of the target feature (e.g., 'Processed_Close_Price').
       
    Returns:
       train_df, test_df: Processed training and testing DataFrames.
    """
    print("Starting data preprocessing.")
    df = raw_dataframe.copy()
    
    # Step 1: Compute technical features.
    print("Step 1: Computing technical features.")
    #df = compute_features(df)
    print("Technical features computed.")
    
    # Ensure data is sorted by DateTime.
    if "DateTime" in df.columns:
        print("Sorting DataFrame by 'DateTime'.")
        df.sort_values("DateTime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("DataFrame sorted by 'DateTime'.")
    
    # Step 2: Feature Engineering via VMD-MIC+FE Pipeline.
    from features.feature_engineering import (
        vmd_feature_extraction, 
        fuzzy_entropy_feature_extraction, 
        composite_feature_creation,
        correlation_based_feature_selection,
        integrate_features
    )
    
    # Define which feature(s) to decompose via VMD. For example, we use "Close_Price".
    vmd_features = ["Close_Price"]
    vmd_params_dict = {
        "Close_Price": {"alpha":2000, "tau":0, "DC":0, "init":1, "tol":1e-7}
    }
    print("Step 2a: VMD feature extraction.")
    imfs_dict = vmd_feature_extraction(df, vmd_features, vmd_params_dict, k_range=range(2, 10))
    print("VMD feature extraction completed.")
    print("Step 2b: Fuzzy entropy feature extraction.")
    fe_values_dict, imfs_processed = fuzzy_entropy_feature_extraction(imfs_dict, m=2, r=0.2)
    print("Fuzzy entropy feature extraction completed.")
    # Define fuzzy entropy thresholds for grouping IMFs.
    fe_thresholds = {"Composite_Low": (0, 0.5), "Composite_High": (0.5, np.inf)}
    print("Step 2c: Composite feature creation.")
    composite_features_df = composite_feature_creation(fe_values_dict, imfs_processed, fe_thresholds)
    print("Composite feature creation completed.")
    
    # Select remaining indicator features via MIC.
    print("Step 2d: Correlation-based feature selection.")
    selected_indicator_features = correlation_based_feature_selection(df, target_feature, mic_threshold=0.5)
    print(f"Correlation-based feature selection completed. Selected features: {selected_indicator_features}")
    
    # Integrate the selected indicator features with composite features.
    print("Step 2e: Integrating features.")
    final_features_df = integrate_features(df, composite_features_df, selected_indicator_features)
    # Append the target feature.
    final_features_df[target_feature] = df[target_feature].values
    print("Features integrated.")
    
    # Step 3: Time Scaling (if DateTime exists).
    if "DateTime" in df.columns:
        print("Step 3: Time scaling.")
        n = len(final_features_df)
        final_features_df["Scaled_Time"] = np.linspace(0, 5000, n)
        print("Time scaling completed.")
    
    # Step 4: Split the data (90% train, 10% test).
    print("Step 4: Splitting data into training and testing sets (90/10 split).")
    split_idx = int(len(final_features_df) * 0.9)
    train_df = final_features_df.iloc[:split_idx].reset_index(drop=True)
    test_df = final_features_df.iloc[split_idx:].reset_index(drop=True)
    print(f"Data split. Training set shape: {train_df.shape}, Testing set shape: {test_df.shape}")
    
    # Step 5: Standardize features (using training set statistics).
    print("Step 5: Standardizing features.")
    scaler = StandardScaler()
    # Exclude target_feature from scaling.
    feature_cols = train_df.columns.drop(target_feature)
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    print("Features standardized.")
    
    print("Data preprocessing completed.")
    return train_df, test_df

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time series forecasting using a rolling window approach.
    """
    def __init__(self, df, features, target, seq_len, pred_len):
        print("Initializing TimeSeriesDataset.")
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.X = df[self.features].values
        self.y = df[self.target].values
        print(f"TimeSeriesDataset initialized with seq_len={seq_len}, pred_len={pred_len}, features={features}, target={target}. Data shape: {self.X.shape}")

    def __len__(self):
        dataset_len = len(self.X) - self.seq_len - self.pred_len + 1
        logging.debug(f"__len__ called, returning {dataset_len}")
        return dataset_len

    def __getitem__(self, idx):
        logging.debug(f"__getitem__ called with index {idx}")
        x_seq = self.X[idx : idx + self.seq_len]
        y_seq = self.y[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        y_seq = torch.tensor(y_seq, dtype=torch.float32)
        logging.debug(f"__getitem__ returning input sequence shape: {x_seq.shape}, output sequence shape: {y_seq.shape}")
        return x_seq, y_seq

def create_dataloaders(train_df, test_df, batch_size, seq_len, pred_len):
    # Create memory-mapped arrays in parallel
    def create_mmap(df, filename):
        mmap = np.memmap(filename, dtype=np.float32, mode='w+', shape=df.shape)
        mmap[:] = df.values
        return mmap
    
    with ThreadPoolExecutor() as executor:
        train_future = executor.submit(create_mmap, train_df, 'train.dat')
        test_future = executor.submit(create_mmap, test_df, 'test.dat')
        train_mmap = train_future.result()
        test_mmap = test_future.result()
    
    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(train_mmap))
    test_dataset = TensorDataset(torch.from_numpy(test_mmap))
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Uncomment the following lines to perform a quick test:
# if __name__ == "__main__":
#     raw_df = load_raw_data("data/raw/your_data.csv")
#     features = ["Open_Price", "High_Price", "Low_Price", "Close_Price", "Stock_Volume"]
#     target = "Processed_Close_Price"
#     train_df, test_df = preprocess_data(raw_df, features, target)
#     train_loader, test_loader = create_dataloaders(train_df, test_df, batch_size=32, seq_len=256, pred_len=64)
#     for x, y in train_loader:
#         print("Input sequence shape:", x.shape, "Output sequence shape:", y.shape)
#         break 