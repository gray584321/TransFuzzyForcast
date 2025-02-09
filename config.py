#!/usr/bin/env python
"""
Configuration File

Defines hyperparameters and file paths for the project.
"""

import os
import torch
import numpy as np

config = {
    "model": {
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "attention_factor": 5.0,
        "max_seq_len": 256,
        "pred_len": 10,
        "feature_dim": 20,  # Adjust according to your feature engineering output
        "output_dim": 1,
        "timestamp_vocab_size": None
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 1e-5,
        "epochs": 280,
        "early_stopping_patience": 20,
        "gradient_clip_value": 1.0,
        "beta_initial": 0.5,  # For Dynamic Loss Function
        "c": 1.0
    },
    "vmd": {
        "alpha": 2000,
        "tau": 0,
        "DC": 0,
        "init": 1,
        "tol": 1e-7
    },
    "data": {
        "raw_data_path": "data/raw_data.csv",
        "processed_data_path": "data/ready_for_train/merged_stock_data.csv"
    },
    "device": torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
    "output": {
        "model_save_path": "best_model.pth",
        "results_dir": "results/",
        "plots_dir": "results/plots/"
    }
}

# Add these other required parameters that are referenced in main.py
merged_data_path = "data/ready_for_train/merged_stock_data.csv"
features_to_use = ['Open_Price', 'High_Price', 'Low_Price', 'Close_Price', 'Stock_Volume']
target_feature = 'Close_Price'
features_to_decompose = ['Close_Price']

# Add the rest of the required parameters (example values shown)
vmd_params_dict = {'alpha': 2000, 'tau': 0, 'K': 3, 'chunk_size': 2000}
fuzzy_m = 2
fuzzy_r = 0.2
fuzzy_n = 2
fe_thresholds = {'threshold1': 0.5}
mic_threshold = 0.6
input_dim = 10
feature_dim = 64
output_dim = 1
d_model = 512
n_heads = 8
e_layers = 3
d_ff = 2048
dropout = 0.1
activation = 'gelu'
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.01
amsgrad = False
gc_momentum = 0.5
beta_initial = 0.5
c = 1.0
epochs = 250
early_stopping_patience = 50
output_dir = "results/"

# Add these configuration parameters to your config.py
# Example configuration parameters - adjust values according to your needs
batch_size = 1024
learning_rate = 0.00001
seq_len = 12     # Input sequence length
pred_len = 6     # Prediction length
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Reduce default K range
k_range = range(5, 9)  # Instead of larger ranges like 3-10

# Set default float precision
DEFAULT_DTYPE = np.float32 

# Add parallel processing parameters
parallel_config = {
    "feature_workers": 16,      # Number of parallel feature processors
    "vmd_workers": 16,          # Number of parallel VMD processors
    "dataloader_workers": 16    # Number of data loading workers
}

# Update memory parameters
vmd_params_dict = {'alpha': 2000, 'tau': 0, 'K': 3, 'chunk_size': 500}  # Smaller chunks 

# Model Architecture
feature_dim = 21    # Should match final feature matrix dimension
output_dim = 1      # Single-step prediction
d_model = 512       # Dimension of model embeddings
n_heads = 8         # Number of attention heads
e_layers = 3        # Number of encoder layers
d_ff = 2048         # Dimension of feed-forward network
dropout = 0.1
activation = 'gelu' 