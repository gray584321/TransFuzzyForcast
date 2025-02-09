#!/usr/bin/env python
"""
main_preprocess.py

This version of main loads preprocessed pickle files and immediately starts model training.
It expects a pickle file named 'final_data.pkl' in the cache directory (config.output_dir/cache)
containing a tuple: (final_feature_matrix, full_target).
"""

import os
import json
import pickle
import gc
import psutil
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config  # Assumes a config.py file with necessary hyperparameters/settings.

# ---------------------------
# I. Load Preprocessed Data
# ---------------------------

print("Starting main_preprocess...")

# Define cache directory and final data pickle path.
cache_dir = os.path.join(config.output_dir, 'cache')
final_data_path = os.path.join(cache_dir, 'final_data.pkl')

if not os.path.exists(final_data_path):
    raise FileNotFoundError(
        f"Preprocessed data file not found at {final_data_path}. "
        "Please run your full preprocessing pipeline first to generate this file."
    )

# Load preprocessed final integrated features and target.
with open(final_data_path, 'rb') as f:
    # Expected to be a tuple: (final_feature_matrix, full_target)
    final_feature_matrix, full_target = pickle.load(f)

print(f"Loaded preprocessed data: final_feature_matrix shape: {final_feature_matrix.shape}, target shape: {full_target.shape}")

# ---------------------------
# II. Split and Standardize Data
# ---------------------------

# Split the data into training and testing sets (90/10 split)
split_idx = int(len(final_feature_matrix) * 0.9)
if isinstance(final_feature_matrix, pd.DataFrame):
    train_feature_matrix = final_feature_matrix.iloc[:split_idx]
    test_feature_matrix = final_feature_matrix.iloc[split_idx:]
else:
    train_feature_matrix = final_feature_matrix[:split_idx]
    test_feature_matrix = final_feature_matrix[split_idx:]

train_target = full_target[:split_idx]
test_target = full_target[split_idx:]

# Standardize the feature matrices
feature_scaler = StandardScaler()
if isinstance(train_feature_matrix, pd.DataFrame):
    train_feature_matrix_scaled = pd.DataFrame(
        feature_scaler.fit_transform(train_feature_matrix),
        columns=train_feature_matrix.columns
    )
    test_feature_matrix_scaled = pd.DataFrame(
        feature_scaler.transform(test_feature_matrix),
        columns=test_feature_matrix.columns
    )
else:
    train_feature_matrix_scaled = feature_scaler.fit_transform(train_feature_matrix)
    test_feature_matrix_scaled = feature_scaler.transform(test_feature_matrix)

# Standardize the targets
target_scaler = StandardScaler()
train_target_scaled = target_scaler.fit_transform(train_target.reshape(-1, 1)).flatten()
test_target_scaled = target_scaler.transform(test_target.reshape(-1, 1)).flatten()

# ---------------------------
# III. Create Sliding Window Pairs
# ---------------------------

def create_window_pairs(data, target, seq_len, pred_len):
    """
    Converts a 2D array (time steps, features) and corresponding target array into sliding window pairs.
    Returns:
        features: (num_windows, seq_len, num_features)
        targets: (num_windows, pred_len)
    where num_windows = len(data) - seq_len - pred_len + 1.
    """
    X, Y = [], []
    num_windows = len(data) - seq_len - pred_len + 1
    for i in range(num_windows):
        X.append(data[i : i + seq_len])
        Y.append(target[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(Y)

# Create sliding window pairs for training and testing.
train_features, train_labels = create_window_pairs(train_feature_matrix_scaled, train_target_scaled, config.seq_len, config.pred_len)
test_features, test_labels = create_window_pairs(test_feature_matrix_scaled, test_target_scaled, config.seq_len, config.pred_len)

print(f"Sliding window pairs created: Train features: {train_features.shape}, Train labels: {train_labels.shape}; Test features: {test_features.shape}, Test labels: {test_labels.shape}")

# Convert to tensors.
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(-1)  # Adjusting dimension if needed.
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(-1)

# Create DataLoaders.
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

print("DataLoaders created.")

# Save final processed data for quick reloading to avoid re-running preprocessing steps.
final_data_cache_path = os.path.join(config.output_dir, 'final_data_cache.pkl')
final_data = {
    "train_features": train_features,
    "train_labels": train_labels,
    "test_features": test_features,
    "test_labels": test_labels,
}
with open(final_data_cache_path, "wb") as f:
    pickle.dump(final_data, f)
print(f"Final processed data saved to: {final_data_cache_path}")

# ---------------------------
# IV. Model, Optimizer & Loss Function Initialization
# ---------------------------

from models.lftsformer import EnhancedLFTSformer
from models.optimization import GCAdam
from training.loss_functions import DynamicLossFunction
from training.trainer import train_model
from evaluation.evaluation_metrics import calculate_mae, calculate_mse, calculate_rmse, calculate_r2
from evaluation.visualizations import plot_predictions, generate_heatmap, generate_boxplot

print("Initializing EnhancedLFTSformer model.")
model = EnhancedLFTSformer(
    feature_dim=config.feature_dim,      # Number of input features.
    d_model=config.d_model,              # Hidden dimensionality.
    d_ff=config.d_ff,                    # Feed-forward network hidden dimension.
    n_heads=config.n_heads,              # Number of attention heads.
    attention_factor=5.0,                # Attention factor; adjust as needed.
    dropout=config.dropout,              # Dropout rate.
    max_seq_len=config.seq_len,          # Maximum sequence length.
    pred_len=config.pred_len,            # Prediction length.
    output_dim=config.output_dim,        # Number of output features.
    timestamp_vocab_size=None            # Optional: set if using timestamp embeddings.
)
model = model.to(config.device)
print("Model initialized and moved to device.")

# Initialize loss function and move to device.
loss_function = DynamicLossFunction(beta_initial=config.beta_initial, c=config.c)
loss_function = loss_function.to(config.device)

# Initialize the GCAdam optimizer with model and loss function parameters.
optimizer = GCAdam(
    list(model.parameters()) + list(loss_function.parameters()),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)
print("Optimizer and loss function initialized.")

# Log memory usage.
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss/1e6:.2f} MB")
    gc.collect()

log_memory_usage()

# ---------------------------
# V. Model Training
# ---------------------------

print("Starting model training.")
trained_model, training_history = train_model(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    loss_function=loss_function,
    optimizer=optimizer,
    epochs=config.epochs,
    device=config.device,
    early_stopping_patience=config.early_stopping_patience
)
print("Model training completed.")

# ---------------------------
# VI. Model Evaluation
# ---------------------------
trained_model.eval()
all_true = []
all_preds = []
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        outputs = trained_model(inputs)
        all_true.append(labels.cpu().numpy())
        all_preds.append(outputs.cpu().numpy())

all_true = np.concatenate(all_true, axis=0)
all_preds = np.concatenate(all_preds, axis=0)

mae = calculate_mae(all_true, all_preds)
mse = calculate_mse(all_true, all_preds)
rmse = calculate_rmse(all_true, all_preds)
r2 = calculate_r2(all_true, all_preds)
print("Evaluation Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# ---------------------------
# VII. Save Outputs and Visualizations
# ---------------------------
# Save predictions plot, heatmap, and boxplot.
plot_predictions(all_true, all_preds, save_path=os.path.join(config.output_dir, "predictions_preprocess.png"))
generate_heatmap(all_true, all_preds, save_path=os.path.join(config.output_dir, "heatmap_preprocess.png"))
generate_boxplot(all_true, all_preds, save_path=os.path.join(config.output_dir, "boxplot_preprocess.png"))
print("Evaluations and visualizations saved.")

if __name__ == "__main__":
    # M1/MPS-specific initialization.
    if torch.backends.mps.is_available():
        pass  # Limit memory usage if needed.
    print("Training complete. Exiting.")
    torch.mps.empty_cache()  # Final cleanup. 