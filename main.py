#!/usr/bin/env python
"""
main.py

Orchestrates data loading, feature engineering, model training, and evaluation.
Cached pickle files are used for all heavy computations to speed up subsequent runs.
"""

import os
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # Use new OpenMP API instead of deprecated omp_set_nested

import json
import logging
import torch
import numpy as np
import pandas as pd
import gc
import psutil
import pickle
from sklearn.preprocessing import StandardScaler

print("here")
# Add this line right here, at the very top of your executable code in main.py
print("This is a test log from main.py at the very beginning!")

# ---------------------------
# I. Configuration Loading
# ---------------------------
import config
print("Configuration module imported.")

# Example: load key hyperparameters and settings from config
batch_size = config.batch_size
learning_rate = config.learning_rate
seq_len = config.seq_len
pred_len = config.pred_len
device = config.device
print("Hyperparameters and settings loaded from config.")

# ---------------------------
# II. Data Loading and Preprocessing
# ---------------------------
# Import functions from data/dataloader.py
from data.dataloader import load_raw_data, preprocess_data, create_dataloaders
print("Dataloader functions imported.")

# Load the raw data using the path defined in config
print("Loading raw data...")
raw_df = load_raw_data(config.merged_data_path)
print("Raw data loaded successfully.")

# Preprocess the raw data: returns processed training and testing DataFrames.
# (We then combine these to form the full dataset for feature engineering.)
print("Preprocessing data...")
train_df, test_df = preprocess_data(raw_df, config.features_to_use, config.target_feature)
print("Data preprocessing completed.")
full_df = pd.concat([train_df, test_df], ignore_index=True)
print("Combined full dataset shape for feature engineering:", full_df.shape)

# (Optional) You might initially create dataloaders here, but we will re-create them
# after feature engineering.
# train_dataloader, test_dataloader = create_dataloaders(train_df, test_df, batch_size, seq_len, pred_len)

# ---------------------------
# III. Feature Engineering (VMD-MIC+FE)
# ---------------------------
# Import feature engineering functions (which internally uses vmd_mic.py)
from features.feature_engineering import (
    vmd_feature_extraction,
    fuzzy_entropy_feature_extraction,
    composite_feature_creation,
    correlation_based_feature_selection,
    integrate_features,
    determine_optimal_k,
    variational_mode_decomposition
)
print("Feature engineering functions imported.")

# Setup cache directory and file for feature extraction
cache_dir = os.path.join(config.output_dir, 'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
cache_file = os.path.join(cache_dir, 'feature_cache.pkl')

def load_or_compute(cache_path, compute_func):
    """
    Checks if the cache file exists. If so, load and return the cached result.
    Otherwise, compute the result using compute_func(), store it in the cache, then return it.
    """
    if os.path.exists(cache_path):
        print(f"Loading cache from {cache_path}")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        return result
    else:
        result = compute_func()
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached results saved to {cache_path}")
        return result

def compute_imfs():
    results = []
    for feature in config.features_to_decompose:
        print(f"Processing {feature} for training...")
        train_signal = train_df[feature].values.astype(np.float32)
        optimal_k = determine_optimal_k(train_signal, config.k_range, config.vmd_params_dict)
        imfs_train = variational_mode_decomposition(
            train_signal, 
            K=optimal_k, 
            **{k: v for k, v in config.vmd_params_dict.items() if k not in ['K', 'chunk_size']}
        )
        results.append((feature, optimal_k, imfs_train.astype(np.float32)))
    
    optimal_k_dict = {feature: k for feature, k, _ in results}
    imfs_dict_train = {feature: imfs for feature, _, imfs in results}
    
    print("Processing testing data for IMFs...")
    imfs_dict_test = {}
    for feature in config.features_to_decompose:
        print(f"Processing {feature} (test)...")
        test_signal = test_df[feature].values.astype(np.float32)
        optimal_k = optimal_k_dict[feature]
        imfs_test = variational_mode_decomposition(
            test_signal, 
            K=optimal_k, 
            **{k: v for k, v in config.vmd_params_dict.items() if k not in ['K', 'chunk_size']}
        )
        imfs_dict_test[feature] = imfs_test.astype(np.float32)
        del test_signal
        gc.collect()
    
    return optimal_k_dict, imfs_dict_train, imfs_dict_test

imfs_cache_path = os.path.join(cache_dir, 'imfs_cache.pkl')
def compute_imfs_full():
    results = []
    for feature in config.features_to_decompose:
        print(f"Processing {feature} for VMD on full dataset...")
        signal = full_df[feature].values.astype(np.float32)
        optimal_k = determine_optimal_k(signal, config.k_range, config.vmd_params_dict)
        imfs = variational_mode_decomposition(
            signal, 
            K=optimal_k, 
            **{k: v for k, v in config.vmd_params_dict.items() if k not in ['K', 'chunk_size']}
        )
        results.append((feature, optimal_k, imfs.astype(np.float32)))
    optimal_k_dict = {feature: k for feature, k, _ in results}
    imfs_dict = {feature: imfs for feature, _, imfs in results}
    return optimal_k_dict, imfs_dict

imf_result = load_or_compute(imfs_cache_path, compute_imfs_full)
if isinstance(imf_result, tuple) and len(imf_result) > 2:
    imf_result = imf_result[:2]
optimal_k_dict, imfs_dict = imf_result

# After processing imfs_dict_train and imfs_dict_test, add:
from features.feature_engineering import (
    fuzzy_entropy_feature_extraction,
    composite_feature_creation,
    correlation_based_feature_selection,
    integrate_features
)

# Generate feature matrices
print("Creating final feature matrices...")

# Get both return values from fuzzy entropy extraction
def compute_fuzzy_entropy():
    return fuzzy_entropy_feature_extraction(imfs_dict)

fe_cache_path = os.path.join(cache_dir, 'fe_cache.pkl')
fe_values, processed_imfs = load_or_compute(fe_cache_path, compute_fuzzy_entropy)

def compute_composite_features():
    return composite_feature_creation(fe_values, processed_imfs, config.fe_thresholds)

composite_cache = os.path.join(cache_dir, 'composite_cache.pkl')
composite_features = load_or_compute(composite_cache, compute_composite_features)

# Perform MIC-based correlation feature selection on the full dataset
selected_features = correlation_based_feature_selection(full_df, config.target_feature)

# Integrate features on the full dataset
final_features_cache = os.path.join(cache_dir, 'final_features_full.pkl')
final_feature_matrix = load_or_compute(final_features_cache, lambda: integrate_features(full_df, composite_features, selected_features))
print("Final integrated feature matrix shape:", final_feature_matrix.shape)

# Split the final integrated features and target into train and test sets (90/10 split)
split_idx = int(len(final_feature_matrix) * 0.9)
train_feature_matrix = final_feature_matrix.iloc[:split_idx]
test_feature_matrix = final_feature_matrix.iloc[split_idx:]

full_target = full_df[config.target_feature].values
y_train = full_target[:split_idx]
y_test = full_target[split_idx:]

# Standardize the feature matrices
feature_scaler = StandardScaler()
train_feature_matrix = pd.DataFrame(
    feature_scaler.fit_transform(train_feature_matrix),
    columns=train_feature_matrix.columns)
test_feature_matrix = pd.DataFrame(
    feature_scaler.transform(test_feature_matrix),
    columns=test_feature_matrix.columns)

# Standardize the targets as well (optional, but recommended for stability)
target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Add this helper function to create sliding window pairs (features and labels)
def create_window_pairs(data, target, seq_len, pred_len):
    """
    Converts a 2D array (time steps, features) and a target array into sliding window pairs.
    The output shapes will be:
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

# --------- Create DataLoaders from the Final Feature Matrices ---------
from torch.utils.data import TensorDataset, DataLoader
print("Imported TensorDataset and DataLoader from torch.utils.data.")

# Convert feature matrices to numpy if they are pandas DataFrames
print("Converting feature matrices to numpy arrays if necessary.")
if isinstance(train_feature_matrix, pd.DataFrame):
    train_feature_matrix = train_feature_matrix.values
if isinstance(test_feature_matrix, pd.DataFrame):
    test_feature_matrix = test_feature_matrix.values

# Create sliding window pairs (features and corresponding targets) for training and testing.
train_features, train_labels = create_window_pairs(train_feature_matrix, y_train, config.seq_len, config.pred_len)
test_features, test_labels = create_window_pairs(test_feature_matrix, y_test, config.seq_len, config.pred_len)

print(f"Created sliding window pairs: Train features shape {train_features.shape}, Train labels shape {train_labels.shape}; Test features shape {test_features.shape}, Test labels shape {test_labels.shape}")

# Convert to tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
# If output_dim is 1, add an extra dimension for targets.
train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(-1)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(-1)

# Create PyTorch datasets and dataloaders
print("Creating PyTorch datasets and dataloaders.")
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("PyTorch datasets and dataloaders created.")

# ---------------------------
# IV. Model Initialization
# ---------------------------
from models.lftsformer import EnhancedLFTSformer
print("Imported EnhancedLFTSformer model.")

print("Initializing EnhancedLFTSformer model.")
model = EnhancedLFTSformer(
    feature_dim=config.feature_dim,      # Number of input features
    d_model=config.d_model,              # Hidden dimension
    d_ff=config.d_ff,                    # Feed-forward network hidden dimension
    n_heads=config.n_heads,              # Number of attention heads
    attention_factor=5.0,                # Attention factor (adjust as needed)
    dropout=config.dropout,              # Dropout rate
    max_seq_len=config.seq_len,          # Maximum sequence length
    pred_len=config.pred_len,            # Prediction length (as defined in config)
    output_dim=config.output_dim,        # Output dimension (e.g., 1 for closing price)
    timestamp_vocab_size=None            # Optional, set if you use timestamp embeddings
)
model = model.to(device)
print("EnhancedLFTSformer model initialized and moved to device.")

# Synchronize the prediction horizon. Use the model's actual prediction length.
pred_len = model.pred_len

# ---------------------------
# V. Optimizer and Loss Function Initialization
# ---------------------------
from models.optimization import GCAdam
from training.loss_functions import DynamicLossFunction
print("Imported optimizer (GCAdam) and loss function (DynamicLossFunction).")

# Initialize loss function FIRST
print("Initializing DynamicLossFunction.")
loss_function = DynamicLossFunction(beta_initial=config.beta_initial, c=config.c)
loss_function = loss_function.to(device)
print("DynamicLossFunction initialized and moved to device.")

# THEN initialize optimizer with loss function parameters
print("Initializing optimizer (GCAdam).")
optimizer = GCAdam(
    list(model.parameters()) + list(loss_function.parameters()),
    lr=learning_rate,
    weight_decay=config.weight_decay
)
print("Optimizer (GCAdam) initialized.")

# Add this function definition BEFORE the training section
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss/1e6:.2f} MB")
    if torch.backends.mps.is_available():
        print(f"MPS Memory: {torch.mps.current_allocated_memory()/1e6:.2f} MB")
    # Force garbage collection
    gc.collect()
    torch.mps.empty_cache()

# ---------------------------
# VI. Model Training
# ---------------------------
from training.trainer import train_model
print("Imported train_model function.")

# Add memory logging before training (NOW THIS COMES AFTER FUNCTION DEFINITION)
log_memory_usage()
print("Starting model training.")

trained_model, training_history = train_model(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    loss_function=loss_function,
    optimizer=optimizer,
    epochs=config.epochs,
    device=device,
    early_stopping_patience=config.early_stopping_patience
)
print("Model training completed.")

# ---------------------------
# VII. Model Evaluation
# ---------------------------
from evaluation.evaluation_metrics import calculate_mae, calculate_mse, calculate_rmse, calculate_r2
from evaluation.visualizations import plot_predictions, generate_heatmap, generate_boxplot
print("Imported evaluation metrics and visualization functions.")

# Load the best saved model weights (if they have been saved by train_model)
best_model_path = os.path.join(config.output_dir, "best_model.pth")
if os.path.exists(best_model_path):
    print("Loading best model weights from disk.")
    trained_model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Best model weights loaded.")
trained_model.eval()

all_true = []
all_preds = []
print("Starting model evaluation.")
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = trained_model(inputs)
        all_true.append(labels.cpu().numpy())
        all_preds.append(outputs.cpu().numpy())

all_true = np.concatenate(all_true, axis=0)
all_preds = np.concatenate(all_preds, axis=0)
print("Model predictions obtained.")

# Calculate evaluation metrics
print("Calculating evaluation metrics.")
mae = calculate_mae(all_true, all_preds)
mse = calculate_mse(all_true, all_preds)
rmse = calculate_rmse(all_true, all_preds)
r2 = calculate_r2(all_true, all_preds)
print("Evaluation metrics calculated.")

print("Evaluation Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# Generate visualizations (plots will be saved to the output directory)
print("Generating prediction plot.")
plot_predictions(all_true, all_preds, save_path=os.path.join(config.output_dir, "predictions.png"))
print("Prediction plot generated and saved.")
print("Generating heatmap.")
generate_heatmap(all_true, all_preds, save_path=os.path.join(config.output_dir, "heatmap.png"))
print("Heatmap generated and saved.")
print("Generating boxplot.")
generate_boxplot(all_true, all_preds, save_path=os.path.join(config.output_dir, "boxplot.png"))
print("Boxplot generated and saved.")

# ---------------------------
# VIII. Save Results and Logs
# ---------------------------
# Ensure the output directory exists
if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)

# Save evaluation metrics
print("Saving evaluation metrics to file.")
metrics_file = os.path.join(config.output_dir, "evaluation_metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"MSE: {mse}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")
print("Evaluation metrics saved.")

# Save training history to a JSON file
print("Saving training history to JSON file.")
history_file = os.path.join(config.output_dir, "training_history.json")
with open(history_file, "w") as f:
    json.dump(training_history, f, indent=4)
print("Training history saved.")

# Save the trained model weights
print("Saving trained model weights.")
model_weights_file = os.path.join(config.output_dir, "trained_model_weights.pth")
torch.save(trained_model.state_dict(), model_weights_file)
print("Trained model weights saved.")

print("Experiment completed successfully.")
print(f"Evaluation Metrics: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
print("Training history and model weights saved.")

# Save the cached feature extraction data so subsequent runs can load it instead
with open(cache_file, 'wb') as f:
    pickle.dump((optimal_k_dict, imfs_dict), f)
print("Feature extraction data cached.")

if __name__ == "__main__":
    # M1/MPS specific initialization
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.5)  # Limit memory usage if needed
    
    def main():
        # Existing code from the script...
        print("Experiment completed successfully.")
    
    main()
    torch.mps.empty_cache()  # Final cleanup 