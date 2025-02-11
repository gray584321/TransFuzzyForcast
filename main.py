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
import glob
from tqdm import tqdm


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
# --- Device-Agnostic Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("Hyperparameters and settings loaded from config.")

# ---------------------------
# II. Data Loading and Preprocessing from Individual CSV Files
# ---------------------------
print("Checking for cached processed features...")
if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)
features_cache_path = os.path.join(config.output_dir, "processed_features.pkl")
# Define cache_file
cache_file = os.path.join(config.output_dir, "feature_extraction_cache.pkl")

if os.path.exists(features_cache_path):
    print(f"Loading cached processed features from {features_cache_path}")
    with open(features_cache_path, 'rb') as f:
        cached_data = pickle.load(f)
        full_features = cached_data['features']
        full_target = cached_data['target']
    print("Loaded cached features successfully.")
    print("Combined feature matrix shape from cache:", full_features.shape)
else:
    print("No cached features found. Processing CSV files from the processed folder individually...")
    processed_folder = 'data/processed'
    csv_files = glob.glob(os.path.join(processed_folder, '*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in processed folder: {processed_folder}")

    from features.feature_engineering import (
        variational_mode_decomposition,
        determine_optimal_k,
        fuzzy_entropy_feature_extraction,
        composite_feature_creation,
        correlation_based_feature_selection,
        integrate_features
    )

    def find_best_k(signal):
        best_k = determine_optimal_k(signal, config.k_range, config.vmd_params_dict)
        return best_k

    def process_single_csv(file_path, selected_features=None, is_first_file=False, optimal_k_dict=None):
        print(f"\nProcessing CSV file: {file_path}")
        df = pd.read_csv(file_path)
        imfs_dict = {}
        
        # Calculate optimal k only for the first file
        if is_first_file:
            optimal_k_dict = {}
            print("Calculating optimal k values...")
            for feature in tqdm(config.features_to_decompose, desc="Finding optimal k"):
                if feature in df.columns:
                    signal = df[feature].values.astype(np.float32)
                    optimal_k = find_best_k(signal)
                    optimal_k_dict[feature] = optimal_k
        
        # Process features using either calculated or provided optimal k values
        print("Performing VMD decomposition...")
        for feature in tqdm(config.features_to_decompose, desc="VMD Processing"):
            if feature in df.columns:
                signal = df[feature].values.astype(np.float32)
                k_value = optimal_k_dict.get(feature)
                if k_value is None:
                    print(f"Warning: No optimal k found for {feature}, skipping")
                    continue
                
                imfs = variational_mode_decomposition(
                    signal, 
                    K=k_value, 
                    **{k: v for k, v in config.vmd_params_dict.items() if k not in ['K', 'chunk_size']}
                )
                imfs_dict[feature] = imfs.astype(np.float32)
            else:
                print(f"Warning: {feature} not found in {file_path}")
        
        print("Calculating fuzzy entropy...")
        fe_values, processed_imfs = fuzzy_entropy_feature_extraction(imfs_dict)
        
        print("Creating composite features...")
        composite_features = composite_feature_creation(fe_values, processed_imfs, config.fe_thresholds)
        
        if is_first_file:
            print("Performing feature selection on first file...")
            selected_features = correlation_based_feature_selection(df, config.target_feature)
            print(f"Selected features from first file: {selected_features}")
        
        print("Integrating features...")
        final_feature_matrix = integrate_features(df, composite_features, selected_features)
        return final_feature_matrix, df[config.target_feature].values.astype(np.float32), selected_features, optimal_k_dict

    # Process first file to get selected features and optimal k values
    print("\nProcessing first file to determine optimal k values and selected features...")
    first_file = csv_files[0]
    first_features, first_target, selected_features, optimal_k_dict = process_single_csv(first_file, is_first_file=True)
    feature_matrices = [first_features]
    target_arrays = [first_target]

    # Process remaining files using the same selected features and optimal k values
    print("\nProcessing remaining files using parameters from first file...")
    for csv_file in tqdm(csv_files[1:], desc="Processing files"):
        features, target, _, _ = process_single_csv(
            csv_file, 
            selected_features=selected_features, 
            is_first_file=False, 
            optimal_k_dict=optimal_k_dict
        )
        feature_matrices.append(features)
        target_arrays.append(target)

    print("\nCombining all processed features...")
    full_features = pd.concat(feature_matrices, ignore_index=True)
    full_target = np.concatenate(target_arrays)
    print("Combined feature matrix shape for training:", full_features.shape)
    
    # Save processed features to cache
    print(f"Saving processed features to cache: {features_cache_path}")
    with open(features_cache_path, 'wb') as f:
        pickle.dump({
            'features': full_features,
            'target': full_target,
        }, f)
    print("Saved processed features to cache.")

# ---------------------------
# III. DataLoader Setup
# ---------------------------
print("Setting up data loaders...")

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(full_features)
scaled_features = torch.FloatTensor(scaled_features)
full_target = torch.FloatTensor(full_target)

# Create sequences for training
def create_sequences(features, targets, seq_length, pred_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(features) - seq_length - pred_length + 1):
        X.append(features[i:(i + seq_length)])
        y.append(targets[i + seq_length:i + seq_length + pred_length])
    return torch.stack(X), torch.stack(y)

# Create sequences
print("Creating sequences...")
X, y = create_sequences(scaled_features, full_target, seq_len, pred_len)
print(f"Sequence shapes - X: {X.shape}, y: {y.shape}")

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Train/Test split - Train size: {len(X_train)}, Test size: {len(X_test)}")

# Create DataLoader instances
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Set to 0 for Apple Silicon and non-CUDA
    pin_memory=True if device.type == 'cuda' else False  # Only pin memory for CUDA
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # Set to 0 for Apple Silicon and non-CUDA
    pin_memory=True if device.type == 'cuda' else False  # Only pin memory for CUDA
)

print(f"Created dataloaders - Train batches: {len(train_dataloader)}, Test batches: {len(test_dataloader)}")

# Save scaler for inference
if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)
scaler_path = os.path.join(config.output_dir, "feature_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Saved feature scaler to {scaler_path}")

# ---------------------------
# IV. Model Initialization
# ---------------------------
from models.lftsformer import EnhancedLFTSformer
print("Imported EnhancedLFTSformer model.")

# Get the actual feature dimension from the processed data
actual_feature_dim = scaled_features.shape[1]
print(f"Actual feature dimension from data: {actual_feature_dim}")

print("Initializing EnhancedLFTSformer model.")
model = EnhancedLFTSformer(
    feature_dim=actual_feature_dim,      # Use actual feature dimension instead of config
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
model = model.to(device) # Move model to the selected device
print(f"EnhancedLFTSformer model initialized with feature_dim={actual_feature_dim} and moved to device.")

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
loss_function = loss_function.to(device) # Move loss function to device
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
    if device.type == 'mps': # Only relevant for MPS
        print(f"MPS Memory: {torch.mps.current_allocated_memory()/1e6:.2f} MB")
    # Force garbage collection
    gc.collect()
    if device.type == 'mps': # Only relevant for MPS
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
    device=device,  # Pass the device to the training function
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
        inputs = inputs.to(device)  # Move inputs to device
        labels = labels.to(device)  # Move labels to device
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

# ---------------------------
# IX. Run Inference (Example)
# ---------------------------
print("Running inference example...")
import subprocess

# Construct the command to run inference.py
#  We'll use the test data for this example, but you could use any CSV.
#  Adjust paths as necessary.

# Find a test file to use for inference
test_csv_files = glob.glob(os.path.join('data/processed', '*.csv'))
if not test_csv_files:
    print("No CSV files found in processed folder for inference.")
else:
    inference_data_path = test_csv_files[0]  # Use the first CSV file found
    inference_command = [
        "python",
        "inference.py",
        "--data_path", inference_data_path,
        "--model_path", os.path.join(config.output_dir, "trained_model_weights.pth"),
        "--scaler_path", os.path.join(config.output_dir, "feature_scaler.pkl"),
        "--output_path", os.path.join(config.output_dir, "example_predictions.csv")
    ]

    try:
        subprocess.run(inference_command, check=True)
        print("Inference completed.  Example predictions saved.")
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")

if __name__ == "__main__":
    # M1/MPS specific initialization
    if device.type == 'mps': # Only relevant for MPS
        torch.mps.set_per_process_memory_fraction(0.5)  # Limit memory usage if needed
    
    def main():
        # Existing code from the script...
        print("Experiment completed successfully.")
    
    main()
    if device.type == 'mps': # Only relevant for MPS
        torch.mps.empty_cache()  # Final cleanup 