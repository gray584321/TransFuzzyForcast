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
# II. Data Loading and Preprocessing from Individual CSV Files
# ---------------------------
print("Processing CSV files from the processed folder individually...")
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

def process_single_csv(file_path):
    print(f"Processing CSV file: {file_path}")
    df = pd.read_csv(file_path)
    optimal_k_dict = {}
    imfs_dict = {}
    for feature in config.features_to_decompose:
        if feature in df.columns:
            print(f"Processing VMD for feature: {feature}")
            signal = df[feature].values.astype(np.float32)
            optimal_k = determine_optimal_k(signal, config.k_range, config.vmd_params_dict)
            optimal_k_dict[feature] = optimal_k
            imfs = variational_mode_decomposition(
                signal, 
                K=optimal_k, 
                **{k: v for k, v in config.vmd_params_dict.items() if k not in ['K', 'chunk_size']}
            )
            imfs_dict[feature] = imfs.astype(np.float32)
        else:
            print(f"Warning: {feature} not found in {file_path}")
    fe_values, processed_imfs = fuzzy_entropy_feature_extraction(imfs_dict)
    composite_features = composite_feature_creation(fe_values, processed_imfs, config.fe_thresholds)
    selected_features = correlation_based_feature_selection(df, config.target_feature)
    final_feature_matrix = integrate_features(df, composite_features, selected_features)
    return final_feature_matrix, df[config.target_feature].values

feature_matrices = []
target_arrays = []
for csv_file in csv_files:
    features, target = process_single_csv(csv_file)
    feature_matrices.append(features)
    target_arrays.append(target)

full_features = pd.concat(feature_matrices, ignore_index=True)
full_target = np.concatenate(target_arrays)
print("Combined feature matrix shape for training:", full_features.shape)

# Feature engineering already applied per CSV file. Using the processed features directly.

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