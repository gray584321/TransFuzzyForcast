#!/usr/bin/env python
"""
inference.py

Performs inference using a trained EnhancedLFTSformer model.  Loads the
trained model, feature scaler, and performs inference on a given CSV file.
"""

import torch
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from models.lftsformer import EnhancedLFTSformer  # Ensure this import matches your project structure
import config  # Import your configuration


def load_model(model_path, model_config):
    """Loads the trained model."""
    model = EnhancedLFTSformer(**model_config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(config.device)))
    model.to(config.device)
    model.eval()  # Set to evaluation mode
    return model

def load_scaler(scaler_path):
    """Loads the feature scaler."""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def preprocess_data(data_path, scaler, seq_len, pred_len, features):
    """
    Preprocesses the input data for inference.

    Args:
        data_path (str): Path to the input CSV file.
        scaler (StandardScaler): Trained feature scaler.
        seq_len (int): Input sequence length.
        pred_len (int): Prediction length.
        features (list): List of feature names to use.

    Returns:
        torch.Tensor: Preprocessed input tensor for the model.
        pd.DataFrame: The original dataframe (for plotting, etc.)
    """
    df = pd.read_csv(data_path)

    # Feature selection and ordering (MUST match training)
    df = df[features]

    # Scaling
    scaled_data = scaler.transform(df.values)
    scaled_data = torch.FloatTensor(scaled_data).to(config.device)

    # Create sequences.  We only need the *last* sequence for a single prediction.
    X = scaled_data[-seq_len:]  # Get the last 'seq_len' rows

    # Reshape for the model (add batch dimension)
    X = X.unsqueeze(0)  # Add batch dimension: [1, seq_len, feature_dim]

    return X, df


def run_inference(model, input_tensor):
    """Runs inference on the given input tensor."""
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_tensor)
    return output.cpu().numpy()  # Move to CPU and convert to NumPy array

def postprocess_output(output, scaler, target_feature):
    """
    Inverse transforms the output to the original scale.

    Args:
        output (np.ndarray): Model output.
        scaler (StandardScaler): Trained feature scaler.
        target_feature (str): The name of the target feature.

    Returns:
        np.ndarray: Inverse-transformed predictions.
    """

    # Create a dummy array with the same shape as the scaler's input
    dummy_array = np.zeros((output.shape[0], scaler.n_features_in_))

    # Find the index of the target feature
    target_index = config.features_to_use.index(target_feature)

    # Place the predictions into the dummy array at the target feature's column
    dummy_array[:, target_index] = output.flatten()

    # Inverse transform
    inverse_transformed = scaler.inverse_transform(dummy_array)

    # Extract the target feature
    predictions = inverse_transformed[:, target_index]

    return predictions


def main(data_path, model_path, scaler_path, output_path):
    """Main inference function."""

    # Load model configuration from config.py
    model_config = {
        "feature_dim": config.feature_dim,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "n_heads": config.n_heads,
        "attention_factor": config.model["attention_factor"],
        "dropout": config.dropout,
        "max_seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "output_dim": config.output_dim,
        "timestamp_vocab_size": config.model["timestamp_vocab_size"]
    }

    # 1. Load model and scaler
    model = load_model(model_path, model_config)
    scaler = load_scaler(scaler_path)

    # 2. Preprocess data
    input_tensor, df = preprocess_data(data_path, scaler, config.seq_len, config.pred_len, config.features_to_use)

    # 3. Run inference
    output = run_inference(model, input_tensor)

    # 4. Postprocess output
    predictions = postprocess_output(output, scaler, config.target_feature)

    # 5. Save or display results
    print("Predictions:", predictions)
    # Save to CSV (optional)
    output_df = pd.DataFrame({'predictions': predictions})
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # You could also add plotting/visualization here, comparing predictions
    # to the last 'pred_len' values of the 'target_feature' in the input df.

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with the trained model.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input CSV data file.")
    parser.add_argument("--model_path", type=str, default=os.path.join(config.output_dir, "trained_model_weights.pth"),
                        help="Path to the trained model weights (.pth file).")
    parser.add_argument("--scaler_path", type=str, default=os.path.join(config.output_dir, "feature_scaler.pkl"),
                        help="Path to the feature scaler (.pkl file).")
    parser.add_argument("--output_path", type=str, default="predictions.csv",
                        help="Path to save the predictions CSV file.")

    args = parser.parse_args()

    main(args.data_path, args.model_path, args.scaler_path, args.output_path) 