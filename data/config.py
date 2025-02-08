import logging
import os
import sys  # Import the sys module
import torch

# Define output directory for logs and results
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure logging
log_file = os.path.join(output_dir, "experiment.log")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Add a StreamHandler to send logs to the console
stdout_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout to print to standard output
stdout_handler.setLevel(logging.INFO) # Set the level for the handler (can be different from the basicConfig level)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s') # Use the same format as basicConfig, or customize
stdout_handler.setFormatter(formatter)
logging.getLogger('').addHandler(stdout_handler) # Get the root logger and add the handler

logging.info("Starting script execution.")

TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "JPM",
    "V",
    "UNH",
    "JNJ",
    "PFE",
    "XOM",
    "CVX",
    "WMT",
    "HD",
    "DIS",
    "NFLX",
    "BA",
    "CAT",
    "KO",
    "NKE",
    "TMUS",
    "GOOG",
]


# --- Hyperparameters ---
seq_len = 256
pred_len = 64
batch_size = 32 # try 16, 32, 64
learning_rate = 0.001 # try 0.0001, 0.001, 0.01
epochs = 50 # number of training epochs
early_stopping_patience = 10 # patience for early stopping
gradient_clip_value = 5.0 # Gradient clipping value (e.g., 5.0) or None to disable

# --- Feature Engineering Parameters ---
features_to_use = [
    'Weighted_Price', 'High_Minus_Low', 'Return_Lag1', 'Return_Lag2',
    'Return_Lag3', 'Return_Lag4', 'Return_Lag5', 'Return_Lag6',
    'Correlation_MA5_MA30', 'Sum3', 'Sum5',
    'Sum5_Squared_Minus_Sum3_Squared_Sqrt', 'RSI_6', 'RSI_14',
    'Rate_of_Change_9', 'Rate_of_Change_14', 'Williams_R', 'ATR_5',
    'ATR_10', 'CCI', 'DEMA', 'Year', 'Month', 'Day', 'Hour', 'Minute',
    'Scaled_Time',  # Include Scaled_Time
    'Composite_Low', 'Composite_High' # Composite features
]
target_feature = 'Processed_Close_Price'
features_to_decompose = [
    'feature1', # Keep only essential features for decomposition
    'feature2',
    # Remove less important features from this list
]
vmd_params_dict = {
    'alpha': 2000,        # moderate bandwidth constraint
    'tau': 0.,            # noise-tolerance (no strict fidelity enforcement)
    'K': 3,             # Reduced number of modes (was possibly higher or None)
    'DC': 0,             # no DC part imposed
    'init': 1,           # initialize omegas uniformly
    'tol': 1e-7,
    'kMax': 30,          # Maximum K if K is None (automatic K estimation)
    'energy_loss_coefficient': 0.01 # Energy loss coefficient for automatic K estimation
}
fuzzy_m = 2 # Fuzzy entropy embedding dimension
fuzzy_r = 0.2 # Fuzzy entropy tolerance ratio (r = r_multiplier * std)
fuzzy_n = 2 # Fuzzy entropy step size
fe_thresholds = {"Composite_Low": (0, 0.5), "Composite_High": (0.5, float('inf'))} # Fuzzy entropy thresholds for composite features
mic_threshold = 0.5 # MIC threshold for feature selection

# --- Model Parameters ---
input_size = len(features_to_use) # Number of input features
hidden_size = 128 # Number of LSTM units
num_layers = 2 # Number of LSTM layers
output_size = pred_len # Prediction length for forecasting
dropout_rate = 0.2 # Dropout rate
model_type = 'LSTM' # Model type: 'LSTM' or 'Transformer'

# --- Data Paths ---
raw_data_dir = "data/raw" # Directory for raw CSV data
processed_data_dir = "data/processed" # Directory for processed CSV data
merged_data_path = os.path.join(processed_data_dir, "merged_stock_data.csv") # Path to merged processed data

# --- Output Paths ---
output_dir = "output" # Defined at the beginning
model_output_path = os.path.join(output_dir, "trained_model.pth") # Path to save trained model
metrics_output_path = os.path.join(output_dir, "evaluation_metrics.json") # Path to save evaluation metrics
predictions_output_path = os.path.join(output_dir, "predictions.csv") # Path to save predictions
training_history_path = os.path.join(output_dir, "training_history.json") # Path to save training history

# --- Device Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu" # Use CUDA if available, otherwise CPU

# --- Validation Tracking ---
VALIDATED_FILES = { # Dictionary to track validated files, initialized as empty
}
VALIDATED_FILES['SPY.csv'] = True  # Validated on 2025-02-05 14:16:53.023620
VALIDATED_FILES['QQQ.csv'] = True  # Validated on 2025-02-05 14:14:00.469831
VALIDATED_FILES['AAPL.csv'] = True  # Validated on 2025-02-05 14:18:31.212258
VALIDATED_FILES['MSFT.csv'] = True  # Validated on 2025-02-05 14:14:38.867633
VALIDATED_FILES['GOOGL.csv'] = True  # Validated on 2025-02-05 14:17:02.920084
VALIDATED_FILES['AMZN.csv'] = True  # Validated on 2025-02-05 14:14:00.469831
VALIDATED_FILES['TSLA.csv'] = True  # Validated on 2025-02-05 14:16:38.308641
VALIDATED_FILES['NVDA.csv'] = True  # Validated on 2025-02-05 14:15:10.874828
VALIDATED_FILES['JPM.csv'] = True  # Validated on 2025-02-05 14:16:34.544454
VALIDATED_FILES['V.csv'] = True  # Validated on 2025-02-05 14:18:20.930113
VALIDATED_FILES['UNH.csv'] = True  # Validated on 2025-02-05 14:18:17.546863
VALIDATED_FILES['JNJ.csv'] = True  # Validated on 2025-02-05 14:18:57.279843
VALIDATED_FILES['PFE.csv'] = True  # Validated on 2025-02-05 14:17:38.775836
VALIDATED_FILES['XOM.csv'] = True  # Validated on 2025-02-05 14:15:31.996238
VALIDATED_FILES['CVX.csv'] = True  # Validated on 2025-02-05 14:15:35.850281
VALIDATED_FILES['WMT.csv'] = True  # Validated on 2025-02-05 14:18:40.598579
VALIDATED_FILES['HD.csv'] = True  # Validated on 2025-02-05 14:15:22.030701
VALIDATED_FILES['DIS.csv'] = True  # Validated on 2025-02-05 14:16:17.026539
VALIDATED_FILES['NFLX.csv'] = True  # Validated on 2025-02-05 14:16:30.732791
VALIDATED_FILES['BA.csv'] = True  # Validated on 2025-02-05 14:14:51.549546
VALIDATED_FILES['CAT.csv'] = True  # Validated on 2025-02-05 14:13:53.661424
VALIDATED_FILES['KO.csv'] = True  # Validated on 2025-02-05 14:14:27.394420
VALIDATED_FILES['NKE.csv'] = True  # Validated on 2025-02-05 14:15:52.425896
VALIDATED_FILES['TMUS.csv'] = True  # Validated on 2025-02-05 14:18:37.253186
VALIDATED_FILES['GOOG.csv'] = True  # Validated on 2025-02-05 14:17:33.861703


logging.info("Configuration loaded successfully.")
