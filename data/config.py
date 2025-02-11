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
    "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V",
    "UNH", "JNJ", "PFE", "XOM", "CVX", "WMT", "HD", "DIS", "NFLX", "BA",
    "CAT", "KO", "NKE", "TMUS", "ACN", "ORCL", "IBM", "INTC", "AMD", "ADBE",
    "CRM", "PYPL", "CMCSA", "T", "VZ", "CSCO", "QCOM", "TXN", "BMY", "COST",
    "GE", "GS", "AXP", "MMM", "LMT", "HON", "UPS", "FDX", "USB", "SCHW",
    "BLK", "AMGN", "GILD", "MDT", "LLY", "ABT", "CVS", "UNP", "SBUX", "MCD",
    "LOW", "SPGI", "AMAT", "ADI", "MU", "LRCX", "ISRG", "ZTS", "VRTX", "BIIB",
    "ILMN", "EXC", "AEP", "DUK", "SO", "NEE", "D", "ED", "PNW", "EIX",
    "DTE", "SRE", "PEG", "XEL", "PCG", "APD", "LIN", "DD", "EMR", "ITW",
    "F", "GM", "TGT", "C", "WFC", "MS", "PGR", "ALL", "PRU", "TRV",
    "CB", "MET", "AIG", "VLO", "MPC", "COP", "SLB", "OXY", "HAL", "FANG",
    "MRO", "FTI", "PSX", "HES", "WMB", "FLS", "VMC", "NUE", "DOV", "ALB",
    "CL", "KMB", "PEP", "SJM", "CAG", "CPB", "SYY", "ADM", "MOS", "K",
    "CF", "CPRT", "DLR", "AMT", "WY", "SPG", "SBAC", "EXR", "AVB", "ESS",
    "EQR", "MAA", "PEAK", "PLD", "EQIX", "CINF", "MMC", "PFG", "AMP", "UNM",
    "STT", "WLTW", "AFL", "CME", "ICE", "NDAQ", "FTNT", "INTU", "ANSS", "CDNS",
    "WDAY", "SNPS", "VRSN", "CTSH", "FIS", "FISV", "MSCI", "DXCM", "COO", "MTB",
    "TFC", "KEY", "BEN", "BK", "PNC", "BAC", "C", "BBT", "CFG", "FITB",
    "HBAN", "ZION", "L", "AON", "EXPE", "BKNG", "HLT", "MAR", "CMG", "MGM",
    "DRI", "YUM", "CLX", "KHC", "DG", "STLA", "XPEV", "LI", "NIO", "BYD",
    "LULU", "ROST", "GPS", "CTXS", "DGX", "SIRI", "EA", "ATVI", "TTWO", "Z",
    "EBAY", "NOW", "OKTA", "TEAM", "DOCU", "SNOW", "ZS", "SQ", "SHOP", "ZM",
    "FSLY", "PINS", "NET", "WORK", "COUP", "UAA", "BIDU", "NTES", "JD", "PDD",
    "BABA", "TCEHY", "VOD", "ERIC", "NOK", "HMC", "TM", "NEM", "ECL", "SHW",
    "MLM", "FCX", "FMC", "BLL", "IP", "NTR", "X", "ALXN", "REGN", "INCY"
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

VALIDATED_FILES['CSCO.csv'] = True  # Validated on 2025-02-08 23:57:36.333042

VALIDATED_FILES['BB.csv'] = True  # Validated on 2025-02-08 23:57:40.072743

VALIDATED_FILES['AXP.csv'] = True  # Validated on 2025-02-08 23:57:49.505392

VALIDATED_FILES['AVGO.csv'] = True  # Validated on 2025-02-08 23:57:52.821647

VALIDATED_FILES['BABA.csv'] = True  # Validated on 2025-02-08 23:57:56.841227

VALIDATED_FILES['ISRG.csv'] = True  # Validated on 2025-02-10 21:52:04.854216

VALIDATED_FILES['PDD.csv'] = True  # Validated on 2025-02-10 21:52:30.727325

VALIDATED_FILES['VRTX.csv'] = True  # Validated on 2025-02-10 21:53:15.884981

VALIDATED_FILES['GILD.csv'] = True  # Validated on 2025-02-10 21:53:37.278538

VALIDATED_FILES['EQIX.csv'] = True  # Validated on 2025-02-10 21:53:57.161193

VALIDATED_FILES['MDT.csv'] = True  # Validated on 2025-02-10 21:54:17.679466

VALIDATED_FILES['TM.csv'] = True  # Validated on 2025-02-10 21:54:37.977875

VALIDATED_FILES['CDNS.csv'] = True  # Validated on 2025-02-10 21:55:19.733873

VALIDATED_FILES['MSCI.csv'] = True  # Validated on 2025-02-10 21:55:39.489510

VALIDATED_FILES['EIX.csv'] = True  # Validated on 2025-02-10 21:55:59.708878

VALIDATED_FILES['BYD.csv'] = True  # Validated on 2025-02-10 21:56:19.792646

VALIDATED_FILES['LI.csv'] = True  # Validated on 2025-02-10 21:56:45.396182

VALIDATED_FILES['DTE.csv'] = True  # Validated on 2025-02-10 21:57:05.711263

VALIDATED_FILES['C.csv'] = True  # Validated on 2025-02-10 21:57:29.506033

VALIDATED_FILES['T.csv'] = True  # Validated on 2025-02-10 21:57:54.638247

VALIDATED_FILES['CF.csv'] = True  # Validated on 2025-02-10 21:58:15.413112

VALIDATED_FILES['MGM.csv'] = True  # Validated on 2025-02-10 21:58:36.850492

VALIDATED_FILES['ZS.csv'] = True  # Validated on 2025-02-10 21:58:58.871721

VALIDATED_FILES['CFG.csv'] = True  # Validated on 2025-02-10 22:10:41.438816

VALIDATED_FILES['STLA.csv'] = True  # Validated on 2025-02-10 22:11:04.875376

VALIDATED_FILES['FSLY.csv'] = True  # Validated on 2025-02-10 22:11:26.989851

VALIDATED_FILES['SYY.csv'] = True  # Validated on 2025-02-10 22:11:46.984294

VALIDATED_FILES['FCX.csv'] = True  # Validated on 2025-02-10 22:12:10.249024

VALIDATED_FILES['ADM.csv'] = True  # Validated on 2025-02-10 22:12:30.892837

VALIDATED_FILES['BAC.csv'] = True  # Validated on 2025-02-10 22:12:56.690824

VALIDATED_FILES['PSX.csv'] = True  # Validated on 2025-02-10 22:13:17.511463

VALIDATED_FILES['ESS.csv'] = True  # Validated on 2025-02-10 22:13:37.038327

VALIDATED_FILES['HAL.csv'] = True  # Validated on 2025-02-10 22:13:58.737292

VALIDATED_FILES['FLS.csv'] = True  # Validated on 2025-02-10 22:14:18.712187

VALIDATED_FILES['ADI.csv'] = True  # Validated on 2025-02-10 22:14:39.228061

VALIDATED_FILES['F.csv'] = True  # Validated on 2025-02-10 22:15:06.068128

VALIDATED_FILES['ADBE.csv'] = True  # Validated on 2025-02-10 22:15:28.068162

VALIDATED_FILES['CPRT.csv'] = True  # Validated on 2025-02-10 22:15:48.732891

VALIDATED_FILES['CB.csv'] = True  # Validated on 2025-02-10 22:16:08.853801

VALIDATED_FILES['PEP.csv'] = True  # Validated on 2025-02-10 22:16:30.143704

VALIDATED_FILES['PEG.csv'] = True  # Validated on 2025-02-10 22:16:50.276235

VALIDATED_FILES['NOW.csv'] = True  # Validated on 2025-02-10 22:17:10.872533

VALIDATED_FILES['LLY.csv'] = True  # Validated on 2025-02-10 22:17:32.662187

VALIDATED_FILES['COST.csv'] = True  # Validated on 2025-02-10 22:17:53.677306

VALIDATED_FILES['LOW.csv'] = True  # Validated on 2025-02-10 22:18:14.404505

VALIDATED_FILES['BKNG.csv'] = True  # Validated on 2025-02-10 22:18:33.578709

VALIDATED_FILES['FMC.csv'] = True  # Validated on 2025-02-10 22:18:53.303830

VALIDATED_FILES['XEL.csv'] = True  # Validated on 2025-02-10 22:19:13.413232

VALIDATED_FILES['WDAY.csv'] = True  # Validated on 2025-02-10 22:19:33.974431

VALIDATED_FILES['MET.csv'] = True  # Validated on 2025-02-10 22:19:54.145191

VALIDATED_FILES['DLR.csv'] = True  # Validated on 2025-02-10 22:20:14.249543

VALIDATED_FILES['TCEHY.csv'] = True  # Validated on 2025-02-10 22:20:33.980139

VALIDATED_FILES['MPC.csv'] = True  # Validated on 2025-02-10 22:20:54.354005

VALIDATED_FILES['D.csv'] = True  # Validated on 2025-02-10 22:21:14.972854

VALIDATED_FILES['KHC.csv'] = True  # Validated on 2025-02-10 22:21:36.632274

VALIDATED_FILES['UNP.csv'] = True  # Validated on 2025-02-10 22:21:56.831257

VALIDATED_FILES['ORCL.csv'] = True  # Validated on 2025-02-10 22:22:19.774619

VALIDATED_FILES['ECL.csv'] = True  # Validated on 2025-02-10 22:22:39.881692

VALIDATED_FILES['EBAY.csv'] = True  # Validated on 2025-02-10 22:23:00.624476

VALIDATED_FILES['SBUX.csv'] = True  # Validated on 2025-02-10 22:23:23.236711

VALIDATED_FILES['AMT.csv'] = True  # Validated on 2025-02-10 22:23:43.599042

VALIDATED_FILES['INTU.csv'] = True  # Validated on 2025-02-10 22:24:03.855453

VALIDATED_FILES['MCD.csv'] = True  # Validated on 2025-02-10 22:25:16.169486

VALIDATED_FILES['INTC.csv'] = True  # Validated on 2025-02-10 22:26:13.495895

VALIDATED_FILES['DOCU.csv'] = True  # Validated on 2025-02-10 22:26:35.677673

VALIDATED_FILES['DXCM.csv'] = True  # Validated on 2025-02-10 22:26:56.419234

VALIDATED_FILES['EXR.csv'] = True  # Validated on 2025-02-10 22:27:16.394840

VALIDATED_FILES['XPEV.csv'] = True  # Validated on 2025-02-10 22:27:43.724627

VALIDATED_FILES['GM.csv'] = True  # Validated on 2025-02-10 22:28:07.183128

VALIDATED_FILES['TXN.csv'] = True  # Validated on 2025-02-10 22:28:28.323250

VALIDATED_FILES['SJM.csv'] = True  # Validated on 2025-02-10 22:28:48.319813

VALIDATED_FILES['OXY.csv'] = True  # Validated on 2025-02-10 22:29:14.012445

VALIDATED_FILES['MMM.csv'] = True  # Validated on 2025-02-10 22:29:35.706587

VALIDATED_FILES['MOS.csv'] = True  # Validated on 2025-02-10 22:29:56.434834

VALIDATED_FILES['FTNT.csv'] = True  # Validated on 2025-02-10 22:30:17.884414

VALIDATED_FILES['ED.csv'] = True  # Validated on 2025-02-10 22:30:37.890716

VALIDATED_FILES['VOD.csv'] = True  # Validated on 2025-02-10 22:31:00.180845

VALIDATED_FILES['IP.csv'] = True  # Validated on 2025-02-10 22:31:20.723235

VALIDATED_FILES['EXPE.csv'] = True  # Validated on 2025-02-10 22:31:41.132226

VALIDATED_FILES['PYPL.csv'] = True  # Validated on 2025-02-10 22:32:30.363400

VALIDATED_FILES['NEE.csv'] = True  # Validated on 2025-02-10 22:45:48.241086

VALIDATED_FILES['UPS.csv'] = True  # Validated on 2025-02-10 22:46:09.549138

VALIDATED_FILES['BIDU.csv'] = True  # Validated on 2025-02-10 22:46:33.870810

VALIDATED_FILES['EMR.csv'] = True  # Validated on 2025-02-10 22:46:54.134202

VALIDATED_FILES['ANSS.csv'] = True  # Validated on 2025-02-10 22:47:41.879370

VALIDATED_FILES['OKTA.csv'] = True  # Validated on 2025-02-10 22:48:03.236550

VALIDATED_FILES['NTES.csv'] = True  # Validated on 2025-02-10 22:48:24.290238

VALIDATED_FILES['DD.csv'] = True  # Validated on 2025-02-10 22:48:44.736486

VALIDATED_FILES['ACN.csv'] = True  # Validated on 2025-02-10 22:49:05.407352

VALIDATED_FILES['VRSN.csv'] = True  # Validated on 2025-02-10 22:49:25.446816

VALIDATED_FILES['CMG.csv'] = True  # Validated on 2025-02-10 22:49:46.070681

VALIDATED_FILES['COO.csv'] = True  # Validated on 2025-02-10 22:50:05.480761

VALIDATED_FILES['SHW.csv'] = True  # Validated on 2025-02-10 22:50:25.360251

VALIDATED_FILES['AMAT.csv'] = True  # Validated on 2025-02-10 22:50:47.784106

VALIDATED_FILES['MLM.csv'] = True  # Validated on 2025-02-10 22:51:07.529434

VALIDATED_FILES['EA.csv'] = True  # Validated on 2025-02-10 22:51:27.823935

VALIDATED_FILES['SPG.csv'] = True  # Validated on 2025-02-10 22:51:47.984692

VALIDATED_FILES['AMD.csv'] = True  # Validated on 2025-02-10 22:52:16.563301

VALIDATED_FILES['NDAQ.csv'] = True  # Validated on 2025-02-10 22:52:37.544499

VALIDATED_FILES['PNC.csv'] = True  # Validated on 2025-02-10 22:52:57.980587

VALIDATED_FILES['PINS.csv'] = True  # Validated on 2025-02-10 22:53:20.299070

VALIDATED_FILES['BIIB.csv'] = True  # Validated on 2025-02-10 22:53:40.367860

VALIDATED_FILES['EXC.csv'] = True  # Validated on 2025-02-10 22:54:30.669486

VALIDATED_FILES['HES.csv'] = True  # Validated on 2025-02-10 22:54:50.678892

VALIDATED_FILES['ALB.csv'] = True  # Validated on 2025-02-10 22:55:33.127827

VALIDATED_FILES['VLO.csv'] = True  # Validated on 2025-02-10 22:55:53.661878

VALIDATED_FILES['AON.csv'] = True  # Validated on 2025-02-10 22:56:13.449652

VALIDATED_FILES['ZTS.csv'] = True  # Validated on 2025-02-10 22:56:33.378741

VALIDATED_FILES['FDX.csv'] = True  # Validated on 2025-02-10 22:56:53.963735

VALIDATED_FILES['DG.csv'] = True  # Validated on 2025-02-10 22:57:14.729196

VALIDATED_FILES['CAG.csv'] = True  # Validated on 2025-02-10 22:57:35.056026

VALIDATED_FILES['INCY.csv'] = True  # Validated on 2025-02-10 22:57:55.084872

VALIDATED_FILES['SCHW.csv'] = True  # Validated on 2025-02-10 22:58:17.314097

VALIDATED_FILES['SO.csv'] = True  # Validated on 2025-02-10 22:58:58.349350

VALIDATED_FILES['CME.csv'] = True  # Validated on 2025-02-10 22:59:18.229148

VALIDATED_FILES['AMP.csv'] = True  # Validated on 2025-02-10 23:00:01.691692

VALIDATED_FILES['PNW.csv'] = True  # Validated on 2025-02-10 23:09:49.569900

VALIDATED_FILES['USB.csv'] = True  # Validated on 2025-02-10 23:09:57.612506

VALIDATED_FILES['LULU.csv'] = True  # Validated on 2025-02-10 23:09:58.243374

VALIDATED_FILES['CVS.csv'] = True  # Validated on 2025-02-10 23:10:04.065596

VALIDATED_FILES['MAA.csv'] = True  # Validated on 2025-02-10 23:10:11.143787

VALIDATED_FILES['ICE.csv'] = True  # Validated on 2025-02-10 23:10:13.689928

VALIDATED_FILES['FIS.csv'] = True  # Validated on 2025-02-10 23:10:19.892545

VALIDATED_FILES['LIN.csv'] = True  # Validated on 2025-02-10 23:10:22.345343

VALIDATED_FILES['WFC.csv'] = True  # Validated on 2025-02-10 23:10:25.036253

VALIDATED_FILES['GE.csv'] = True  # Validated on 2025-02-10 23:10:31.175105

VALIDATED_FILES['MMC.csv'] = True  # Validated on 2025-02-10 23:10:34.937930

VALIDATED_FILES['NET.csv'] = True  # Validated on 2025-02-10 23:10:40.385024

VALIDATED_FILES['FANG.csv'] = True  # Validated on 2025-02-10 23:10:44.724551

VALIDATED_FILES['ITW.csv'] = True  # Validated on 2025-02-10 23:10:46.363997

VALIDATED_FILES['GS.csv'] = True  # Validated on 2025-02-10 23:10:50.251235

VALIDATED_FILES['ALL.csv'] = True  # Validated on 2025-02-10 23:10:55.416600

VALIDATED_FILES['ABT.csv'] = True  # Validated on 2025-02-10 23:10:56.791513

VALIDATED_FILES['SQ.csv'] = True  # Validated on 2025-02-10 23:11:04.375350

VALIDATED_FILES['BEN.csv'] = True  # Validated on 2025-02-10 23:11:04.605977

VALIDATED_FILES['MAR.csv'] = True  # Validated on 2025-02-10 23:11:09.272886

VALIDATED_FILES['KMB.csv'] = True  # Validated on 2025-02-10 23:11:10.792509

VALIDATED_FILES['CLX.csv'] = True  # Validated on 2025-02-10 23:11:14.275598

VALIDATED_FILES['SNOW.csv'] = True  # Validated on 2025-02-10 23:11:21.609873

VALIDATED_FILES['SBAC.csv'] = True  # Validated on 2025-02-10 23:11:29.405150

VALIDATED_FILES['CMCSA.csv'] = True  # Validated on 2025-02-10 23:11:30.236291

VALIDATED_FILES['NIO.csv'] = True  # Validated on 2025-02-10 23:11:31.742721

VALIDATED_FILES['PLD.csv'] = True  # Validated on 2025-02-10 23:11:34.652318

VALIDATED_FILES['SHOP.csv'] = True  # Validated on 2025-02-10 23:11:35.325408

VALIDATED_FILES['SPGI.csv'] = True  # Validated on 2025-02-10 23:11:35.971502

VALIDATED_FILES['DGX.csv'] = True  # Validated on 2025-02-10 23:11:39.035732

VALIDATED_FILES['DRI.csv'] = True  # Validated on 2025-02-10 23:11:46.627943

VALIDATED_FILES['PCG.csv'] = True  # Validated on 2025-02-10 23:11:56.309962

VALIDATED_FILES['SIRI.csv'] = True  # Validated on 2025-02-10 23:11:58.451112

VALIDATED_FILES['CINF.csv'] = True  # Validated on 2025-02-10 23:12:00.804161

VALIDATED_FILES['COP.csv'] = True  # Validated on 2025-02-10 23:12:01.561496

VALIDATED_FILES['IBM.csv'] = True  # Validated on 2025-02-10 23:12:02.051331

VALIDATED_FILES['AVB.csv'] = True  # Validated on 2025-02-10 23:12:03.940672

VALIDATED_FILES['NEM.csv'] = True  # Validated on 2025-02-10 23:12:06.213371

VALIDATED_FILES['ILMN.csv'] = True  # Validated on 2025-02-10 23:12:12.726639

VALIDATED_FILES['VMC.csv'] = True  # Validated on 2025-02-10 23:12:21.632984

VALIDATED_FILES['MTB.csv'] = True  # Validated on 2025-02-10 23:12:27.219967

VALIDATED_FILES['STT.csv'] = True  # Validated on 2025-02-10 23:12:27.710932

VALIDATED_FILES['FITB.csv'] = True  # Validated on 2025-02-10 23:12:28.686388

VALIDATED_FILES['HMC.csv'] = True  # Validated on 2025-02-10 23:12:29.588982

VALIDATED_FILES['AFL.csv'] = True  # Validated on 2025-02-10 23:12:31.996257

VALIDATED_FILES['JD.csv'] = True  # Validated on 2025-02-10 23:12:34.298850

VALIDATED_FILES['SRE.csv'] = True  # Validated on 2025-02-10 23:12:47.332199

VALIDATED_FILES['META.csv'] = True  # Validated on 2025-02-10 23:12:47.420771

VALIDATED_FILES['PRU.csv'] = True  # Validated on 2025-02-10 23:12:52.927735

VALIDATED_FILES['NTR.csv'] = True  # Validated on 2025-02-10 23:12:53.778768

VALIDATED_FILES['BK.csv'] = True  # Validated on 2025-02-10 23:12:55.355144

VALIDATED_FILES['X.csv'] = True  # Validated on 2025-02-10 23:12:56.789523

VALIDATED_FILES['PFG.csv'] = True  # Validated on 2025-02-10 23:12:57.726493

VALIDATED_FILES['DUK.csv'] = True  # Validated on 2025-02-10 23:12:58.442058

VALIDATED_FILES['HLT.csv'] = True  # Validated on 2025-02-10 23:13:00.094153

VALIDATED_FILES['ERIC.csv'] = True  # Validated on 2025-02-10 23:13:14.721916

VALIDATED_FILES['BMY.csv'] = True  # Validated on 2025-02-10 23:13:15.365102

VALIDATED_FILES['WMB.csv'] = True  # Validated on 2025-02-10 23:13:18.537101

VALIDATED_FILES['ZM.csv'] = True  # Validated on 2025-02-10 23:13:22.250877

VALIDATED_FILES['REGN.csv'] = True  # Validated on 2025-02-10 23:13:23.047995

VALIDATED_FILES['ZION.csv'] = True  # Validated on 2025-02-10 23:13:23.412869

VALIDATED_FILES['QCOM.csv'] = True  # Validated on 2025-02-10 23:13:23.630222

VALIDATED_FILES['SLB.csv'] = True  # Validated on 2025-02-10 23:13:24.827796

VALIDATED_FILES['SNPS.csv'] = True  # Validated on 2025-02-10 23:13:40.014161

VALIDATED_FILES['NUE.csv'] = True  # Validated on 2025-02-10 23:13:41.039842

VALIDATED_FILES['EQR.csv'] = True  # Validated on 2025-02-10 23:13:43.772236

VALIDATED_FILES['CL.csv'] = True  # Validated on 2025-02-10 23:13:48.171084

VALIDATED_FILES['TFC.csv'] = True  # Validated on 2025-02-10 23:13:48.886506

VALIDATED_FILES['ROST.csv'] = True  # Validated on 2025-02-10 23:13:49.120245

VALIDATED_FILES['HBAN.csv'] = True  # Validated on 2025-02-10 23:13:49.672751

VALIDATED_FILES['TGT.csv'] = True  # Validated on 2025-02-10 23:13:52.214437

VALIDATED_FILES['AIG.csv'] = True  # Validated on 2025-02-10 23:14:08.798444

VALIDATED_FILES['NOK.csv'] = True  # Validated on 2025-02-10 23:14:10.383178

VALIDATED_FILES['MS.csv'] = True  # Validated on 2025-02-10 23:14:15.614510

VALIDATED_FILES['VZ.csv'] = True  # Validated on 2025-02-10 23:14:17.551303

VALIDATED_FILES['LMT.csv'] = True  # Validated on 2025-02-10 23:14:19.115226

VALIDATED_FILES['HON.csv'] = True  # Validated on 2025-02-10 23:14:33.963423

VALIDATED_FILES['CTSH.csv'] = True  # Validated on 2025-02-10 23:14:35.082363

VALIDATED_FILES['YUM.csv'] = True  # Validated on 2025-02-10 23:14:38.024147

VALIDATED_FILES['TEAM.csv'] = True  # Validated on 2025-02-10 23:14:40.162204

VALIDATED_FILES['CPB.csv'] = True  # Validated on 2025-02-10 23:14:40.385991

VALIDATED_FILES['AMGN.csv'] = True  # Validated on 2025-02-10 23:14:42.744016

VALIDATED_FILES['KEY.csv'] = True  # Validated on 2025-02-10 23:14:45.179656

VALIDATED_FILES['K.csv'] = True  # Validated on 2025-02-10 23:14:47.932719

VALIDATED_FILES['WY.csv'] = True  # Validated on 2025-02-10 23:14:52.347862

VALIDATED_FILES['TTWO.csv'] = True  # Validated on 2025-02-10 23:14:57.151673

VALIDATED_FILES['L.csv'] = True  # Validated on 2025-02-10 23:15:00.330575

VALIDATED_FILES['MU.csv'] = True  # Validated on 2025-02-10 23:15:04.391111

VALIDATED_FILES['CRM.csv'] = True  # Validated on 2025-02-10 23:15:04.711720

VALIDATED_FILES['UNM.csv'] = True  # Validated on 2025-02-10 23:15:06.754750

VALIDATED_FILES['APD.csv'] = True  # Validated on 2025-02-10 23:15:13.932340

VALIDATED_FILES['AEP.csv'] = True  # Validated on 2025-02-10 23:15:21.782447

VALIDATED_FILES['TRV.csv'] = True  # Validated on 2025-02-10 23:15:25.882221

VALIDATED_FILES['PGR.csv'] = True  # Validated on 2025-02-10 23:15:25.955878

VALIDATED_FILES['LRCX.csv'] = True  # Validated on 2025-02-10 23:15:48.127195

VALIDATED_FILES['DOV.csv'] = True  # Validated on 2025-02-10 23:16:04.647562

VALIDATED_FILES['UAA.csv'] = True  # Validated on 2025-02-10 23:16:10.092916

VALIDATED_FILES['FTI.csv'] = True  # Validated on 2025-02-10 23:16:25.590092

VALIDATED_FILES['Z.csv'] = True  # Validated on 2025-02-10 23:16:47.111797
