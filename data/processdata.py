#!/usr/bin/env python
"""
This script processes stock data read from a CSV file and computes various technical features.
The input CSV is expected to have the following columns:
    Ticker, DateTime, Open_Price, High_Price, Low_Price, Close_Price, Stock_Volume, RMB_Volume, vwap, num_trades
Additional columns will be computed and the final output CSV will have the columns in this order:

Ticker, DateTime, Open_Price, High_Price, Low_Price, Close_Price, Stock_Volume, RMB_Volume, vwap, num_trades,
Weighted_Price, High_Minus_Low, Return_Lag1, Return_Lag2, Return_Lag3, Return_Lag4,
Return_Lag5, Return_Lag6, Correlation_MA5_MA30, Sum3, Sum5,
Sum5_Squared_Minus_Sum3_Squared_Sqrt, RSI_6, RSI_14, Rate_of_Change_9, Rate_of_Change_14,
Williams_R, ATR_5, ATR_10, CCI, DEMA, Year, Month, Day, Hour, Minute, Processed_Close_Price

Feature calculations include:
  - Weighted_Price: Calculated as (High_Price + Low_Price + Close_Price)/3
    [*Note*: the paper's note suggests a volume weighted formula; adjust accordingly if needed].
  - High_Minus_Low: High_Price minus Low_Price.
  - Return_Lag1 to Return_Lag6: Percentage returns compared to previous periods.
  - Correlation_MA5_MA30: The rolling Pearson correlation (window=30) between the 5-period
    and 30-period moving average of the Close_Price.
  - Sum3 and Sum5: Rolling sums of Close_Price over 3- and 5-period windows.
  - Sum5_Squared_Minus_Sum3_Squared_Sqrt: Square root of (Sum5² - Sum3²).
  - RSI_6 and RSI_14: Relative Strength Index computed on the Close_Price series.
  - Rate_of_Change_9 and Rate_of_Change_14: Percent change over 9- and 14-periods.
  - Williams_R: Williams %R over a 14-period window.
  - ATR_5 and ATR_10: Average True Range over 5 and 10 periods.
  - CCI: Commodity Channel Index (using a 20-period window).
  - DEMA: Double Exponential Moving Average on Close_Price (using period 10).
  - Temporal features: Year, Month, Day, Hour, Minute (extracted from DateTime).
  - Processed_Close_Price: Calculated as the standardized value of log-differenced Close_Price.

Usage:
    python process_data.py --input path/to/your/TICKER01.csv --output processed_TICKER01.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging

# Initialize logging
logging.getLogger(__name__)

def compute_features(df):
    print("Starting feature computation.")
    # Ensure the DataFrame has a 'DateTime' column.
    # If not, check for a case-insensitive match (e.g., "datetime") and rename it.
    if "DateTime" not in df.columns:
        for col in df.columns:
            if col.lower() == "datetime":
                df.rename(columns={col: "DateTime"}, inplace=True)
                print(f"Renamed column '{col}' to 'DateTime'.")
                break
        else:
            error_msg = "Input CSV does not have a 'DateTime' column."
            logging.error(error_msg)
            raise KeyError(error_msg)

    # Now safely convert the 'DateTime' column to datetime
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    logging.debug("Converted 'DateTime' column to datetime objects.")

    # 1. Weighted_Price: using the simple average of (High_Price, Low_Price, Close_Price)
    df["Weighted_Price"] = (df["High_Price"] + df["Low_Price"] + df["Close_Price"]) / 3
    logging.debug("Computed 'Weighted_Price'.")

    # 2. High_Minus_Low
    df["High_Minus_Low"] = df["High_Price"] - df["Low_Price"]
    logging.debug("Computed 'High_Minus_Low'.")

    # 3. Return_Lag1 to Return_Lag6
    for lag in range(1, 7):
        df[f"Return_Lag{lag}"] = df["Close_Price"].pct_change(periods=lag)
    logging.debug("Computed 'Return_Lag1' to 'Return_Lag6'.")

    # 4. Moving Averages and rolling correlation
    df["MA5"] = df["Close_Price"].rolling(5).mean()
    df["MA30"] = df["Close_Price"].rolling(30).mean()
    df["Correlation_MA5_MA30"] = df["MA5"].rolling(30).corr(df["MA30"])
    logging.debug("Computed Moving Averages (MA5, MA30) and 'Correlation_MA5_MA30'.")

    # 5. Sum3 and Sum5 for Close_Price
    df["Sum3"] = df["Close_Price"].rolling(3).sum()
    df["Sum5"] = df["Close_Price"].rolling(5).sum()
    logging.debug("Computed 'Sum3' and 'Sum5'.")

    # 6. Sum5_Squared_Minus_Sum3_Squared_Sqrt
    df["Sum5_Squared_Minus_Sum3_Squared_Sqrt"] = np.sqrt(np.maximum((df["Sum5"]**2 - df["Sum3"]**2), 0))
    logging.debug("Computed 'Sum5_Squared_Minus_Sum3_Squared_Sqrt'.")

    # 7. RSI calculations
    def compute_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Use a simple moving average for gains and losses
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df["RSI_6"] = compute_rsi(df["Close_Price"], 6)
    df["RSI_14"] = compute_rsi(df["Close_Price"], 14)
    logging.debug("Computed 'RSI_6' and 'RSI_14'.")

    # 8. Rate of Change calculations
    df["Rate_of_Change_9"] = (df["Close_Price"] - df["Close_Price"].shift(9)) / df["Close_Price"].shift(9) * 100
    df["Rate_of_Change_14"] = (df["Close_Price"] - df["Close_Price"].shift(14)) / df["Close_Price"].shift(14) * 100
    logging.debug("Computed 'Rate_of_Change_9' and 'Rate_of_Change_14'.")

    # 9. Williams_R: using a 14 period window
    highest_high = df["High_Price"].rolling(14).max()
    lowest_low = df["Low_Price"].rolling(14).min()
    df["Williams_R"] = ((highest_high - df["Close_Price"]) / (highest_high - lowest_low)) * -100
    logging.debug("Computed 'Williams_R'.")

    # 10. ATR (Average True Range)
    high_low = df["High_Price"] - df["Low_Price"]
    high_close = (df["High_Price"] - df["Close_Price"].shift(1)).abs()
    low_close = (df["Low_Price"] - df["Close_Price"].shift(1)).abs()
    df["True_Range"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_5"] = df["True_Range"].rolling(5).mean()
    df["ATR_10"] = df["True_Range"].rolling(10).mean()
    df.drop(columns=["True_Range"], inplace=True)
    logging.debug("Computed 'ATR_5' and 'ATR_10'.")

    # 11. CCI (Commodity Channel Index) using a 20-period window.
    tp = (df["High_Price"] + df["Low_Price"] + df["Close_Price"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df["CCI"] = (tp - sma_tp) / (0.015 * mad)
    logging.debug("Computed 'CCI'.")

    # 12. DEMA (Double Exponential Moving Average) using period=10
    ema = df["Close_Price"].ewm(span=10, adjust=False).mean()
    ema_of_ema = ema.ewm(span=10, adjust=False).mean()
    df["DEMA"] = 2 * ema - ema_of_ema
    logging.debug("Computed 'DEMA'.")

    # 13. Temporal features: Year, Month, Day, Hour, Minute from DateTime
    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["Hour"] = df["DateTime"].dt.hour
    df["Minute"] = df["DateTime"].dt.minute
    logging.debug("Computed temporal features (Year, Month, Day, Hour, Minute).")

    # 14. Processed_Close_Price:
    #     a. Calculate the log of Close_Price and then the first-order difference.
    #     b. Standardize the resulting series.
    df["Close_Log"] = np.log(df["Close_Price"])
    df["Close_Log_Diff"] = df["Close_Log"].diff()
    mean_diff = df["Close_Log_Diff"].mean()
    std_diff = df["Close_Log_Diff"].std()
    df["Processed_Close_Price"] = (df["Close_Log_Diff"] - mean_diff) / std_diff
    logging.debug("Computed 'Processed_Close_Price'.")

    # Drop intermediate columns that are not part of the final output
    df.drop(columns=["MA5", "MA30", "Close_Log", "Close_Log_Diff"], inplace=True)
    logging.debug("Dropped intermediate columns.")

    print("Feature computation completed.")
    return df

def validate_file(df, filename):
    print(f"Validating file structure for {filename}.")
    """Validate the CSV file structure and return True if valid."""
    required_columns = {
        "DateTime",
        "Open_Price",
        "High_Price",
        "Low_Price",
        "Close_Price",
        "Stock_Volume"
    }

    # Check if all required columns exist (after renaming)
    current_columns = set(df.columns)
    missing_columns = required_columns - current_columns

    if missing_columns:
        error_msg = f"Error in {filename}: Missing required columns: {missing_columns}"
        logging.error(error_msg)
        print(error_msg)
        return False

    print(f"File {filename} structure validated successfully.")
    return True

def main():
    print("Starting main function in processdata.py.")
    # Import config at runtime to access the VALIDATED_FILES
    import sys
    sys.path.append('data')
    import config

    # Initialize VALIDATED_FILES if it doesn't exist
    if not hasattr(config, 'VALIDATED_FILES'):
        config.VALIDATED_FILES = {}

    input_folder = "data/raw"
    output_folder = "data/processed"
    merged_output_folder = "data/ready_for_train"  # New folder for merged data

    # Create directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(merged_output_folder, exist_ok=True)
    print(f"Output directories created: {output_folder}, {merged_output_folder}")

    processed_files = []  # Track processed output file paths

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.csv'):
            logging.debug(f"Skipping non-CSV file: {filename}")
            continue

        input_filepath = os.path.join(input_folder, filename)
        print(f"Processing file: {input_filepath}")
        print(f"Processing {input_filepath} ...")

        # Read CSV
        df = pd.read_csv(input_filepath)
        logging.debug(f"CSV file read into DataFrame. Shape: {df.shape}")

        # Rename columns from the input CSV to the standard column names
        rename_dict = {
            "datetime": "DateTime",
            "open": "Open_Price",
            "high": "High_Price",
            "low": "Low_Price",
            "close": "Close_Price",
            "volume": "Stock_Volume"
        }
        df.rename(columns=rename_dict, inplace=True)
        logging.debug(f"Columns renamed using dictionary: {rename_dict}")

        # Check if file needs validation
        if filename not in config.VALIDATED_FILES:
            print(f"Validating {filename} for the first time...")
            print(f"Validating {filename} for the first time...")
            if not validate_file(df, filename):
                logging.warning(f"Validation failed for {filename}. Skipping file.")
                print(f"Skipping {filename} due to validation failure")
                continue

            # Mark file as validated and save to config
            config.VALIDATED_FILES[filename] = True
            print(f"File {filename} validated successfully.")

            # Save the updated VALIDATED_FILES to config.py
            with open('data/config.py', 'a') as f:
                f.write(f"\nVALIDATED_FILES['{filename}'] = True  # Validated on {pd.Timestamp.now()}\n")
            print(f"Updated VALIDATED_FILES in config.py for {filename}.")
        else:
            print(f"Skipping validation for previously validated file: {filename}")
            print(f"Skipping validation for previously validated file: {filename}")

        # Continue with the rest of the processing
        # If the Ticker column is missing, derive it from the filename.
        if "Ticker" not in df.columns:
            ticker_from_filename = os.path.splitext(filename)[0]
            df["Ticker"] = ticker_from_filename
            logging.debug(f"Ticker column missing, derived from filename: {ticker_from_filename}")

        # Compute RMB_Volume if missing and if the 'vwap' column exists.
        if "RMB_Volume" not in df.columns:
            if "vwap" in df.columns:
                df["RMB_Volume"] = df["vwap"] * df["Stock_Volume"]
                logging.debug("Computed 'RMB_Volume' using 'vwap' and 'Stock_Volume'.")
            else:
                df["RMB_Volume"] = np.nan
                logging.warning("'RMB_Volume' column missing and 'vwap' column not found. Setting 'RMB_Volume' to NaN.")

        # Process the data by computing all features.
        print("Computing features for the data.")
        df_processed = compute_features(df)
        print("Features computed.")

        # Reorder the columns as specified in the requirements.
        final_columns = [
            "Ticker",
            "DateTime",
            "Open_Price",
            "High_Price",
            "Low_Price",
            "Close_Price",
            "Stock_Volume",
            "RMB_Volume",
            "vwap",
            "num_trades",
            "Weighted_Price",
            "High_Minus_Low",
            "Return_Lag1",
            "Return_Lag2",
            "Return_Lag3",
            "Return_Lag4",
            "Return_Lag5",
            "Return_Lag6",
            "Correlation_MA5_MA30",
            "Sum3",
            "Sum5",
            "Sum5_Squared_Minus_Sum3_Squared_Sqrt",
            "RSI_6",
            "RSI_14",
            "Rate_of_Change_9",
            "Rate_of_Change_14",
            "Williams_R",
            "ATR_5",
            "ATR_10",
            "CCI",
            "DEMA",
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "Processed_Close_Price"
        ]
        df_processed = df_processed.loc[:, final_columns]
        logging.debug("Columns reordered to final column list.")

        # Remove rows with any missing values - use .dropna() instead of inplace
        df_processed = df_processed.dropna()
        logging.debug("Rows with missing values dropped.")

        # Save processed file and track path
        output_filepath = os.path.join(output_folder, filename)
        df_processed.to_csv(output_filepath, index=False)
        processed_files.append(output_filepath)  # Add to processed files list
        print(f"Processed data saved to: {output_filepath}")
        print(f"Processed data saved to {output_filepath}")

    # Merge all processed files into one CSV
    if processed_files:
        print("Merging all processed files.")
        merged_df = pd.concat(
            (pd.read_csv(f) for f in processed_files),
            ignore_index=True
        )
        # Save to ready_for_train folder instead of processed
        merged_path = os.path.join(merged_output_folder, "merged_stock_data.csv")
        merged_df.to_csv(merged_path, index=False)
        print(f"Successfully merged all files into: {merged_path}")
        print(f"\nSuccessfully merged all files into {merged_path}")
    else:
        logging.warning("No files processed - nothing to merge.")
        print("\nNo files processed - nothing to merge")

    print("Main function in processdata.py completed.")

if __name__ == "__main__":
    main()
