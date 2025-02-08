import os
import csv
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
from config import TICKERS
import logging

# Initialize logging
logging.getLogger(__name__)

# Configuration
def get_date_range():
    print("Getting date range for data fetching.")
    today = datetime.utcnow().date()
    end_date = today - timedelta(days=5)  # 5 days ago
    start_date = end_date - timedelta(days=365*2)  # 2 years before end_date
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    print(f"Date range set: Start Date = {start_date_str}, End Date = {end_date_str}")
    return start_date_str, end_date_str

# Get the date range when module is loaded
START_DATE, END_DATE = get_date_range()

# Add rate limit tracking
class RateLimiter:
    def __init__(self, calls_per_min=5):
        print(f"Initializing RateLimiter with {calls_per_min} calls per minute.")
        self.calls_per_min = calls_per_min
        self.calls = []
        self.last_call_time = None
    
    def wait_if_needed(self):
        logging.debug("Checking if rate limit wait is needed.")
        now = datetime.now()
        
        # Always wait 13 seconds from last call
        if self.last_call_time:
            elapsed = (now - self.last_call_time).total_seconds()
            if elapsed < 13:
                sleep_time = 13 - elapsed
                logging.debug(f"Waiting {sleep_time:.2f} seconds due to 13-second interval rule.")
                time.sleep(sleep_time)
        
        # Update last call time
        self.last_call_time = datetime.now()
        
        # Also maintain the rolling window for safety
        self.calls = [call for call in self.calls if (now - call).total_seconds() < 60]
        if len(self.calls) >= self.calls_per_min:
            sleep_time = 60 - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                logging.warning(f"Rate limit exceeded. Waiting {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            self.calls = self.calls[1:]
        
        self.calls.append(now)
        logging.debug("RateLimiter wait completed.")

def fetch_data(ticker, api_key, rate_limiter, start_date=None, end_date=None):
    """
    Retrieve 1-minute aggregated data for a given ticker from Polygon.io.
    Data range is from 2 years ago until yesterday (no current day data).
    """
    print(f"Fetching data for ticker: {ticker}, from {start_date} to {end_date}.")
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    
    # Ensure we never try to get today's data
    today = datetime.utcnow().date()
    end_datetime = min(
        datetime.strptime(end_date, "%Y-%m-%d").date(),
        today - timedelta(days=1)
    )
    end_date = end_datetime.strftime("%Y-%m-%d")
    
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # Add retry counter
    max_retries = 3  # Maximum number of retry attempts per chunk
    
    # Start with smaller chunks for recent data, then use larger chunks for older data
    while current_date <= end_datetime:
        retry_attempts = 0  # Reset retry counter for each new chunk
        chunk_size = None  # Initialize chunk_size
        
        while True:  # Inner loop for retries
            # Use 7-day chunks for the most recent month, 30-day chunks for older data
            days_from_end = (end_datetime - current_date).days
            chunk_size = 7 if days_from_end <= 30 else 30 if chunk_size is None else chunk_size
            chunk_end = min(current_date + timedelta(days=chunk_size), end_datetime)
            
            log_msg = f"Processing {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
            print(log_msg)
            print(log_msg)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{current_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
            params = {
                "limit": 50000,
                "sort": "asc"
            }
            
            total_records = 0
            while url:
                rate_limiter.wait_if_needed()
                
                response = requests.get(url, headers=headers, params=params if "?" not in url else None)
                
                if response.status_code != 200:
                    error_msg = f"Error {response.status_code}: {response.text[:100]}..."
                    logging.error(error_msg)
                    print(f"✗ {error_msg}")
                    if response.status_code == 429:  # Rate limit error
                        logging.warning("Rate limit hit, waiting 65 seconds...")
                        print("Rate limit hit, waiting 65 seconds...")
                        time.sleep(65)
                        continue  # Retry the same request
                    else:
                        time.sleep(30)
                        break
                    
                data = response.json()
                if data.get("status") != "OK":
                    api_error_msg = f"API Error: {data.get('error', 'Unknown error')}"
                    logging.error(api_error_msg)
                    print(f"✗ {api_error_msg}")
                    break
                    
                if "results" in data:
                    records = data["results"]
                    total_records += len(records)
                    save_to_csv(ticker, records)
                
                url = data.get("next_url")
                if url and "?" in url:
                    url = url.split("&apiKey=")[0]
            
            log_msg = f"Retrieved {total_records} total records for current chunk"
            print(log_msg)
            print(log_msg)
            if total_records == 0 and chunk_size > 1 and retry_attempts < max_retries:
                log_msg = f"No data received. Reducing chunk size (attempt {retry_attempts+1}/{max_retries})..."
                logging.warning(log_msg)
                print(log_msg)
                chunk_size = max(1, chunk_size // 2)
                retry_attempts += 1
                continue
                
            # If still no data after retries, break and move on
            if total_records == 0 and retry_attempts >= max_retries:
                log_msg = f"No data after {max_retries} attempts. Moving to next date chunk."
                logging.warning(log_msg)
                print(log_msg)
                current_date = chunk_end + timedelta(days=1)
                break
                
            # If we have data or exhausted retries, move forward
            current_date = chunk_end + timedelta(days=1)
            break  # Exit retry loop
    print(f"Data fetching completed for ticker: {ticker}.")

def save_to_csv(ticker, data):
    print(f"Saving {len(data)} records to CSV for ticker: {ticker}.")
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create path relative to script location
    output_dir = os.path.join(script_dir, "raw")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        error_msg = f"Error creating directory: {e}"
        logging.error(error_msg)
        print(f"Error creating directory: {e}")
        return
        
    output_file = os.path.join(output_dir, f"{ticker}.csv")
    file_exists = os.path.exists(output_file)

    try:
        with open(output_file, "a", newline="") as csvfile:
            fieldnames = ["datetime", "open", "high", "low", "close", "volume", "vwap", "num_trades"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for entry in data:
                dt = datetime.utcfromtimestamp(entry["t"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow({
                    "datetime": dt,
                    "open": entry.get("o"),
                    "high": entry.get("h"),
                    "low": entry.get("l"),
                    "close": entry.get("c"),
                    "volume": entry.get("v"),
                    "vwap": entry.get("vw"),
                    "num_trades": entry.get("n"),
                })
        log_msg = f"Saved {len(data)} records to {ticker}.csv"
        print(log_msg)
        print(f"✓ {log_msg}")
    except Exception as e:
        error_msg = f"Error writing to file: {e}"
        logging.error(error_msg)
        print(f"✗ {error_msg}")

def get_last_datetime_from_csv(ticker):
    print(f"Getting last datetime from CSV for ticker: {ticker}.")
    """
    Reads the existing CSV file for the given ticker (if it exists) and returns
    the maximum datetime (as a datetime object) found in the "datetime" column.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "raw", f"{ticker}.csv")
    
    if not os.path.exists(file_path):
        print(f"No existing CSV file found for ticker: {ticker}.")
        return None
        
    last_dt = None
    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
            if last_dt is None or dt > last_dt:
                last_dt = dt
    print(f"Last datetime found in CSV for ticker {ticker}: {last_dt}")
    return last_dt

def validate_ticker(ticker, api_key, rate_limiter):
    """
    Validate if a ticker exists and is active using Polygon.io's Ticker Details endpoint
    Now includes rate limiting
    """
    print(f"Validating ticker: {ticker}.")
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    
    try:
        rate_limiter.wait_if_needed()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "OK" and data.get("results"):
                is_active = data["results"].get("active", False)
                print(f"Ticker {ticker} validation successful. Active status: {is_active}")
                return is_active
        elif response.status_code == 404:
            log_msg = f"{ticker} not found"
            logging.warning(log_msg)
            print(f"✗ {log_msg}")
            return False
        else:
            error_msg = f"Error validating {ticker}: {response.status_code}, {response.text}"
            logging.error(error_msg)
            print(f"✗ {error_msg}")
            return False
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logging.error(error_msg)
        print(f"✗ {error_msg}")
        return False
    print(f"Ticker {ticker} validation failed or ticker not active.")
    return False

def main():
    print("Starting data fetcher main function.")
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    
    if not api_key:
        error_msg = "Missing API key. Create a .env file with POLYGON_API_KEY=your_key"
        logging.error(error_msg)
        print(error_msg)
        exit(1)

    log_msg = f"\n🚀 Starting data fetch for {len(TICKERS)} tickers"
    print(log_msg)
    print(log_msg)
    log_msg = f"⏳ Historical range: {START_DATE} to {END_DATE}\n"
    print(log_msg)
    print(log_msg)

    # Create a single rate limiter instance to be shared across all requests
    rate_limiter = RateLimiter(calls_per_min=5)  # Free tier limit

    for i, ticker in enumerate(TICKERS):
        log_msg = f"\n🔍 [{i+1}/{len(TICKERS)}] Checking {ticker}"
        print(log_msg)
        print(log_msg)
        
        if not validate_ticker(ticker, api_key, rate_limiter):
            log_msg = f"⏩ Skipping {ticker}"
            print(log_msg)
            print(log_msg)
            continue
            
        last_dt = get_last_datetime_from_csv(ticker)
        
        if last_dt:
            # Check if last data point is more than 168 hours (1 week) old
            time_since_last = datetime.now() - last_dt
            hours_old = time_since_last.total_seconds() / 3600
            if time_since_last.total_seconds() < 168 * 3600:  # 168 hours in seconds
                log_msg = f"Data is recent (last update: {last_dt.strftime('%Y-%m-%d %H:%M:%S')}, {hours_old:.1f} hours old). Skipping {ticker}"
                print(log_msg)
                print(log_msg)
                continue
                
            last_dt = last_dt + timedelta(minutes=1)
            fetch_start = last_dt.strftime("%Y-%m-%d")
            log_msg = f"Data is old (last update: {last_dt.strftime('%Y-%m-%d %H:%M:%S')}, {hours_old:.1f} hours old)"
            print(log_msg)
            print(log_msg)
            log_msg = f"Fetching new data from {fetch_start} to {END_DATE}"
            print(log_msg)
            print(log_msg)
        else:
            fetch_start = START_DATE
            log_msg = f"No existing data for {ticker}"
            print(log_msg)
            print(log_msg)
            log_msg = f"Fetching full history from {fetch_start} to {END_DATE}"
            print(log_msg)
            print(log_msg)

        fetch_data(ticker, api_key, rate_limiter, start_date=fetch_start, end_date=END_DATE)
        
        completion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"Completed processing {ticker} at {completion_time}"
        print(log_msg)
        print(log_msg)

    completion_all_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"\nCompleted all processing at {completion_all_time}"
    print(log_msg)
    print(log_msg)
    print("Data fetcher main function completed.")

if __name__ == "__main__":
    main()
