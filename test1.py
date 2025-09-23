import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Set your proxy settings
proxy_server = 'http://proxy1.doj.gov.hk:8080'
os.environ['HTTP_PROXY'] = proxy_server
os.environ['HTTPS_PROXY'] = proxy_server

# List of Hong Kong stock tickers
hk_stocks = ['0001.HK', '0005.HK', '0011.HK', '0700.HK', '0883.HK', '0027.HK']  # Example tickers

# Calculate the date range for the last year
end_date = datetime.now().date()  # Set end date to today
start_date = end_date - timedelta(days=365)  # One year back

# Set Pandas display options to show all rows
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Directory to save CSV files
output_dir = 'hk_stock_data'  # Change this to your desired output directory
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

for ticker in hk_stocks:
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)  # Get last 1 year of data
        print(f"\nData for {ticker}:\n", data)
        
        # Save data to CSV file
        csv_file_path = os.path.join(output_dir, f"{ticker}.csv")
        data.to_csv(csv_file_path)  # Save to CSV
        print(f"Data for {ticker} saved to {csv_file_path}")
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")

# Reset display options to default after use (optional)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')