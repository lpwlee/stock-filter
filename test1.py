import os
import yfinance as yf
from datetime import datetime, timedelta

# Set your proxy settings
proxy_server = 'http://proxy1.doj.gov.hk:8080'
os.environ['HTTP_PROXY'] = proxy_server
os.environ['HTTPS_PROXY'] = proxy_server

# List of Hong Kong stock tickers
hk_stocks = ['0001.HK', '0005.HK', '0011.HK', '0700.HK', '0883.HK']  # Example tickers

# Calculate the date range for the last year
end_date = datetime.now() - timedelta(days=1)  # Set end date to yesterday
start_date = end_date - timedelta(days=365)  # One year back

start_date = "2024-01-01"
end_date = "2025-01-01"

for ticker in hk_stocks:
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)  # Get last 1 year of data
        print(f"\nData for {ticker}:\n", data.head())
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")