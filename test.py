# Set up proxy
proxies = {
    "http": "http://proxy1.doj.gov.hk:8080",
    #"https": "https://proxy1.doj.gov.hk:8080"
}
import yfinance as yf
import pandas as pd
import requests

# Set up proxy
proxies = {
    "http": "http://proxy1.doj.gov.hk:8080",
    #"https": "https://proxy1.doj.gov.hk:8080"
}

# Function to load stock data
def load_hk_stock_data(tickers):
    stock_data = {}
    for ticker in tickers:
        try:
            # Create a session
            session = requests.Session()
            session.proxies.update(proxies)

            # URL for the stock data
            url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=0&period2=9999999999&interval=1d&events=history'

            # Attempt to fetch data using proxy for HTTP, but bypass for HTTPS
            response = session.get(url, proxies={"http": proxies["http"], "https": None}, timeout=30)  # Bypass the proxy for HTTPS
            response.raise_for_status()  # Raise an error for bad responses

            # Read the CSV response into a DataFrame
            data = pd.read_csv(pd.compat.StringIO(response.text))
            stock_data[ticker] = data
            print(f"Loaded data for {ticker}")
        except requests.exceptions.ProxyError:
            print(f"Proxy error for {ticker}. Trying without proxy.")
            # Try again without a proxy
            try:
                response = requests.get(url)  # Direct request without proxy
                response.raise_for_status()  # Raise an error for bad responses

                # Read the CSV response into a DataFrame
                data = pd.read_csv(pd.compat.StringIO(response.text))
                stock_data[ticker] = data
                print(f"Loaded data for {ticker} without proxy")
            except Exception as e:
                print(f"Error loading data for {ticker}: {e}")
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
    return stock_data

# List of Hong Kong stock tickers
hk_stocks = ['0001.HK', '0005.HK', '0011.HK', '0700.HK', '0883.HK']  # Example tickers

# Load the stock data
hk_stock_data = load_hk_stock_data(hk_stocks)

# Example: Display the data for the first stock
for ticker, data in hk_stock_data.items():
    print(f"\nData for {ticker}:\n", data.head())