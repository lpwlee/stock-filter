from futu import OpenQuoteContext, SubType
from datetime import datetime, timedelta
import pandas as pd  # Ensure pandas is imported

# Set up the API connection
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# Define the stock ticker (e.g., Hong Kong stock)
stock_code = 'HK.00027'  # Example: Galaxy Entertainment Group Limited

# Calculate the date range for the last two years
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # Two years ago

# Retrieve historical K-line data for the past two years
try:
    # Request historical K-line data
    ret, kline_data, _ = quote_ctx.request_history_kline(
        stock_code,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        ktype=SubType.K_DAY  # Daily candlesticks
    )
    
    if ret == 0:
        # Set pandas display options to show all rows
        pd.set_option('display.max_rows', None)  # Show all rows in DataFrame
        
        # Print the historical K-line data
        print(f"Historical K-line data for {stock_code}:\n", kline_data)

        # Save DataFrame to CSV
        kline_data.to_csv(f'historical_data_{stock_code}.csv', index=False)
        print(f"Data saved to historical_data_{stock_code}.csv")
    else:
        print(f"Error fetching historical data: {kline_data}")
finally:
    # Close the connection
    quote_ctx.close()