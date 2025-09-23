import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load historical data
file_path = 'hk_stock_data/0027.HK.csv'  # Update with your file path
data = pd.read_csv(file_path, skiprows=2)

# Rename columns based on the actual structure
data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open']

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.dropna(subset=['Date'], inplace=True)
data.set_index('Date', inplace=True)

# Convert relevant columns to numeric
data[['Price', 'Close', 'High', 'Low', 'Open']] = data[['Price', 'Close', 'High', 'Low', 'Open']].apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values
data.dropna(inplace=True)

# Identify the most recent high and low
recent_high = data['High'].max()
recent_low = data['Low'].min()

# Calculate Fibonacci levels
diff = recent_high - recent_low
level_0 = recent_high
level_1 = recent_high - 0.236 * diff
level_2 = recent_high - 0.382 * diff
level_3 = recent_high - 0.618 * diff
level_4 = recent_low

# Print Fibonacci levels
print(f"Recent High: {recent_high}")
print(f"Recent Low: {recent_low}")
print(f"Fibonacci Levels:")
print(f"0%: {level_0}")
print(f"23.6%: {level_1}")
print(f"38.2%: {level_2}")
print(f"61.8%: {level_3}")
print(f"100%: {level_4}")

# Plotting the data with Fibonacci levels
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.axhline(level_0, color='red', linestyle='--', label='Fibonacci 0%')
plt.axhline(level_1, color='orange', linestyle='--', label='Fibonacci 23.6%')
plt.axhline(level_2, color='yellow', linestyle='--', label='Fibonacci 38.2%')
plt.axhline(level_3, color='green', linestyle='--', label='Fibonacci 61.8%')
plt.axhline(level_4, color='purple', linestyle='--', label='Fibonacci 100%')
plt.title('Fibonacci Retracement Levels')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Conclusion
current_price = data['Close'].iloc[-1]
print(f"\nCurrent Price: {current_price:.2f}")

if current_price > level_3:
    print("The stock is currently above the 61.8% Fibonacci level, indicating potential bullish momentum.")
elif level_2 < current_price <= level_3:
    print("The stock is between the 38.2% and 61.8% Fibonacci levels, suggesting consolidation with potential for both upward and downward movement.")
elif current_price <= level_2:
    print("The stock is below the 38.2% Fibonacci level, indicating potential bearish momentum.")

print("Overall, the trend for the next 1-2 months will depend on how the stock reacts to these Fibonacci levels.")