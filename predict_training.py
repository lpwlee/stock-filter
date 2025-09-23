import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load historical data, skipping unnecessary header rows
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

# Calculate Technical Indicators
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26):
    data['EMA_Short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_Long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    return data['EMA_Short'] - data['EMA_Long']

# Add indicators to the DataFrame
data['RSI'] = calculate_rsi(data)
data['MACD'] = calculate_macd(data)
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

# Lagged features for the next day's prediction
data['Open_lag'] = data['Open'].shift(1)
data['Close_lag'] = data['Close'].shift(1)

# Drop NaN values to clean the dataset
data.dropna(inplace=True)

# Prepare features and target
features = data[['RSI', 'MACD', 'SMA_20', 'EMA_20', 'Open_lag', 'Close_lag']]
target = data[['Open', 'Close']]

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)  # Use 80% for training
X_train = features[:train_size]
y_train = target[:train_size]
X_test = features[train_size:]
y_test = target[train_size:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.2f}')

# Predicting the next day's Open and Close prices
latest_data = features.iloc[-1].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)
predicted_open_close = model.predict(latest_data_scaled)

print(f'Predicted Open: {predicted_open_close[0][0]:.2f}, Predicted Close: {predicted_open_close[0][1]:.2f}')

# Optional: Visualize the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size:], y_test['Close'], label='Actual Close Prices', color='blue')
plt.plot(data.index[train_size:], predictions[:, 1], label='Predicted Close Prices', color='orange', linewidth=2)
plt.title('Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()