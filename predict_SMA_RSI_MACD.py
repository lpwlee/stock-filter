import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load historical data
file_path = 'hk_stock_data/0027.HK.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path, skiprows=2)

# Rename columns correctly
data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Volume']

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate the Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate indicators
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['RSI'] = calculate_rsi(data)

data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Additional Features
data['Volume_Change'] = data['Volume'].pct_change()

# Create lagged features
for lag in range(1, 6):
    data[f'Close_lag_{lag}'] = data['Close'].shift(lag)

# Drop NaN values
data.dropna(inplace=True)

# Check for infinite values
if np.isinf(data).values.any():
    print("Infinite values found; replacing with NaN.")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any rows that may have been converted to NaN
data.dropna(inplace=True)

# Check for any remaining NaN or infinite values
print("Checking for NaN or infinite values after cleaning:")
print(data.isnull().sum())
print(data.isin([np.inf, -np.inf]).sum())

# Prepare features and target
features = data[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'Volume_Change'] + [f'Close_lag_{lag}' for lag in range(1, 6)]]
target = data['Close']

# Check for extreme values in features
print("Checking for extreme values in features:")
print(features.describe())

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
X_train, X_test = features_scaled[:train_size], features_scaled[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.2f}')

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size:], y_test, label='Actual Prices', color='blue')
plt.plot(data.index[train_size:], predictions, label='Predicted Prices', color='orange', linewidth=2)
plt.title('Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()