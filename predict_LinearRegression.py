import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Load the historical data, skipping the first two rows
file_path = 'hk_stock_data/0027.HK.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path, skiprows=2)

# Rename columns to match the data structure
data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open']  

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Ensure numeric columns are of the correct type
data[['Price', 'Close', 'High', 'Low', 'Open']] = data[['Price', 'Close', 'High', 'Low', 'Open']].apply(pd.to_numeric)

# Create lag features for Open and Close prices
data['Open_Lag1'] = data['Open'].shift(1)
data['Close_Lag1'] = data['Close'].shift(1)

# Drop rows with NaN values
data.dropna(inplace=True)

# Features and target variables
X = data[['Open_Lag1', 'Close_Lag1']]
y_open = data['Open']
y_close = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_open_train, y_open_test = train_test_split(X, y_open, test_size=0.2, random_state=42)
X_train, X_test, y_close_train, y_close_test = train_test_split(X, y_close, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model for Opening Price
model_open = LinearRegression()
model_open.fit(X_train_scaled, y_open_train)

# Train the model for Closing Price
model_close = LinearRegression()
model_close.fit(X_train_scaled, y_close_train)

# Prepare the latest data for prediction
latest_data = data.iloc[-1]
latest_features = np.array([[latest_data['Open_Lag1'], latest_data['Close_Lag1']]])

# Scale the latest features
latest_features_scaled = scaler.transform(latest_features)

# Predict the next day's Opening and Closing prices
next_open = model_open.predict(latest_features_scaled)[0]
next_close = model_close.predict(latest_features_scaled)[0]

# Calculate the next trading day
next_trading_day = data.index[-1] + timedelta(days=1)

# Display the predictions
print(f"Predicted Opening Price for {next_trading_day.date()}: {next_open:.2f}")
print(f"Predicted Closing Price for {next_trading_day.date()}: {next_close:.2f}")