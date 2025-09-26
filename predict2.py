import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# Load historical data from CSV
file_name = 'historical_data_HK.00027.csv'
data = pd.read_csv(file_name)

# Convert 'time_key' to datetime
data['time_key'] = pd.to_datetime(data['time_key'])

# Create lagged features and moving averages
for lag in range(1, 6):  # Create lagged features for the last 5 days
    data[f'close_lag_{lag}'] = data['close'].shift(lag)

# Create moving averages
data['moving_average_5'] = data['close'].rolling(window=5).mean()
data['moving_average_10'] = data['close'].rolling(window=10).mean()

# Drop NaN values created by lagging and moving averages
data.dropna(inplace=True)

# Features and target variable
features = data[['open', 'high', 'low', 'close', 'volume', 
                 'close_lag_1', 'close_lag_2', 'close_lag_3', 
                 'close_lag_4', 'close_lag_5', 
                 'moving_average_5', 'moving_average_10']].values
targets = data['close'].values

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Prepare data for LSTM using the entire dataset
X = []
y = []

look_back = 20  # Number of previous days to consider

for i in range(look_back, len(scaled_features)):
    X.append(scaled_features[i-look_back:i])
    y.append(scaled_features[i, 3])  # Predicting the 'close' price

X, y = np.array(X), np.array(y)

# Use all data for training
X_train = X
y_train = y

# Build, compile, and train the model as before

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for the close price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict the next day's prices using the last available data
latest_data = scaled_features[-look_back:].reshape(1, look_back, -1)
predicted_price = model.predict(latest_data)

# Inverse transform the predicted price to original scale
predicted_close = scaler.inverse_transform(np.concatenate((np.zeros((1, 11)), predicted_price), axis=1))[:, 3][0]

# Get today's date and calculate the next calendar day
today = datetime.now().date()
next_calendar_day = today + timedelta(days=1)

# Display the prediction for the next calendar day
print(f"Predicted prices for {next_calendar_day}:\n"
      f"Close price: {predicted_close:.2f}")