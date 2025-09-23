import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Load historical data from CSV
file_name = 'historical_data_HK.00027.csv'
data = pd.read_csv(file_name)

# Convert 'time_key' to datetime
data['time_key'] = pd.to_datetime(data['time_key'])

# Feature engineering: create additional features if necessary
data['price_change'] = data['close'].shift(-1) - data['close']  # Next day price change
data.dropna(inplace=True)  # Drop rows with NaN values

# Features and targets
features = data[['open', 'close', 'high', 'low', 'volume']]
targets = data[['open', 'close']].shift(-1).dropna()  # Next day's open and close
features = features[:-1]  # Align features with targets

# Train the model
model = LinearRegression()
model.fit(features, targets)

# Prepare input for the next day's prediction using the last row of historical data
latest_data = features.iloc[-1].values.reshape(1, -1)  # Get the last row as input

# Create a DataFrame to match the model's expected input format
latest_data_df = pd.DataFrame(latest_data, columns=features.columns)

# Predict the next day's prices
predicted_prices = model.predict(latest_data_df)

# Get today's date and calculate the next calendar day
today = datetime.now().date()
next_calendar_day = today + timedelta(days=1)

# Display the prediction for the next calendar day
predicted_open, predicted_close = predicted_prices[0]
print(f"Predicted prices for {next_calendar_day}:\n"
      f"Open price: {predicted_open:.2f}\n"
      f"Close price: {predicted_close:.2f}")