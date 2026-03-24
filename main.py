import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Fix randomness for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 1. DATA INGESTION
symbol = "USDTRY=X"
df = yf.download(symbol, start="2015-01-01", end="2026-03-24", interval="1d")

# 2. PREPROCESSING
df.columns = [col[0] for col in df.columns] # Flatten MultiIndex columns
data = df[['Close']].values

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. WINDOWING (Sliding Window)
X = []
y = []
window_size = 60 

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # Reshape for LSTM [samples, time steps, features]

# 4. DATA SPLITTING (80% Train, 20% Test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Preparation Complete! Train set: {X_train.shape}, Test set: {X_test.shape}")

# 5. LSTM MODEL ARCHITECTURE
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Model Training
print("Training model, please wait...")
model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

# 6. EVALUATION ON TEST DATA
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions) # Scale back to original values

y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. PERFORMANCE METRICS (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
print(f"Model Root Mean Squared Error (RMSE): {rmse:.2f} TL")

# Visualization
plt.figure(figsize=(14, 5))
plt.plot(y_test_unscaled, color='blue', label='Actual USD/TRY Price')
plt.plot(predictions, color='red', label='LSTM Prediction')
plt.title('USD/TRY Price Prediction Performance')
plt.xlabel('Time (Test Days)')
plt.ylabel('Price (TL)')
plt.legend()
plt.show()

# 8. FUTURE 30-DAY FORECASTING
last_60_days = scaled_data[-60:]
future_predictions = []
current_batch = last_60_days.reshape((1, 60, 1))

for i in range(30):
    current_pred = model.predict(current_batch)[0]
    future_predictions.append(current_pred)
    # Update batch: remove first day, append predicted day
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

future_predictions_unscaled = scaler.inverse_transform(future_predictions)

print("\n--- Next 30 Days Forecast ---")
for i, price in enumerate(future_predictions_unscaled):
    print(f"Day {i+1}: {price[0]:.2f} TL")
