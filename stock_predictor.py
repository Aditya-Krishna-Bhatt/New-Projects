import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# 1. DATA ACQUISITION: Download historical data for Nifty 50 or Apple (AAPL)
print("📥 Fetching historical market data...")
ticker = "AAPL"
data = yf.download(ticker, start="2016-01-01", end="2026-01-01")

# Clean multi-level column names if returned by yfinance
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# 2. FEATURE ENGINEERING: Mathematical indicators showcasing your Math & Computing skills
print("⚙️ Engineering mathematical indicators...")
data['Target'] = data['Close'].shift(-1) # What we want to predict: Tomorrow's close price

# Technical Indicators
data['SMA_10'] = data['Close'].rolling(window=10).mean() # Simple Moving Average
data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean() # Exponential Moving Average
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=10).std()

# Drop rows with NaN values created by rolling calculations and shifts
data.dropna(inplace=True)

# Define features (X) and Target (y)
feature_cols = ['Close', 'SMA_10', 'EMA_5', 'Daily_Return', 'Volatility']
X = data[feature_cols]
y = data['Target']

# 3. TRAINING & TESTING SPLIT: Sequential split for Time-Series (No random shuffling!)
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 4. MODEL BUILDING
print("🤖 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. EVALUATION METRICS
predictions = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n📊 Model Evaluation Metrics:")
print(f"👉 Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
print(f"👉 Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 6. VISUALIZATION: Save plot to look professional on GitHub
plt.figure(figsize=(12, 6))
plt.plot(data.index[split_index:], y_test, label="Actual Price", color="blue", alpha=0.6)
plt.plot(data.index[split_index:], predictions, label="Predicted Price", color="orange", alpha=0.9)
plt.title(f"{ticker} Stock Price Prediction Engine - Time Series Analysis")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.savefig("prediction_chart.png", dpi=300)
print("💾 Prediction plot saved as 'prediction_chart.png'")
