import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Set API Key (ensure to keep this secure)
API_KEY = "833b7f0c6c7243b6b751715b243e4802"  # Store this securely

# Custom On-Balance Volume (OBV) function
def custom_on_balance_volume(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df

# Save the trained model
def save_model(model, filename="stock_model.pkl"):
    joblib.dump(model, filename)

# Load the trained model
def load_model(filename="stock_model.pkl"):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None

# Fetch stock data
@st.cache_data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    data['RSI'] = data['Close'].pct_change().rolling(14).mean()
    data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()
    data = custom_on_balance_volume(data)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data

# Fetch real-time stock price
def fetch_real_time_price(symbol):
    stock = yf.Ticker(symbol)
    real_time_data = stock.history(period="1d", interval="1m")
    return real_time_data['Close'][-1]

# Fetch EPS and earnings report
def fetch_fundamentals(symbol):
    stock = yf.Ticker(symbol)
    eps = stock.info.get("trailingEps", None)
    earnings = stock.quarterly_earnings
    return eps, earnings

# Train or update the model
def train_or_update_model(data, eps, model=None):
    # Prepare the new data for training
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']]
    
    if eps:
        data['EPS'] = eps  # Add EPS as a constant column
        features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 'EPS']]
    
    labels = data['Target']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # If no model exists, initialize a new model
    if model is None:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(features_scaled, labels)
    else:
        # Append new data to the existing features and labels (Random Forest will retrain with all data)
        model.fit(features_scaled, labels)  # For RandomForest, it will retrain with all data each time

    save_model(model)  # Save the updated model

    accuracy = model.score(features_scaled, labels) * 100
    return model, accuracy

# Streamlit UI
st.markdown("<h1 style='color: white;'>💰 AI Stock Options Predictor 💰</h1>", unsafe_allow_html=True)
symbol = st.text_input("Enter Stock Symbol", "AAPL")

# Set the background to dark
st.markdown("""
    <style>
        body {
            background-color: #2e2e2e;
            color: white;
        }
        .stTextInput>label {
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

if symbol:
    stock_data = fetch_stock_data(symbol)
    eps, earnings = fetch_fundamentals(symbol)
    
    # Load the existing model and update it
    model = load_model()
    model, accuracy = train_or_update_model(stock_data, eps, model)
    
    real_time_price = fetch_real_time_price(symbol)

    st.subheader(f"📈 Stock Data for {symbol}")
    st.write(f"### Real-Time Price: **${real_time_price:.2f}**")
    if eps:
        st.write(f"### EPS (Earnings Per Share): **${eps:.2f}**")
    else:
        st.write("### EPS: Not available")
    st.write(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")

    # Charts
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Stock Price", "RSI", "OBV"))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['OBV'], mode='lines', name='OBV'), row=3, col=1)
    st.plotly_chart(fig)

    # Earnings Report Table
    if earnings is not None:
        st.write("### Earnings Report")
        st.dataframe(earnings)
    else:
        st.write("No earnings report available.")

    # Plot Accuracy Over Time (if model has been updated multiple times)
    if os.path.exists("model_accuracy.txt"):
        accuracy_data = []
        with open("model_accuracy.txt", "r") as f:
            for line in f:
                date, accuracy = line.split(": ")
                accuracy = float(accuracy.strip().replace("%", ""))
                accuracy_data.append((date, accuracy))

        accuracy_df = pd.DataFrame(accuracy_data, columns=["Date", "Accuracy"])
        accuracy_df['Date'] = pd.to_datetime(accuracy_df['Date'])

        fig, ax = plt.subplots()
        ax.plot(accuracy_df['Date'], accuracy_df['Accuracy'], marker='o', linestyle='-', color='b')
        ax.set_title("Model Accuracy Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Accuracy (%)")
        st.pyplot(fig)
