import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import io
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set API Key directly
API_KEY = "833b7f0c6c7243b6b751715b243e4802"  # Ensure you handle the API key securely

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
    return real_time_data['Close'][-1]  # Latest closing price

# Fetch sentiment score from news articles
@st.cache_data
def fetch_sentiment(symbol):
    try:
        newsapi = NewsApiClient(api_key=API_KEY)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")
        return 0

# Train Machine Learning Model
@st.cache_resource
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']]
    labels = data['Target']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid={'n_estimators': [50, 100, 200]}, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_score_ * 100, X_test, y_test

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model, symbol):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['ATR'], latest_data['OBV'], latest_data['SMA_20'], latest_data['SMA_50']]])
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    return option, strike_price, expiration_date, latest_data

# Streamlit UI
st.title("💰 AI Stock Options Predictor by Shriyan Kandula 💰")

# Explanation section
st.markdown("""
### Explanation of Indicators:
- **RSI (Relative Strength Index)**: Measures the strength of a stock's price movement. RSI above 70 indicates the stock might be overbought (potential for price to go down). RSI below 30 indicates it might be oversold (potential for price to go up).
- **ATR (Average True Range)**: Indicates the volatility of the stock. Higher ATR values suggest the stock is more volatile, and lower ATR values suggest less volatility.
- **OBV (On-Balance Volume)**: Tracks volume flow and can help predict price movements. A rising OBV with rising prices suggests strength, while falling OBV may suggest weakness.
- **SMA (Simple Moving Averages)**: Indicates trends based on past closing prices. A crossover of SMA_20 above SMA_50 suggests an uptrend, while the reverse suggests a downtrend.
""")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy, X_test, y_test = train_model(stock_data)
    option, strike_price, expiration, latest_data = generate_recommendation(stock_data, sentiment_score, model, symbol)

    # Fetch and display the real-time stock price
    real_time_price = fetch_real_time_price(symbol)

    st.subheader(f"📈 Option Recommendation for {symbol}")
    st.write(f"**Recommended Option:** {option}")
    st.write(f"**Strike Price:** ${strike_price}")
    st.write(f"**Expiration Date:** {expiration}")
    st.write(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    test_accuracy = model.score(X_test, y_test) * 100
    st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")
    st.write(f"### Real-Time Price: **${real_time_price:.2f}**")

    st.download_button("Download Stock Data", data=stock_data.to_csv(index=True), file_name=f"{symbol}_stock_data.csv", mime="text/csv")

    # Add chart with explanations
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Stock Price', 'RSI'))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    st.plotly_chart(fig)

    # Explanation of prediction reasoning
    st.markdown("""
    ### Why the Prediction?
    The model is predicting the stock direction based on multiple factors:
    - **RSI**: If RSI is above 70, it's typically overbought, indicating the potential for a price decrease.
    - **ATR**: Higher volatility indicates a higher chance of significant price movement.
    - **OBV**: A rising OBV signals that the buying volume is higher, possibly indicating the stock is bullish.
    - **SMA Crossover**: If the shorter SMA (SMA_20) is above the longer SMA (SMA_50), it indicates a bullish signal, and vice versa for a bearish signal.
    
    Additionally, sentiment from recent news articles can tilt the recommendation to confirm whether the stock is likely to go up or down.
    """)
