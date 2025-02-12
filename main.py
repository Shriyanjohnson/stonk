import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set API Key
API_KEY = "833b7f0c6c7243b6b751715b243e4802"  # Your provided API key

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

# Fetch stock data with earnings
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    data['RSI'] = data['Close'].pct_change().rolling(14).mean()
    data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()
    data = custom_on_balance_volume(data)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Fetch earnings data (quarterly earnings per share)
    try:
        earnings_data = stock.earnings
        if earnings_data is not None and len(earnings_data) > 0:
            # Take the most recent earnings value
            data['Earnings'] = earnings_data.iloc[0]['Earnings']
        else:
            data['Earnings'] = np.nan  # If earnings data is not available, set it as NaN
    except Exception as e:
        data['Earnings'] = np.nan  # If fetching earnings fails, set it as NaN

    # Drop rows with NaN values and reset the index
    data = data.dropna()
    return data

# Fetch real-time stock price
def fetch_real_time_price(symbol):
    stock = yf.Ticker(symbol)
    real_time_data = stock.history(period="1d", interval="1m")
    return real_time_data['Close'][-1]  # Latest closing price

# Fetch sentiment score from news articles
def fetch_sentiment(symbol):
    try:
        newsapi = NewsApiClient(api_key=API_KEY)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except:
        return 0

# Train Machine Learning Model
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 'Earnings']]

    # Drop any rows with NaN or infinite values in features before training the model
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    labels = data['Target']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model, symbol):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['ATR'], latest_data['OBV'], latest_data['SMA_20'], latest_data['SMA_50'], latest_data['Earnings']]])
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
st.title("💰 AI Stock Options Predictor 💰")
symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, X_test, y_test = train_model(stock_data)
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

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Stock Price', 'RSI'))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    st.plotly_chart(fig)

# Bottom section explaining the functionality and models
st.subheader("🔍 Explanation of the Model and Features")
st.write("""
    This AI-powered stock prediction model utilizes machine learning algorithms to predict the best option (Call or Put) for a given stock based on historical data, technical indicators, and real-time sentiment from the news.
    
    **Key Features and Indicators:**
    - **Close Price:** The stock's closing price.
    - **RSI (Relative Strength Index):** A momentum oscillator that measures the speed and change of price movements.
    - **ATR (Average True Range):** A measure of volatility.
    - **OBV (On-Balance Volume):** A volume-based indicator to confirm price trends.
    - **SMA (Simple Moving Average):** The average price over a specific period (20-day, 50-day).
    - **Earnings:** The most recent earnings per share (EPS) reported by the company.

    The model is trained using a **Random Forest Classifier**, a powerful ensemble machine learning algorithm that works by constructing multiple decision trees during training and outputs the mode of the classes (Call/Put) for classification problems.

    **Why This Model is Better:**
    - **Incorporates Real-Time Data:** Unlike traditional models, this one leverages both technical indicators and real-time news sentiment for more informed predictions.
    - **Accurate & Dynamic:** The model is updated continuously with new data, making it adaptive and capable of adjusting to changing market conditions.
    - **Earnings as a Metric:** Earnings data, a key financial indicator, is used to predict stock price movements, giving the model a more comprehensive view of the company's performance.

    By combining various technical indicators, machine learning, and real-time sentiment, this tool provides you with data-driven insights for better trading decisions.
""")


    
