import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
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

# Performance Tracking Function
def track_performance(model, X_test, y_test):
    predicted = model.predict(X_test)
    accuracy = (predicted == y_test).mean() * 100
    return accuracy

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
    
    # Automated Alert (threshold check)
    if prediction_prob > 0.75:  # Threshold for alert (e.g., high probability prediction)
        st.warning(f"🔔 Alert! High probability of a **{option}** position for {symbol} based on current data!")
    
    return option, strike_price, expiration_date, latest_data

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰\n Made by Shriyan Kandula")

# Disclaimer message
st.write("**Note:** The predictions and recommendations made by this tool may not be entirely accurate and should be considered as one of many factors when making investment decisions.")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy, X_test, y_test = train_model(stock_data)
    option, strike_price, expiration, latest_data = generate_recommendation(stock_data, sentiment_score, model, symbol)

    # Performance Metrics: Tracking model performance on test data
    test_accuracy = track_performance(model, X_test, y_test)

    # Fetch and display the real-time stock price
    real_time_price = fetch_real_time_price(symbol)

    st.subheader(f"📈 Option Recommendation for {symbol}")
    st.write(f"**Recommended Option:** {option}")
    st.write(f"**Strike Price:** ${strike_price}")
    st.write(f"**Expiration Date:** {expiration}")
    st.write(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")
    st.write(f"### Real-Time Price: **${real_time_price:.2f}**")

    st.download_button("Download Stock Data", data=stock_data.to_csv(index=True), file_name=f"{symbol}_stock_data.csv", mime="text/csv")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Stock Price', 'RSI'))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    st.plotly_chart(fig)

    # Displaying the indicator values and their significance
    st.subheader("📊 Indicator Analysis")
    st.write(f"**RSI (Relative Strength Index):** {latest_data['RSI']:.2f} - RSI values above 70 may suggest an overbought condition (uptrend possible reversal), and below 30 may indicate an oversold condition (uptrend likely).")
    st.write(f"**ATR (Average True Range):** {latest_data['ATR']:.2f} - A high ATR value suggests higher volatility, potentially signaling larger price movements.")
    st.write(f"**SMA_20 (20-period Simple Moving Average):** {latest_data['SMA_20']:.2f} - A rising SMA_20 could indicate an uptrend, while a falling SMA_20 could suggest a downtrend.")
    st.write(f"**SMA_50 (50-period Simple Moving Average):** {latest_data['SMA_50']:.2f} - A crossover above the 50-day SMA could indicate bullish movement (uptrend).")
    st.write(f"**OBV (On-Balance Volume):** {latest_data['OBV']:.2f} - A rising OBV suggests increasing buying pressure (uptrend), while a falling OBV signals a downtrend.")

# Explanation of functionality and benefits over competitors
st.subheader("🔍 How It Works & Why It's Better Than Competitors")
st.write("""
This tool combines a range of advanced features to give you the best insights for stock option predictions:

- **Comprehensive Data Sources:** Integrates historical stock data, real-time price updates, and sentiment analysis from news articles to give a well-rounded view of the market.
- **Custom Indicators:** Utilizes On-Balance Volume (OBV) and Average True Range (ATR), alongside standard indicators like RSI and moving averages, to enhance the accuracy of predictions.
- **Machine Learning Power:** Employs a Random Forest model trained on real data, with a high level of accuracy achieved through hyperparameter tuning. Unlike many competitors, this tool continuously refines its predictions for improved accuracy.
- **Sentiment Analysis:** News sentiment is factored into decision-making, providing an extra layer of market insight that most competitors don't offer.
- **Easy-to-Use Interface:** Designed with simplicity in mind, this app presents crucial stock information in a clean and user-friendly format. It's perfect for both novice and experienced traders.
- **Automated Alerts:** The tool alerts you when there’s a high probability of a certain stock movement, allowing you to make decisions quickly.

Unlike other stock prediction tools, this one takes into account real-time market sentiment and incorporates comprehensive technical indicators for smarter, data-backed decisions.
""")
