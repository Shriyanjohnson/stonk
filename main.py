import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='833b7f0c6c7243b6b751715b243e4802')

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="180d")  # Last 180 days for better analysis
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    bollinger = BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['Moving_Avg'] = data['Close'].rolling(window=20).mean()
    data.dropna(inplace=True)  # Remove NaN values
    return data

# Function to fetch news sentiment
def fetch_sentiment(symbol):
    try:
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        articles = all_articles.get('articles', [])
        if not articles:
            return 0

        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score

    except Exception:
        return 0  # Default to neutral if API fails

# Function to train ML model
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)  # 1 = Call, 0 = Put

    features = data[['Close', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Moving_Avg']]
    labels = data['Target']

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    accuracy = model.score(features, labels) * 100
    return model, accuracy

# Function to generate option recommendation
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['BB_High'], latest_data['BB_Low'], latest_data['Moving_Avg']]])

    prediction = model.predict(latest_features)[0]
    option = "Call" if prediction == 1 else "Put"

    # Adjust based on sentiment
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"

    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()

    return option, strike_price, expiration_date

# Streamlit UI
st.title("AI Stock Options Predictor")
st.image("https://www.pngmart.com/files/21/Money-PNG-HD.png", use_column_width=True)
st.markdown("### Created by Shriyan K")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy = train_model(stock_data)
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    st.metric(label="Model Accuracy", value=f"{accuracy:.2f}%")
    st.write(f"### Option Recommendation: **{option}**")
    st.write(f"Strike Price: **${strike_price}**")
    st.write(f"Expiration Date: **{expiration}**")
    
    # Plot stock data
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name='Market Data'))
    fig.update_layout(title=f"{symbol} Stock Price Chart",
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Disclaimer:** This application is for informational purposes only and does not constitute financial advice.
    Please conduct your own due diligence before making any investment decisions.
    """)
