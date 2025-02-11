import yfinance as yf
import streamlit as st
from ta import add_all_ta_features
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
from sklearn.ensemble import RandomForestClassifier
from newsapi import NewsApiClient

# Your NewsAPI Key
NEWS_API_KEY = 'your_api_key_here'

# Function to fetch stock data and calculate technical indicators
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="60d")  # Last 60 days
    data = add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume')
    return data

# Function to analyze market sentiment based on news headlines
def fetch_sentiment(symbol):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
    headlines = [article['title'] for article in all_articles['articles']]
    sentiment_score = 0
    analyzer = SentimentIntensityAnalyzer()
    for headline in headlines:
        sentiment_score += analyzer.polarity_scores(headline)['compound']
    sentiment_score /= len(headlines) if len(headlines) > 0 else 1
    return sentiment_score

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to predict the stock option recommendation (Call or Put)
def generate_recommendation(data, sentiment_score):
    # Calculate technical features (e.g., RSI)
    rsi = calculate_rsi(data)
    latest_rsi = rsi.iloc[-1]
    features = [
        data['volume'].mean(),
        latest_rsi,
        sentiment_score
    ]
    model = RandomForestClassifier()
    model.fit([[features[0], features[1], features[2]]], [1])  # Dummy training for now
    prediction = model.predict([features])
    option = "Call" if prediction[0] == 1 else "Put"
    strike_price = round(data['Close'].iloc[-1] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    return option, strike_price, expiration_date

# Streamlit UI
st.title("AI Stock Options Predictor")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score)
    
    st.write(f"Option Recommendation: {option}")
    st.write(f"Strike Price: ${strike_price}")
    st.write(f"Expiration Date: {expiration}")
