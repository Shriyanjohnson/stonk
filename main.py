import yfinance as yf
import streamlit as st
from ta import add_all_ta_features
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to fetch stock data and calculate technical indicators
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="60d")  # Last 60 days of stock data for better analysis
    data = add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume')
    return data

# Function to analyze market sentiment based on news headlines
def fetch_sentiment(symbol):
    # Using the NewsAPI to fetch news for the symbol
    newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')  # Replace with your NewsAPI key
    all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
    
    headlines = [article['title'] for article in all_articles['articles']]
    
    sentiment_score = 0
    analyzer = SentimentIntensityAnalyzer()
    for headline in headlines:
        sentiment_score += analyzer.polarity_scores(headline)['compound']
    
    # Calculate the average sentiment score
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

# Function to create a label based on stock data
def create_label(data, sentiment_score):
    # Calculate RSI
    rsi = calculate_rsi(data)
    latest_rsi = rsi.iloc[-1]
    
    # Define basic rules for "Call" (1) and "Put" (0)
    if latest_rsi < 30:  # Oversold condition
        return 1  # Call
    elif latest_rsi > 70:  # Overbought condition
        return 0  # Put
    else:
        return 1 if sentiment_score > 0 else 0  # Sentiment-based decision if RSI is in the neutral range

# Function to prepare features for training
def prepare_features(symbol):
    # Fetch data and sentiment
    data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    
    # Calculate features (RSI, volume, etc.)
    rsi = calculate_rsi(data)
    latest_rsi = rsi.iloc[-1]
    avg_volume = data['Volume'].mean()
    
    # Create label
    label = create_label(data, sentiment_score)
    
    # Return features and label
    return [avg_volume, latest_rsi, sentiment_score], label

# Function to train the ML model
def train_model():
    # Define a list of symbols to train on
    symbols = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT"]
    
    # Create lists for features and labels
    X = []
    y = []
    
    for symbol in symbols:
        features, label = prepare_features(symbol)
        X.append(features)
        y.append(label)
    
    # Convert to numpy arrays for ML
    X = np.array(X)
    y = np.array(y)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Test the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100}%")
    
    return model

# Function to make predictions using the trained model
def make_prediction(symbol, model):
    features, _ = prepare_features(symbol)
    prediction = model.predict([features])
    return "Call" if prediction[0] == 1 else "Put"

# Train the model once when the app starts (this might take time)
model = train_model()

# Streamlit UI
st.title("AI Stock Options Predictor")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    option = make_prediction(symbol, model)
    st.write(f"Option Recommendation: {option}")
    
    # Fetch latest stock data for Strike Price and Expiration Date
    stock_data = fetch_stock_data(symbol)
    strike_price = round(stock_data['Close'].iloc[-1] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()  # Next Friday
    
    st.write(f"Strike Price: ${strike_price}")
    st.write(f"Expiration Date: {expiration_date}")
