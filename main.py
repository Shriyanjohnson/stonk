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
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to fetch stock data and calculate technical indicators
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="60d")  # Last 60 days of stock data for better analysis
    data = add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume')
    return data

# Function to analyze market sentiment based on news headlines
def fetch_sentiment(symbol, api_key):
    try:
        # Using the NewsAPI to fetch news for the symbol
        newsapi = NewsApiClient(api_key=api_key)  # Your NewsAPI key
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        
        headlines = [article['title'] for article in all_articles['articles']]
        
        sentiment_score = 0
        analyzer = SentimentIntensityAnalyzer()
        for headline in headlines:
            sentiment_score += analyzer.polarity_scores(headline)['compound']
        
        sentiment_score /= len(headlines) if len(headlines) > 0 else 1
        return sentiment_score
    
    except Exception as e:
        print(f"Error fetching sentiment: {e}")
        return 0  # Return 0 in case of error (neutral sentiment)

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Prepare features and labels for model training
def prepare_features(symbol, api_key):
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol, api_key)
    
    # Calculate RSI
    rsi = calculate_rsi(stock_data)
    stock_data['RSI'] = rsi
    
    # Calculate price movement (1 for up, 0 for down)
    stock_data['Price_Change'] = stock_data['Close'].shift(-1) > stock_data['Close']
    stock_data.dropna(subset=['RSI', 'Price_Change'], inplace=True)
    
    # Define features and labels
    features = stock_data[['RSI', 'trend_ema_fast', 'trend_ema_slow', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'momentum_rsi', 'momentum_stoch_rsi']]
    labels = stock_data['Price_Change'].astype(int)  # 1 for 'up' (Call), 0 for 'down' (Put)
    
    return features, labels

# Train the ML model
def train_model(symbol, api_key):
    features, labels = prepare_features(symbol, api_key)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Function to predict the stock option recommendation (Call or Put)
def generate_recommendation(symbol, api_key, model):
    features, _ = prepare_features(symbol, api_key)
    last_data = features.iloc[-1].values.reshape(1, -1)
    
    # Predict the stock movement (1 for Call, 0 for Put)
    prediction = model.predict(last_data)
    
    # Determine option based on model prediction
    option = "Call" if prediction[0] == 1 else "Put"
    
    # Calculate strike price as nearest 10 of the last closing price
    stock_data = fetch_stock_data(symbol)
    strike_price = round(stock_data['Close'].iloc[-1] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()  # Next Friday
    
    return option, strike_price, expiration_date

# Plot Candlestick chart with technical indicators
def plot_candlestick(symbol):
    stock_data = fetch_stock_data(symbol)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'], name="Candlestick"), row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['trend_ema_fast'], mode='lines', name='EMA Fast', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['trend_ema_slow'], mode='lines', name='EMA Slow', line=dict(color='red')), row=1, col=1)

    fig.update_layout(title=f"{symbol} Stock with Technical Indicators", xaxis_rangeslider_visible=False)
    return fig

# Streamlit UI
st.title("AI Stock Options Predictor")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
api_key = "833b7f0c6c7243b6b751715b243e4802"  # Your NewsAPI key here

if symbol:
    try:
        # Train model and display accuracy
        model, accuracy = train_model(symbol, api_key)  
        
        st.header(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        option, strike_price, expiration = generate_recommendation(symbol, api_key, model)
        
        st.write(f"Option Recommendation: {option}")
        st.write(f"Strike Price: ${strike_price}")
        st.write(f"Expiration Date: {expiration}")
        
        # Plot the candlestick chart with indicators
        fig = plot_candlestick(symbol)
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
