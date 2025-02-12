import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
import pandas as pd
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Fetch Earnings Data (P/E ratio, Earnings Surprise, etc.)
def fetch_earnings_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        earnings = stock.earnings
        latest_earnings = earnings.iloc[-1] if len(earnings) > 0 else None
        if latest_earnings is not None:
            eps = latest_earnings['Earnings']
            revenue = latest_earnings['Revenue']
            pe_ratio = stock.info.get('trailingPE', None)
            return eps, revenue, pe_ratio
        else:
            return 0, 0, 0
    except Exception as e:
        st.warning(f"Warning: Unable to retrieve earnings data for {symbol}. Error: {e}")
        return 0, 0, 0

# Custom OBV Implementation
def custom_on_balance_volume(df):
    obv = [0]  # initialize with 0 for the first row
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])  # price goes up, OBV increases
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])  # price goes down, OBV decreases
        else:
            obv.append(obv[-1])  # price unchanged, OBV remains the same
    df['On_Balance_Volume'] = obv
    return df

# Function to fetch stock data
@st.cache_data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="90d")
        
        # Technical Indicators
        data['RSI'] = RSIIndicator(data['Close']).rsi()
        data['MACD'] = MACD(data['Close']).macd()
        data['Volatility'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        data = custom_on_balance_volume(data)  # Using the custom OBV function
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Handle NaN values
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.warning(f"Warning: Unable to retrieve stock data for {symbol}. Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if data retrieval fails

# Function to fetch sentiment score
@st.cache_data
def fetch_sentiment(symbol):
    try:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            st.warning("No API key found for NewsAPI. Unable to fetch sentiment data.")
            return 0
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception as e:
        st.warning(f"Warning: Unable to retrieve sentiment data for {symbol}. Error: {e}")
        return 0

# Train Model with GridSearchCV
@st.cache_resource
def train_model(data, symbol):
    try:
        data['Price Change'] = data['Close'].diff()
        data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
        features = data[['Close', 'RSI', 'MACD', 'Volatility', 'On_Balance_Volume', 'SMA_20', 'SMA_50']]
        
        # Add Earnings-related Features
        eps, revenue, pe_ratio = fetch_earnings_data(symbol)
        features['EPS'] = eps
        features['Revenue'] = revenue
        features['P/E Ratio'] = pe_ratio
        
        # Model features and labels
        labels = data['Target']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train-test split and cross-validation
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # GridSearchCV for tuning model parameters
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Cross-validation score
        cross_val_accuracy = cross_val_score(best_model, X_train, y_train, cv=5).mean() * 100
        
        return best_model, cross_val_accuracy, X_test, y_test
    except Exception as e:
        st.warning(f"Warning: An error occurred while training the model. Error: {e}")
        return None, 0, None, None

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model):
    try:
        latest_data = data.iloc[-1]
        latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'],
                                     latest_data['Volatility'], latest_data['On_Balance_Volume'], 
                                     latest_data['SMA_20'], latest_data['SMA_50'], 
                                     latest_data['EPS'], latest_data['Revenue'], latest_data['P/E Ratio']]])
        
        prediction_prob = model.predict_proba(latest_features)[0][1]
        option = "Call" if prediction_prob > 0.5 else "Put"
        
        if sentiment_score > 0.2 and option == "Put":
            option = "Call"
        elif sentiment_score < -0.2 and option == "Call":
            option = "Put"
        
        strike_price = round(latest_data['Close'] / 10) * 10
        expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
        
        return option, strike_price, expiration_date, latest_data
    except Exception as e:
        st.warning(f"Warning: An error occurred while generating the option recommendation. Error: {e}")
        return "N/A", 0, None, None

# Streamlit UI
st.markdown('<div class="title">💰 AI Stock Options Predictor 💰</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Developed by Shriyan Kandula, a sophomore at Shaker High School.</div>', unsafe_allow_html=True)

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    stock_data = fetch_stock_data(symbol)
    if stock_data.empty:
        st.warning(f"Unable to retrieve stock data for {symbol}. Please try another symbol.")
    else:
        sentiment_score = fetch_sentiment(symbol)
        
        model, accuracy, X_test, y_test = train_model(stock_data, symbol)
        if model:
            current_price = stock_data['Close'].iloc[-1]
            st.markdown(f'<div class="current-price">Current Price of {symbol}: **${current_price:.2f}**</div>', unsafe_allow_html=True)
            
            option, strike_price, expiration, latest_data = generate_recommendation(stock_data, sentiment_score, model)
            
            st.markdown(f""" 
            <div class="recommendation"> 
                <h3>Option Recommendation: **{option}**</h3>
                <p>Strike Price: **${strike_price}**</p>
                <p>Expiration Date: **{expiration}**</p>
            </div> """, unsafe_allow_html=True)
            
            st.markdown(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
            test_accuracy = model.score(X_test, y_test) * 100
            st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")
            
            # Displaying Technical Indicators
            st.subheader('Technical Indicators and Their Current Values')
            st.write(f"**RSI** (Relative Strength Index): {latest_data['RSI']:.2f} - An RSI above 70 indicates overbought conditions, while below 30 indicates oversold.")
            st.write(f"**MACD** (Moving Average Convergence Divergence): {latest_data['MACD']:.2f} - Indicates momentum. Positive values suggest upward momentum.")
            st.write(f"**On-Balance Volume (OBV)**: {latest_data['On_Balance_Volume']:.2f} - Shows the cumulative buying and selling pressure.")
            st.write(f"**SMA-20** (Simple Moving Average - 20 days): {latest_data['SMA_20']:.2f}")
            st.write(f"**SMA-50** (Simple Moving Average - 50 days): {latest_data['SMA_50']:.2f}")
            st.write(f"**Volatility** (Average True Range): {latest_data['Volatility']:.2f} - A measure of price fluctuations over a given period.")
            
            # Visualizations
            st.subheader('Stock Data and Technical Indicators')
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=('Stock Price', 'RSI Indicator', 'MACD Indicator'))
            fig.add_trace(go.Candlestick(x=stock_data.index, ...), row=1, col=1)  # Add more visualization code here

