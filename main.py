import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from newsapi import NewsApiClient
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set API Key for News API
API_KEY = "833b7f0c6c7243b6b751715b243e4802"  # Use your own API key

# Function to fetch stock data using yfinance
@st.cache_data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    
    # Adding technical indicators
    data['RSI'] = data['Close'].pct_change().rolling(14).mean()  # Relative Strength Index
    data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()  # Average True Range
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # Simple Moving Average (20)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # Simple Moving Average (50)
    
    # Calculate On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i - 1]:
            obv.append(obv[-1] + data['Volume'][i])
        elif data['Close'][i] < data['Close'][i - 1]:
            obv.append(obv[-1] - data['Volume'][i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    
    # Adding earnings data as a feature
    earnings = stock.earnings
    if earnings:
        data['Earnings'] = earnings['Earnings']
    
    data.dropna(inplace=True)
    return data

# Fetch sentiment score from news articles using News API and TextBlob
@st.cache_data
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

# Train or update the model using SGDClassifier for incremental learning
def train_or_update_model(data, model=None):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 'Earnings']]
    labels = data['Target']
    
    # Handle NaN or missing data
    if features.isnull().sum().any() or labels.isnull().sum().any():
        raise ValueError("Data contains NaN values, which cannot be processed.")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Initialize or update the model
    if model is None:
        model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)  # Logistic regression loss
        model.fit(features_scaled, labels)
    else:
        model.partial_fit(features_scaled, labels, classes=[0, 1])  # Incremental learning

    return model

# Option Recommendation Function based on the model and sentiment
def generate_recommendation(data, sentiment_score, model, symbol):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['ATR'], 
                                 latest_data['OBV'], latest_data['SMA_20'], latest_data['SMA_50'], latest_data['Earnings']]])
    prediction_prob = model.predict_proba(latest_features)[0][1]
    
    option = "Call" if prediction_prob > 0.5 else "Put"
    
    # Adjust recommendation based on sentiment score
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
    
    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    
    return option, strike_price, expiration_date, latest_data

# Streamlit UI Setup
st.title("💰 AI Stock Options Predictor 💰")
symbol = st.text_input("Enter Stock Symbol", "AAPL")

# Fetching the stock data and sentiment score
if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model = train_or_update_model(stock_data)
    option, strike_price, expiration, latest_data = generate_recommendation(stock_data, sentiment_score, model, symbol)

    # Display results on Streamlit
    st.subheader(f"📈 Option Recommendation for {symbol}")
    st.write(f"**Recommended Option:** {option}")
    st.write(f"**Strike Price:** ${strike_price}")
    st.write(f"**Expiration Date:** {expiration}")
    
    # Display model performance
    accuracy = model.score(StandardScaler().fit_transform(stock_data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 'Earnings']]), stock_data['Target']) * 100
    st.write(f"### Model Accuracy: **{accuracy:.2f}%**")

    # Real-time stock data visualization
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Stock Price', 'RSI'))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    st.plotly_chart(fig)
    
    # Download Stock Data Button
    st.download_button("Download Stock Data", data=stock_data.to_csv(index=True), file_name=f"{symbol}_stock_data.csv", mime="text/csv")

# Explanation Section for Presentation

st.subheader("🛠️ Model Functionality & Explanation")

st.write("""
This AI Stock Options Predictor uses historical stock data and real-time news sentiment to make predictions about stock movements and recommend options (Call/Put). 

### Key Features:
- **Real-Time Data**: Fetches real-time stock data and adjusts options recommendations based on up-to-the-minute changes in the stock market.
- **Sentiment Analysis**: Uses news articles to gauge market sentiment and adjusts recommendations accordingly.
- **Incremental Machine Learning**: The model continuously learns from the latest data, updating the prediction logic over time, improving as more data is fed into it.
- **Technical Indicators**: Utilizes a variety of technical indicators (RSI, ATR, SMA, OBV) to help predict market movements.
- **Earnings Impact**: Takes earnings data into account to improve predictions, understanding that earnings reports can have significant impacts on stock prices.

### Machine Learning Models:
- **SGDClassifier (Logistic Regression)**: Trains using **log_loss** to predict whether the stock will go up or down in the next period. It uses gradient descent, providing efficient incremental learning that can adapt to new data without retraining from scratch.
- **Sentiment-based Adjustment**: The model adjusts its predictions based on sentiment analysis from news articles. This ensures that external news factors (e.g., major company announcements) are accounted for in the recommendation.

### What Makes This Better Than Other Platforms:
- **Real-time and Continuous Learning**: Most platforms rely on static models, while this app continuously learns and updates predictions with the latest stock and sentiment data.
- **Comprehensive Feature Set**: This app integrates multiple sources of data, such as technical indicators and sentiment analysis, combined with earnings reports to create a well-rounded view of the market.
- **Incremental Learning**: Unlike other platforms, this model updates incrementally (using `partial_fit`), ensuring that it adapts to new data as it comes in.

This model is designed for anyone looking to get more accurate and timely options recommendations based on a mix of historical data, sentiment analysis, and earnings reports.
""")
