import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
import pandas as pd
import numpy as np
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Custom HTML & CSS Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #ecf0f1;
            font-size: 36px;
            font-weight: bold;
            margin-top: 20px;
        }
        .subtitle {
            text-align: center;
            color: #bdc3c7;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .current-price, .recommendation {
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 8px;
            margin-top: 30px;
            text-align: center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            font-size: 24px;
            color: #2c3e50;
        }
        .recommendation h3 {
            color: #e74c3c;
            font-size: 22px;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 50px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to calculate On-Balance Volume (OBV)
def on_balance_volume(df):
    obv = [0]  # The first OBV value is set to 0
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])  # No change if close price is the same
    return pd.Series(obv, index=df.index)

# Fetch stock data and calculate OBV manually
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    
    # Technical Indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['OnBalanceVolume'] = on_balance_volume(data)  # Using custom OBV function
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Handle NaN values
    data.dropna(inplace=True)
    return data

# Function to fetch sentiment score
def fetch_sentiment(symbol):
    try:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return 0
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except:
        return 0

# Train Model with Hyperparameter Tuning
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    
    features = data[['Close', 'RSI', 'MACD', 'OnBalanceVolume', 'Volatility', 'SMA_20', 'SMA_50']]
    labels = data['Target']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    
    # RandomForest with GridSearch for hyperparameter tuning
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_score_ * 100, X_test, y_test

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['OnBalanceVolume'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50']]])
    
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"

    # Modify prediction based on sentiment
    reasoning = f"Stock Prediction: The model suggests a **{option}** option because of the following reasons:\n"
    reasoning += f"- **RSI**: {latest_data['RSI']:.2f}, indicating whether the stock is overbought or oversold.\n"
    reasoning += f"- **MACD**: {latest_data['MACD']:.2f}, showing trend changes.\n"
    reasoning += f"- **On-Balance Volume**: {latest_data['OnBalanceVolume']:.2f}, indicating buying or selling pressure.\n"
    reasoning += f"- **Volatility**: {latest_data['Volatility']:.4f}, showing price fluctuation.\n"
    reasoning += f"- **SMA-20 vs SMA-50**: Suggests short vs long-term price trends.\n"
    
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
        reasoning += "\nSentiment analysis suggests a positive market sentiment, prompting a Call option."
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
        reasoning += "\nSentiment analysis suggests a negative market sentiment, prompting a Put option."
    
    strike_price = round(latest_data['Close'] / 5) * 5  # Round to nearest 5 for better strike price logic
    if option == "Call" and strike_price < latest_data['Close']:
        strike_price = strike_price + 5
    elif option == "Put" and strike_price > latest_data['Close']:
        strike_price = strike_price - 5

    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    
    return option, strike_price, expiration_date, reasoning

# Streamlit UI
st.markdown('<div class="title">💰 AI Stock Options Predictor 💰</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Developed by Shriyan Kandula, a sophomore at Shaker High School.</div>', unsafe_allow_html=True)

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy, X_test, y_test = train_model(stock_data)

    current_price = stock_data['Close'].iloc[-1]

    st.markdown(f'<div class="current-price">Current Price of {symbol}: **${current_price:.2f}**</div>', unsafe_allow_html=True)

    option, strike_price, expiration, reasoning = generate_recommendation(stock_data, sentiment_score, model)

    st.markdown(f"""
        <div class="recommendation">
            <h3>Option Recommendation: **{option}**</h3>
            <p>Strike Price: **${strike_price}**</p>
            <p>Expiration Date: **{expiration}**</p>
            <p><strong>Reasoning:</strong></p>
            <pre>{reasoning}</pre>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    test_accuracy = model.score(X_test, y_test) * 100
    st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")

# Footer
st.markdown("""
    <div class="footer">
        Created by **Shriyan Kandula**, a sophomore at **Shaker High School** | 💻 Stock Predictions & Insights
    </div>
""", unsafe_allow_html=True)
