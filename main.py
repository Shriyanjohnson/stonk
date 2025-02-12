import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
symbol = 'AAPL'
lookback = 90  # Lookback period in days
test_size = 0.2  # Test size percentage
n_estimators = 200  # Number of estimators for Random Forest

# Fetching the stock data
@st.cache
def fetch_stock_data(symbol, lookback):
    stock = yf.Ticker(symbol)
    data = stock.history(period=f"{lookback}d")
    
    # Technical indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd_diff()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_10'] = EMAIndicator(data['Close'], window=10).ema_indicator()
    data['EMA_50'] = EMAIndicator(data['Close'], window=50).ema_indicator()
    data['Bollinger_High'] = BollingerBands(data['Close']).bollinger_hband()
    data['Bollinger_Low'] = BollingerBands(data['Close']).bollinger_lband()
    data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data['Stochastic'] = StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()

    # Custom OBV calculation (if ta.volume doesn't work)
    data = on_balance_volume(data)
    
    data.dropna(inplace=True)
    return data

# Custom OBV Function
def on_balance_volume(df):
    """
    Calculate the On-Balance Volume (OBV) for a given DataFrame with stock data.
    """
    obv = [0]  # Initialize OBV list with 0
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])  # Volume added to OBV if the close price is higher
        elif df['Close'][i] < df['Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])  # Volume subtracted from OBV if the close price is lower
        else:
            obv.append(obv[-1])  # No change if the close price is the same
    df['OBV'] = obv
    return df

# Fetch the data
data = fetch_stock_data(symbol, lookback)

# Feature Engineering
features = ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_50', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'Stochastic', 'OBV']
X = data[features]
y = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 for up, 0 for down (next day's close)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# Hyperparameter Tuning for Random Forest and XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

# GridSearchCV for XGBoost
grid_search_xgb = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

# Best Models
best_rf_model = grid_search_rf.best_estimator_
best_xgb_model = grid_search_xgb.best_estimator_

# Ensemble Method: Voting Classifier
ensemble_model = VotingClassifier(estimators=[('rf', best_rf_model), ('xgb', best_xgb_model)], voting='hard')
ensemble_model.fit(X_train, y_train)

# Model Evaluation
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# Results
st.write(f"### Test Accuracy: **{accuracy:.2f}%**")

# Feature Importance Plot for Random Forest
importance = best_rf_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
st.pyplot(fig)

# Fetching latest news and performing sentiment analysis
def fetch_news_sentiment(symbol):
    newsapi = NewsApiClient(api_key='833b7f0c6c7243b6b751715b243e4802')
    today = datetime.date.today()
    articles = newsapi.get_everything(q=symbol, from_param=today, to=today, language='en', sort_by='relevancy')
    headlines = [article['title'] for article in articles['articles']]
    
    sentiment_scores = []
    for headline in headlines:
        sentiment = TextBlob(headline).sentiment.polarity
        sentiment_scores.append(sentiment)
        
    average_sentiment = np.mean(sentiment_scores)
    return average_sentiment

# Calculate news sentiment for AAPL
sentiment = fetch_news_sentiment(symbol)

# Show sentiment
st.write(f"### Sentiment for {symbol}: {sentiment:.2f}")

# Visualizations
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name='Candlestick'),
                     go.Scatter(x=data.index, y=data['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'),
                     go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='blue', width=1), name='SMA 50')])

fig.update_layout(title=f'{symbol} Stock Price Analysis with Indicators', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)

# Explanation
st.write("""
### Explanation of Results:
1. **RSI (Relative Strength Index)**: This indicates whether a stock is overbought (>70) or oversold (<30).
2. **MACD (Moving Average Convergence Divergence)**: Shows the difference between short-term and long-term moving averages.
3. **SMA (Simple Moving Average)**: Averages the stock price over a fixed window (e.g., 20 or 50 days).
4. **EMA (Exponential Moving Average)**: Similar to SMA but gives more weight to recent prices.
5. **Bollinger Bands**: Show volatility and overbought/oversold conditions.
6. **ATR (Average True Range)**: Measures market volatility.
7. **Stochastic Oscillator**: Compares a stock's closing price to its price range over a certain period.
8. **OBV (On-Balance Volume)**: Tracks volume flow to help confirm price trends.
9. **Sentiment Analysis**: News sentiment is captured from headlines and used to assess the market mood.
10. **Accuracy**: This percentage indicates how well the model performs on unseen test data. A higher value means better predictive power.
""")

# Footer
st.write("""
--- 
Created by: **Shriyan Kandula**, a Sophomore at Shaker High School. This stock prediction model uses machine learning and technical analysis to provide insights into market trends.
""")

