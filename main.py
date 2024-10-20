import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import seaborn as sns

# Load environment variables
load_dotenv()

# Get API Keys from environment variables
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate API keys
if not NEWSAPI_KEY:
    st.error("NewsAPI Key not found. Please set it in the `.env` file.")
    st.stop()

if not OPENAI_API_KEY:
    st.error("OpenAI API Key not found. Please set it in the `.env` file.")
    st.stop()

# Initialize API clients
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit App Configuration
st.set_page_config(layout="wide", page_title="Advanced Stock Analysis Dashboard")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title('📈 Advanced Stock Insights Dashboard')

# Sidebar Configuration
st.sidebar.header('🔍 Analysis Parameters')

def get_user_input():
    stock_symbol = st.sidebar.text_input('Stock Symbol', 'AAPL').upper()
    
    # Date range selection with preset options
    date_ranges = {
        '1 Week': 7,
        '1 Month': 30,
        '3 Months': 90,
        '6 Months': 180,
        '1 Year': 365,
        'Custom': 0
    }
    
    selected_range = st.sidebar.selectbox('Select Time Range', list(date_ranges.keys()))
    
    if selected_range == 'Custom':
        start_date = st.sidebar.date_input('Start Date', 
                                         datetime.today() - timedelta(days=30))
        end_date = st.sidebar.date_input('End Date', datetime.today())
    else:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=date_ranges[selected_range])
    
    # Technical Analysis Parameters
    st.sidebar.subheader('Technical Indicators')
    show_sma = st.sidebar.checkbox('Show Simple Moving Averages', True)
    show_rsi = st.sidebar.checkbox('Show RSI', True)
    show_macd = st.sidebar.checkbox('Show MACD', True)
    show_bollinger = st.sidebar.checkbox('Show Bollinger Bands', True)
    
    return (stock_symbol, start_date, end_date, 
            show_sma, show_rsi, show_macd, show_bollinger)

# Get user inputs
(stock_symbol, start_date, end_date, 
 show_sma, show_rsi, show_macd, show_bollinger) = get_user_input()

# Technical Analysis Functions
def calculate_technical_indicators(data):
    """Calculate comprehensive technical indicators"""
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def analyze_patterns(data):
    """Analyze trading patterns and signals"""
    patterns = []
    
    # Ensure sufficient data for analysis
    if len(data) < 50:
        return patterns
    
    # Moving Average Crossovers
    if (data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] and 
        data['SMA_20'].iloc[-2] <= data['SMA_50'].iloc[-2]):
        patterns.append("Golden Cross detected (bullish)")
    elif (data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1] and 
          data['SMA_20'].iloc[-2] >= data['SMA_50'].iloc[-2]):
        patterns.append("Death Cross detected (bearish)")
    
    # RSI Signals
    current_rsi = data['RSI'].iloc[-1]
    if current_rsi > 70:
        patterns.append(f"Overbought conditions (RSI: {current_rsi:.2f})")
    elif current_rsi < 30:
        patterns.append(f"Oversold conditions (RSI: {current_rsi:.2f})")
    
    # MACD Signals
    if (data['MACD'].iloc[-1] > data['Signal_Line'].iloc[-1] and 
        data['MACD'].iloc[-2] <= data['Signal_Line'].iloc[-2]):
        patterns.append("MACD bullish crossover")
    elif (data['MACD'].iloc[-1] < data['Signal_Line'].iloc[-1] and 
          data['MACD'].iloc[-2] >= data['Signal_Line'].iloc[-2]):
        patterns.append("MACD bearish crossover")
    
    # Bollinger Band Signals
    last_close = data['Close'].iloc[-1]
    if last_close > data['BB_upper'].iloc[-1]:
        patterns.append("Price above upper Bollinger Band (potential reversal)")
    elif last_close < data['BB_lower'].iloc[-1]:
        patterns.append("Price below lower Bollinger Band (potential reversal)")
    
    return patterns

# Data Loading Functions
@st.cache_data(show_spinner=False)
def load_stock_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if data.empty:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Error fetching stock info: {e}")
        return None

# Load Data
with st.spinner('Fetching market data...'):
    stock_data = load_stock_data(stock_symbol, start_date, end_date)
    stock_info = load_stock_info(stock_symbol)

if stock_data is None:
    st.error(f"No data found for {stock_symbol}. Please check the symbol and try again.")
    st.stop()

# Calculate Technical Indicators
tech_data = calculate_technical_indicators(stock_data)
patterns = analyze_patterns(tech_data)

# Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 {stock_symbol} Price Analysis")
    
    # Create interactive price chart using Plotly
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='OHLC'
    ))
    
    # Add technical indicators based on user selection
    if show_sma:
        fig.add_trace(go.Scatter(
            x=tech_data['Date'],
            y=tech_data['SMA_20'],
            name='SMA 20',
            line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=tech_data['Date'],
            y=tech_data['SMA_50'],
            name='SMA 50',
            line=dict(color='blue')
        ))
    
    if show_bollinger:
        fig.add_trace(go.Scatter(
            x=tech_data['Date'],
            y=tech_data['BB_upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=tech_data['Date'],
            y=tech_data['BB_lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ))
    
    fig.update_layout(
        title=f'{stock_symbol} Price Chart',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Quick Stats")
    
    if stock_info:
        metrics = {
            "Current Price": stock_info.get('currentPrice', 'N/A'),
            "Market Cap": f"${stock_info.get('marketCap', 0):,}",
            "P/E Ratio": stock_info.get('trailingPE', 'N/A'),
            "52W High": stock_info.get('fiftyTwoWeekHigh', 'N/A'),
            "52W Low": stock_info.get('fiftyTwoWeekLow', 'N/A'),
            "Volume": f"{stock_info.get('volume', 0):,}"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)

# Technical Analysis Section
st.subheader("📊 Technical Analysis")

if show_rsi:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=tech_data['Date'],
        y=tech_data['RSI'],
        name='RSI'
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(
        title='Relative Strength Index (RSI)',
        yaxis_title='RSI Value',
        height=300
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

if show_macd:
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(
        x=tech_data['Date'],
        y=tech_data['MACD'],
        name='MACD'
    ))
    fig_macd.add_trace(go.Scatter(
        x=tech_data['Date'],
        y=tech_data['Signal_Line'],
        name='Signal Line'
    ))
    fig_macd.add_bar(
        x=tech_data['Date'],
        y=tech_data['MACD_Histogram'],
        name='MACD Histogram'
    )
    fig_macd.update_layout(
        title='MACD Indicator',
        yaxis_title='Value',
        height=300
    )
    st.plotly_chart(fig_macd, use_container_width=True)

# Pattern Analysis
st.subheader("🎯 Pattern Analysis")
if patterns:
    for pattern in patterns:
        st.info(pattern)
else:
    st.write("No significant patterns detected in the current timeframe.")

# News Section
@st.cache_data(show_spinner=False)
def load_news(ticker, from_date, to_date):
    try:
        all_articles = newsapi.get_everything(
            q=ticker,
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=10
        )
        return all_articles.get('articles', [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

st.subheader("📰 Latest News & Analysis")

with st.spinner('Fetching news...'):
    news_articles = load_news(stock_symbol, start_date, end_date)

if news_articles:
    for article in news_articles:
        st.markdown(f"### {article['title']}")
        st.write(
            f"**Source:** {article['source']['name']}  |  **Published At:** {article['publishedAt']}")
        st.write(article.get('description', 'No description available.'))
        st.write(f"[Read more...]({article['url']})")
        st.markdown("---")
else:
    st.write('No news articles found for this date range.')

# AI Insights Generation
def generate_ai_insights(symbol, data, articles, patterns, stock_info):
    """Enhanced AI insights generation"""
    # Prepare technical analysis summary
    latest_close = data['Close'].iloc[-1]
    if len(data) >= 2:
        prev_close = data['Close'].iloc[-2]
    else:
        prev_close = latest_close  # If not enough data, use latest close
    price_change = latest_close - prev_close
    if prev_close != 0:
        price_change_pct = (price_change / prev_close) * 100
    else:
        price_change_pct = 0.0

    # Prepare market context
    market_cap = stock_info.get('marketCap', 'N/A')
    pe_ratio = stock_info.get('trailingPE', 'N/A')

    # Prepare news summary
    news_summary = "\n".join([
        f"- {article['title']}" for article in articles[:5]
    ])

    # Prepare patterns
    pattern_summary = ', '.join(patterns) if patterns else 'No significant patterns detected'

    # Enhanced prompt for more detailed analysis
    prompt = f"""As a senior financial analyst, provide a comprehensive analysis of {symbol}:

Technical Analysis:
- Current Price: ${latest_close:.2f} ({price_change_pct:.2f}% change)
- Market Cap: {market_cap}
- P/E Ratio: {pe_ratio}
- Recent Patterns: {pattern_summary}

News Highlights:
{news_summary}

Based on the above data and market trends, provide insights and potential future outlook for {symbol}.
"""

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if you don't have access to GPT-4
            messages=[
                {"role": "system", "content": "You are a seasoned financial analyst providing detailed stock analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        analysis = chat_completion.choices[0].message.content.strip()
        return analysis
    except openai.APIConnectionError as e:
        return f"Connection error: {e}"
    except openai.RateLimitError as e:
        return f"Rate limit exceeded: {e}"
    except openai.APIStatusError as e:
        return f"API error {e.status_code}: {e.response}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Generate and Display AI Insights
st.subheader("🤖 AI-Powered Analysis")

with st.spinner('Generating AI insights...'):
    ai_insights = generate_ai_insights(stock_symbol, tech_data, news_articles, patterns, stock_info)

st.write(ai_insights)
