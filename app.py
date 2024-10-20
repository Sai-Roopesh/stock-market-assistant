import streamlit as st
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get NewsAPI Key from environment variable
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

if not NEWSAPI_KEY:
    st.error("NewsAPI Key not found. Please set it in the `.env` file.")
    st.stop()

# Initialize NewsApiClient
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# Streamlit App
st.set_page_config(layout="wide")
st.title('📈 Stock Insights Dashboard')

# Sidebar for user input
st.sidebar.header('🔍 User Input Parameters')


def get_user_input():
    stock_symbol = st.sidebar.text_input('Stock Symbol', 'AAPL').upper()
    start_date = st.sidebar.date_input(
        'Start Date', datetime.today() - timedelta(days=10))
    end_date = st.sidebar.date_input('End Date', datetime.today())
    return stock_symbol, start_date, end_date


stock_symbol, start_date, end_date = get_user_input()

# Validate Date Range
if start_date > end_date:
    st.sidebar.error("Start Date must be before End Date.")
    st.stop()

# Fetch stock data


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


with st.spinner('Fetching stock data...'):
    stock_data = load_stock_data(stock_symbol, start_date, end_date)

if stock_data is None:
    st.error(
        f"No data found for {stock_symbol} between {start_date} and {end_date}. Please check the stock symbol and date range.")
    st.stop()

st.subheader(f'📊 Historical Stock Data for {stock_symbol}')
st.dataframe(stock_data.tail())

# Plot closing price
st.subheader(f'📈 {stock_symbol} Closing Price')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(stock_data['Date'], stock_data['Close'],
        label='Close Price', color='blue')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title(f'{stock_symbol} Closing Price from {start_date} to {end_date}')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Fetch news articles


@st.cache_data(show_spinner=False)
def load_news(ticker, from_date, to_date):
    all_articles = newsapi.get_everything(
        q=ticker,
        from_param=from_date.strftime('%Y-%m-%d'),
        to=to_date.strftime('%Y-%m-%d'),
        language='en',
        sort_by='relevancy',
        page_size=10
    )
    articles = all_articles.get('articles', [])
    return articles


with st.spinner('Fetching news articles...'):
    news_articles = load_news(
        stock_symbol, start_date - timedelta(days=1), end_date)

st.subheader(f'📰 Recent News Articles about {stock_symbol}')

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

# Optional: Download data as CSV


@st.cache_data(show_spinner=False)
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


csv = convert_df_to_csv(stock_data)

st.download_button(
    label="📥 Download Stock Data as CSV",
    data=csv,
    file_name=f"{stock_symbol}_stock_data_{start_date}_to_{end_date}.csv",
    mime='text/csv',
)
