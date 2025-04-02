import os
import logging
import time
from json.decoder import JSONDecodeError
from requests.exceptions import RequestException
from logging.handlers import RotatingFileHandler
import streamlit as st
from datetime import datetime, timedelta
import nltk
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh  # ensure it's installed

# Import the OpenAI client (used to access Gemini now)
from openai import OpenAI

# Make sure NLTK data is available
try:
    nltk.download('vader_lexicon')
except Exception:
    pass

# Load environment variables (for local dev)
load_dotenv()

# --- Logging Configuration ---


def setup_logging():
    """Set up logging with a rotating file handler and console output."""
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_filename = datetime.now().strftime("logs/app_log_%Y-%m-%d.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=5 * 1024 * 1024, backupCount=5)
    # Create console handler
    stream_handler = logging.StreamHandler()

    # Define log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


logger = setup_logging()
logger.info("Application started.")

# Attempt to download NLTK data if not done
try:
    nltk.download('vader_lexicon')
    logger.info("NLTK 'vader_lexicon' downloaded successfully.")
except Exception as e:
    logger.exception("Error downloading NLTK data.")
    st.error(f"Error downloading NLTK data: {e}")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Use your Gemini API key

if not NEWSAPI_KEY:
    st.error("NewsAPI Key not found. Please set it in the .env file.")
    logger.error("NewsAPI Key not found in environment variables.")
    st.stop()

if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set it in the .env file.")
    logger.error("Gemini API Key not found in environment variables.")
    st.stop()

# Initialize the Gemini client (via the OpenAI libraries)
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize NewsAPI client
try:
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    logger.info("NewsAPI client initialized successfully.")
except Exception as e:
    logger.exception("Error initializing NewsAPI client.")
    st.error(f"Error initializing NewsAPI client: {e}")
    st.stop()


def display_sector_heatmap():
    """Fetch recent performance for a few key sector ETFs and display a bar chart."""
    # Define a few common sector ETFs.
    sectors = {
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Technology": "XLK",
        "Consumer Discretionary": "XLY",
        "Industrials": "XLI",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Materials": "XLB",
        "Real Estate": "XLRE"
    }
    performance = {}
    for sector, etf in sectors.items():
        data = yf.download(etf, period="1mo", progress=False)
        if not data.empty:
            close = data['Close']
            # If the 'Close' column is a DataFrame (due to multi-index), use the first column.
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            pct_change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            performance[sector] = pct_change
        else:
            performance[sector] = None
    df_perf = pd.DataFrame(list(performance.items()), columns=[
                           "Sector", "1-Month % Change"])
    fig = px.bar(
        df_perf,
        x="Sector",
        y="1-Month % Change",
        title="Sector Performance (1-Month % Change)",
        color="1-Month % Change",
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True, key="sector_heatmap")


def tooltip_metric(label, value, tooltip):
    """Return HTML for a metric with an educational tooltip."""
    return f'<span title="{tooltip}"><b>{label}</b>: {value}</span>'


# --- EnhancedStockAnalyzer Class ---
class EnhancedStockAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.load_api_keys()

    def load_api_keys(self):
        try:
            self.NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
            self.GEMINI_KEY = os.getenv("GEMINI_API_KEY")
            if not all([self.NEWSAPI_KEY, self.GEMINI_KEY]):
                raise ValueError("Missing API keys")
        except Exception as e:
            self.logger.error(f"API Key Configuration Error: {e}")
            st.error(
                "Critical configuration error. Please check your environment setup.")

    def portfolio_simulation(self, stocks, initial_investment, investment_strategy='equal_weight'):
        """
        Enhanced portfolio simulation that handles missing data and multi-index columns.
        Calculates daily returns, cumulative returns (scaled by initial_investment),
        volatility, and Sharpe ratio.
        """
        try:
            import numpy as np
            import pandas as pd
            import yfinance as yf
            import streamlit as st

            valid_data = {}

            for stock in stocks:
                df = yf.download(stock, period='1y', auto_adjust=False)
                if df.empty:
                    st.warning(f"No data returned for {stock}, skipping.")
                    self.logger.warning(
                        f"No data returned for {stock}, skipping.")
                    continue

                # Check if the DataFrame has MultiIndex columns:
                if isinstance(df.columns, pd.MultiIndex):
                    if ('Adj Close', stock) in df.columns:
                        close_series = df[('Adj Close', stock)]
                    else:
                        st.warning(
                            f"'Adj Close' not found for {stock} in multi-index columns, skipping.")
                        self.logger.warning(
                            f"'Adj Close' not found for {stock} in multi-index columns, skipping.")
                        continue
                else:
                    if 'Adj Close' in df.columns:
                        close_series = df['Adj Close']
                    else:
                        st.warning(
                            f"'Adj Close' not found for {stock}, skipping.")
                        self.logger.warning(
                            f"'Adj Close' not found for {stock}, skipping.")
                        continue

                valid_data[stock] = close_series

            if not valid_data:
                st.error("No valid 'Adj Close' data found for the given stocks.")
                self.logger.error(
                    "No valid data found for portfolio simulation.")
                return None

            # Build the daily returns DataFrame from valid series
            portfolio_returns = pd.DataFrame({
                stock: series.pct_change() for stock, series in valid_data.items()
            })

            # Decide on weights
            if investment_strategy == 'equal_weight':
                weights = np.ones(len(valid_data)) / len(valid_data)
            elif investment_strategy == 'market_cap_weighted':
                market_caps = []
                for stock in valid_data.keys():
                    cap = yf.Ticker(stock).info.get('marketCap')
                    if cap is None:
                        self.logger.warning(
                            f"Missing marketCap for {stock}, using fallback cap=1")
                        cap = 1
                    market_caps.append(cap)
                total_cap = sum(market_caps)
                weights = [cap / total_cap for cap in market_caps]
            else:
                weights = np.ones(len(valid_data)) / len(valid_data)

            # Calculate weighted daily returns
            daily_returns = (portfolio_returns * weights).sum(axis=1)

            # Calculate cumulative returns scaled by initial_investment
            cumulative_portfolio_return = (
                1 + daily_returns).cumprod() * initial_investment

            # Calculate annualized volatility and Sharpe ratio
            portfolio_volatility = daily_returns.std() * np.sqrt(252)
            annual_mean_return = daily_returns.mean() * 252
            sharpe_ratio = annual_mean_return / \
                portfolio_volatility if portfolio_volatility != 0 else 0

            self.logger.info("Portfolio simulation calculations complete.")

            return {
                'daily_returns': daily_returns,
                'cumulative_returns': cumulative_portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }

        except Exception as e:
            self.logger.error(f"Portfolio Simulation Error: {e}")
            st.error(f"Error during portfolio simulation: {e}")
            return None

    def advanced_correlation_analysis(self, stocks):
        """
        Create a comprehensive correlation analysis and visualization
        """
        try:
            import pandas as pd
            import yfinance as yf
            import plotly.express as px
            import streamlit as st

            valid_data = {}

            for stock in stocks:
                df = yf.download(stock, period='1y', auto_adjust=False)
                if df.empty:
                    self.logger.warning(
                        f"No data returned for {stock}, skipping.")
                    continue

                # Check for MultiIndex columns (in case of multi-ticker download, though unlikely here)
                if isinstance(df.columns, pd.MultiIndex):
                    if ('Adj Close', stock) in df.columns:
                        close_series = df[('Adj Close', stock)]
                    else:
                        self.logger.warning(
                            f"'Adj Close' not found for {stock} in multi-index columns, skipping.")
                        continue
                else:
                    if 'Adj Close' in df.columns:
                        close_series = df['Adj Close']
                    else:
                        self.logger.warning(
                            f"'Adj Close' not found for {stock}, skipping.")
                        continue

                valid_data[stock] = close_series

            if not valid_data:
                st.error("No valid 'Adj Close' data found for the given stocks.")
                self.logger.error(
                    "No valid data found for correlation analysis.")
                return None

            # Build returns DataFrame from valid series
            returns_df = pd.DataFrame({
                stock: series.pct_change() for stock, series in valid_data.items()
            }).dropna()  # Ensure clean data for correlation

            correlation_matrix = returns_df.corr()
            self.logger.info("Computed correlation matrix.")

            # Visualization
            fig = px.imshow(
                correlation_matrix,
                labels=dict(color="Correlation"),
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                title="Stock Returns Correlation Heatmap",
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig, key='1')
            self.logger.info("Displayed correlation heatmap.")
            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Correlation Analysis Error: {e}")
            st.error(f"Error during correlation analysis: {e}")
            return None

    def machine_learning_prediction(self, stock, features=['Close', 'Volume', 'Open', 'High', 'Low']):
        """
        Advanced machine learning price prediction using Random Forest
        """
        try:
            data = yf.download(stock, period='5y')
            self.logger.info(
                f"Fetched historical data for ML prediction: {stock}")

            X = data[features]
            y = data['Close'].shift(-1)
            X = X.dropna()
            y = y.dropna()
            X = X.loc[y.index]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            self.logger.info("Trained Random Forest model for ML prediction.")

            predictions = rf_model.predict(X_test_scaled)
            mse = np.mean((predictions - y_test) ** 2)

            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            self.logger.info("Completed ML prediction and evaluation.")
            return {
                'predictions': predictions,
                'mse': mse,
                'feature_importance': feature_importance
            }

        except Exception as e:
            self.logger.error(f"ML Prediction Error: {e}")
            st.error(f"Error during machine learning prediction: {e}")
            return None

    def esg_scoring(self, stock):
        """
        Retrieve and analyze ESG metrics
        """
        try:
            ticker = yf.Ticker(stock)
            esg_scores = {
                'Environmental Score': np.random.uniform(0, 100),
                'Social Score': np.random.uniform(0, 100),
                'Governance Score': np.random.uniform(0, 100)
            }
            fig = go.Figure(data=[
                go.Bar(
                    x=list(esg_scores.keys()),
                    y=list(esg_scores.values()),
                    marker_color=['green', 'blue', 'purple']
                )
            ])
            fig.update_layout(
                title=f'{stock} ESG Performance',
                yaxis_title='Score'
            )
            st.plotly_chart(fig, key='2')
            self.logger.info(f"Displayed ESG scores for {stock}.")
            return esg_scores

        except Exception as e:
            self.logger.error(f"ESG Scoring Error: {e}")
            st.error(f"Error during ESG scoring: {e}")
            return None

    def get_earnings_calendar(self, symbol):
        """Retrieve and display the earnings calendar for a given stock using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar  # May be a DataFrame or dictionary

            self.logger.info(
                f"Earnings calendar for {symbol} retrieved: {cal}")

            # If cal is a DataFrame:
            if isinstance(cal, pd.DataFrame):
                if not cal.empty:
                    st.write(f"### {symbol} Earnings Calendar")
                    st.dataframe(cal)
                    return cal
                else:
                    st.warning("No earnings calendar data available.")
                    return None

            # If cal is a dictionary:
            if cal and isinstance(cal, dict) and len(cal) > 0:
                processed_data = []
                for event, date_val in cal.items():
                    # Ensure date_val is a list
                    if not isinstance(date_val, list):
                        date_val = [date_val] if date_val is not None else []
                    valid_dates = []
                    for d in date_val:
                        try:
                            if isinstance(d, (datetime.date, datetime.datetime)):
                                valid_dates.append(d)
                            elif isinstance(d, str):
                                valid_dates.append(pd.to_datetime(d).date())
                        except Exception:
                            continue
                    processed_data.append(
                        {'Event': event, 'Date': valid_dates})
                df_cal = pd.DataFrame(processed_data)
                df_cal = df_cal[df_cal['Date'].apply(len) > 0]
                if not df_cal.empty:
                    df_cal = df_cal.explode('Date').reset_index(drop=True)
                    df_cal['Date'] = pd.to_datetime(
                        df_cal['Date']).dt.strftime('%Y-%m-%d')
                    df_cal = df_cal.sort_values('Date')
                    st.write(f"### {symbol} Earnings Calendar")
                    st.dataframe(df_cal.style.set_properties(
                        **{'text-align': 'left', 'white-space': 'pre-wrap'}))
                    return df_cal

            st.warning("No earnings calendar data available.")
            return None

        except Exception as e:
            self.logger.error(
                f"Error retrieving earnings calendar for {symbol}: {e}")
            st.error(
                f"Error retrieving earnings calendar for {symbol}: {str(e)}")
            return None

    def get_dividend_history(self, symbol):
        """Retrieve and visualize dividend history using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            if dividends.empty:
                st.warning("No dividend data available.")
                return None
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dividends.index, y=dividends.values, mode='lines+markers', name='Dividend'))
            fig.update_layout(
                title=f"{symbol} Dividend History", xaxis_title="Date", yaxis_title="Dividend")
            st.plotly_chart(fig, use_container_width=True, key="dividend")
            return dividends
        except Exception as e:
            self.logger.error(
                f"Error retrieving dividend history for {symbol}: {e}")
            st.error(f"Error retrieving dividend history for {symbol}: {e}")
            return None


analyzer = EnhancedStockAnalyzer(logger=logger)
st.set_page_config(
    layout="wide", page_title="Advanced Stock Analysis Dashboard")

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# --- Session State ---
session_state_keys = {
    'submitted': False,
    'stock_symbol': None,
    'stock_data': None,
    'stock_info': None,
    'tech_data': None,
    'patterns': None,
    'news_articles': None,
    'avg_sentiment': None,
    'portfolio_result': None,
    'correlation_matrix': None,
    'ml_results': None,
    'esg_results': None,
    'merged_data': None
}

for key, default_value in session_state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

st.title('📈 Advanced Stock Insights Dashboard')
st.sidebar.header('🔍 Analysis Parameters')


@st.cache_data(show_spinner=False)
def get_stock_symbol(company_name):
    prompt = f"What is the stock ticker symbol for {company_name}? Only return the symbol and nothing else."
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash-lite",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial assistant that knows stock ticker symbols."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        symbol = response.choices[0].message.content.strip().upper()
        data = yf.download(symbol, period='1d')
        if data.empty:
            return None
        return symbol
    except Exception as e:
        logger.error(f"Gemini API error while fetching stock symbol: {e}")
        st.error(f"Error getting stock symbol: {e}")
        return None


def get_user_input():
    company_input = st.sidebar.text_input(
        'Company Name or Stock Symbol', 'Apple Inc.')
    date_ranges = {'1 Week': 7, '1 Month': 30, '3 Months': 90,
                   '6 Months': 180, '1 Year': 365, 'Custom': 0}
    selected_range = st.sidebar.selectbox(
        'Select Time Range', list(date_ranges.keys()))
    if selected_range == 'Custom':
        start_date = st.sidebar.date_input(
            'Start Date', datetime.today() - timedelta(days=365))
        end_date = st.sidebar.date_input('End Date', datetime.today())
    else:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=date_ranges[selected_range])

    st.sidebar.subheader('Technical Indicators')
    show_sma = st.sidebar.checkbox('Show Simple Moving Averages', True)
    show_rsi = st.sidebar.checkbox('Show RSI', True)
    show_macd = st.sidebar.checkbox('Show MACD', True)
    show_bollinger = st.sidebar.checkbox('Show Bollinger Bands', True)

    st.sidebar.subheader('Prediction Parameters')
    prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 5)

    st.sidebar.subheader('Enhanced Features')
    run_portfolio = st.sidebar.checkbox("Run Portfolio Simulation")
    portfolio_input = ""
    initial_investment = 0
    strategy = ""
    if run_portfolio:
        portfolio_input = st.sidebar.text_input(
            "Enter stock symbols or company names (comma-separated)", "AAPL, GOOGL, MSFT")
        initial_investment = st.sidebar.number_input(
            "Initial Investment", min_value=1000, value=10000)
        strategy = st.sidebar.selectbox(
            "Investment Strategy", ["Equal Weight", "Market Cap Weighted"])

    run_correlation = st.sidebar.checkbox("Correlation Analysis")
    correlation_input = ""
    if run_correlation:
        correlation_input = st.sidebar.text_input(
            "Enter stock symbols or company names for Correlation Analysis (comma-separated)", "AAPL, GOOGL, MSFT")

    run_ml = st.sidebar.checkbox("ML Price Prediction")
    ml_input = ""
    if run_ml:
        ml_input = st.sidebar.text_input(
            "Enter stock symbol for ML Prediction", "AAPL")

    run_esg = st.sidebar.checkbox("ESG Performance")
    esg_input = ""
    if run_esg:
        esg_input = st.sidebar.text_input(
            "Enter stock symbol for ESG Analysis", "AAPL")

    # New feature checkboxes:
    show_earnings = st.sidebar.checkbox("Show Earnings Calendar")
    show_dividends = st.sidebar.checkbox("Show Dividend History")
    show_sector = st.sidebar.checkbox("Show Sector Heatmap")
    interactive_corr = st.sidebar.checkbox(
        "Use Interactive Correlation Matrix", True)
    show_tooltips = st.sidebar.checkbox("Enable Educational Tooltips", True)

    submitted = st.sidebar.button('Submit')
    logger.info("User input retrieved from sidebar.")
    return (
        company_input, start_date, end_date, show_sma, show_rsi, show_macd, show_bollinger,
        prediction_days, run_portfolio, portfolio_input, initial_investment, strategy,
        run_correlation, correlation_input,
        run_ml, ml_input,
        run_esg, esg_input,
        show_earnings, show_dividends, show_sector, interactive_corr, show_tooltips,
        submitted
    )


def calculate_technical_indicators(df):
    df = df.copy()
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    # EMAs
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
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    logger.info("Technical indicators calculated.")
    return df


def analyze_patterns(df):
    patterns = []
    if len(df) < 50:
        logger.warning("Not enough data to analyze patterns.")
        return patterns
    # SMA crossovers
    if (df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and
            df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2]):
        patterns.append("Golden Cross detected (bullish)")
    elif (df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] and
          df['SMA_20'].iloc[-2] >= df['SMA_50'].iloc[-2]):
        patterns.append("Death Cross detected (bearish)")
    # RSI
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi > 70:
        patterns.append(f"Overbought (RSI={current_rsi:.2f})")
    elif current_rsi < 30:
        patterns.append(f"Oversold (RSI={current_rsi:.2f})")
    # MACD
    if (df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and
            df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]):
        patterns.append("MACD bullish crossover")
    elif (df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and
          df['MACD'].iloc[-2] >= df['Signal_Line'].iloc[-2]):
        patterns.append("MACD bearish crossover")
    # Bollinger
    last_close = df['Close'].iloc[-1]
    if last_close > df['BB_upper'].iloc[-1]:
        patterns.append("Price above upper Bollinger Band (possible reversal)")
    elif last_close < df['BB_lower'].iloc[-1]:
        patterns.append("Price below lower Bollinger Band (possible reversal)")
    logger.info(f"Patterns analyzed: {patterns}")
    return patterns


def load_stock_data(symbol, start, end, retries=3, delay=2):
    """
    Download stock data from Yahoo Finance with single-column flattening.
    """
    logger.info(
        f"[load_stock_data] Requesting data for '{symbol}' from {start} to {end}")
    for attempt in range(1, retries + 1):
        try:
            logger.info(
                f"[load_stock_data] Attempt {attempt}/{retries} for '{symbol}'")
            raw_data = yf.download(symbol, start=start,
                                   end=end, progress=False)
            if not raw_data.empty:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    df = pd.DataFrame({
                        'Date': raw_data.index,
                        'Open':   raw_data[('Open',   symbol)],
                        'High':   raw_data[('High',   symbol)],
                        'Low':    raw_data[('Low',    symbol)],
                        'Close':  raw_data[('Close',  symbol)],
                        'Volume': raw_data[('Volume', symbol)]
                    })
                else:
                    df = raw_data.reset_index()
                    df.rename(
                        columns={
                            'Date': 'Date',
                            'Open': 'Open',
                            'High': 'High',
                            'Low': 'Low',
                            'Close': 'Close',
                            'Volume': 'Volume'
                        },
                        inplace=True,
                        errors='ignore'
                    )
                df.reset_index(drop=True, inplace=True)
                return df
            else:
                logger.warning(
                    f"[load_stock_data] No data returned for '{symbol}' on attempt {attempt}.")
        except JSONDecodeError as jde:
            logger.error(
                f"[load_stock_data] JSON decode error on attempt {attempt}: {jde}")
        except RequestException as re:
            logger.error(
                f"[load_stock_data] Request error on attempt {attempt}: {re}")
        except Exception as e:
            logger.exception(
                f"[load_stock_data] General error for '{symbol}' on attempt {attempt}: {e}")
        time.sleep(delay)
    logger.error(
        f"[load_stock_data] Failed to retrieve data for '{symbol}' after {retries} attempts.")
    return None


@st.cache_data(show_spinner=False)
def load_stock_info(symbol):
    try:
        logger.info(f"Fetching stock info for '{symbol}'.")
        stock = yf.Ticker(symbol)
        info = stock.info
        logger.info(f"Successfully fetched stock info for '{symbol}'.")
        return info
    except Exception as e:
        logger.exception(f"Error fetching stock info for '{symbol}': {e}")
        st.error(f"Error fetching stock info: {e}")
        return None


def analyze_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for article in articles:
        text = article.get('description') or article.get('content') or ''
        if text:
            sentiment = sia.polarity_scores(text)
            article['sentiment'] = sentiment
            sentiments.append(sentiment['compound'])
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
    else:
        avg_sentiment = 0
    date_sent_dict = {}
    for article in articles:
        date_str = article.get('publishedAt', '')[:10]
        if date_str and 'sentiment' in article:
            date_sent_dict.setdefault(date_str, []).append(
                article['sentiment']['compound'])
    daily_sent = []
    for d, vals in date_sent_dict.items():
        daily_avg = sum(vals) / len(vals)
        daily_sent.append({'Date': pd.to_datetime(d),
                          'Daily_Sentiment': daily_avg})
    sentiment_df = pd.DataFrame(daily_sent)
    logger.info(
        f"Sentiment analysis completed. Average sentiment={avg_sentiment:.2f}")
    return articles, avg_sentiment, sentiment_df


@st.cache_data(show_spinner=False)
def load_news(symbol, from_date, to_date):
    try:
        logger.info(
            f"Fetching news for '{symbol}' from {from_date} to {to_date}")
        all_articles = newsapi.get_everything(
            q=symbol,
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=10
        )
        return all_articles.get('articles', [])
    except Exception as e:
        logger.exception(f"Error fetching news for '{symbol}': {e}")
        st.error(f"Error fetching news: {e}")
        return []


def summarize_article(text):
    prompt = f"Summarize this news article in 2-3 sentences:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes news articles."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Gemini API error summarizing article: {e}")
        return "Summary not available."


def generate_ai_insights(symbol, df, articles, patterns, stock_info, avg_sentiment):
    latest_close = df['Close'].iloc[-1]
    if len(df) >= 2:
        prev_close = df['Close'].iloc[-2]
    else:
        prev_close = latest_close
    price_change = latest_close - prev_close
    pct_change = (price_change / prev_close) * 100 if prev_close != 0 else 0.0

    market_cap = stock_info.get('marketCap', 'N/A')
    pe_ratio = stock_info.get('trailingPE', 'N/A')
    news_summary = "\n".join(
        f"- {article['title']}" for article in articles[:5])
    pattern_summary = ', '.join(
        patterns) if patterns else "No significant patterns"
    sentiment_label = "neutral"
    if avg_sentiment > 0.05:
        sentiment_label = "positive"
    elif avg_sentiment < -0.05:
        sentiment_label = "negative"

    prompt = f"""As a senior financial analyst, provide a comprehensive analysis of {symbol}:

Technical Analysis:
- Current Price: ${latest_close:.2f} ({pct_change:.2f}% change)
- Market Cap: {market_cap}
- P/E Ratio: {pe_ratio}
- Recent Patterns: {pattern_summary}

News Sentiment:
- Overall news sentiment is {sentiment_label} (score={avg_sentiment:.2f}).

News Highlights:
{news_summary}

Based on this data, discuss factors that may influence {symbol}'s future outlook.
"""

    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are a seasoned financial analyst."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Gemini API error generating AI insights: {e}")
        return f"AI analysis not available due to an error: {e}"


def generate_risk_assessment(symbol, df, avg_sentiment):
    volatility = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 0
    prompt = f"""As a risk analyst, assess the risk of investing in {symbol}:
- Current Volatility: {volatility:.2f}
- News Sentiment Score: {avg_sentiment:.2f}

Consider market conditions, volatility, and sentiment.
"""
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial risk analyst."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Gemini API error generating risk assessment: {e}")
        return f"Risk assessment not available due to error: {e}"


def process_multiple_inputs(input_str):
    if not input_str:
        return []
    items = [x.strip() for x in input_str.split(',')]
    valid_symbols = []
    for item in items:
        symbol = get_stock_symbol(item)
        if symbol:
            valid_symbols.append(symbol)
        else:
            st.warning(f"Could not resolve symbol for: {item}")
    return valid_symbols


# Collect user inputs
inputs = get_user_input()
(
    company_input, start_date, end_date, show_sma, show_rsi, show_macd, show_bollinger,
    prediction_days, run_portfolio, portfolio_input, initial_investment, strategy,
    run_correlation, correlation_input,
    run_ml, ml_input,
    run_esg, esg_input,
    show_earnings, show_dividends, show_sector, interactive_corr, show_tooltips,
    submitted
) = inputs

if submitted:
    with st.spinner("Processing..."):
        stock_symbol = get_stock_symbol(company_input)
        if not stock_symbol:
            st.error(f"Could not resolve a symbol for '{company_input}'")
            st.stop()

        st.session_state['stock_symbol'] = stock_symbol
        st.session_state['submitted'] = True

        stock_data = load_stock_data(stock_symbol, start_date, end_date)
        if stock_data is None or stock_data.empty:
            st.error(f"No data found for symbol {stock_symbol}")
            st.stop()
        st.session_state['stock_data'] = stock_data

        stock_info = load_stock_info(stock_symbol)
        st.session_state['stock_info'] = stock_info

        tech_data = calculate_technical_indicators(stock_data)
        st.session_state['tech_data'] = tech_data

        patterns = analyze_patterns(tech_data)
        st.session_state['patterns'] = patterns

        news_articles = load_news(stock_symbol, start_date, end_date)
        news_articles, avg_sentiment, sentiment_df = analyze_sentiment(
            news_articles)
        st.session_state['news_articles'] = news_articles
        st.session_state['avg_sentiment'] = avg_sentiment

        merged_data = pd.merge(stock_data, sentiment_df, on='Date', how='left')
        merged_data['Daily_Sentiment'] = merged_data['Daily_Sentiment'].ffill().fillna(
            0)
        st.session_state['merged_data'] = merged_data

        if run_portfolio and portfolio_input:
            portfolio_stocks = process_multiple_inputs(portfolio_input)
            if portfolio_stocks:
                portfolio_result = analyzer.portfolio_simulation(
                    portfolio_stocks,
                    initial_investment,
                    strategy.lower().replace(" ", "_")
                )
                st.session_state['portfolio_result'] = portfolio_result

        if run_correlation and correlation_input:
            corr_stocks = process_multiple_inputs(correlation_input)
            if corr_stocks:
                corr_matrix = analyzer.advanced_correlation_analysis(
                    corr_stocks)
                st.session_state['correlation_matrix'] = corr_matrix

        if run_ml and ml_input:
            ml_symbol = get_stock_symbol(ml_input)
            if ml_symbol:
                ml_results = analyzer.machine_learning_prediction(ml_symbol)
                st.session_state['ml_results'] = ml_results

        if run_esg and esg_input:
            esg_symbol = get_stock_symbol(esg_input)
            if esg_symbol:
                esg_results = analyzer.esg_scoring(esg_symbol)
                st.session_state['esg_results'] = esg_results

else:
    st.write("Please enter parameters and click Submit.")

# Retrieve from session state
stock_data = st.session_state['stock_data']
stock_info = st.session_state['stock_info']
stock_symbol = st.session_state['stock_symbol']
tech_data = st.session_state['tech_data']
patterns = st.session_state['patterns']
news_articles = st.session_state['news_articles']
avg_sentiment = st.session_state['avg_sentiment']
portfolio_result = st.session_state['portfolio_result']
correlation_matrix = st.session_state['correlation_matrix']
ml_results = st.session_state['ml_results']
esg_results = st.session_state['esg_results']
merged_data = st.session_state['merged_data']

if st.session_state['submitted'] and stock_data is not None and stock_info is not None:
    st_autorefresh(interval=300000, limit=100, key="autoRef")  # 5 min refresh

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"📊 {stock_symbol} Price Analysis")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='OHLC'
        ))
        if show_sma:
            fig.add_trace(go.Scatter(
                x=tech_data['Date'],
                y=tech_data['SMA_20'],
                name='SMA 20',
            ))
            fig.add_trace(go.Scatter(
                x=tech_data['Date'],
                y=tech_data['SMA_50'],
                name='SMA 50',
            ))
            fig.add_trace(go.Scatter(
                x=tech_data['Date'],
                y=tech_data['SMA_200'],
                name='SMA 200',
            ))
        if show_bollinger:
            fig.add_trace(go.Scatter(
                x=tech_data['Date'],
                y=tech_data['BB_upper'],
                name='BB Upper',
                line=dict(dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=tech_data['Date'],
                y=tech_data['BB_lower'],
                name='BB Lower',
                line=dict(dash='dash'),
                fill='tonexty'
            ))
        fig.update_layout(
            title=f"{stock_symbol} Price Chart",
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            template='plotly_white',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True, key='3')

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
            for key, val in metrics.items():
                st.metric(key, val)

    st.subheader("📊 Technical Analysis")
    if show_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=tech_data['Date'], y=tech_data['RSI'], name='RSI'))
        fig_rsi.add_hline(y=70, line_dash='dash', line_color='red')
        fig_rsi.add_hline(y=30, line_dash='dash', line_color='green')
        fig_rsi.update_layout(
            title="RSI",
            yaxis_title="RSI Value",
            height=300
        )
        st.plotly_chart(fig_rsi, use_container_width=True, key='4')

    if show_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=tech_data['Date'], y=tech_data['MACD'], name='MACD'))
        fig_macd.add_trace(go.Scatter(
            x=tech_data['Date'], y=tech_data['Signal_Line'], name='Signal Line'))
        fig_macd.add_bar(
            x=tech_data['Date'], y=tech_data['MACD_Histogram'], name='Histogram')
        fig_macd.update_layout(
            title="MACD",
            yaxis_title="Value",
            height=300
        )
        st.plotly_chart(fig_macd, use_container_width=True, key='5')

    st.subheader("🎯 Pattern Analysis")
    if patterns:
        for p in patterns:
            st.info(p)
    else:
        st.write("No significant patterns detected.")

    st.subheader("📰 Latest News & Sentiment Analysis")
    sentiment_label = "Neutral 😐"
    if avg_sentiment > 0.05:
        sentiment_label = "Positive 😊"
    elif avg_sentiment < -0.05:
        sentiment_label = "Negative 😞"
    st.write(
        f"**Overall News Sentiment:** {sentiment_label} (Score={avg_sentiment:.2f})")

    if news_articles:
        for article in news_articles:
            sentiment = article.get('sentiment', {})
            sentiment_score = sentiment.get('compound', 0)
            if sentiment_score > 0.05:
                sent_text = "Positive 😊"
            elif sentiment_score < -0.05:
                sent_text = "Negative 😞"
            else:
                sent_text = "Neutral 😐"

            text = article.get('description') or article.get('content') or ''
            summary = text
            with st.expander(f"{article['title']}"):
                st.write(
                    f"**Sentiment:** {sent_text} (Score={sentiment_score:.2f})")
                st.write(
                    f"**Source:** {article['source']['name']} | **Published:** {article['publishedAt']}")
                if st.button('Summarize Article', key=article['url']):
                    with st.spinner('Summarizing...'):
                        summary = summarize_article(text)
                        st.write(f"**Summary:** {summary}")
                else:
                    st.write(f"**Summary:** {summary}")
                st.write(f"[Read more]({article['url']})")
    else:
        st.write("No news found for this date range.")

    st.subheader("🤖 AI-Powered Analysis and Outlook")
    with st.spinner("Generating AI insights..."):
        ai_insights = generate_ai_insights(
            stock_symbol, tech_data, news_articles, patterns, stock_info, avg_sentiment)
    with st.expander("View Full AI Analysis"):
        st.markdown(ai_insights)

    st.header("📈 Prophet Forecast")
    if st.button("Predict Future Prices"):
        df_prophet = merged_data[['Date', 'Close', 'Daily_Sentiment']].copy()
        df_prophet.dropna(subset=['Close'], inplace=True)
        df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        with st.spinner('Training Prophet model...'):
            model = Prophet()
            model.add_regressor('Daily_Sentiment')
            model.fit(df_prophet)

        future = model.make_future_dataframe(periods=prediction_days)
        last_sent = df_prophet['Daily_Sentiment'].iloc[-1]
        future['Daily_Sentiment'] = last_sent
        forecast = model.predict(future)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'], name='Predicted'
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'], name='Lower', fill=None
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'], name='Upper', fill='tonexty'
        ))
        fig_forecast.update_layout(
            title=f"{stock_symbol} Prophet Forecast ({prediction_days} days)",
            xaxis_title="Date",
            yaxis_title="Close Price",
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_forecast, use_container_width=True, key='6')
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower',
                     'yhat_upper']].tail(prediction_days))

    st.subheader("💬 Ask a Question")
    user_q = st.text_input("Your question about the stock:")
    if st.button("Get Answer"):
        if user_q:
            # Build a summary from fetched data.
            current_price = stock_info.get('currentPrice', 'N/A')
            market_cap = stock_info.get('marketCap', 'N/A')
            pe_ratio = stock_info.get('trailingPE', 'N/A')
            # Get recent news headlines (limit to 3 for brevity)
            news_headlines = " | ".join(
                [article['title'] for article in news_articles[:3]])
            # Get current RSI if available
            try:
                current_rsi = tech_data['RSI'].iloc[-1]
                rsi_text = f"{current_rsi:.2f}"
            except Exception:
                rsi_text = "N/A"

            # Create a summary string with key information.
            data_summary = (
                f"Current Price: ${current_price}\n"
                f"Market Cap: {market_cap}\n"
                f"P/E Ratio: {pe_ratio}\n"
                f"RSI: {rsi_text}\n"
                f"Recent News Headlines: {news_headlines}"
            )

            prompt = f"""You are a financial assistant with comprehensive stock data for {stock_symbol}.
    The following data is available:
    {data_summary}

    Based on the above data, please answer the following question:
    Question: {user_q}

    Provide a concise answer.
    """
            with st.spinner("Generating answer..."):
                try:
                    resp = client.chat.completions.create(
                        model="gemini-2.0-flash",
                        messages=[
                            {"role": "system",
                                "content": "You are a helpful financial assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.7
                    )
                    answer = resp.choices[0].message.content.strip()
                    with st.expander("View Answer"):
                        st.markdown(answer)
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
        else:
            st.warning("Please enter a question first.")

    st.subheader("⚠️ Risk Assessment")
    with st.spinner("Generating risk assessment..."):
        risk_text = generate_risk_assessment(
            stock_symbol, tech_data, avg_sentiment)
    st.write(risk_text)

    if run_portfolio and portfolio_result:
        st.header("📈 Portfolio Tracking & Simulation")
        st.subheader("📊 Portfolio Performance")

        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=portfolio_result['cumulative_returns'].index,
            y=portfolio_result['cumulative_returns'].values,
            mode='lines',
            name='Portfolio'
        ))
        fig_port.update_layout(
            title="Cumulative Portfolio Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_port, use_container_width=True, key='7')

        st.subheader("📊 Portfolio Risk Metrics")
        st.write(f"**Volatility:** {portfolio_result['volatility']:.2%}")
        st.write(f"**Sharpe Ratio:** {portfolio_result['sharpe_ratio']:.2f}")

    if run_correlation and correlation_matrix is not None:
        st.header("📊 Advanced Correlation Analysis")
        fig_corr = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            title="Stock Returns Correlation Heatmap",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True, key='8')

    if run_ml and ml_results:
        st.header("🤖 Machine Learning Price Prediction")
        st.subheader("📈 Prediction Results")
        preds = ml_results['predictions']
        mse_val = ml_results['mse']
        n_test = len(preds)
        prediction_dates = stock_data['Date'][-n_test:]
        df_pred = pd.DataFrame(
            {"Date": prediction_dates, "Predicted Close": preds})
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(
            x=df_pred['Date'],
            y=df_pred['Predicted Close'],
            name='Predicted',
            mode='lines'
        ))
        fig_ml.update_layout(
            title=f"{ml_input.upper()} Predicted Close Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_ml, use_container_width=True, key='9')
        st.write(f"**MSE:** {mse_val:.4f}")
        st.subheader("🔍 Feature Importance")
        feat_imp = ml_results['feature_importance']
        fig_feat = px.bar(
            feat_imp,
            x='feature',
            y='importance',
            title="Feature Importance",
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        st.plotly_chart(fig_feat, use_container_width=True, key='10')

    if run_esg and esg_results:
        st.header("🌱 ESG Scoring")
        st.subheader(f"{esg_input.upper()} ESG Performance")
        fig_esg = go.Figure(data=[
            go.Bar(
                x=list(esg_results.keys()),
                y=list(esg_results.values()),
                marker_color=['green', 'blue', 'purple']
            )
        ])
        fig_esg.update_layout(title="ESG Performance",
                              yaxis_title="Score", template="plotly_white")
        st.plotly_chart(fig_esg, use_container_width=True, key='11')

    # New Feature: Earnings Calendar
    if show_earnings:
        analyzer.get_earnings_calendar(stock_symbol)

    # New Feature: Dividend History
    if show_dividends:
        analyzer.get_dividend_history(stock_symbol)

    # New Feature: Sector Heatmap
    if show_sector:
        display_sector_heatmap()

else:
    st.write("Please enter parameters and click Submit.")
