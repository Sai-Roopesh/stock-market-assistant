# 📈 Stock Insights Dashboard

A Streamlit-based dashboard that provides insights into stocks by fetching historical data and recent news articles. Users can visualize stock price trends, view relevant news, and download the stock data in CSV format.

## Features

- Fetch stock data using `yfinance`
- Display historical stock prices with interactive plots
- Fetch recent news articles related to the stock symbol using the NewsAPI
- Download stock data as a CSV file
- User-friendly interface with customizable stock symbol and date range

## Installation

### Requirements

- Python 3.9 or above
- Conda (for managing the environment)

### Setting up the environment

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a Conda environment:**

   ```bash
   conda create --name stock_dashboard python=3.9
   conda activate stock_dashboard
   ```

3. **Install the required packages:**

   All the required packages are listed in the `requirements.txt` file.

   **Using Conda**:

   Install the packages available on `conda-forge` and `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   ```bash
   conda install -c conda-forge yfinance
   ```

4. **Set up the environment variables:**

   Create a `.env` file in the project root and add your NewsAPI key:

   ```bash
   NEWSAPI_KEY=your_newsapi_key_here
   ```

5. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter a stock symbol (e.g., AAPL for Apple) and specify the date range in the sidebar.
2. View historical stock prices in the dashboard.
3. Fetch and read recent news articles related to the stock.
4. Download the stock data as a CSV file for further analysis.

## Dependencies

The project depends on the following libraries:

- `yfinance`: For fetching historical stock data
- `pandas`: Data manipulation
- `matplotlib`: Plotting stock price trends
- `newsapi-python`: Fetching news articles from NewsAPI
- `python-dotenv`: For managing environment variables
- `streamlit`: Web framework for the dashboard

## Note

To install `yfinance`, ensure you are using the following command to avoid issues:

```bash
conda install -c conda-forge yfinance
```
