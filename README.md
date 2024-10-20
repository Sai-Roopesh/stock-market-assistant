# 📈 Advanced Stock Insights Dashboard

Welcome to the **Advanced Stock Insights Dashboard**! This powerful Streamlit application provides comprehensive stock analysis, combining financial data, technical indicators, news sentiment, and AI-driven insights to help you make informed investment decisions.

## Table of Contents

- [📈 Advanced Stock Insights Dashboard](#-advanced-stock-insights-dashboard)
  - [Table of Contents](#table-of-contents)
  - [🔍 Features](#-features)
  - [🚀 Demo](#-demo)
  - [🛠️ Installation](#️-installation)
  - [🔧 Setup](#-setup)
  - [💡 Usage](#-usage)
  - [📚 Technologies Used](#-technologies-used)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [📞 Contact](#-contact)

## 🔍 Features

- **Stock Data Retrieval**: Fetches real-time and historical stock data using Yahoo Finance.
- **Technical Analysis**: Calculates and visualizes key technical indicators like SMA, RSI, MACD, Bollinger Bands, and more.
- **Pattern Recognition**: Detects trading patterns such as Golden Cross, Death Cross, and RSI signals.
- **News Integration**: Retrieves the latest news articles related to the selected stock and performs sentiment analysis.
- **AI-Powered Insights**: Generates comprehensive analysis and outlooks using OpenAI's GPT models.
- **Interactive Visualizations**: Provides dynamic and interactive charts using Plotly for in-depth data exploration.
- **User-Friendly Interface**: Intuitive sidebar for customizing analysis parameters and easy navigation.

## 🚀 Demo

![Dashboard Screenshot](./screenshots/dashboard.png)

*Note: Replace the above image path with actual screenshots of your application.*

## 🛠️ Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure you have Python installed. You can download it from [here](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/advanced-stock-insights-dashboard.git
cd advanced-stock-insights-dashboard
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **macOS and Linux:**

  ```bash
  source venv/bin/activate
  ```

### Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## 🔧 Setup

### 1. Obtain API Keys

The application requires API keys for **NewsAPI** and **OpenAI**. Follow the steps below to obtain them:

- **NewsAPI Key:**
  - Sign up for a free account at [NewsAPI](https://newsapi.org/).
  - Navigate to your account dashboard to retrieve your API key.

- **OpenAI API Key:**
  - Sign up or log in to your account at [OpenAI](https://platform.openai.com/).
  - Navigate to the API section to generate your API key.

### 2. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your API keys:

```bash
NEWSAPI_KEY=your_newsapi_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Example `.env` File:**

```
NEWSAPI_KEY=abcd1234efgh5678ijkl9012mnop3456
OPENAI_API_KEY=sk-YourOpenAIKeyHere
```

*Ensure that you replace the placeholder values with your actual API keys.*

### 3. Download NLTK Data

The application uses NLTK's VADER for sentiment analysis. The necessary data is downloaded automatically when you run the app for the first time.

## 💡 Usage

Run the Streamlit application using the following command:

```bash
streamlit run main.py
```

This command will start the application and open it in your default web browser. If it doesn't open automatically, navigate to [http://localhost:8501](http://localhost:8501) in your browser.

### Using the Dashboard

1. **Company Selection:**
   - Use the sidebar to input the company name or stock symbol you wish to analyze (e.g., "Apple Inc." or "AAPL").

2. **Time Range:**
   - Select a predefined time range (e.g., 1 Week, 1 Month) or choose a custom date range for analysis.

3. **Technical Indicators:**
   - Toggle the visibility of various technical indicators such as SMA, RSI, MACD, and Bollinger Bands.

4. **View Analysis:**
   - The main dashboard displays interactive charts, technical analysis, pattern detection, latest news with sentiment analysis, AI-powered insights, and risk assessments.

5. **Ask Questions:**
   - Use the Q&A section to ask specific questions about the selected stock. The AI assistant will provide informative answers based on the available data.

## 📚 Technologies Used

- **[Streamlit](https://streamlit.io/)**: Framework for building interactive web applications.
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis.
- **[NumPy](https://numpy.org/)**: Numerical computing.
- **[yfinance](https://pypi.org/project/yfinance/)**: Accessing Yahoo Finance data.
- **[NewsAPI](https://newsapi.org/)**: Fetching news articles.
- **[NLTK](https://www.nltk.org/)**: Natural Language Processing for sentiment analysis.
- **[OpenAI API](https://platform.openai.com/docs/api-reference/introduction)**: Generating AI-powered insights and Q&A.
- **[Plotly](https://plotly.com/python/)**: Interactive data visualizations.
- **[Matplotlib](https://matplotlib.org/)** & **[Seaborn](https://seaborn.pydata.org/)**: Additional plotting libraries.
- **[dotenv](https://pypi.org/project/python-dotenv/)**: Managing environment variables.

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

