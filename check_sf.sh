#!/bin/bash

echo "Checking yfinance version..."
python -c "import yfinance as yf; print('yfinance version:', yf.__version__)"

echo "Attempting to download AAPL data for 1 day..."
python -c "import yfinance as yf; data = yf.download('AAPL', period='1d'); print(data)"

