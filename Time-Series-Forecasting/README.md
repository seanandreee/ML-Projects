# Time Series Forecasting with LSTM

Authored by Sean Andre Membrido

This project implements a time series forecasting model using an LSTM (Long Short-Term Memory) neural network. The model is trained on historical stock price data and used to predict future stock prices.


## Project Structure

- `data/`: Directory to store the stock price dataset (e.g., NVDA stock data).
- `market-prediction.py`: Main script to download data, preprocess, define, train, evaluate the LSTM model, and forecast future stock prices.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- yfinance
- seaborn
- scikit-learn

You can install the required packages using pip:

```bash
pip install yfinance numpy pandas matplotlib seaborn scikit-learn torch
