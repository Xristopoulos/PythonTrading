# PythonTrading
This is a Python-based trading tool I developed that leverages AI and a data-driven approach to make informed decisions on when to enter and exit trades. By using advanced algorithms and machine learning techniques, the tool analyzes market data to optimize trading strategies, aiming to make smarter and more profitable trades.


Trading Strategies Using Downloaded Candles

Overview

This set of Python scripts allows you to download historical candlestick data, apply trading strategies, and backtest the performance. The overall workflow involves:

Downloading candle data from a cryptocurrency exchange using Download_Candles.py.
Testing a simple buy-and-hold strategy or a more advanced long/short strategy using the downloaded data.
The dt-Test.py file implements a simple buy and hold strategy until a good opportunity arises to sell.
The LongShort_demo.py implements a strategy that takes both long and short positions based on multiple technical indicators.

Parameters have already been optimized using Optuna. These are the most suitable parameter values for BTC/USDT for each respective program.

Requirements
Python 3.x
Libraries:
ccxt: for interacting with the Binance API to download market data.
pandas: for data manipulation.
numpy: for numerical computations.
matplotlib: for plotting graphs.


Step 1: Download Candles
The Download_Candles.py script is responsible for downloading cryptocurrency candlestick data for a given symbol (like BTC/USDT) over a period of one year.

Usage
i) Modify the list of coins inside Download_Candles.py to include the desired cryptocurrency pairs (e.g., ['BTC/USDT', 'XRP/USDT']).
ii) Run the script

This will download the 1-minute OHLCV (Open, High, Low, Close, Volume) data for the specified coins and save them as CSV files in the same directory. If the data already exists, it will load from the cache.

Step 2: Backtest with dt-Test.py (Buy and Hold Strategy)
The dt-Test.py script implements a basic buy-and-hold strategy. It monitors market conditions and executes buy and sell actions based on specific conditions, such as technical indicators like RSI, MACD, and Stochastic Oscillator.

Usage
i) Load the downloaded data in dt-Test.py.
The strategy runs on the available data with a set of parameters for entering and exiting positions.
ii) Run the script

This will backtest the strategy and provide insights such as total profits, win rates, and drawdowns.

Step 3: Backtest with LongShort_demo.py (Long and Short Strategy)
The LongShort_demo.py script implements a more advanced trading strategy that uses both long and short positions. It applies multiple technical indicators such as RSI, Stochastic, ADX, and Bollinger Bands to decide when to enter and exit trades.

Key Components
Indicators: It computes several technical indicators including RSI, Stochastic Oscillator, ADX, and Bollinger Bands.
Position Type: The strategy can take both long (buy) and short (sell) positions.
Backtesting: The script runs backtests to calculate potential profits, win rates, and other performance metrics.
Usage
i) Load the downloaded market data (make sure Download_Candles.py has been run and saved the data).
ii) Customize parameters within LongShort_demo.py, like risk per trade, leverage, stop loss, and take profit levels.
iii) Run the script:

The script will backtest the long/short strategy and output the results, including equity curves and trade logs.







I developed this project as a university student with a deep interest in AI and quantitative trading. It’s more than just a set of technical scripts—it's a way for me to show off my skills, my dedication to always learning, and my excitement for algorithmic trading. Using Optuna for parameter optimization, along with advanced trading strategies and machine learning methods, really shows my passion for finding new ways to approach finance. I hope this project not only helps me learn but also inspires others who want to bring AI into the world of financial markets.




