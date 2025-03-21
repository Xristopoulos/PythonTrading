import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures

# Generalized function to fetch candle data for any coin
def fetch_coin_data(symbol):
    CACHE_FILE = f'{symbol.replace("/", "_").lower()}_1m_1yr.csv'

    # Check if cached data exists
    if os.path.exists(CACHE_FILE):
        print(f"Loading {symbol} data from cache...")
        df = pd.read_csv(CACHE_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    exchange = ccxt.binance()

    # Calculate the timestamp for 1 year ago (365 days)
    one_year_ago = datetime.now() - timedelta(days=365)
    start_timestamp = int(one_year_ago.timestamp() * 1000)  # Convert to milliseconds

    # Fetch the latest candle to determine the current timestamp
    latest_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1)
    latest_timestamp = latest_ohlcv[0][0]  # Latest candle's timestamp in ms

    # Fetch the data in chunks using pagination
    limit = 1000  # Number of candles per request
    all_ohlcv = []
    since = start_timestamp

    print(f"Downloading {symbol} data from 1 year ago to present...")
    while since < latest_timestamp:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=limit, since=since)
        if not ohlcv:
            break  # No more data available
        all_ohlcv.extend(ohlcv)

        since = ohlcv[-1][0] + 60 * 1000
        print(f"Downloaded {len(all_ohlcv)} candles for {symbol}...")

    # Create DataFrame and convert timestamp column to datetime
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Calculate the percentage of candles downloaded over the specified period
    total_available_candles = (latest_timestamp - start_timestamp) // (60 * 1000)
    downloaded_candles = len(df)
    percentage_downloaded = (downloaded_candles / total_available_candles) * 100

    print(f"Total available candles (last year): {total_available_candles}")
    print(f"Candles downloaded: {downloaded_candles}")
    print(f"Percentage downloaded: {percentage_downloaded:.2f}%")

    # Save the data to a cache file
    df.to_csv(CACHE_FILE, index=False)
    print(f"Data saved to {CACHE_FILE}")

    return df

# Function to run download for multiple coins in parallel
def download_multiple_coins(coins):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_coin_data, coins))

    # Print the result for each coin
    for coin, df in zip(coins, results):
        print(f"Data for {coin} downloaded and saved.")

# List of coins
coins = ['BTC/USDT','XRP/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOGE/USDT', 'TRX/USDT', 'PEPE/USDT']  # List of coins you want to download data for

download_multiple_coins(coins)
