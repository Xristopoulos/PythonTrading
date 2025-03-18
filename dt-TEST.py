import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Binance Trading Fees & Arbitrary Slippage
BINANCE_FEE = 0.001  # 0.1% per trade
SLIPPAGE = 0.0005  # 0.05% per trade

# File to cache the downloaded data

ORDERS_FILE = 'orders.txt'
DEBUG_FILE = 'debug.txt'


def add_indicators(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # 1) Existing Indicators: RSI, MACD, Stoch
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(7) / 7, mode='valid')
    avg_loss = np.convolve(loss, np.ones(7) / 7, mode='valid')
    avg_loss[avg_loss == 0] = 1e-10  # Prevent divide-by-zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = np.append([np.nan] * (7 - 1), rsi)

    # MACD (6, 13, 5)
    ema6 = pd.Series(close).ewm(span=6, adjust=False).mean().values
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    macd = ema6 - ema13
    macd_signal = pd.Series(macd).ewm(span=5, adjust=False).mean().values
    df['macd'] = macd
    df['macd_signal'] = macd_signal

    # Stochastic Oscillator (5, 3, 3)
    low_min = pd.Series(low).rolling(window=5).min().values
    high_max = pd.Series(high).rolling(window=5).max().values
    stoch_k = 100 * (close - low_min) / (high_max - low_min)
    stoch_d = pd.Series(stoch_k).rolling(window=3).mean().values
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    # 2) Bollinger Bands (20 period, ±2 std)
    df['bb_middle'] = pd.Series(close).rolling(window=20).mean()
    df['bb_std'] = pd.Series(close).rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # 3) ATR & ADX for Trend Strength
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['up_move'] = df['high'].diff()
    df['down_move'] = df['low'].diff() * -1
    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'], 0
    )
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'], 0
    )

    period = 14
    df['plus_dm_sm'] = df['plus_dm'].rolling(period).mean()
    df['minus_dm_sm'] = df['minus_dm'].rolling(period).mean()
    df['tr_sm'] = df['tr'].rolling(period).mean()

    df['plus_di'] = (df['plus_dm_sm'] / df['tr_sm']) * 100
    df['minus_di'] = (df['minus_dm_sm'] / df['tr_sm']) * 100
    df['dx'] = (np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
    df['adx'] = df['dx'].rolling(period).mean()

    return df


def apply_trading_fees_and_slippage(price, trade_type='buy'):

    if trade_type == 'buy':
        return price * (1 + BINANCE_FEE + SLIPPAGE)
    elif trade_type == 'sell':
        return price * (1 - BINANCE_FEE - SLIPPAGE)
    return price


def backtest(df):

    initial_balance = 1000.0
    cash_balance = initial_balance

    # Single-position tracking
    in_position = False
    entry_price = 0.0
    position_size = 0.0
    allocated = 0.0  # cost of open trade (entry cost)
    tp_level = 0.0
    entry_time = None

    # Metrics
    trade_count = 0
    profits = []
    trade_durations = []

    # For plotting
    cash_balance_history = []
    position_value_history = []
    real_balance_history = []

    # Track the highest close so far (ATH)
    ath_so_far = 0.0

    # optimization results
    rsi_base_threshold = 46.151105287033765
    rsi_strict_threshold = 44.06544846876969
    stoch_base_threshold = 52.488355330728524
    stoch_strict_threshold = 44.783943023110496
    adx_threshold = 18.453930054990742
    tp_percentage = 2.999278809950785
    near_ath_factor = 0.8903608440958564

    print(f"Total Candles Loaded: {len(df)}")

    with open(ORDERS_FILE, 'w') as orders_file, open(DEBUG_FILE, 'w') as debug_file:
        for i in range(len(df)):
            if i < 200:
                ath_so_far = max(ath_so_far, df['close'][i])
                continue

            current_time = df['timestamp'][i]
            current_price = df['close'][i]
            ath_so_far = max(ath_so_far, current_price)

            # Calculate position_value if in trade
            if in_position:
                position_value = position_size * current_price
            else:
                position_value = 0.0

            real_balance = cash_balance + position_value

            # Save balances for plotting
            cash_balance_history.append(cash_balance)
            position_value_history.append(position_value)
            real_balance_history.append(real_balance)

            # 1) Check exit condition if in position (TP reached)
            if in_position:
                if current_price >= tp_level:
                    sell_price = apply_trading_fees_and_slippage(current_price, 'sell')
                    proceeds = position_size * sell_price
                    cash_balance += proceeds  # realize gains

                    pnl = proceeds - allocated
                    profits.append(pnl)

                    duration = pd.to_datetime(current_time) - pd.to_datetime(entry_time)
                    trade_durations.append(duration)

                    pnl_percent = (pnl / allocated) * 100 if allocated != 0 else 0

                    orders_file.write(
                        f"Trade from {entry_time} to {current_time}: "
                        f"Buy={entry_price:.2f}, Sell={sell_price:.2f}, "
                        f"PnL={pnl:.2f}, PnL%={pnl_percent:.2f}%\n"
                    )
                    debug_file.write(
                        f"EXIT: Entered={entry_time}, Exit={current_time}, "
                        f"SellPrice={sell_price:.2f}, PnL={pnl:.2f}\n"
                    )

                    # Reset position
                    in_position = False
                    entry_price = 0.0
                    position_size = 0.0
                    allocated = 0.0
                    tp_level = 0.0
                    entry_time = None
                    continue

            # 2) If not in position, check entry conditions
            else:
                # near_ATH: current price within the specified factor of ATH
                near_ath = (current_price >= near_ath_factor * ath_so_far)
                base_condition = (
                    (df['rsi'][i] < rsi_base_threshold) and
                    (df['stoch_k'][i] < stoch_base_threshold) and
                    (df['stoch_d'][i] < stoch_base_threshold) and
                    (df['macd'][i] > df['macd_signal'][i]) and
                    (df['adx'][i] > adx_threshold) and
                    (current_price < df['bb_middle'][i])
                )
                stricter_condition = (
                    (df['rsi'][i] < rsi_strict_threshold) and
                    (df['stoch_k'][i] < stoch_strict_threshold) and
                    (df['stoch_d'][i] < stoch_strict_threshold) and
                    (df['macd'][i] > df['macd_signal'][i]) and
                    (df['adx'][i] > adx_threshold) and
                    (current_price < df['bb_middle'][i])
                )
                can_enter = stricter_condition if near_ath else base_condition

                if can_enter:
                    allocated = cash_balance
                    if allocated <= 0:
                        continue

                    buy_price = apply_trading_fees_and_slippage(current_price, 'buy')
                    position_size = allocated / buy_price
                    entry_price = buy_price
                    entry_time = current_time
                    in_position = True
                    trade_count += 1

                    cash_balance -= allocated  # allocate funds

                    # Use the optimized take profit percentage:
                    tp_level = buy_price * (1 + tp_percentage / 100)

                    orders_file.write(
                        f"Buy at {buy_price:.2f} on {current_time}, allocated={allocated:.2f}\n"
                    )
                    debug_file.write(
                        f"ENTRY: time={current_time}, RSI={df['rsi'][i]:.2f}, "
                        f"StochK={df['stoch_k'][i]:.2f}, StochD={df['stoch_d'][i]:.2f}, "
                        f"MACD={df['macd'][i]:.2f}, ADX={df['adx'][i]:.2f}, "
                        f"nearATH={near_ath}, close={current_price:.2f}, "
                        f"BB_mid={df['bb_middle'][i]:.2f}, BuyPrice={buy_price:.2f}, "
                        f"Allocated={allocated:.2f}, TP={tp_level:.2f}\n"
                    )

        # End of main loop

    # Final balances and performance metrics are printed below...
    final_cash_balance = cash_balance
    if in_position:
        final_position_value = position_size * df['close'].iloc[-1]
        final_real_balance = allocated  # Reflect the entry cost if still in a trade
        unrealized_pnl = final_position_value - allocated
    else:
        final_position_value = 0.0
        final_real_balance = cash_balance
        unrealized_pnl = 0.0

    total_profit = final_real_balance - initial_balance

    print(f"Total Trades Executed: {trade_count}")
    print(f"Final Cash Balance: {final_cash_balance:.2f}")
    print(f"Final Position Value: {final_position_value:.2f} (Unrealized PNL: {unrealized_pnl:.2f})")
    print(f"Final Real Balance: {final_real_balance:.2f} (Total PNL: {total_profit:.2f})")

    if profits:
        wins = sum(1 for p in profits if p > 0)
        win_rate = (wins / len(profits)) * 100
    else:
        win_rate = 0.0
    print(f"Win Rate: {win_rate:.2f}%")

    max_drawdown = min(profits) if profits else 0
    print(f"Max Drawdown: {max_drawdown:.2f}")
    avg_duration = (np.mean(trade_durations).total_seconds() / 60) if trade_durations else 0
    print(f"Average Time Per Trade: {avg_duration:.2f} minutes")

    # Plot the balance histories
    timestamps = pd.to_datetime(df['timestamp'])[200: 200 + len(cash_balance_history)]
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, cash_balance_history, label='Cash Balance')
    plt.plot(timestamps, position_value_history, label='Position Value', linestyle=':')
    plt.plot(timestamps, real_balance_history, label='Real Balance', linestyle='--')
    plt.xlabel("Timestamp")
    plt.ylabel("Balance")
    plt.title("Single-Position Backtest (Optimized Parameters)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    return profits

# -------------------------------------------
CACHE_FILE = ['btc_usdt_1m_1yr.csv']

for cache in CACHE_FILE:
    df = pd.read_csv(cache)
    df = add_indicators(df)
    print(f"Total Candles Loaded: {len(df)} for {cache}")
    df_used = df.iloc[200:]  # skipping first 200 for indicator warm-up
    print(f"Total Candles Used in Backtest: {len(df_used)}")

    profits = backtest(df)
