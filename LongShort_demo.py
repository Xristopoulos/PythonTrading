import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


# Define PositionType Enum
class PositionType(Enum):
    LONG = "long"
    SHORT = "short"


# **Original Indicator Functions**
def compute_rsi(series, period=14):
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
    avg_loss[avg_loss == 0] = 1e-10
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate(([np.nan] * (period - 1), rsi))


def compute_adx(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    plus_di = 100 * (plus_dm.rolling(period).sum() / tr.rolling(period).sum())
    minus_di = 100 * (minus_dm.rolling(period).sum() / tr.rolling(period).sum())
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(period).mean()
    return adx


def add_indicators(df):
    df['rsi'] = compute_rsi(df['close'].values, period=14)
    df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) /
                     (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['adx'] = compute_adx(df)
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
    return df


# **Liquidation Price Calculation**
def calculate_liquidation_price(entry_price, leverage, position_type=PositionType.LONG):
    if position_type == PositionType.LONG:
        return entry_price * (1 - (1 / leverage))
    elif position_type == PositionType.SHORT:
        return entry_price * (1 + (1 / leverage))


# **Extracted Trade Logic Functions**
def should_enter_trade(df, i, position_type, params):
    price = df['close'].iloc[i]
    if position_type == PositionType.LONG:
        return (df['rsi'].iloc[i] < params['rsi_base_threshold'] and
                df['stoch_k'].iloc[i] < params['stoch_base_threshold'] and
                df['adx'].iloc[i] > params['adx_threshold'] and
                price < df['bb_middle'].iloc[i])
    elif position_type == PositionType.SHORT:
        return (df['rsi'].iloc[i] > params['short_rsi_base_threshold'] and
                df['stoch_k'].iloc[i] > params['short_stoch_base_threshold'] and
                df['adx'].iloc[i] > params['short_adx_threshold'] and
                price > df['bb_middle'].iloc[i])
    return False


def enter_trade(df, i, balance, position_type, params):
    price = df['close'].iloc[i]
    risk_key = 'risk_per_trade' if position_type == PositionType.LONG else 'short_risk_per_trade'
    leverage_key = 'leverage' if position_type == PositionType.LONG else 'short_leverage'

    margin_per_trade = balance * params[risk_key]
    if margin_per_trade > balance:
        return None, None, None, None

    allocated = margin_per_trade
    position_size = (allocated * params[leverage_key]) / price
    entry_price = price
    liquidation_price = calculate_liquidation_price(entry_price, params[leverage_key], position_type)

    notional_value = position_size * entry_price
    entry_fee = notional_value * params.get('fee_rate', 0)

    if position_type == PositionType.LONG:
        tp_level = entry_price * (1 + (params['tp_percentage'] * params['near_ath_factor']) / 100)
        stop_loss_level = entry_price * (1 - params['stop_loss_percent'] / 100)
        if stop_loss_level >= liquidation_price:
            print(f"Warning: Stop-loss above liquidation price! Adjusting SL.")
            stop_loss_level = liquidation_price * 0.99
    else:  # SHORT
        tp_level = entry_price * (1 - (params['short_tp_percentage'] * params['short_near_ath_factor']) / 100)
        stop_loss_level = entry_price * (1 + params['short_stop_loss_percent'] / 100)
        if stop_loss_level <= liquidation_price:
            print(f"Warning: Stop-loss below liquidation price! Adjusting SL.")
            stop_loss_level = liquidation_price * 1.01

    return position_size, entry_price, tp_level, stop_loss_level, liquidation_price, allocated, entry_fee


# **Backtesting Function with Fixed Fee Deduction**
def backtest_fixed_leverage(df, params):
    INITIAL_BALANCE = params['initial_balance']
    balance = INITIAL_BALANCE
    in_position = False
    trade_count, wins, losses = 0, 0, 0
    equity_curve = []
    order_log = []
    trade_timestamps = []
    total_fees = 0
    gross_profit_loss = 0

    for i in range(20, len(df)):
        price = df['close'].iloc[i]
        timestamp = df.index[i]

        if in_position:
            min_max_price = min(df['close'].iloc[i - 1:i + 1].min(),
                                price) if position_type == PositionType.LONG else max(
                df['close'].iloc[i - 1:i + 1].max(), price)
            if ((position_type == PositionType.LONG and (
                    price >= tp_level or price <= stop_loss_level or price <= liquidation_price)) or
                    (position_type == PositionType.SHORT and (
                            price <= tp_level or price >= stop_loss_level or price >= liquidation_price))):
                sell_price = price
                proceeds = position_size * sell_price
                exit_fee = proceeds * params.get('fee_rate', 0)
                total_fees += exit_fee
                gross_pl = proceeds - (position_size * entry_price) if position_type == PositionType.LONG else (
                                                                                                                           position_size * entry_price) - proceeds
                gross_profit_loss += gross_pl
                profit_loss = gross_pl - exit_fee
                balance += allocated + profit_loss
                wins += 1 if profit_loss > 0 else 0
                losses += 1 if profit_loss < 0 else 0
                order_log.append(
                    f"EXIT: time={timestamp}, SellPrice={sell_price:.2f}, GrossPL={gross_pl:.2f}, Profit/Loss={profit_loss:.2f}, FeesPaid={entry_fee + exit_fee:.2f}, New Balance={balance:.2f}")
                in_position = False

        else:
            for pos_type in [PositionType.LONG, PositionType.SHORT]:
                if should_enter_trade(df, i, pos_type, params):
                    result = enter_trade(df, i, balance, pos_type, params)
                    if result[0] is None:
                        continue
                    position_size, entry_price, tp_level, stop_loss_level, liquidation_price, allocated, entry_fee = result
                    balance -= allocated + entry_fee  # Entry fee deducted here
                    total_fees += entry_fee
                    in_position = True
                    position_type = pos_type
                    trade_count += 1
                    order_log.append(
                        f"ENTRY: time={timestamp}, BuyPrice={entry_price:.2f}, PositionSize={position_size:.2f}, Fee={entry_fee:.2f}")
                    trade_timestamps.append((timestamp, 'entry', pos_type.value))
                    break

        equity_curve.append(balance)

    gross_pnl = gross_profit_loss
    net_pnl = balance - INITIAL_BALANCE
    win_rate = (wins / trade_count) * 100 if trade_count > 0 else 0
    max_drawdown = min(equity_curve) - INITIAL_BALANCE

    with open("LeverageOrder.txt", "w") as file:
        file.write("=" * 50 + "\n")
        file.write(f"{'TRADE LOGS':^50}\n")
        file.write("=" * 50 + "\n")
        for log in order_log:
            file.write(log + "\n")

    return net_pnl, win_rate, max_drawdown, equity_curve, trade_count, balance, order_log, trade_timestamps, total_fees, gross_pnl


# **Enhanced Visualization with BTC Price and Improved Trade Markers**
def plot_equity_curve(equity_curve, trade_timestamps, df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot equity curve on the left y-axis
    ax1.plot(range(len(equity_curve)), equity_curve, label="Equity Curve", color="blue", linewidth=2)
    ax1.set_xlabel("Time (adjusted index)")
    ax1.set_ylabel("Balance ($)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)

    # Plot trade markers (sample every 5th trade to reduce clutter)
    sampled_trades = trade_timestamps[::5]
    for timestamp, trade_type, pos_type in sampled_trades:
        idx = df.index.get_loc(timestamp) - 20
        color = 'green' if pos_type == 'long' else 'red'
        marker = '^' if pos_type == 'long' else 'v'
        label = "Long Entry" if pos_type == 'long' and (timestamp, trade_type, pos_type) == sampled_trades[0] else (
            "Short Entry" if pos_type == 'short' and (timestamp, trade_type, pos_type) == sampled_trades[0] else "")
        ax1.plot(idx, equity_curve[idx], marker=marker, color=color, markersize=8, label=label)

    # Create a second y-axis for BTC price
    ax2 = ax1.twinx()
    btc_price = df['close'].iloc[20:].reset_index(drop=True)
    ax2.plot(range(len(equity_curve)), btc_price, label="BTC Price", color="orange", linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("BTC Price ($)", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Balance and BTC Price Evolution with Trade Entries")
    plt.show()


# **Main Execution**
if __name__ == "__main__":
    # Load Data
    CACHE_FILE = 'btc_usdt_1m_1yr.csv'
    df = pd.read_csv(CACHE_FILE, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = add_indicators(df)

    # Parameters
    params = {
        'initial_balance': 1000,
        'fee_rate': 0.001,
        # Long params
        'rsi_base_threshold': 16.373657650173044,
        'stoch_base_threshold': 30.288536623934057,
        'adx_threshold': 22.0889582141916,
        'tp_percentage': 3.4606411972383313,
        'stop_loss_percent': 26.090526913613548,
        'near_ath_factor': 1.4837723048162779,
        'leverage': 19.723177376997505,
        'risk_per_trade': 0.04730074878845295,
        # Short params
        'short_rsi_base_threshold': 77.01447270843248,
        'short_stoch_base_threshold': 88.97794712065782,
        'short_adx_threshold': 49.10155770160169,
        'short_tp_percentage': 1.6177948183923818,
        'short_stop_loss_percent': 14.748412455027468,
        'short_near_ath_factor': 1.424929553837893,
        'short_leverage': 19.45790942145642,
        'short_risk_per_trade': 0.008694680346875462
    }

    # Run Backtest
    net_pnl, win_rate, max_drawdown, equity_curve, trade_count, balance, order_log, trade_timestamps, total_fees, gross_pnl = backtest_fixed_leverage(
        df, params)

    # Print Results
    print("=" * 50)
    print(f"{'Backtest Results':^50}")
    print("=" * 50)
    print(f"Total Trades Executed: {trade_count}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Gross PNL (before fees): ${gross_pnl:.2f}")
    print(f"Total Fees Paid: ${total_fees:.2f}")
    print(f"Net PNL (after fees): ${net_pnl:.2f}")
    print(f"Fees as % of Gross PNL: {((total_fees / gross_pnl) * 100 if gross_pnl > 0 else 0):.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Max Drawdown: ${max_drawdown:.2f}")
    print("=" * 50)
    print("✅ Orders have been saved to 'LeverageOrder.txt'")

    # Plot with BTC Price
    plot_equity_curve(equity_curve, trade_timestamps, df)