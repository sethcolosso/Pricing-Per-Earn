import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
import warnings
import time
warnings.filterwarnings("ignore")

# ==========================
# CONFIGURATION
# ==========================
FRED_API_KEY = "cf6a075550a88f00f3a045cd7a70cd46"
fred = Fred(api_key=FRED_API_KEY)

# ==========================
# FETCH DATA FUNCTIONS
# ==========================
def fetch_asset_data(ticker, start="2010-01-01", end=None):
    """Fetch price data with robust handling"""
    print(f"Fetching data for {ticker} from {start} to {end or 'today'} ...")
    for attempt in range(3):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            # Handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            # Ensure 'Adj Close' exists
            if 'Adj Close' not in data.columns:
                if 'Close' in data.columns:
                    data['Adj Close'] = data['Close']
                else:
                    raise KeyError("No price data found")
            data = data[['Adj Close']].copy()
            data["Adj Close"] = data["Adj Close"].ffill().bfill()
            if len(data) > 0:
                print(f"Success: Fetched {len(data)} days of data.")
                return data
            else:
                print("Empty data, retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(3)
    raise ValueError(f"Failed to fetch data for {ticker} after 3 attempts.")

def fetch_earnings(ticker, start, end=None):
    """Fetch quarterly earnings and resample to daily"""
    print(f"Fetching earnings for {ticker} ...")
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.quarterly_earnings
        if earnings.empty or 'Earnings' not in earnings.columns:
            print(f"No earnings data for {ticker}. Using trailing EPS or default.")
            return None
        # Ensure index is datetime
        earnings['Date'] = pd.to_datetime(earnings.index)
        earnings = earnings.set_index('Date')['Earnings']
        # Resample to daily
        date_range = pd.date_range(start=start, end=end or datetime.today(), freq='D')
        earnings_daily = earnings.reindex(date_range).ffill()
        if earnings_daily.empty:
            return None
        return earnings_daily
    except Exception as e:
        print(f"Failed to fetch earnings for {ticker}: {e}")
        return None

def fetch_vix_data(start, end=None):
    """Fetch CBOE VIX data"""
    print("Fetching VIX data ...")
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False)
        vix_series = vix["Close"]
        vix_series.name = "VIX"
        return vix_series
    except:
        print("Failed to fetch VIX data.")
        return None

def fetch_yield_curve(start, end=None):
    """Fetch 10Y minus 2Y Treasury yield spread"""
    print("Fetching yield curve data from FRED ...")
    try:
        t10y = fred.get_series("DGS10")
        t2y = fred.get_series("DGS2")
        yc = (t10y - t2y).dropna()
        yc.name = "Yield_Curve"
        yc = yc.loc[start:end]
        return yc
    except:
        print("Failed to fetch yield curve data.")
        return None

def fetch_current_data(ticker):
    """Fetch current price, EPS, VIX, and yield curve"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Use provided price for AAPL if available
        price = 262.82 if ticker == "AAPL" else info.get('regularMarketPrice', info.get('currentPrice'))
        eps = info.get('trailingEps')
        vix = yf.Ticker("^VIX").info.get('regularMarketPrice', np.nan)
        t10y = fred.get_series("DGS10").iloc[-1]
        t2y = fred.get_series("DGS2").iloc[-1]
        yc = t10y - t2y
        return price, eps, vix, yc
    except Exception as e:
        print(f"Failed to fetch current data: {e}")
        return None, None, np.nan, np.nan

# ==========================
# COMPUTATION FUNCTIONS
# ==========================
def compute_pe_ratio(price, eps, prev_pe=None, prev_prev_pe=None):
    """Compute P/E with fallbacks to previous day or average of past two days"""
    if eps and eps != 0 and not np.isnan(eps):
        return price / eps
    print("⚠️ EPS missing — using previous P/E or average")
    if prev_pe is not None and not np.isnan(prev_pe):
        return prev_pe
    if prev_pe is not None and prev_prev_pe is not None:
        return np.mean([prev_pe, prev_prev_pe])
    return 20.0  # Default if no data

def generate_signal(pe, vix, yield_curve):
    """Generate trading signal based on P/E, VIX, and Yield Curve"""
    score = 0
    if pe is not None and not np.isnan(pe):
        if pe < 15:
            score += 1
        elif pe > 25:
            score -= 1
    if vix is not None and not np.isnan(vix):
        if vix < 15:
            score += 1
        elif vix > 25:
            score -= 1
    if yield_curve is not None and not np.isnan(yield_curve):
        if yield_curve > 0:
            score += 1
        else:
            score -= 1
    if score >= 2:
        signal = "BUY"
    elif score <= -1:
        signal = "SELL"
    else:
        signal = "HOLD"
    return score, signal

def backtest(ticker, start, end=None):
    """Run backtest with statistical analysis and risk management"""
    print("Starting backtest...")
    # Fetch data
    data = fetch_asset_data(ticker, start, end)
    earnings = fetch_earnings(ticker, start, end)
    vix = fetch_vix_data(start, end)
    yc = fetch_yield_curve(start, end)

    # Merge data
    df = pd.DataFrame(index=data.index)
    df["Price"] = data["Adj Close"]
    if earnings is not None:
        df = df.join(earnings.rename("EPS"), how="left")
    if vix is not None:
        df = df.join(vix, how="inner")
    if yc is not None:
        df = df.join(yc, how="inner")

    # Handle missing data
    if 'EPS' not in df.columns:
        df['EPS'] = np.nan
    if 'VIX' not in df.columns:
        df['VIX'] = np.nan
    if 'Yield_Curve' not in df.columns:
        df['Yield_Curve'] = np.nan

    # Compute P/E
    df['PE'] = np.nan
    prev_pe = None
    prev_prev_pe = None
    for i, row in df.iterrows():
        pe = compute_pe_ratio(row['Price'], row.get('EPS'), prev_pe, prev_prev_pe)
        df.loc[i, 'PE'] = pe
        prev_prev_pe = prev_pe
        prev_pe = pe

    # Generate signals
    df['Score'] = 0
    df['Signal'] = "HOLD"
    for i, row in df.iterrows():
        score, signal = generate_signal(row['PE'], row.get('VIX'), row.get('Yield_Curve'))
        df.loc[i, 'Score'] = score
        df.loc[i, 'Signal'] = signal

    # Backtest with risk management
    transaction_cost = 0.001  # 0.1% per trade
    df['Market_Return'] = df['Price'].pct_change().fillna(0)
    df['Strategy_Return'] = 0.0
    in_position = False
    num_trades = 0
    for i in range(1, len(df)):
        signal = df['Signal'].iloc[i-1]
        current_return = df['Market_Return'].iloc[i]
        strategy_return = 0.0
        if signal == "BUY" and not in_position:
            in_position = True
            num_trades += 1
            strategy_return = current_return - transaction_cost
        elif signal == "SELL" and in_position:
            in_position = False
            num_trades += 1
            strategy_return = current_return - transaction_cost
        elif in_position:
            strategy_return = current_return
            # Risk management: 5% stop-loss
            if current_return < -0.05:
                strategy_return -= transaction_cost
                in_position = False
                num_trades += 1
        df.loc[df.index[i], 'Strategy_Return'] = strategy_return

    # Calculate statistics
    df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    total_return_strategy = df['Cumulative_Strategy'].iloc[-1] - 1
    total_return_market = df['Cumulative_Market'].iloc[-1] - 1
    num_days = len(df)
    annual_return_strategy = (1 + total_return_strategy) ** (252 / num_days) - 1 if num_days > 0 else 0
    annual_return_market = (1 + total_return_market) ** (252 / num_days) - 1 if num_days > 0 else 0
    annual_vol_strategy = df['Strategy_Return'].std() * np.sqrt(252) if num_days > 1 else 0
    sharpe_strategy = annual_return_strategy / annual_vol_strategy if annual_vol_strategy != 0 else 0
    peak = df['Cumulative_Strategy'].cummax()
    drawdown = (df['Cumulative_Strategy'] - peak) / peak
    max_drawdown = drawdown.min()

    # Print stats
    print("\n===== BACKTEST STATISTICS =====")
    print(f"Total Return (Strategy): {total_return_strategy:.2%}")
    print(f"Total Return (Buy & Hold): {total_return_market:.2%}")
    print(f"Annualized Return (Strategy): {annual_return_strategy:.2%}")
    print(f"Annualized Return (Buy & Hold): {annual_return_market:.2%}")
    print(f"Annualized Volatility (Strategy): {annual_vol_strategy:.2%}")
    print(f"Sharpe Ratio (Strategy): {sharpe_strategy:.2f}")
    print(f"Max Drawdown (Strategy): {max_drawdown:.2%}")
    print(f"Number of Trades: {num_trades}")
    print("===============================")

    return df

# ==========================
# MAIN EXECUTION
# ==========================
def main():
    print("Market Sentiment Backtester")
    print("="*50)
    ticker = input("Enter ticker Symbol (e.g., AAPL, SPY, ^GSPC): ").strip().upper()
    start = input("Start date (YYYY-MM-DD, default 2015-01-01): ").strip() or "2015-01-01"
    end = None  # Current date

    try:
        results = backtest(ticker, start, end)
        last = results.iloc[-1]
        print("\n===== MOST RECENT ANALYSIS =====")
        print(f"Ticker: {ticker}")
        print(f"P/E Ratio: {last['PE']:.2f}")
        print(f"VIX: {last['VIX']:.2f}" if not np.isnan(last.get('VIX')) else "VIX: Unavailable")
        print(f"Yield Curve (10Y-2Y): {last['Yield_Curve']:.2f}" if not np.isnan(last.get('Yield_Curve')) else "Yield Curve: Unavailable")
        print(f"Market Sentiment Score: {int(last['Score'])}")
        print(f"Trade Signal: 🚀 {last['Signal']}")
        print("================================")
        results.to_csv(f"{ticker}_PE_VIX_YieldCurve_Backtest.csv")
        print(f"\nBacktest results saved as {ticker}_PE_VIX_YieldCurve_Backtest.csv")
    except Exception as e:
        print(f"Error: Cannot access historical data for {ticker}. {e}")
        print("Proceeding with limited current analysis...")
        price, eps, vix, yc = fetch_current_data(ticker)
        if price is None:
            print("Unable to fetch current price. Aborting.")
            return
        pe = compute_pe_ratio(price, eps)
        score, signal = generate_signal(pe, vix, yc)
        print("\n===== LIMITED CURRENT ANALYSIS (NO BACKTEST) =====")
        print(f"Ticker: {ticker}")
        print(f"Current Price: ${price:.2f}")
        print(f"P/E Ratio (approx): {pe:.2f}")
        print(f"VIX: {vix:.2f}" if not np.isnan(vix) else "VIX: Unavailable")
        print(f"Yield Curve (10Y-2Y): {yc:.2f}" if not np.isnan(yc) else "Yield Curve: Unavailable")
        print(f"Market Sentiment Score: {score}")
        print(f"Trade Signal: 🚀 {signal}")
        print("================================")
if __name__ == "__main__":
    main()
