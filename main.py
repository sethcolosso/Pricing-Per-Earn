import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
import warnings
import time
warnings.filterwarnings("ignore")

FRED_API_KEY = "cf6a075550a88f00f3a045cd7a70cd46"
fred = Fred(api_key=FRED_API_KEY)
# ==========================
# FETCH DATA FUNCTIONS
# ==========================
def fetch_asset_data(ticker, start="2010-01-01", end=None):
    """Fetch price data for a given asset with retries"""
    print(f"Fetching data for {ticker} from {start} to {end or 'today'} ...")
    for attempt in range(3):
        try:
            data = yf.download(tickers=[ticker], start=start, end=end)
            if isinstance(data.columns, pd.MultiIndex):
                data = data.xs(ticker, axis=1, level=1)
            data["Adj Close"] = data["Adj Close"].ffill()
            if not data.empty:
                return data
            else:
                print("Data is empty, retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(5)
    raise ValueError(f"Failed to fetch data for {ticker} after 3 attempts.")

def fetch_earnings(ticker, start="2010-01-01", end=None):
    """Fetch historical earnings data"""
    print(f"Fetching earnings for {ticker} ...")
    if ticker in ['^GSPC', 'SPY']:
        # Shiller data for S&P 500
        url = 'https://datahub.io/core/s-and-p-500/r/data.csv'
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').dropna(subset=['Earnings'])
        daily_index = pd.date_range(start=df.index.min(), end=datetime.today(), freq='D')
        df_daily = df.reindex(daily_index).ffill()
        df_daily = df_daily.loc[start:end]
        return df_daily['Earnings']
    else:
        # For individual stocks
        try:
            stock = yf.Ticker(ticker)
            earnings_df = stock.quarterly_earnings
            earnings_df.index = pd.to_datetime(earnings_df.index)
            earnings = earnings_df['Earnings'].resample('D').ffill().loc[start:end]
            return earnings
        except Exception as e:
            print(f"Failed to fetch earnings for {ticker}: {e}")
            return None

def fetch_vix_data(start="2010-01-01", end=None):
    """Fetch CBOE VIX data"""
    print("Fetching VIX data ...")
    vix = yf.download("^VIX", start=start, end=end)
    vix_series = vix["Close"]
    vix_series.name = "VIX"
    return vix_series

def fetch_yield_curve(start="2010-01-01", end=None):
    """Fetch 10Y minus 2Y Treasury yield spread (Yield Curve)"""
    print("Fetching yield curve data from FRED ...")
    t10y = fred.get_series("DGS10")
    t2y = fred.get_series("DGS2")
    yc = (t10y - t2y).dropna()
    yc.name = "Yield_Curve"
    yc = yc.loc[start:end]
    return yc
# ==========================
# COMPUTATION FUNCTIONS
# ==========================
def compute_pe_ratio(price, eps, prev_pe=None, prev_prev_pe=None):
    """Compute Price to Earnings ratio, with fallback to previous PE or average of past 2"""
    if eps is not None and eps != 0 and not np.isnan(eps):
        return price / eps
    else:
        print("⚠️ EPS is missing or invalid — using previous PE or average")
        if prev_pe is not None and not np.isnan(prev_pe):
            return prev_pe
        elif prev_pe is not None and prev_prev_pe is not None:
            return np.mean([prev_pe, prev_prev_pe])
        else:
            return np.nan  # If no previous, remain nan (handle later)

def generate_signal(pe, vix, yield_curve):
    """Generate trading signal based on PE, VIX, and Yield Curve"""
    score = 0

    # PE scoring
    if pe < 15:
        score += 1
    elif pe > 25:
        score -= 1

    # VIX scoring
    if vix < 15:
        score += 1
    elif vix > 25:
        score -= 1

    # Yield Curve scoring
    if yield_curve > 0:
        score += 1
    else:
        score -= 1

    # Interpret signal
    if score >= 2:
        signal = "BUY"
    elif score <= -1:
        signal = "SELL"
    else:
        signal = "HOLD"

    return score, signal

def backtest(ticker, start="2015-01-01", end=None):
    """Run backtest on historical data with statistical analysis"""
    # Fetch data
    data = fetch_asset_data(ticker, start, end)
    earnings = fetch_earnings(ticker, start, end)
    vix = fetch_vix_data(start, end)
    yc = fetch_yield_curve(start, end)

    # Merge all
    combined = pd.DataFrame(index=data.index)
    combined["Price"] = data["Adj Close"]
    if earnings is not None:
        combined = combined.join(earnings, how="left")
    combined = combined.join(vix, how="inner")
    combined = combined.join(yc, how="inner")

    # Compute PE with fallbacks in a loop
    signals = []
    prev_pe = None
    prev_prev_pe = None
    for date, row in combined.iterrows():
        eps = row.get("Earnings", np.nan)
        price = row["Price"]
        pe = compute_pe_ratio(price, eps, prev_pe, prev_prev_pe)
        if np.isnan(pe):
            pe = 20.0  # Default average if no previous (unlikely)
        prev_prev_pe = prev_pe
        prev_pe = pe
        vix_val = row["VIX"]
        yc_val = row["Yield_Curve"]
        score, signal = generate_signal(pe, vix_val, yc_val)
        signals.append({"Date": date, "PE": pe, "VIX": vix_val, "YieldCurve": yc_val, "Score": score, "Signal": signal, "Price": price})

    results = pd.DataFrame(signals).set_index("Date")

    # Add statistical backtesting (simple long-only strategy with transaction costs)
    transaction_cost = 0.001  # 0.1% per trade
    results['Market_Return'] = results['Price'].pct_change()
    results['Strategy_Return'] = 0.0
    in_position = False
    num_trades = 0
    for i in range(1, len(results)):
        signal = results['Signal'].iloc[i-1]  # Use previous day's signal to avoid lookahead
        current_return = results['Market_Return'].iloc[i]
        if signal == "BUY" and not in_position:
            in_position = True
            num_trades += 1
            results['Strategy_Return'].iloc[i] = current_return - transaction_cost
        elif signal == "SELL" and in_position:
            in_position = False
            num_trades += 1
            results['Strategy_Return'].iloc[i] = current_return - transaction_cost
        elif in_position:
            results['Strategy_Return'].iloc[i] = current_return

    # Cumulative returns
    results['Cumulative_Market'] = (1 + results['Market_Return']).cumprod().fillna(1)
    results['Cumulative_Strategy'] = (1 + results['Strategy_Return']).cumprod().fillna(1)

    # Statistics
    total_return_strategy = results['Cumulative_Strategy'].iloc[-1] - 1
    total_return_market = results['Cumulative_Market'].iloc[-1] - 1
    num_days = len(results)
    annual_return_strategy = (1 + total_return_strategy) ** (252 / num_days) - 1 if num_days > 0 else 0
    annual_return_market = (1 + total_return_market) ** (252 / num_days) - 1 if num_days > 0 else 0
    annual_vol_strategy = results['Strategy_Return'].std() * np.sqrt(252) if num_days > 1 else 0
    sharpe_strategy = annual_return_strategy / annual_vol_strategy if annual_vol_strategy != 0 else 0
    # Max drawdown
    peak = results['Cumulative_Strategy'].cummax()
    drawdown = (results['Cumulative_Strategy'] - peak) / peak
    max_drawdown = drawdown.min()

    # Risk management: Simple trailing stop example (sell if daily loss > 5% while in position)
    # Enhance the loop with stop loss
    results['Strategy_Return'] = 0.0  # Reset to re-compute with risk mgmt
    in_position = False
    num_trades = 0
    for i in range(1, len(results)):
        signal = results['Signal'].iloc[i-1]
        current_return = results['Market_Return'].iloc[i]
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
            # Risk management: Stop loss if daily return < -5%
            if strategy_return < -0.05:
                strategy_return -= transaction_cost  # Extra cost for forced sell
                in_position = False
                num_trades += 1
        results['Strategy_Return'].iloc[i] = strategy_return

    # Re-compute cumulatives and stats after risk mgmt
    results['Cumulative_Strategy'] = (1 + results['Strategy_Return']).cumprod().fillna(1)
    total_return_strategy = results['Cumulative_Strategy'].iloc[-1] - 1
    annual_return_strategy = (1 + total_return_strategy) ** (252 / num_days) - 1 if num_days > 0 else 0
    annual_vol_strategy = results['Strategy_Return'].std() * np.sqrt(252) if num_days > 1 else 0
    sharpe_strategy = annual_return_strategy / annual_vol_strategy if annual_vol_strategy != 0 else 0
    peak = results['Cumulative_Strategy'].cummax()
    drawdown = (results['Cumulative_Strategy'] - peak) / peak
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
    return results

# ==========================
# MAIN EXECUTION
# ==========================
def main():
    ticker = input("Enter the stock, ETF, or asset ticker (e.g., SPY, ^GSPC, AAPL): ").strip()
    start = input("Enter start date (YYYY-MM-DD, default 2015-01-01): ").strip() or "2015-01-01"
    end = None  # Current date by default

    try:
        results = backtest(ticker, start, end)
        last = results.iloc[-1]

        print("\n===== MOST RECENT ANALYSIS =====")
        print(f"Ticker: {ticker}")
        print(f"P/E Ratio: {last['PE']:.2f}")
        print(f"VIX: {last['VIX']:.2f}")
        print(f"Yield Curve (10Y-2Y): {last['YieldCurve']:.2f}")
        print(f"Market Sentiment Score: {last['Score']}")
        print(f"Trade Signal: 🚀 {last['Signal']}")
        print("================================")

        # Save output
        results.to_csv(f"{ticker}_PE_VIX_YieldCurve_Backtest.csv")
        print(f"\nBacktest results saved as {ticker}_PE_VIX_YieldCurve_Backtest.csv")
    except Exception as e:
        print(f"Error: Cannot access historical data for {ticker}. {e}")
        print("Proceeding with limited calculations minus the missing data.")
        # Optional: Implement a fallback for current signal if possible
        # For example, fetch current values
        try:
            stock = yf.Ticker(ticker)
            price = stock.info.get('regularMarketPrice', np.nan)
            if np.isnan(price):
                raise ValueError("Cannot fetch current price.")
            
            # Current VIX
            vix_data = yf.Ticker("^VIX").info
            vix = vix_data.get('regularMarketPrice', np.nan)
            
            # Current yield curve (approx, since FRED is series, fetch latest)
            t10y_latest = fred.get_series("DGS10").iloc[-1]
            t2y_latest = fred.get_series("DGS2").iloc[-1]
            yc = t10y_latest - t2y_latest
            
            # EPS: For S&P use Shiller latest, else trailing EPS
            if ticker in ['^GSPC', 'SPY']:
                earnings = fetch_earnings(ticker, start, end)
                eps = earnings.iloc[-1] if earnings is not None else np.nan
            else:
                eps = stock.info.get('trailingEps', np.nan)
            
            pe = compute_pe_ratio(price, eps)
            if np.isnan(pe):
                pe = 20.0
            
            score, signal = generate_signal(pe, vix, yc)
            
            print("\n===== LIMITED CURRENT ANALYSIS (NO BACKTEST) =====")
            print(f"Ticker: {ticker}")
            print(f"Current Price: {price:.2f}")
            print(f"P/E Ratio (approx): {pe:.2f}")
            print(f"VIX: {vix:.2f}")
            print(f"Yield Curve (10Y-2Y): {yc:.2f}")
            print(f"Market Sentiment Score: {score}")
            print(f"Trade Signal: 🚀 {signal}")
            print("================================")
        except Exception as fallback_e:
            print(f"Fallback failed: {fallback_e}")
            print("Unable to perform any calculations without data.")

# ==========================
# RUN
# ==========================
if __name__ == "__main__":

    main()
