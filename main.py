import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import warnings
import time
warnings.filterwarnings("ignore")

# ==========================
# CONFIG
# ==========================
SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"

# ==========================
# FETCH DATA
# ==========================
def fetch_price(ticker, start, end=None):
    print(f"Fetching price for {ticker}...")
    end = end or datetime.today().strftime('%Y-%m-%d')
    for _ in range(3):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data.empty:
                raise ValueError("Empty data")
            data = data[['Close']].copy()
            data.rename(columns={'Close': 'Price'}, inplace=True)
            print(f"Success: {len(data)} days")
            return data
        except Exception as e:
            print(f"Retry... {e}")
            time.sleep(3)
    raise ValueError("Price fetch failed")

def fetch_earnings(ticker, start, end=None):
    print(f"Fetching earnings for {ticker}...")
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.quarterly_earnings
        if earnings is None or earnings.empty:
            raise ValueError("No data")
        earnings = earnings.copy()
        earnings.index = pd.to_datetime(earnings.index)
        earnings = earnings[['Earnings']]
        full_range = pd.date_range(start=start, end=end, freq='D')
        daily = earnings.reindex(full_range).ffill()
        print(f"Earnings: {len(earnings)} quarters")
        return daily
    except Exception as e:
        print(f"Earnings failed: {e}. Using trailing EPS...")
        try:
            eps = stock.info.get('trailingEps', 6.0)
            series = pd.Series(eps, index=pd.date_range(start=start, end=end, freq='D'))
            print(f"Trailing EPS: {eps:.2f}")
            return series
        except:
            return None

def fetch_shiller_cape(start, end=None):
    print("Fetching Shiller CAPE...")
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        response = requests.get(SHILLER_URL, timeout=10)
        df = pd.read_excel(BytesIO(response.content), sheet_name="Data", skiprows=7)
        df = df.iloc[:, [0, 7]]
        df.columns = ['Date', 'CAPE']
        df['Date'] = pd.to_datetime(df['Date'].astype(str).str.split('.').str[0] + '-01')
        df = df.set_index('Date').dropna()
        full = pd.date_range(start=start, end=end, freq='D')
        cape = df['CAPE'].reindex(full).ffill()
        print(f"CAPE: {len(df)} months")
        return cape
    except Exception as e:
        print(f"CAPE failed: {e}")
        return None

def fetch_vix(start, end=None):
    print("Fetching VIX...")
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False)['Close']
        vix.name = 'VIX'
        return vix
    except:
        print("VIX fallback")
        return pd.Series(20.0, index=pd.date_range(start=start, end=end, freq='D'))

def fetch_yield_curve(start, end=None):
    print("Fetching Yield Curve...")
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        t10 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", parse_dates=['DATE'])
        t2 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2", parse_dates=['DATE'])
        yc = t10.set_index('DATE')['DGS10'] - t2.set_index('DATE')['DGS2']
        yc = yc.reindex(pd.date_range(start=start, end=end, freq='D')).ffill()
        yc.name = 'Yield_Curve'
        return yc
    except:
        print("Yield curve fallback")
        return pd.Series(0.5, index=pd.date_range(start=start, end=end, freq='D'))

# ==========================
# ENHANCED SIGNAL
# ==========================
def generate_signal(pe, vix, yc, rsi):
    score = 0
    if pe < 30: score += 1
    elif pe > 40: score -= 1
    if vix < 18: score += 1
    elif vix > 30: score -= 1
    if yc > 0.3: score += 1
    if rsi < 30: score += 1
    if rsi > 70: score -= 1
    if score >= 2: return score, "BUY"
    elif score <= -2: return score, "SELL"
    else: return score, "HOLD"

# ==========================
# BACKTEST
# ==========================
def backtest(ticker, start, end=None, plot=True):
    end = end or datetime.today().strftime('%Y-%m-%d')
    print("\n" + "="*60)
    print("BACKTEST STARTED")
    print("="*60)
    
    price = fetch_price(ticker, start, end)
    vix = fetch_vix(start, end)
    yc = fetch_yield_curve(start, end)
    
    try:
        stock = yf.Ticker(ticker)
        eps = stock.info.get('trailingEps', 6.0)
        print(f"Using trailing EPS: {eps:.2f}")
    except:
        eps = 6.0
        print("Using default EPS: 6.0")

    df = price.copy()
    df['EPS'] = eps
    df['PE'] = df['Price'] / eps

    df['VIX'] = vix.reindex(df.index).ffill().bfill()
    df['Yield_Curve'] = yc.reindex(df.index).ffill().bfill()

    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    df['Signal'] = 'HOLD'
    for i in range(len(df)):
        pe = df['PE'].iloc[i]
        vix_val = df['VIX'].iloc[i]
        yc_val = df['Yield_Curve'].iloc[i]
        rsi = df['RSI'].iloc[i]
        _, sig = generate_signal(pe, vix_val, yc_val, rsi)
        df.iloc[i, df.columns.get_loc('Signal')] = sig

    df['Return'] = df['Price'].pct_change().fillna(0)
    df['Strategy'] = 0.0
    in_pos = False
    cost = 0.001
    trades = 0

    for i in range(1, len(df)):
        sig = df['Signal'].iloc[i-1]
        ret = df['Return'].iloc[i]
        strat = 0.0
        if sig == "BUY" and not in_pos:
            in_pos = True
            trades += 1
            strat = ret - cost
        elif sig == "SELL" and in_pos:
            in_pos = False
            trades += 1
            strat = ret - cost
        elif in_pos:
            strat = ret
            if ret < -0.05:
                strat -= cost
                in_pos = False
                trades += 1
        df.iloc[i, df.columns.get_loc('Strategy')] = strat

    df['Cum_Strat'] = (1 + df['Strategy']).cumprod()
    df['Cum_Market'] = (1 + df['Return']).cumprod()

    total = df['Cum_Strat'].iloc[-1] - 1
    market = df['Cum_Market'].iloc[-1] - 1
    days = len(df)
    ann = (1 + total) ** (252/days) - 1 if days else 0
    vol = df['Strategy'].std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dd = (df['Cum_Strat'] / df['Cum_Strat'].cummax() - 1).min()

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Period: {start} to {end}")
    print(f"Total Return: {total:+.2%}")
    print(f"Buy & Hold: {market:+.2%}")
    print(f"Annualized: {ann:+.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {dd:.2%}")
    print(f"Trades: {trades}")
    print("="*50)

    if plot:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['Cum_Strat'], label='Strategy', color='blue', linewidth=2)
        plt.plot(df.index, df['Cum_Market'], label='Buy & Hold', color='gray', alpha=0.7)
        plt.title(f'{ticker} Strategy vs Buy & Hold')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['PE'], label='P/E Ratio', color='purple')
        plt.axhline(30, color='green', linestyle='--', label='Buy Zone')
        plt.axhline(40, color='red', linestyle='--', label='Sell Zone')
        plt.title('P/E Ratio Over Time')
        plt.ylabel('P/E')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{ticker}_backtest.png", dpi=150)
        plt.show()
        print(f"Plot saved: {ticker}_backtest.png")

    return df

# ==========================
# MAIN (FIXED PRINT ERROR)
# ==========================
def main():
    print("Market Sentiment Backtester (P/E + VIX + Yield Curve + RSI)")
    print("="*60)
    ticker = input("Ticker: ").strip().upper()
    start = input("Start (YYYY-MM-DD): ").strip() or "2015-01-01"

    try:
        results = backtest(ticker, start)
        last = results.iloc[-1]
        pe = last['PE'].item() if isinstance(last['PE'], (pd.Series, np.generic)) else last['PE']
        rsi = last['RSI'].item() if isinstance(last['RSI'], (pd.Series, np.generic)) else last['RSI']
        print(f"\nLATEST SIGNAL: {last['Signal']} | P/E: {pe:.1f} | RSI: {rsi:.0f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Try again in 5 minutes.")

if __name__ == "__main__":
    main()
