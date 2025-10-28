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
# SIGNAL
# ==========================
def generate_signal(pe, vix, yc):
    score = 0
    if pe < 15: score += 1
    elif pe > 25: score -= 1
    if vix < 15: score += 1
    elif vix > 25: score -= 1
    if yc > 0: score += 1
    else: score -= 1
    if score >= 2: return score, "BUY"
    elif score <= -1: return score, "SELL"
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
    
    # P/E
    if ticker in ['SPY', '^GSPC']:
        pe_series = fetch_shiller_cape(start, end)
        use_cape = True
    else:
        eps = fetch_earnings(ticker, start, end)
        pe_series = None
        use_cape = False

    df = price.copy()
    df['VIX'] = vix.reindex(df.index).ffill().bfill()
    df['Yield_Curve'] = yc.reindex(df.index).ffill().bfill()

    if use_cape and pe_series is not None:
        df['PE'] = pe_series.reindex(df.index).ffill()
    else:
        df['EPS'] = eps.reindex(df.index).ffill() if eps is not None else np.nan
        df['PE'] = np.nan
        prev = None
        for i in range(len(df)):
            eps_val = df['EPS'].iloc[i]
            pe = df['Price'].iloc[i] / eps_val if pd.notna(eps_val) and eps_val > 0 else (prev or 20.0)
            df.iloc[i, df.columns.get_loc('PE')] = pe
            prev = pe

    # Signals
    df['Signal'] = 'HOLD'
    for i in range(len(df)):
        _, sig = generate_signal(df['PE'].iloc[i], df['VIX'].iloc[i], df['Yield_Curve'].iloc[i])
        df.iloc[i, df.columns.get_loc('Signal')] = sig

    # Backtest
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

    print("\nRESULTS")
    print(f"Total Return: {total:+.2%}")
    print(f"Buy & Hold: {market:+.2%}")
    print(f"Annualized: {ann:+.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max DD: {dd:.2%}")
    print(f"Trades: {trades}")

    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['Cum_Strat'], label='Strategy', color='blue')
        plt.plot(df.index, df['Cum_Market'], label='Buy & Hold', color='gray')
        plt.title(f'{ticker} Backtest')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{ticker}_curve.png")
        plt.show()

    return df

# ==========================
# MAIN
# ==========================
def main():
    print("Market Sentiment Backtester")
    print("="*60)
    ticker = input("Ticker: ").strip().upper()
    start = input("Start (YYYY-MM-DD): ").strip() or "2015-01-01"

    try:
        results = backtest(ticker, start)
        last = results.iloc[-1]
        print(f"\nSIGNAL: {last['Signal']} | P/E: {last['PE']:.1f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Try again in 5 mins")

if __name__ == "__main__":
    main()
