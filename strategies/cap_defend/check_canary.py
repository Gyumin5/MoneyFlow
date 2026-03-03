import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import yfinance as yf
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

tickers = ['VT', 'EEM']
print(f"Checking Canary Assets: {tickers}")
try:
    data = yf.download(tickers, period="2y", progress=False)['Close']
    
    if data.empty:
        print("Error: No data fetched.")
    else:
        for t in tickers:
            if t not in data.columns:
                print(f"Error: {t} not found in data columns: {data.columns}")
                continue
                
            series = data[t].dropna()
            if len(series) < 200:
                print(f"[{t}] Not enough data ({len(series)} days)")
                continue

            ma200 = series.rolling(200).mean().iloc[-1]
            curr = series.iloc[-1]
            last_date = series.index[-1]
            
            print(f"\n[{t}] (Last Date: {last_date.date()})")
            print(f"Current Price: {curr:.2f}")
            print(f"MA200:         {ma200:.2f}")
            
            if curr > ma200:
                print("✅ Risk-On (Bullish) - Price > MA200")
            else:
                print("🚨 Risk-Off (Bearish) - Price < MA200")
                diff = (curr - ma200) / ma200
                print(f"   Diff: {diff:.2%}")

except Exception as e:
    print(f"An error occurred: {e}")
