"""
Cap Defend V13 Recommendation Script (Standard Version - Clean)
=============================================================
V13: Multi Bonus Scoring - RSI(45-70→+0.2), MACD hist>0→+0.2, BB %B>0.5→+0.2
- Pure Strategy Recommendation (No Personal Asset Data)
- Dynamic Coin Universe (CoinGecko Top 50 + Upbit Filter)
- No Manual Ticker Mapping (e.g. No POL->MATIC)
- Generates 'portfolio_result.html' with only Strategy Weights
- Cash Buffer: 0% (Standard)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timezone
import pyupbit
import json

# --- 1. Constants & Configuration ---
DATA_DIR = "./data"
STOCK_RATIO, COIN_RATIO = 0.60, 0.40
CASH_ASSET = 'Cash'
# No Cash Buffer for standard report
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN', 'FDUSD']

# Stock V10 Configuration
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA', 'GLD', 'PDBC', 'QUAL', 'MTUM', 'IQLT', 'IMTM']
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY_ASSETS = ['VT', 'EEM']

# Coin V11 Configuration
VOLATILITY_WINDOW = 90
VOL_CAP_FILTER = 0.10
N_SELECTED_COINS = 5

# --- 2. Dynamic Coin Universe ---
def get_dynamic_coin_universe(log: list) -> (list, dict):
    print("\n--- 🛰️ Step 1: Coin Universe Selection (V12) ---")
    log.append("<h2>🛰️ Step 1: 코인 유니버스 선정 (V12: Live CoinGecko Top 50)</h2>")
    
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    FETCH_LIMIT = 100 
    MIN_TRADE_VALUE_KRW = 1_000_000_000 
    DAYS_TO_CHECK = 260 
    headers = {"accept": "application/json", "User-Agent": "Mozilla/5.0"}
    
    UNIVERSE_CACHE_FILE = os.path.join(DATA_DIR, "universe_cache.json")
    cg_data = []
    
    # 1. Try Fetching from CoinGecko (Retry 5 times)
    for attempt in range(1, 6):
        try:
            log.append(f"<div class='info'>Fetching CoinGecko Top {FETCH_LIMIT} (Attempt {attempt}/5)...</div>")
            cg_params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': FETCH_LIMIT, 'page': 1}
            cg_response = requests.get(COINGECKO_URL, params=cg_params, headers=headers, timeout=20)
            
            if cg_response.status_code == 200:
                cg_data = cg_response.json()
                # Save to Cache
                with open(UNIVERSE_CACHE_FILE, 'w') as f:
                    json.dump(cg_data, f)
                log.append(f"<div class='success'>✅ CoinGecko Data Fetched & Cached.</div>")
                break
            elif cg_response.status_code == 429:
                wait_time = 30 * attempt # 30s, 60s, 90s...
                log.append(f"<div class='warning'>⚠️ Rate Limit (429). Waiting {wait_time}s...</div>")
                time.sleep(wait_time)
            else:
                log.append(f"<div class='error'>⚠️ API Error ({cg_response.status_code}).</div>")
                time.sleep(10)
                
        except Exception as e:
            log.append(f"<div class='error'>⚠️ Connection Error: {e}</div>")
            time.sleep(10)

    # 2. If Failed, Load from Cache
    if not cg_data:
        if os.path.exists(UNIVERSE_CACHE_FILE):
            log.append("<p class='warning'>⚠️ Loading Universe from Local Cache (Fallback).</p>")
            try:
                with open(UNIVERSE_CACHE_FILE, 'r') as f:
                    cg_data = json.load(f)
            except: pass
            
    if not cg_data:
        log.append("<p class='error'>❌ All Methods Failed. Using Hardcoded Fallback.</p>")
        return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD'], {}
    
    cg_symbol_to_id_map = {f"{item['symbol'].upper()}-USD": item['id'] for item in cg_data}
    
    try:
        upbit_krw_tickers = pyupbit.get_tickers(fiat="KRW")
        upbit_symbols = {t.split('-')[1] for t in upbit_krw_tickers}
    except: upbit_symbols = set()
    
    final_universe = []
    for item in cg_data:
        symbol = item['symbol'].upper()
        if symbol in STABLECOINS: continue
        if symbol not in upbit_symbols: continue
        
        upbit_ticker = f"KRW-{symbol}"
        try:
            df = pyupbit.get_ohlcv(ticker=upbit_ticker, interval="day", count=DAYS_TO_CHECK)
            time.sleep(0.05)
            if df is None or len(df) < 253: continue
            
            avg_val = df['value'].iloc[-30:].mean()
            if avg_val < MIN_TRADE_VALUE_KRW: continue
            
            ticker_usd = f"{symbol}-USD"
            if ticker_usd not in final_universe: final_universe.append(ticker_usd)
        except: continue
        
        if len(final_universe) >= 50: break
    
    log.append(f"<p>선정된 유니버스 ({len(final_universe)}개)</p>")
    return final_universe, cg_symbol_to_id_map

# --- 3. Data Download ---
def download_required_data(tickers: list, log: list, coin_id_map: dict):
    """
    데이터 다운로드 - Yahoo Finance 우선, CoinGecko 폴백
    [V12.3] 데이터 안정화: File Cache, Retry(10회), Stale Data Fallback 적용
    """
    print("\n--- 📥 Step 2: 데이터 다운로드 (Yahoo Priority + Quality Check) ---")
    log.append("<h2>📥 Step 2: 데이터 다운로드 (Yahoo 우선 + 품질검증)</h2>")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # USD-KRW 환율 조회 (품질 검증용)
    usd_krw_rate = 1450.0
    try:
        usdt_price = pyupbit.get_current_price("KRW-USDT")
        if usdt_price: usd_krw_rate = usdt_price
    except: pass
    
    # [V12.3] Robust Session with Retries
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # Retry 10 times, backoff factor 0.5
    retries = Retry(total=10, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    today_str = datetime.now().date()
    
    for ticker in list(set(tickers)):
        if ticker == CASH_ASSET: continue
        fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        success = False
        
        # 1. Fresh Cache Check
        if os.path.exists(fp):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(fp)).date()
                if mtime == today_str:
                    success = True
            except: pass
        
        if success:
            continue

        # 2. Download from Yahoo
        try:
            current_timestamp = int(datetime.now(timezone.utc).timestamp())
            start_timestamp = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {"period1": start_timestamp, "period2": current_timestamp, "interval": "1d", "includeAdjustedClose": "true"}
            
            resp = session.get(url, params=params, timeout=30)
            
            if resp.status_code == 200:
                res = resp.json()['chart']['result'][0]
                df = pd.DataFrame({'Date': pd.to_datetime(res['timestamp'], unit='s').date, 'Adj_Close': res['indicators']['adjclose'][0]['adjclose']})
                df = df.dropna().drop_duplicates('Date')
                
                # [Quality Check V12.2]
                if ticker.endswith('-USD') and len(df) > 0:
                    symbol = ticker.replace('-USD', '')
                    try:
                        upbit_ohlcv = pyupbit.get_ohlcv(f"KRW-{symbol}", interval="day", count=1)
                        if upbit_ohlcv is not None and len(upbit_ohlcv) > 0:
                            upbit_close_krw = upbit_ohlcv['close'].iloc[-1]
                            yahoo_last_usd = df['Adj_Close'].iloc[-1]
                            yahoo_last_krw = yahoo_last_usd * usd_krw_rate
                            
                            if upbit_close_krw > 0 and yahoo_last_krw > 0:
                                diff_pct = abs(yahoo_last_krw - upbit_close_krw) / upbit_close_krw
                                if diff_pct > 0.10:
                                    print(f"  ⚠️ {ticker}: 가격 불일치 (Yahoo {yahoo_last_krw:,.0f}원 vs Upbit {upbit_close_krw:,.0f}원, diff={diff_pct:.0%}) - 제외")
                                    continue 
                    except: pass
                
                df.to_csv(fp, index=False)
                success = True
                print(f"  - Downloaded {ticker} (Yahoo)")
                
        except: pass
            
        # 3. Fallback to CoinGecko
        if not success and ticker in coin_id_map:
             try:
                cid = coin_id_map[ticker]
                url = f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
                resp = requests.get(url, params={'vs_currency':'usd','days':'500'}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json().get('prices', [])
                    df = pd.DataFrame(data, columns=['ts', 'Adj_Close'])
                    df['Date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
                    df[['Date','Adj_Close']].drop_duplicates('Date').to_csv(fp, index=False)
                    print(f"  - Downloaded {ticker} (CoinGecko)")
                    success = True
             except: pass
        
        # 4. Final Safety Net: Stale Data
        if not success:
            if os.path.exists(fp):
                file_date = datetime.fromtimestamp(os.path.getmtime(fp)).date()
                print(f"  ⚠️ Failed to update {ticker}, using STALE data from {file_date}")
                log.append(f"<p class='warning'>Used stale data for {ticker} ({file_date})</p>")
            else:
                log.append(f"<p class='error'>XXX Failed to download: {ticker}</p>")

# --- 4. Logic Engines ---
def load_price(ticker):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=['Date'])
        # [Validation] Check Data Recency
        if df.empty: return pd.Series(dtype=float)
        
        last_date = df['Date'].iloc[-1].date() if hasattr(df['Date'].iloc[-1], 'date') else df['Date'].iloc[-1]
        today = datetime.now().date()
        
        # 만약 데이터 마지막 날짜가 7일 이상 지났으면 쓸모없는 데이터 (상폐/티커변경 등)
        if (today - last_date).days > 7:
            return pd.Series(dtype=float)
            
        return df.set_index('Date')['Adj_Close'].sort_index()
    except: return pd.Series(dtype=float)

def calc_sma(s, w): return s.rolling(w).mean().iloc[-1] if len(s) >= w else np.nan
def calc_ret(s, d): return s.iloc[-1]/s.iloc[-1-d] - 1 if len(s) >= d+1 and s.iloc[-1-d]!=0 else np.nan
def calc_sharpe(s, d):
    if len(s) < d+1: return 0
    ret = s.pct_change().iloc[-d:]
    return (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
def calc_weighted_mom(s):
    if len(s) < 253: return -np.inf
    r3, r6, r12 = calc_ret(s, 63), calc_ret(s, 126), calc_ret(s, 252)
    return 0.5*r3 + 0.3*r6 + 0.2*r12

# --- Multi Bonus Indicators (V13) ---
def calc_rsi(s, period=14):
    if len(s) < period + 1: return np.nan
    delta = s.diff().iloc[-period-1:]
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0: return 100.0
    return 100 - (100 / (1 + gain / loss))

def calc_macd_hist(s):
    if len(s) < 35: return np.nan
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return (macd_line - signal).iloc[-1]

def calc_bb_pctb(s, period=20):
    if len(s) < period: return np.nan
    sma = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = upper.iloc[-1] - lower.iloc[-1]
    if band_width == 0: return 0.5
    return (s.iloc[-1] - lower.iloc[-1]) / band_width

def calc_multi_bonus_score(s):
    base = calc_sharpe(s, 126) + calc_sharpe(s, 252)
    rsi = calc_rsi(s)
    macd_h = calc_macd_hist(s)
    pctb = calc_bb_pctb(s)
    if pd.notna(rsi) and 45 <= rsi <= 70: base += 0.2
    if pd.notna(macd_h) and macd_h > 0: base += 0.2
    if pd.notna(pctb) and pctb > 0.5: base += 0.2
    return base, rsi, macd_h, pctb

def run_stock_strategy_v11(log, all_prices):
    log.append("<h2>📈 주식 포트폴리오 분석 (V11)</h2>")
    vt, eem = all_prices.get('VT'), all_prices.get('EEM')
    
    if len(vt) >= 200 and len(eem) >= 200:
        vt_cur, eem_cur = vt.iloc[-1], eem.iloc[-1]
        vt_sma, eem_sma = vt.rolling(200).mean().iloc[-1], eem.rolling(200).mean().iloc[-1]
        risk_on = vt_cur > vt_sma and eem_cur > eem_sma
        log.append(f"<p><b>[Canary]</b> VT: ${vt_cur:.2f} (MA {vt_sma:.2f}) | EEM: ${eem_cur:.2f} (MA {eem_sma:.2f})</p>")
    else: 
        risk_on = False
        log.append("<p class='error'>Canary Data Missing</p>")

    if risk_on:
        log.append("<h4>🚀 공격 모드</h4>")
        scores = []
        for t in OFFENSIVE_STOCK_UNIVERSE:
            p = all_prices.get(t)
            if len(p) >= 253: scores.append({'Ticker': t, 'Mom': calc_weighted_mom(p), 'Qual': calc_sharpe(p, 126)})
        df = pd.DataFrame(scores).set_index('Ticker')
        
        try: log.append(f"<div class='table-wrap'>{df.to_html(classes='dataframe small-table')}</div>")
        except: pass

        top_m = df.sort_values('Mom', ascending=False).head(3).index.tolist()
        top_q = df.sort_values('Qual', ascending=False).head(3).index.tolist()
        picks = list(set(top_m + top_q))
        
        log.append(f"<p>Final Picks: <b>{picks}</b></p>")
        return {t: 1.0/len(picks) for t in picks}, "공격 모드"
    else:
        log.append("<h4>🛡️ 수비 모드</h4>")
        res = []
        for t in DEFENSIVE_STOCK_UNIVERSE:
            r = calc_ret(all_prices.get(t), 126)
            if pd.notna(r): res.append({'Ticker': t, '6m Ret': r})
        
        try: log.append(f"<div class='table-wrap'>{pd.DataFrame(res).sort_values('6m Ret', ascending=False).to_html(classes='dataframe small-table')}</div>")
        except: pass
            
        best = sorted(res, key=lambda x: x['6m Ret'], reverse=True)[0]
        if best['6m Ret'] < 0: return {CASH_ASSET: 1.0}, "수비 (현금)"
        return {best['Ticker']: 1.0}, f"수비 ({best['Ticker']})"

def run_coin_strategy_v12(coin_universe, all_prices, target_date, log):
    log.append("<h2>🪙 코인 포트폴리오 (V12)</h2>")
    
    btc = all_prices.get('BTC-USD')
    if len(btc) < 50: return {CASH_ASSET: 1.0}, "데이터 부족"
    
    # BTC 기준 Target Date와 비교
    tgt_dt = target_date.date() if hasattr(target_date, 'date') else target_date
    
    sma50 = btc.rolling(50).mean().iloc[-1]
    cur = btc.iloc[-1]
    log.append(f"<p>[BTC Canary] ${cur:,.0f} vs MA50 ${sma50:,.0f} (Date: {tgt_dt})</p>")
    
    rows = []
    healthy = []
    
    for t in coin_universe:
        p = all_prices.get(t)
        if len(p) < 35: continue
        
        # [Strict Date Check]
        # 데이터의 마지막 날짜가 기준일(BTC 날짜)과 다르면 제외 (오래된 데이터 방지)
        last_dt = p.index[-1].date() if hasattr(p.index[-1], 'date') else p.index[-1]
        diff_days = (tgt_dt - last_dt).days
        if diff_days != 0: # 날짜가 하루라도 다르면 데이터 믹스 위험 -> 제외
            # log.append(f"<p class='error'>Drop {t}: Date Mismatch (BTC:{tgt_dt} vs {last_dt})</p>")
            continue
        
        sma30 = p.rolling(30).mean().iloc[-1]
        mom21 = calc_ret(p, 21)
        cur_p = p.iloc[-1]
        vol90 = p.pct_change().iloc[-90:].std()
        
        is_ok = (cur_p > sma30) and (mom21 > 0) and (vol90 <= VOL_CAP_FILTER)
        
        rows.append({'Coin': t, 'Price': f"${cur_p:.4f}", 'SMA30': f"${sma30:.4f}", 'Mom21': f"{mom21:.2%}", 'Vol90': f"{vol90:.4f}", 'Status': "🟢" if is_ok else "🔴"})
        if is_ok: healthy.append(t)
            
    try: log.append(f"<div class='table-wrap'>{pd.DataFrame(rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except: pass

    if cur <= sma50: return {CASH_ASSET: 1.0}, "Risk-Off"
    
    if not healthy: return {CASH_ASSET: 1.0}, "No Healthy"
    
    # V13: Multi Bonus Scoring (Sharpe + RSI/MACD/BB bonuses)
    scores = []
    for t in healthy:
        score, rsi, macd_h, pctb = calc_multi_bonus_score(all_prices[t])
        bonus_flags = []
        if pd.notna(rsi) and 45 <= rsi <= 70: bonus_flags.append('RSI')
        if pd.notna(macd_h) and macd_h > 0: bonus_flags.append('MACD')
        if pd.notna(pctb) and pctb > 0.5: bonus_flags.append('BB')
        scores.append({
            'Coin': t, 'Score': score,
            'RSI': f"{rsi:.1f}" if pd.notna(rsi) else "-",
            'MACD_H': f"{macd_h:.4f}" if pd.notna(macd_h) else "-",
            'BB%B': f"{pctb:.2f}" if pd.notna(pctb) else "-",
            'Bonus': '+'.join(bonus_flags) if bonus_flags else '-'
        })
    score_df = pd.DataFrame(scores).sort_values('Score', ascending=False)

    try:
        log.append(f"<p><b>[V13 Multi Bonus]</b> RSI(45-70→+0.2) | MACD hist>0→+0.2 | BB %B>0.5→+0.2</p>")
        log.append(f"<div class='table-wrap'>{score_df.head(10).to_html(classes='dataframe small-table')}</div>")
    except: pass
    
    top5 = score_df.head(5)['Coin'].tolist()
    
    vols = {t: all_prices[t].pct_change().iloc[-90:].std() for t in top5}
    inv_vols = {t: 1/v for t, v in vols.items() if v > 0}
    tot = sum(inv_vols.values())
    weights = {t: v/tot for t, v in inv_vols.items()} if tot > 0 else {t: 1/len(top5) for t in top5}
    
    # [Added] Weight/Vol/InvVol Table (Match Personal Version)
    w_rows = [{'Coin': t, 'Weight': f"{w:.2%}", 'Vol': f"{vols[t]:.4f}", 'InvVol': f"{inv_vols[t]:.2f}"} for t, w in weights.items()]
    try:
        log.append(f"<div class='table-wrap'>{pd.DataFrame(w_rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except: pass
    
    return weights, "Full Invest"

def save_html(log_global, final_port, s_port, c_port, s_stat, c_stat, date_today):
    filepath = "portfolio_result.html"
    items = []
    for t, w in final_port.items():
        if t == CASH_ASSET:
            asset_type = "현금"
        elif t in c_port:  # Check if in coin portfolio
            asset_type = "코인"
        else:
            asset_type = "주식"
        items.append({'종목': t, '자산군': asset_type, '비중': w})
    items.sort(key=lambda x: (x['자산군']=='현금', -x['비중']))
    
    tbody = "".join([f"<tr><td>{i['종목']}</td><td>{i['자산군']}</td><td>{i['비중']:.2%}</td></tr>" for i in items])
    
    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cap Defend V13 Recommendation</title>
         <style>
            body {{ font-family: -apple-system, sans-serif; background: #f0f2f5; padding: 10px; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 16px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 10px; border-bottom: 1px solid #f1f3f4; text-align: left; font-size: 0.95em; }}
            th {{ background-color: #fafafa; font-weight: 600; color: #555; }}
            .status-bar {{ display: flex; gap: 10px; background: #e8f0fe; padding: 15px; border-radius: 12px; margin-bottom: 20px; color: #1967d2; font-weight: 500; flex-wrap: wrap; }}
            .dataframe {{ width: 100%; border: 1px solid #ddd; border-collapse: collapse; margin: 10px 0; }}
            .dataframe th, .dataframe td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .dataframe tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .small-table {{ font-size: 0.9em; }}
            .table-wrap {{ overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Cap Defend V13</h1>
            <p>기준일: {date_today.strftime('%Y-%m-%d')}</p>
            
            <div class="status-bar">
                <div>📉 주식: {s_stat}</div>
                <div>🪙 코인: {c_stat}</div>
            </div>
            
            <h2>📊 최종 추천 비중 (Stock + Coin)</h2>
            <table><thead><tr><th>종목</th><th>자산군</th><th>비중</th></tr></thead><tbody>{tbody}</tbody></table>
            
            <h2>📜 상세 로그</h2>
            {''.join(log_global)}
        </div>
    </body>
    </html>
    """
    with open(filepath, 'w') as f: f.write(html)
    print(f"Report saved to {filepath}")

# --- 6. Main Execution ---
if __name__ == "__main__":
    log = []
    
    # 1. Universe
    c_univ, ids = get_dynamic_coin_universe(log)
    if 'BTC-USD' not in ids: ids['BTC-USD'] = 'bitcoin'
    
    # 2. Download Data (Pure Ticker)
    all_tickers = set(OFFENSIVE_STOCK_UNIVERSE + DEFENSIVE_STOCK_UNIVERSE + CANARY_ASSETS + c_univ + ['BTC-USD'])
    download_required_data(list(all_tickers), log, ids)
    prices = {t: load_price(t) for t in all_tickers}
    
    if prices['BTC-USD'].empty: sys.exit(1)
    target_date = prices['BTC-USD'].index[-1]
    
    # 3. Strategy
    s_port, s_stat = run_stock_strategy_v11(log, prices)
    c_port, c_stat = run_coin_strategy_v12(c_univ, prices, target_date, log)
    
    # 4. Final Port
    final_port = {CASH_ASSET: 0}
    for t, w in s_port.items(): 
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * STOCK_RATIO
    for t, w in c_port.items(): 
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * COIN_RATIO
        
    save_html(log, final_port, s_port, c_port, s_stat, c_stat, target_date)
