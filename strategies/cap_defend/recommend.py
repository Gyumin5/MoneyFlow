"""
Cap Defend V14 Recommendation Script (Standard Version)
=======================================================
Stock V14: R6 + EEM-only canary (SMA200, 0.5% hyst) + No health + 12M Mom3+Sh3 EW + Defense Top3
- Universe: SPY, QQQ, VEA, EEM, GLD, PDBC (6 ETFs — VGK→VEA for global coverage)
- Canary: EEM > SMA200 (0.5% hysteresis)
- Health: None (anchor-day robustness test로 제거 — Mom21은 Day1 편향)
- Selection: Mom3+Sh3 union (12M momentum Top3 + Sharpe63 Top3), Equal Weight
- Defense: Top 3 by 6M return from (IEF, BIL, BNDX, GLD, PDBC)

Coin V14: K:SMA(60) + H:Mom(21)+Mom(90)+Vol5% + G5 + EW + DD Exit + Blacklist
- Canary: BTC > SMA(60) + 1% hysteresis
- Health: Mom(21)>0 AND Mom(90)>0 AND Vol(90)<=5%
- Selection: 시총순 Top 5, Equal Weight
- DD Exit: 60d peak -25% → sell warning
- Blacklist: -15% daily drop → 7d exclude
- Crash Breaker: BTC daily -10% → cash warning
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

# Stock Configuration (V14: R6 Universe + EEM-only Canary)
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC']
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY_ASSETS = ['EEM']
STOCK_CANARY_MA_PERIOD = 200
STOCK_CANARY_HYST = 0.005  # 0.5% hysteresis

# Coin Configuration
VOLATILITY_WINDOW = 90
VOL_CAP_FILTER = 0.05
N_SELECTED_COINS = 5
CANARY_SMA_PERIOD = 60
BL_THRESHOLD = -0.15
BL_DAYS = 7
DD_EXIT_LOOKBACK = 60
DD_EXIT_THRESHOLD = -0.25
CRASH_THRESHOLD = -0.10
COIN_CANARY_HYST = 0.01  # 1% hysteresis band

# --- 2. Dynamic Coin Universe ---
def get_dynamic_coin_universe(log: list) -> (list, dict):
    print("\n--- 🛰️ Step 1: Coin Universe Selection (V14) ---")
    log.append("<h2>🛰️ Step 1: 코인 유니버스 선정 (V14: Live CoinGecko Top 40)</h2>")
    
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
        
        if len(final_universe) >= 40: break
    
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
    """V14: Pure 12-month momentum."""
    if len(s) < 253: return -np.inf
    return calc_ret(s, 252)


# --- V14 DD Exit / Blacklist ---
def check_dd_exit(s, lookback=DD_EXIT_LOOKBACK, threshold=DD_EXIT_THRESHOLD):
    """Check if coin should be exited: price / max(recent lookback days) - 1 < threshold."""
    if len(s) < lookback: return False, 0.0
    recent = s.iloc[-lookback:]
    peak = recent.max()
    if peak <= 0: return False, 0.0
    dd = s.iloc[-1] / peak - 1
    return dd <= threshold, dd

def check_blacklist(s, threshold=BL_THRESHOLD, lookback_days=BL_DAYS):
    """Check if coin had a daily drop worse than threshold in the last lookback_days."""
    if len(s) < lookback_days + 1: return False, 0.0
    recent = s.iloc[-(lookback_days + 1):]
    daily_rets = recent.pct_change().dropna()
    worst = daily_rets.min()
    return worst <= threshold, worst

def run_stock_strategy_v14(log, all_prices):
    """V14 Stock Strategy: R6 + EEM-only canary (0.5% hyst) + No health + 12M Mom3+Sh3 EW + Defense Top3"""
    log.append("<h2>📈 주식 포트폴리오 분석 (V14: R6+EEM+12M+EW)</h2>")
    eem = all_prices.get('EEM')

    if eem is not None and len(eem) >= STOCK_CANARY_MA_PERIOD:
        eem_cur = eem.iloc[-1]
        eem_sma = eem.rolling(STOCK_CANARY_MA_PERIOD).mean().iloc[-1]
        dist = eem_cur / eem_sma - 1
        # EEM-only canary with 0.5% hysteresis
        if dist > STOCK_CANARY_HYST:
            risk_on = True
        elif dist < -STOCK_CANARY_HYST:
            risk_on = False
        else:
            risk_on = eem_cur > eem_sma  # dead zone: simple comparison
        log.append(f"<p><b>[Canary]</b> EEM: ${eem_cur:.2f} (MA{STOCK_CANARY_MA_PERIOD} ${eem_sma:.2f}, dist {dist:+.2%}, hyst ±{STOCK_CANARY_HYST:.1%})</p>")
        if risk_on:
            log.append("<p>✅ <b>Risk-On</b></p>")
        else:
            log.append("<p>🚨 <b>Risk-Off</b></p>")
    else:
        risk_on = False
        log.append("<p class='error'>Canary Data Missing (EEM)</p>")

    if risk_on:
        log.append("<h4>🚀 공격 모드 (Mom3+Sh3 Union + EW)</h4>")
        scores = []
        for t in OFFENSIVE_STOCK_UNIVERSE:
            p = all_prices.get(t)
            if p is None or len(p) < 253: continue
            scores.append({'Ticker': t, 'Mom12M': calc_weighted_mom(p), 'Sharpe63': calc_sharpe(p, 63)})

        if not scores:
            log.append("<p class='warning'>공격 ETF 데이터 부족 → 수비 전환</p>")
            # Fall through to defense
        else:
            df = pd.DataFrame(scores).set_index('Ticker')
            try: log.append(f"<div class='table-wrap'>{df.to_html(classes='dataframe small-table', float_format='%.4f')}</div>")
            except: pass

            top_m = df.sort_values('Mom12M', ascending=False).head(3).index.tolist()
            top_s = df.sort_values('Sharpe63', ascending=False).head(3).index.tolist()
            picks = list(dict.fromkeys(top_m + top_s))  # deduplicated, order preserved

            log.append(f"<p>Mom3: {top_m} | Sh3: {top_s}</p>")
            log.append(f"<p>Final Picks ({len(picks)}): <b>{picks}</b> (Equal Weight)</p>")
            return {t: 1.0/len(picks) for t in picks}, "공격 모드"

    # Defense mode: Top 3 by 6M return
    log.append("<h4>🛡️ 수비 모드 (Top 3 by 6M Return)</h4>")
    res = []
    for t in DEFENSIVE_STOCK_UNIVERSE:
        p = all_prices.get(t)
        if p is None: continue
        r = calc_ret(p, 126)
        if pd.notna(r): res.append({'Ticker': t, '6m Ret': r})

    try: log.append(f"<div class='table-wrap'>{pd.DataFrame(res).sort_values('6m Ret', ascending=False).to_html(classes='dataframe small-table')}</div>")
    except: pass

    if not res:
        return {CASH_ASSET: 1.0}, "수비 (데이터 없음)"

    res.sort(key=lambda x: x['6m Ret'], reverse=True)
    top3 = [r for r in res[:3] if r['6m Ret'] > 0]
    if not top3:
        return {CASH_ASSET: 1.0}, "수비 (전부 음수)"
    picks = [r['Ticker'] for r in top3]
    log.append(f"<p>Defense Picks: <b>{picks}</b> (Equal Weight)</p>")
    return {t: 1.0/len(picks) for t in picks}, f"수비 ({', '.join(picks)})"

def run_coin_strategy_v14(coin_universe, all_prices, target_date, log):
    log.append("<h2>🪙 코인 포트폴리오 (V14)</h2>")

    btc = all_prices.get('BTC-USD')
    if len(btc) < CANARY_SMA_PERIOD: return {CASH_ASSET: 1.0}, "데이터 부족"

    tgt_dt = target_date.date() if hasattr(target_date, 'date') else target_date

    # --- Crash Breaker (G5): BTC daily -10% in last 3d → cash ---
    CRASH_COOLDOWN = 3
    if len(btc) >= CRASH_COOLDOWN + 1:
        crash_rets = btc.iloc[-(CRASH_COOLDOWN + 1):].pct_change().dropna()
        worst_crash = crash_rets.min()
        if worst_crash <= CRASH_THRESHOLD:
            log.append(f"<p class='error'><b>[CRASH BREAKER]</b> BTC worst {worst_crash:+.1%} in {CRASH_COOLDOWN}d — 현금 대기</p>")
            return {CASH_ASSET: 1.0}, "CRASH (BTC -10%)"

    # --- Canary: BTC > SMA(60) with 1% Hysteresis ---
    sma = btc.rolling(CANARY_SMA_PERIOD).mean().iloc[-1]
    cur = btc.iloc[-1]
    # Stateless hysteresis: use distance from SMA as proxy
    # ON if cur > sma * 1.01, OFF if cur < sma * 0.99, else maintain trend
    dist = cur / sma - 1
    if dist > COIN_CANARY_HYST:
        canary_on = True
    elif dist < -COIN_CANARY_HYST:
        canary_on = False
    else:
        canary_on = cur > sma  # in dead zone, use simple comparison
    risk_label = '<span style="color:green">Risk-On</span>' if canary_on else '<span style="color:red">Risk-Off</span>'
    hyst_info = f" (Hyst ±{COIN_CANARY_HYST:.0%}, dist {dist:+.2%})"
    log.append(f"<p><b>[Canary]</b> BTC ${cur:,.0f} vs SMA({CANARY_SMA_PERIOD}) ${sma:,.0f} → {risk_label}{hyst_info}</p>")

    # --- Blacklist: -15% daily drop → exclude 7d ---
    blacklisted = []
    for t in coin_universe:
        p = all_prices.get(t)
        if p is None or len(p) < 2: continue
        is_bl, daily_ret = check_blacklist(p)
        if is_bl:
            blacklisted.append((t, daily_ret))
    if blacklisted:
        bl_text = ', '.join([f"{t} ({r:+.1%})" for t, r in blacklisted])
        log.append(f"<p class='warning'><b>[Blacklist]</b> {bl_text} — 7일 제외</p>")
    bl_set = {t for t, _ in blacklisted}
    filtered_universe = [t for t in coin_universe if t not in bl_set]

    # --- Health: Mom(21)>0 AND Mom(90)>0 AND Vol(90)<=5% ---
    rows = []
    healthy = []
    for t in filtered_universe:
        p = all_prices.get(t)
        if p is None or len(p) < 91: continue

        last_dt = p.index[-1].date() if hasattr(p.index[-1], 'date') else p.index[-1]
        if (tgt_dt - last_dt).days != 0: continue

        cur_p = p.iloc[-1]
        mom21 = calc_ret(p, 21)
        mom90 = calc_ret(p, 90)
        vol90 = p.pct_change().iloc[-90:].std()

        is_ok = (pd.notna(mom21) and mom21 > 0 and
                 pd.notna(mom90) and mom90 > 0 and
                 vol90 <= VOL_CAP_FILTER)

        rows.append({'Coin': t, 'Price': f"${cur_p:.4f}",
                     'Mom21': f"{mom21:.2%}" if pd.notna(mom21) else "-",
                     'Mom90': f"{mom90:.2%}" if pd.notna(mom90) else "-",
                     'Vol90': f"{vol90:.4f}",
                     'Status': "🟢" if is_ok else "🔴"})
        if is_ok:
            healthy.append(t)

    try: log.append(f"<div class='table-wrap'>{pd.DataFrame(rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except: pass

    log.append(f"<p>🔍 Healthy Coins: <b>{len(healthy)}</b> {healthy}</p>")

    if not canary_on:
        log.append(f"<p><b>→ 카나리아 OFF: 전량 현금 대기</b></p>")
        return {CASH_ASSET: 1.0}, "Risk-Off"

    if not healthy:
        return {CASH_ASSET: 1.0}, "No Healthy"

    # --- Selection: 시총순 Top 5 (universe order = market cap) ---
    top5 = healthy[:N_SELECTED_COINS]
    log.append(f"<p><b>[Selection]</b> 시총순 Top {N_SELECTED_COINS}: {top5}</p>")

    # --- Weighting: Equal Weight ---
    weights = {t: 1.0 / len(top5) for t in top5}
    w_rows = [{'Coin': t, 'Weight': f"{w:.2%}"} for t, w in weights.items()]
    try: log.append(f"<div class='table-wrap'>{pd.DataFrame(w_rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except: pass

    # --- DD Exit Warning: 60d peak -25% ---
    dd_warnings = []
    for t in top5:
        p = all_prices.get(t)
        if p is None: continue
        is_exit, dd = check_dd_exit(p)
        if is_exit:
            dd_warnings.append((t, dd))
    if dd_warnings:
        dd_text = ', '.join([f"<b>{t}</b> ({dd:+.1%})" for t, dd in dd_warnings])
        log.append(f"<p class='error'><b>[DD EXIT]</b> {dd_text} — 매도 필요 (60d 고점 대비 -25% 초과)</p>")
        # Remove DD-exited coins: their weight goes to CASH (not redistributed)
        cash_weight = 0.0
        for t, _ in dd_warnings:
            if t in weights:
                cash_weight += weights[t]
                del weights[t]
        if cash_weight > 0:
            weights[CASH_ASSET] = weights.get(CASH_ASSET, 0.0) + cash_weight
        if not weights:
            weights = {CASH_ASSET: 1.0}

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
        <title>Cap Defend V14 Recommendation</title>
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
            <h1>🚀 Cap Defend V14</h1>
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
    s_port, s_stat = run_stock_strategy_v14(log, prices)
    c_port, c_stat = run_coin_strategy_v14(c_univ, prices, target_date, log)
    
    # 4. Final Port
    final_port = {CASH_ASSET: 0}
    for t, w in s_port.items(): 
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * STOCK_RATIO
    for t, w in c_port.items(): 
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * COIN_RATIO
        
    save_html(log, final_port, s_port, c_port, s_stat, c_stat, target_date)
