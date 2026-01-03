"""
Cap Defend V10 Recommendation Script
====================================
Synthesized "Optimal" Strategy:
- Stock: V3 Universe + V1 Canary + Enhanced Defense (BIL)
- Coin: V10 Strategy (V4 Logic + InvVol + No Crash Filter)

Generates 'recommend_v10_report.html'
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import json
import requests
from datetime import datetime, timezone, timedelta
import pyupbit

# --- 1. Constants & Configuration ---
DATA_DIR = "./data"
STOCK_RATIO, COIN_RATIO = 0.60, 0.40
CASH_ASSET = 'Cash'
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN', 'FDUSD']

# Stock V10 Configuration
# Universe: V3 Expanded (Global + Factors)
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA', 'GLD', 'PDBC', 'QUAL', 'MTUM', 'IQLT', 'IMTM']
# Defense: Enhanced (Adding BIL, BNDX)
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
# Canary: V1 Standard (200 SMA)
CANARY_ASSETS = ['VT', 'EEM']
STOCK_CANARY_MA_PERIOD = 200
N_FACTOR_ASSETS = 3

# Coin V10 Configuration
COIN_CANARY_MA_PERIOD = 50
HEALTH_FILTER_MA_PERIOD = 30
HEALTH_FILTER_RETURN_PERIOD = 21
N_SELECTED_COINS = 5
VOLATILITY_WINDOW = 90

# --- 2. Dynamic Coin Universe (Same as V8) ---
def get_dynamic_coin_universe(log: list) -> (list, dict):
    print("\n--- 🛰️ Step 1: Coin Universe Selection (Live) ---")
    log.append("<h2>🛰️ Step 1: 동적 코인 유니버스 선정 (Live API)</h2>")
    
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    MARKET_CAP_RANK_LIMIT = 25 
    MIN_TRADE_VALUE_KRW = 1_000_000_000 # 10억
    DAYS_TO_CHECK = 31
    headers = {"accept": "application/json"}
    
    try:
        print(f"  - Fetching Top {MARKET_CAP_RANK_LIMIT} from CoinGecko...")
        cg_params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': MARKET_CAP_RANK_LIMIT, 'page': 1}
        cg_response = requests.get(COINGECKO_URL, params=cg_params, headers=headers, timeout=10)
        cg_response.raise_for_status()
        cg_data = cg_response.json()
        cg_symbol_to_id_map = {item['symbol'].upper(): item['id'] for item in cg_data}
        cg_symbols = set(cg_symbol_to_id_map.keys())
        
        print("  - Cross-checking with Upbit KRW Market...")
        upbit_krw_tickers = pyupbit.get_tickers(fiat="KRW")
        upbit_symbols = {t.split('-')[1] for t in upbit_krw_tickers}
        common_symbols = cg_symbols.intersection(upbit_symbols)
        
        final_universe = []
        coin_id_map = {}
        
        for symbol in sorted(list(common_symbols)):
            if symbol in STABLECOINS: continue
            
            upbit_ticker = f"KRW-{symbol}"
            # Liquidity Check
            try:
                df = pyupbit.get_ohlcv(ticker=upbit_ticker, interval="day", count=DAYS_TO_CHECK + 1)
                if df is None or len(df) < DAYS_TO_CHECK: continue
                avg_val = df['value'].iloc[-DAYS_TO_CHECK:].mean()
                if avg_val >= MIN_TRADE_VALUE_KRW:
                    ticker_usd = f"{symbol}-USD"
                    final_universe.append(ticker_usd)
                    if symbol in cg_symbol_to_id_map:
                        coin_id_map[ticker_usd] = cg_symbol_to_id_map[symbol]
                time.sleep(0.1)
            except: pass
            
    except Exception as e:
        print(f"Error: {e}")
        log.append(f"<p class='error'>Error fetching universe: {e}</p>")
        # Fallback
        fallback = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'AVAX-USD', 'LINK-USD']
        return fallback, {t: t.split('-')[0].lower() if t!='BTC-USD' else 'bitcoin' for t in fallback}

    log.append(f"<p>선정된 유니버스 ({len(final_universe)}개): {final_universe}</p>")
    return final_universe, coin_id_map

# --- 3. Data Download ---
def download_required_data(tickers: list, log: list, coin_id_map: dict):
    print("\n--- 📥 Step 2: 데이터 다운로드 ---")
    log.append("<h2>📥 Step 2: 필요 데이터 다운로드 및 업데이트</h2>")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    for ticker in list(set(tickers)):
        if ticker == CASH_ASSET: continue
        fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        
        success = False
        
        # 1. Try CoinGecko if applicable
        if ticker in coin_id_map:
            try:
                cid = coin_id_map[ticker]
                url = f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
                resp = requests.get(url, params={'vs_currency':'usd','days':'500'}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json().get('prices', [])
                    df = pd.DataFrame(data, columns=['ts', 'Adj_Close'])
                    df['Date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
                    df[['Date','Adj_Close']].drop_duplicates('Date').to_csv(fp, index=False)
                    print(f"  - Downloaded {ticker} (Gecko)")
                    success = True
                else:
                    print(f"  - Gecko Failed {ticker}: {resp.status_code}")
            except Exception as e:
                print(f"  - Gecko Error {ticker}: {e}")
        
        # 2. Fallback to Yahoo (Stocks or Failed Coins)
        if not success:
            try:
                current_timestamp = int(datetime.now(timezone.utc).timestamp())
                start_timestamp = int(datetime(2010, 1, 1, tzinfo=timezone.utc).timestamp())
                
                # Yahoo Query
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                params = {"period1": start_timestamp, "period2": current_timestamp, "interval": "1d", "includeAdjustedClose": "true"}
                resp = session.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    res = resp.json()['chart']['result'][0]
                    ts = res['timestamp']
                    adj = res['indicators']['adjclose'][0]['adjclose']
                    df = pd.DataFrame({'Date': pd.to_datetime(ts, unit='s').date, 'Adj_Close': adj})
                    df.dropna().drop_duplicates('Date').to_csv(fp, index=False)
                    print(f"  - Downloaded {ticker} (Yahoo)")
                else:
                    print(f"  - Yahoo Failed {ticker}: {resp.status_code}")
            except Exception as e:
                print(f"  - Yahoo Error {ticker}: {e}")
            
# --- 4. Logic Engines ---

def load_price(ticker):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=['Date'])
        return df.set_index('Date')['Adj_Close'].sort_index()
    except: return pd.Series(dtype=float)

def calc_sma(s, w):
    return s.rolling(w).mean().iloc[-1] if len(s) >= w else np.nan

def calc_ret(s, d):
    if len(s) < d+1: return np.nan
    if s.iloc[-1-d] == 0: return 0
    return s.iloc[-1]/s.iloc[-1-d] - 1

def calc_sharpe(s, d):
    if len(s) < d+1: return 0
    ret = s.pct_change().iloc[-d:]
    if ret.std() == 0: return 0
    return (ret.mean() / ret.std()) * np.sqrt(252)

def calc_kalmar(s, d):
    if len(s) < d+1: return 0
    # Rolling Max Drawdown (Approx over window d)
    roll_max = s.rolling(d).max()
    dd = (s - roll_max) / roll_max
    min_dd = dd.rolling(d).min().abs().iloc[-1]
    if min_dd < 0.01: min_dd = 0.01
    ret = s.iloc[-1]/s.iloc[-1-d] - 1 # Simple Return or CAGR? Simple Return fine for rank
    # Note: Previous script used pct_change sum? No, pct_change(window) is simple ret.
    return ret / min_dd

def calc_weighted_mom(s):
    if len(s) < 253: return -np.inf
    r3, r6, r12 = calc_ret(s, 63), calc_ret(s, 126), calc_ret(s, 252)
    return 0.5*r3 + 0.3*r6 + 0.2*r12

def run_stock_strategy_v10(log, all_prices, target_date):
    """
    V10 Stock: V3 Universe + v1 Canary (200 SMA on VT/EEM)
    Returns: Portfolio, Status, MetaData (Signal Dist, Next Candidates)
    """
    log.append("<h2>📈 주식 포트폴리오 분석 (V10 - V3 Univ + V1 Canary)</h2>")
    
    # 1. Canary
    vt = all_prices.get('VT', pd.Series(dtype=float)).loc[:target_date]
    eem = all_prices.get('EEM', pd.Series(dtype=float)).loc[:target_date]
    
    risk_on = False
    meta = {'signal_dist': {}, 'next_candidates': []}
    
    if len(vt) >= 200 and len(eem) >= 200:
        vt_sma = vt.rolling(200).mean().iloc[-1]
        eem_sma = eem.rolling(200).mean().iloc[-1]
        vt_cur, eem_cur = vt.iloc[-1], eem.iloc[-1]
        
        vt_dist = (vt_cur - vt_sma) / vt_sma
        eem_dist = (eem_cur - eem_sma) / eem_sma
        meta['signal_dist'] = {'VT': vt_dist, 'EEM': eem_dist}
        
        log.append(f"<p><b>[Canary]</b> VT: ${vt_cur:.2f} ({vt_dist:+.2%}) | EEM: ${eem_cur:.2f} ({eem_dist:+.2%}) vs MA200</p>")
        
        if vt_cur > vt_sma and eem_cur > eem_sma:
            risk_on = True
            log.append("<p>✅ <b>Risk-On (공격)</b>: VT와 EEM 모두 200일선 상회</p>")
        else:
            log.append("<p>🚨 <b>Risk-Off (방어)</b>: 200일선 하회 감지</p>")
    else:
        log.append("<p class='error'>Canary 데이터 부족. 방어 모드 발동.</p>")

    # 2. Selection
    if risk_on:
        log.append("<h4>🚀 공격 모드: 팩터 유니버스(V3) 선정</h4>")
        scores = []
        for t in OFFENSIVE_STOCK_UNIVERSE:
            p = all_prices.get(t, pd.Series(dtype=float)).loc[:target_date]
            if len(p) >= 253:
                mom = calc_weighted_mom(p)
                qual = calc_sharpe(p, 126)
                scores.append({'Ticker': t, 'Mom': mom, 'Qual': qual})
        
        df = pd.DataFrame(scores).set_index('Ticker')
        log.append("<h5>팩터 점수:</h5>")
        log.append(f"<div class='table-wrap'>{df.to_html(classes='dataframe small-table')}</div>")
        
        # Momentum Sorting
        df_m = df.sort_values('Mom', ascending=False)
        top_m = df_m.head(3).index.tolist()
        
        # Quality Sorting
        df_q = df.sort_values('Qual', ascending=False)
        top_q = df_q.head(3).index.tolist()
        
        picks = list(set(top_m + top_q))
        
        meta['selection_reason'] = {
            'Mom_Picks': top_m,
            'Qual_Picks': top_q
        }
        
        # Next Candidates (Simple List)
        all_sorted = df.sort_values('Mom', ascending=False).index.tolist()
        meta['next_candidates'] = [t for t in all_sorted if t not in picks][:3]
        
        log.append(f"<p>최종 주식 포트폴리오: <b>{picks}</b></p>")
        return {t: 1.0/len(picks) for t in picks}, "공격 모드", meta
        
    else:
        log.append("<h4>🛡️ 수비 모드: Enhanced Defense (BIL 포함)</h4>")
        best_t, best_r = CASH_ASSET, -999
        res = []
        for t in DEFENSIVE_STOCK_UNIVERSE:
            p = all_prices.get(t, pd.Series(dtype=float)).loc[:target_date]
            r = calc_ret(p, 126)
            if pd.notna(r):
                res.append({'Ticker': t, '6m Ret': r})
                if r > best_r:
                    best_r, best_t = r, t
        
        log.append(f"<div class='table-wrap'>{pd.DataFrame(res).sort_values('6m Ret', ascending=False).to_html(classes='dataframe small-table')}</div>")
        
        if best_r < 0:
            log.append(f"<p>모든 방어 자산(BIL 포함)의 6개월 모멘텀이 음수입니다. <b>현금(Cash)</b>을 보유합니다.</p>")
            return {CASH_ASSET: 1.0}, "수비 (현금)", meta
        
        log.append(f"<p>최종 방어 자산: <b>{best_t}</b> (6m: {best_r:.2%})<br><small>* BIL보다 수익률이 낮거나 음수면 현금 보유</small></p>")
        return {best_t: 1.0}, f"수비 ({best_t})", meta

def run_coin_strategy_v10(coin_universe, all_prices, target_date, log, is_today=True):
    """
    V10 Coin: V4 Mod (No Crash Filter) + InvVol
    Returns: Portfolio, Status, MetaData (Signal Dist, Next Candidates), Log
    """
    date_str = target_date.date()
    log.append(f"<h3>🪙 코인 포트폴리오 (V10) ({date_str})</h3>")
    
    meta = {'signal_dist': {}, 'next_candidates': []}
    
    # 1. Canary
    btc = all_prices.get('BTC-USD', pd.Series(dtype=float)).loc[:target_date]
    if len(btc) < 50: return {CASH_ASSET: 1.0}, "데이터 부족", meta, log
    
    sma50 = btc.rolling(50).mean().iloc[-1]
    cur = btc.iloc[-1]
    dist = (cur - sma50) / sma50
    meta['signal_dist'] = {'BTC': dist}
    
    log.append(f"<p>[BTC Canary] ${cur:,.0f} ({dist:+.2%}) vs MA50 ${sma50:,.0f}</p>")
    
    # 2. Health (No Crash Filter) - Run ALWAYS for visibility
    healthy = []
    rows = []
    
    # Calculate scores first for sorting/display
    scored_coins = []
    
    for t in coin_universe:
        p = all_prices.get(t, pd.Series(dtype=float)).loc[:target_date]
        if len(p) < 35: continue
        
        sma30 = p.rolling(30).mean().iloc[-1]
        mom21 = calc_ret(p, 21)
        cur_p = p.iloc[-1]
        
        # Calculate Score (Sharpe 6m + 12m) for display
        score = 0.0
        if len(p) >= 253:
            s6 = calc_sharpe(p, 126)
            s12 = calc_sharpe(p, 252)
            score = s6 + s12
            
        is_ok = (cur_p > sma30) and (mom21 > 0)
        status = "🟢" if is_ok else "🔴"
        
        scored_coins.append({
            'Coin': t, 
            'Price': cur_p, # Keep numeric for now
            'SMA30': sma30,
            'Mom21': mom21,
            'Score': score,
            'Status': status,
            'IsOk': is_ok
        })

    # Sort by Score for better visibility
    scored_coins.sort(key=lambda x: x['Score'], reverse=True)
    
    for c in scored_coins:
        rows.append({
            'Coin': c['Coin'], 
            'Price': f"${c['Price']:,.2f}", 
            'SMA30': f"${c['SMA30']:,.2f}", 
            'Mom21': f"{c['Mom21']:.2%}", 
            'Score': f"{c['Score']:.2f}",
            'Status': c['Status']
        })
        if c['IsOk']: healthy.append(c['Coin'])
        
    log.append(f"<div class='table-wrap'>{pd.DataFrame(rows).to_html(classes='dataframe small-table', index=False)}</div>")

    if cur <= sma50:
        log.append("<p>🚨 <b>Risk-Off</b>: BTC 약세장. 전량 현금.</p>")
        return {CASH_ASSET: 1.0}, "Risk-Off", meta, log
        
    log.append("<p>✅ <b>Risk-On</b>: 강세장. 코인 투자 진행.</p>")

    
    if not healthy:
        log.append("<p>건강한 코인이 없습니다. 현금 전환.</p>")
        return {CASH_ASSET: 1.0}, "No Healthy", meta, log

    # 3. Selection (Sharpe 126+252)
    scores = []
    for t in healthy:
        p = all_prices[t].loc[:target_date]
        if len(p) < 253: continue
        s = calc_sharpe(p, 126) + calc_sharpe(p, 252)
        scores.append({'Coin': t, 'Score': s})
    
    if not scores:
         return {CASH_ASSET: 1.0}, "No Scores", meta, log

    # Sort by Score
    score_df = pd.DataFrame(scores).sort_values('Score', ascending=False)
    
    # Log Score Table
    log.append("<h4>🏆 최종 선발 (Sharpe V10)</h4>")
    log.append(f"<div class='table-wrap'>{score_df.head(10).to_html(classes='dataframe small-table', float_format=lambda x: f'{x:.2f}')}</div>")
    
    top5 = score_df.head(5)['Coin'].tolist()
    
    # Next Candidates (Next 5)
    meta['next_candidates'] = score_df.iloc[5:10]['Coin'].tolist()
    
    log.append(f"<p>Selection (Top 5 Sharpe): <b>{top5}</b></p>")

    # 4. Weighting (Inverse Volatility)
    vols = {t: all_prices[t].loc[:target_date].pct_change().iloc[-90:].std() for t in top5}
    inv_vols = {t: 1/v for t, v in vols.items() if v > 0}
    tot = sum(inv_vols.values())
    
    weights = {t: v/tot for t, v in inv_vols.items()} if tot > 0 else {t: 1/len(top5) for t in top5}
    
    # Log weights
    w_rows = [{'Coin': t, 'Vol(90d)': f"{vols[t]:.4f}", 'Weight': f"{w:.2%}"} for t, w in weights.items()]
    log.append(f"<div class='table-wrap'>{pd.DataFrame(w_rows).to_html(classes='dataframe small-table', index=False)}</div>")
    
    return weights, "Full Invest", meta, log


# --- 5. HTML Generation ---

def calculate_turnover(p_prev, p_curr):
    keys = set(p_prev.keys()) | set(p_curr.keys())
    return sum(abs(p_curr.get(k,0) - p_prev.get(k,0)) for k in keys) / 2

def save_html(log_global, final_port, s_port, c_port, s_stat, c_stat, turnover, log_today, log_yesterday, date_today, asset_prices_krw, s_meta, c_meta):
    filepath = "portfolio_result.html"
    
    # Sort
    items = []
    for t, w in final_port.items():
        cat = "현금" if t == CASH_ASSET else ("코인" if t in c_port else "주식")
        items.append({'종목': t, '자산군': cat, '비중': w})
        
    items.sort(key=lambda x: (x['자산군']!='현금', x['비중']), reverse=True)
    
    tbody = ""
    for i in items:
        tbody += f"<tr><td>{i['종목']}</td><td>{i['자산군']}</td><td>{i['비중']:.2%}</td></tr>"
        
    final_json = json.dumps({i['종목']: i['비중'] for i in items})
    
    # Normalized Portfolios for individual calculators
    s_norm = {k: v/sum(s_port.values()) for k,v in s_port.items()} if s_port else {}
    c_norm = {k: v/sum(c_port.values()) for k,v in c_port.items()} if c_port else {}
    
    s_json = json.dumps(s_norm)
    c_json = json.dumps(c_norm)
    
    prices_json = json.dumps(asset_prices_krw)
    
    # Mapping for Calculator
    sym_map = {t.replace('-USD',''): t for t in c_port.keys() if t.endswith('-USD')}
    sym_map_json = json.dumps(sym_map)

    # Signal Distances (Safety Buffer)
    signal_html = "<div style='display:flex; gap: 10px; margin-bottom: 20px;'>"
    def get_badge(dist):
        color = '#27ae60' if dist > 0 else '#e74c3c'
        txt = f"+{dist:.2%}" if dist > 0 else f"{dist:.2%}"
        return f"<span style='background:{color}; color:white; padding:4px 8px; border-radius:4px; font-weight:bold;'>{txt}</span>"
    
    if 'BTC' in c_meta['signal_dist']:
        signal_html += f"<div><strong>BTC:</strong> {get_badge(c_meta['signal_dist']['BTC'])} (vs MA50)</div>"
    if 'VT' in s_meta['signal_dist']:
         signal_html += f"<div><strong>VT:</strong> {get_badge(s_meta['signal_dist']['VT'])} (vs MA200)</div>"
    if 'EEM' in s_meta['signal_dist']:
         signal_html += f"<div><strong>EEM:</strong> {get_badge(s_meta['signal_dist']['EEM'])} (vs MA200)</div>"
    signal_html += "</div>"
    
    # Strategy Insights
    reason = s_meta.get('selection_reason', {})
    mom_picks = ", ".join(reason.get('Mom_Picks', []))
    qual_picks = ", ".join(reason.get('Qual_Picks', []))
    
    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cap Defend V10 Recommendation</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 10px; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            h1 {{ color: #1a73e8; font-size: 1.5rem; margin-bottom: 5px; border-bottom: none; }}
            h2 {{ font-size: 1.2rem; color: #202124; margin-top: 25px; border-bottom: 2px solid #f1f3f4; padding-bottom: 10px; }}
            h3 {{ font-size: 1.1rem; margin-top: 0; }}
            p {{ line-height: 1.5; margin: 5px 0; font-size: 0.95rem; }}
            
            /* Responsive Tables */
            .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 10px 0; border-radius: 8px; border: 1px solid #e0e0e0; }}
            table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
            th, td {{ padding: 12px 15px; border-bottom: 1px solid #f1f3f4; text-align: left; font-size: 0.9rem; }}
            th {{ background: #f8f9fa; color: #5f6368; font-weight: 600; position: sticky; top: 0; }}
            tr:last-child td {{ border-bottom: none; }}
            
            /* Cards & Grid */
            .dashboard-grid {{ display: flex; flex-direction: column; gap: 15px; margin-bottom: 20px; }}
            @media (min-width: 600px) {{ .dashboard-grid {{ flex-direction: row; }} }}
            
            .card {{ background: #fff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; flex: 1; }}
            
            /* Status Bar */
            .status-bar {{ display: flex; flex-wrap: wrap; gap: 10px; background: #e8f0fe; padding: 15px; border-radius: 12px; margin-bottom: 20px; color: #1967d2; font-weight: 500; font-size: 0.9rem; }}
            .status-bar div {{ flex: 1 1 auto; white-space: nowrap; }}
            
            /* Calculator */
            .calc-grid {{ display: flex; flex-direction: column; gap: 15px; }}
            @media (min-width: 600px) {{ .calc-grid {{ flex-direction: row; }} }}
            .calc-box {{ background: #f8f9fa; padding: 15px; border-radius: 12px; border: 1px solid #e0e0e0; flex: 1; }}
            
            input, button {{ width: 100%; padding: 12px; margin: 5px 0; border-radius: 8px; border: 1px solid #dadce0; font-size: 1rem; box-sizing: border-box; }}
            button {{ background: #1a73e8; color: white; border: none; font-weight: 600; cursor: pointer; transition: background 0.2s; }}
            button:active {{ background: #1765cc; }}
            
            .calculator {{ background: #fff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; margin-top: 25px; }}
            
            /* Utils */
            .error {{ color: #d93025; background: #fce8e6; padding: 10px; border-radius: 8px; }}
            .small-text {{ font-size: 0.85rem; color: #5f6368; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Cap Defend V10</h1>
            <p class="small-text">기준일: {date_today.strftime('%Y-%m-%d')}</p>
            
            <div class="dashboard-grid">
                <div class="card">
                    <h3>🛡️ Market Signal</h3>
                    {signal_html}
                </div>
                <div class="card">
                    <h3>🧠 Strategy Insights</h3>
                    <p><strong>Top Mom:</strong> {mom_picks}</p>
                    <p><strong>Top Qual:</strong> {qual_picks}</p>
                </div>
            </div>

            <div class="status-bar">
                <div>📉 주식: {s_stat}</div>
                <div>🪙 코인: {c_stat}</div>
                <div>🔄 턴오버: {turnover:.1%}</div>
            </div>

            <h2>📊 최종 추천 포트폴리오</h2>
            <div class="table-wrap">
                <table>
                    <thead><tr><th>종목</th><th>자산군</th><th>비중</th></tr></thead>
                    <tbody>{tbody}</tbody>
                </table>
            </div>

            <h2>🧮 투자 배분 계산기</h2>
            <div class="calc-grid">
                <div class="calc-box">
                    <h3>💰 전체 투자금</h3>
                    <input type="number" id="invest-amt" placeholder="전체 투자금 (원)" style="width:90%">
                    <button onclick="calc('total')" style="width:100%">계산 (6:4 배분)</button>
                </div>
                <div class="calc-box">
                    <h3>📈 주식만 계산</h3>
                    <input type="number" id="stock-amt" placeholder="주식 투자금 (원)" style="width:90%">
                    <button onclick="calc('stock')" style="width:100%">주식 배분</button>
                </div>
                <div class="calc-box">
                    <h3>🪙 코인만 계산</h3>
                    <input type="number" id="coin-amt" placeholder="코인 투자금 (원)" style="width:90%">
                    <button onclick="calc('coin')" style="width:100%">코인 배분</button>
                </div>
            </div>
            <div id="calc-res" class="calculator" style="display:none;"></div>

            <div class="calculator">
                <h3>🪙 코인 턴오버 계산기</h3>
                <p>현재 보유 코인 입력 시 추천 포트폴리오와의 차이를 계산합니다.</p>
                <div id="my-assets">
                    <!-- JS generated rows -->
                </div>
                <button onclick="calcTurnover()">내 턴오버 계산</button>
                <div id="turnover-res"></div>
            </div>

            <h2>📜 상세 분석 로그</h2>
            {''.join(log_global)}
            {''.join(log_today)}

        </div>

        <script>
            const finalP = {final_json};
            const stockP = {s_json};
            const coinP = {c_json};
            
            const prices = {prices_json};
            const symMap = {sym_map_json};

            function fmt(n) {{ return new Intl.NumberFormat('ko-KR').format(Math.round(n)) + ' 원'; }}

            function calc(type) {{
                let amt = 0;
                let port = {{}};
                let title = "";
                
                if(type === 'total') {{
                    amt = parseFloat(document.getElementById('invest-amt').value);
                    port = finalP;
                    title = "전체 포트폴리오 배분 결과 (주식 60% : 코인 40%)";
                }} else if(type === 'stock') {{
                    amt = parseFloat(document.getElementById('stock-amt').value);
                    port = stockP;
                    title = "주식 포트폴리오 배분 결과 (주식 100%)";
                }} else if(type === 'coin') {{
                    amt = parseFloat(document.getElementById('coin-amt').value);
                    port = coinP;
                    title = "코인 포트폴리오 배분 결과 (코인 100%)";
                }}
                
                if(!amt) return;
                
                let h = `<h3>${{title}}</h3><div class="table-wrap">`;
                h += '<table><thead><tr><th>종목</th><th>비중</th><th>금액</th><th>단가</th><th>수량</th></tr></thead><tbody>';
                
                let sortedKeys = Object.keys(port).sort((a,b) => port[b] - port[a]);
                
                for(let k of sortedKeys) {{
                    const w = port[k];
                    const val = amt * w;
                    let pr = prices[k] || 0;
                    let qty = pr > 0 ? (val/pr).toFixed(4) : '-';
                    h += `<tr><td>${{k}}</td><td>${{(w*100).toFixed(1)}}%</td><td>${{fmt(val)}}</td><td>${{fmt(pr)}}</td><td>${{qty}}</td></tr>`;
                }}
                h += '</tbody></table></div>';
                
                const resDiv = document.getElementById('calc-res');
                resDiv.style.display = 'block';
                resDiv.innerHTML = h;
            }}

            // Init User Input Rows
            let rows = '<div class="table-wrap"><table>';
            for(let i=0; i<6; i++) {{
                rows += `<tr><td><input class="u-tick" placeholder="Ticker (e.g. BTC)"></td><td><input class="u-amt" type="number" placeholder="Value (KRW)"></td></tr>`;
            }}
            rows += '</table></div>';
            document.getElementById('my-assets').innerHTML = rows;

            function calcTurnover() {{
               const inputs = document.querySelectorAll('.u-tick');
               const amts = document.querySelectorAll('.u-amt');
               let myP = {{}}, tot = 0;
               
               inputs.forEach((inp, i) => {{
                   let t = inp.value.toUpperCase().trim();
                   let v = parseFloat(amts[i].value);
                   if(t && v > 0) {{
                       if(symMap[t]) t = symMap[t];
                       else if(t !== 'CASH' && !t.endsWith('-USD')) t += '-USD';
                       
                       myP[t] = (myP[t]||0) + v;
                       tot += v;
                   }}
               }});
               
               if(tot===0) return;
               
               let myW = {{}};
               for(let k in myP) myW[k] = myP[k]/tot;
               
               let diff = 0;
                let allK = new Set([...Object.keys(myW), ...Object.keys(coinP)]);
                
                let h = '<div class="table-wrap"><table><thead><tr><th>Asset</th><th>My</th><th>Target</th><th>Diff</th></tr></thead><tbody>';
               
               allK.forEach(k => {{
                   let w1 = myW[k] || 0;
                   let w2 = coinP[k] || 0;
                   let d = Math.abs(w1-w2);
                   diff += d;
                    h += `<tr><td>${{k}}</td><td>${{(w1*100).toFixed(1)}}%</td><td>${{(w2*100).toFixed(1)}}%</td><td>${{(d*100).toFixed(1)}}%</td></tr>`;
                }});
                diff /= 2;
                h += '</tbody></table></div>';
               
               document.getElementById('turnover-res').innerHTML = `<h3>Turnover: ${{ (diff*100).toFixed(2) }}%</h3>` + h;
            }}
        </script>
    </body>
    </html>
    """
    
    with open(filepath, 'w') as f: f.write(html)
    print(f"Report saved to {filepath}")


# --- 6. Main Execution ---
if __name__ == "__main__":
    log = []
    
    # Uni Selection
    c_univ, ids = get_dynamic_coin_universe(log)
    if 'BTC-USD' not in ids: ids['BTC-USD'] = 'bitcoin'
    
    # Download
    all_tickers = set(OFFENSIVE_STOCK_UNIVERSE + DEFENSIVE_STOCK_UNIVERSE + CANARY_ASSETS + c_univ + ['BTC-USD'])
    download_required_data(list(all_tickers), log, ids)
    
    # Load Prices
    prices = {t: load_price(t) for t in all_tickers}
    
    # Dates
    if prices['BTC-USD'].empty:
        print("Fatal: No BTC Data")
        sys.exit(1)
        
    dates = prices['BTC-USD'].index
    today = dates[-1]
    yest = dates[-2]
    
    print(f"\nAnalysis Date: {today}")
    
    # Stock Run
    s_port, s_stat, s_meta = run_stock_strategy_v10(log, prices, today)
    
    # Coin Run (Today & Yest)
    log_today = []
    c_port, c_stat, c_meta, log_today = run_coin_strategy_v10(c_univ, prices, today, log_today, True)
    
    # Yesterday for Turnover
    c_port_y, _, _, _ = run_coin_strategy_v10(c_univ, prices, yest, [], False)
    turnover = calculate_turnover(c_port_y, c_port)
    # Final Combine
    final = {}
    for t, w in s_port.items(): final[t] = final.get(t,0) + w*STOCK_RATIO
    for t, w in c_port.items(): final[t] = final.get(t,0) + w*COIN_RATIO
    
    # --- EVENT TRIGGER CHECK (Canary Flip) ---
    # Check Yesterday's Stock Status
    # We need to re-run s_port logic for yesterday? Yes, cheap.
    s_port_y, s_stat_y, _ = run_stock_strategy_v10([], prices, yest)
    
    is_canary_flip = (s_stat != s_stat_y)
    if is_canary_flip:
        msg = f"🚨 EVENT TRIGGER: 주식 상태 변경됨! ({s_stat_y} -> {s_stat})"
        print(msg)
        log.append(f"<div style='background:#e74c3c; color:white; padding:15px; border-radius:8px; margin:20px 0;'><h3>{msg}</h3><p>즉시 전체 리밸런싱을 권장합니다. (+0.7% Alpha)</p></div>")
    else:
        log.append(f"<p>주식 상태 유지 중 ({s_stat})</p>")
    
    # Get KRW Prices (Approx)
    krw_prices = {}
    rate = 1400.0
    try:
        r = pyupbit.get_current_price("KRW-USDT")
        if r: rate = r
    except: pass
    
    for t in final.keys():
        if t == CASH_ASSET: continue
        try:
            if t.endswith('-USD'):
                s = t.replace('-USD','')
                kp = pyupbit.get_current_price(f"KRW-{s}")
                krw_prices[t] = kp if kp else prices[t].iloc[-1] * rate
            else:
                krw_prices[t] = prices[t].iloc[-1] * rate
        except: pass
        
    save_html(log, final, s_port, c_port, s_stat, c_stat, turnover, log_today, [], today, krw_prices, s_meta, c_meta)
