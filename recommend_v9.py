import pandas as pd
import numpy as np
import os
import sys
import time
import json
import requests
from datetime import datetime, timezone, timedelta
import pyupbit

# --- Configuration (V9 Fusion) ---
DATA_DIR = "./data"
STOCK_RATIO, COIN_RATIO = 0.60, 0.40
CASH_ASSET = 'Cash'

# Stock V3 Universe (Global Balanced)
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'HYG', 'LQD', 
                           'QUAL', 'MTUM', 'IQLT', 'IMTM'] 
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY_ASSETS = ['SPY', 'EEM', 'VT']

# Coin V4 Universe
COIN_CANARY_ASSET = 'BTC-USD'
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN']

# --- Helper Functions ---
def calculate_sma(s, w):
    if s is None or len(s.dropna()) < w: return np.nan
    return s.rolling(window=w).mean().iloc[-1]

def calculate_return(s, d):
    if s is None or len(s.dropna()) < d + 1: return np.nan
    if s.iloc[-1 - d] == 0: return 0
    return (s.iloc[-1] / s.iloc[-1 - d]) - 1

def calculate_sharpe(s, d=126):
    if s is None or len(s.dropna()) < d + 1: return np.nan
    ret = s.pct_change().iloc[-d:].dropna()
    if ret.std() == 0: return 0.0
    return (ret.mean() / ret.std()) * np.sqrt(252)

def calculate_dual_sma_check(s):
    if s is None or len(s.dropna()) < 100: return False
    sma20 = s.rolling(20).mean().iloc[-1]
    sma100 = s.rolling(100).mean().iloc[-1]
    return sma20 > sma100

def check_stock_canary_v3(all_prices, target_date, lookback=7):
    spy = all_prices.get('SPY')
    eem = all_prices.get('EEM')
    if spy is None or eem is None: return False 
    
    valid_dates = spy.loc[:target_date].index
    if len(valid_dates) < lookback + 100: return False
    check_dates = valid_dates[-lookback:]
    
    raw_signals = []
    for d in check_dates:
        spy_sub = spy.loc[:d]
        eem_sub = eem.loc[:d]
        raw_signals.append(calculate_dual_sma_check(spy_sub) and calculate_dual_sma_check(eem_sub))
        
    return sum(raw_signals) > (lookback / 2)

# --- Data Fetching ---
def get_dynamic_coin_universe(log):
    print("\n--- 🛰️ Step 1: Coin Universe Selection (Live) ---")
    log.append("<h2>🛰️ Step 1: Coin Universe (Top 30 Cap)</h2>")
    
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 30, 'page': 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        try:
            upbit_tickers = pyupbit.get_tickers(fiat="KRW")
            upbit_symbols = {t.split('-')[1] for t in upbit_tickers}
        except:
             upbit_symbols = {'BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'DOGE', 'TRX', 'AVAX', 'DOT', 'MATIC', 'LINK', 'SHIB', 'LTC'}

        candidates = []
        coin_id_map = {}
        for item in data:
            symbol = item['symbol'].upper()
            if symbol in STABLECOINS: continue
            if symbol in upbit_symbols:
                ticker = f"{symbol}-USD"
                candidates.append(ticker)
                coin_id_map[ticker] = item['id']
                
        log.append(f"<p>Selected {len(candidates)} coins from Top 30: {candidates}</p>")
        return candidates, coin_id_map
    except Exception as e:
        print(f"Error fetching universe: {e}. Using Fallback.")
        log.append(f"<p class='error'>Universe Fetch Failed: {e}. Using Fallback Top Coins.</p>")
        fallback = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD', 
                    'TRX-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD', 'SHIB-USD', 'LTC-USD', 'BCH-USD', 'UNI-USD']
        return fallback, {}

def download_data(tickers, log, coin_id_map):
    print("\n--- 📥 Step 2: Downloading Data ---")
    log.append("<h2>📥 Step 2: Downloading Data</h2>")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for t in list(set(tickers)):
        if t == CASH_ASSET: continue
        fp = os.path.join(DATA_DIR, f"{t}.csv")
        try:
            success = False
            if t in coin_id_map: 
                cid = coin_id_map[t]
                url = f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
                resp = requests.get(url, params={'vs_currency':'usd','days':'365'}, timeout=10)
                if resp.status_code == 200:
                    prices = resp.json().get('prices',[])
                    df = pd.DataFrame(prices, columns=['ts','Adj_Close'])
                    df['Date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
                    df[['Date','Adj_Close']].to_csv(fp, index=False)
                    print(f"Downloaded {t} (Gecko)")
                    success = True
                time.sleep(2)
            
            if not success:
                y_ticker = t
                end = int(time.time())
                start = end - (86400 * 500)
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y_ticker}?period1={start}&period2={end}&interval=1d"
                headers = {'User-Agent': 'Mozilla/5.0'}
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    res = resp.json()['chart']['result'][0]
                    ts = res['timestamp']
                    adj = res['indicators']['adjclose'][0]['adjclose']
                    df = pd.DataFrame({'Date': pd.to_datetime(ts, unit='s').date, 'Adj_Close': adj})
                    df = df.dropna().drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
                    df.to_csv(fp, index=False)
                    print(f"Downloaded {t} (Yahoo)")
                else:
                    print(f"Failed {t} (Yahoo): {resp.status_code}")
        except Exception as e:
            print(f"Failed {t}: {e}")
            log.append(f"<p class='error'>Failed {t}: {e}</p>")

def load_prices(tickers):
    prices = {}
    for t in tickers:
        fp = os.path.join(DATA_DIR, f"{t}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, parse_dates=['Date'])
            df = df.dropna().drop_duplicates(subset=['Date'], keep='last')
            prices[t] = df.set_index('Date')['Adj_Close'].sort_index()
    return prices

# --- Strategy Engines ---

def run_stock_strategy_v3(log, all_prices, target_date):
    log.append("<h2>📈 Stock Strategy V3 (Global Balanced) - 60%</h2>")
    
    is_risk_on = check_stock_canary_v3(all_prices, target_date)
    
    if is_risk_on:
        log.append("<p><b>[Canary] ✅ Risk-On (Attack)</b>: SPY/EEM Dual SMA Bullish</p>")
        # Offensive Logic
        candidates = [t for t in OFFENSIVE_STOCK_UNIVERSE if t in all_prices]
        scores = []
        for t in candidates:
            p = all_prices[t].loc[:target_date]
            if len(p) < 130: continue
            mom = calculate_return(p, 126)
            qual = calculate_sharpe(p, 126)
            scores.append({'Ticker': t, 'Momentum': mom, 'Quality': qual})
            
        df = pd.DataFrame(scores).set_index('Ticker')
        if df.empty: return {CASH_ASSET: 1.0}, "No Data"
        
        # Log Detailed Table
        log.append("<h5>- [세부] 공격 모드 팩터 점수:</h5>")
        log.append(df.to_html(classes='dataframe small-table'))

        top_m = df.nlargest(3, 'Momentum').index.tolist()
        top_q = df.nlargest(3, 'Quality').index.tolist()
        picks = list(set(top_m + top_q))
        
        log.append(f"<p>- 최종 주식 포트폴리오: {picks}</p>")
        return {t: 1.0/len(picks) for t in picks}, "Attack"
    else:
        log.append("<p><b>[Canary] 🚨 Risk-Off (Defend)</b>: Signal Bearish</p>")
        results = []
        best_t = CASH_ASSET
        best_ret = -999
        for t in DEFENSIVE_STOCK_UNIVERSE:
            if t in all_prices:
                p = all_prices[t].loc[:target_date]
                r = calculate_return(p, 126)
                if pd.notna(r):
                    results.append({'Ticker': t, 'Ret': r})
                    if r > best_ret:
                        best_ret = r
                        best_t = t
        if results:
             log.append(pd.DataFrame(results).sort_values('Ret', ascending=False).to_html(classes='dataframe small-table'))
             
        log.append(f"<p>Best Defense: {best_t} (6m Ret: {best_ret:.2%})</p>")
        return {best_t: 1.0}, "Defend"

def run_coin_strategy_v4(coin_universe, all_prices, target_date, log, is_today=True):
    log.append(f"<h3>Coin Strategy V4 (Aggressive Alpha) (Date: {target_date.date()})</h3>")
    
    btc = all_prices.get('BTC-USD')
    if btc is None or len(btc) < 55: return {CASH_ASSET: 1.0}, "No Data", log
    
    cur = btc.loc[:target_date].iloc[-1]
    sma = btc.loc[:target_date].rolling(50).mean().iloc[-1]
    
    log.append("<h4>1. 카나리 신호 확인</h4>")
    log.append(f"<p>- BTC 기준(종가 {target_date.date()}): ${cur:,.2f} | 50일 MA: ${sma:,.2f}</p>")
    log.append(f"<p>- [데이터 진단] 사용가능 데이터 수: {len(btc.loc[:target_date])}개</p>")
    
    if pd.isna(sma):
        log.append(f"<p class='error'>- [오류] 50일 이평선 계산 불가 (데이터 부족: {len(btc.loc[:target_date])} < 50)</p>")
        return {CASH_ASSET: 1.0}, "데이터 부족", log
    
    if cur <= sma:
        log.append(f"<p><b>- [결과] 🚨 약세장. 코인 비중을 '{CASH_ASSET}'으로 전환합니다.</b></p>")
        return {CASH_ASSET: 1.0}, "Risk-Off", log
        
    log.append(f"<p><b>- [결과] ✅ 강세장. 코인 투자를 진행합니다.</b></p>")
    
    log.append("<h4>2. 헬스 체크 결과</h4>")
    healthy = []
    rows = []
    for t in coin_universe:
        if t not in all_prices: continue
        p = all_prices[t].loc[:target_date]
        if len(p) < 35: continue
        
        sma30 = p.rolling(30).mean().iloc[-1]
        mom21 = calculate_return(p, 21)
        high21 = p.rolling(21).max().iloc[-1]
        
        is_h = (p.iloc[-1] > sma30) and (mom21 > 0) and (p.iloc[-1] > high21 * 0.7)
        rows.append({
            '코인': t, 
            '현재가': f"${p.iloc[-1]:,.2f}", 
            'SMA30': f"${sma30:,.2f}", 
            'Mom21': f"{mom21:.2%}",
            '최종 결과': '🟢 건강' if is_h else '🔴 비건강'
        })
        if is_h: healthy.append(t)
    
    if rows: log.append(pd.DataFrame(rows).to_html(classes='dataframe small-table', index=False))
    
    if not healthy:
        log.append("<p>- 건강한 코인이 없습니다. 현금 전환.</p>")
        return {CASH_ASSET: 1.0}, "No Healthy", log
    
    log.append("<h4>3. 코인 선정 (샤프 지수 랭킹)</h4>")
        
    scores = []
    for t in healthy:
        p = all_prices[t].loc[:target_date]
        if len(p) < 130: continue
        s = calculate_sharpe(p, 126) + calculate_sharpe(p, 252)
        scores.append({'Coin': t, 'Score': s})
        
    top5 = pd.DataFrame(scores).nlargest(5, 'Score')['Coin'].tolist()
    
    log.append(f"<p>- <b>상위 {len(top5)}개 코인 선정:</b> {top5}</p>")
    log.append("<h4>4. 최종 비중 결정 (역변동성)</h4>")
    
    vols = {t: all_prices[t].loc[:target_date].pct_change().iloc[-90:].std() for t in top5}
    inv_vols = {t: 1/v for t, v in vols.items() if v > 0}
    tot = sum(inv_vols.values())
    w = {}
    if tot > 0: w = {t: v/tot for t, v in inv_vols.items()}
    else: w = {t: 1/len(top5) for t in top5}
    
    return w, "Full Invest", log

# --- Report HTML (V8 Style) ---

def calculate_turnover(p_yesterday, p_today):
    all_assets = set(p_yesterday.keys()) | set(p_today.keys())
    return sum(abs(p_today.get(a, 0) - p_yesterday.get(a, 0)) for a in all_assets) / 2

def save_portfolio_to_html(global_log, final_portfolio, stock_portfolio, coin_portfolio_today, stock_status, coin_status_today, portfolio_yesterday_coin_only, turnover, log_yesterday, log_today, date_yesterday, asset_prices_krw):
    filepath = './portfolio_result.html'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst)
    update_time = now_kst.strftime('%Y년 %m월 %d일 %H:%M:%S KST')
    portfolio_date = now_kst.strftime('%Y년 %m월 %d일')

    sorted_final_portfolio_items = []
    for t, w in final_portfolio.items():
        asset_class = "현금"
        if t in coin_portfolio_today and t != CASH_ASSET: asset_class = "코인"
        elif t in stock_portfolio and t != CASH_ASSET: asset_class = "주식"
        sorted_final_portfolio_items.append({'종목': t, '자산군': asset_class, '최종 비중': w})
    
    cash_item = next((item for item in sorted_final_portfolio_items if item['종목'] == CASH_ASSET), None)
    other_items = [item for item in sorted_final_portfolio_items if item['종목'] != CASH_ASSET]
    other_items.sort(key=lambda x: x['최종 비중'], reverse=True)
    if cash_item:
        sorted_final_portfolio_items = [cash_item] + other_items
    else:
        sorted_final_portfolio_items = other_items

    tbody_html = ""
    for item in sorted_final_portfolio_items:
        tbody_html += f"<tr><td>{item['종목']}</td><td>{item['자산군']}</td><td>{item['최종 비중']:.2%}</td></tr>"
    
    total_weight = sum(p['최종 비중'] for p in sorted_final_portfolio_items)

    final_portfolio_json = json.dumps({p['종목']: p['최종 비중'] for p in sorted_final_portfolio_items})
    
    coin_strategy_portfolio_today_normalized = {}
    if coin_portfolio_today:
        total_coin_weight = sum(coin_portfolio_today.values())
        if total_coin_weight > 0:
            coin_strategy_portfolio_today_normalized = {t: w / total_coin_weight for t, w in coin_portfolio_today.items()}
    coin_strategy_json = json.dumps(coin_strategy_portfolio_today_normalized)

    symbol_to_ticker_map = {}
    if coin_strategy_json:
        coin_strategy_portfolio_for_map = json.loads(coin_strategy_json)
        for ticker in coin_strategy_portfolio_for_map.keys():
            if ticker != CASH_ASSET and ticker.endswith('-USD'):
                symbol = ticker.replace('-USD', '')
                symbol_to_ticker_map[symbol] = ticker
    symbol_to_ticker_map_json = json.dumps(symbol_to_ticker_map)
    asset_prices_json = json.dumps(asset_prices_krw)

    html_template = '''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cap Defend V9 포트폴리오 (Fusion)</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; line-height: 1.6; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2, h3, h4, h5 {{ color: #27ae60; border-bottom: 1px solid #eaecef; padding-bottom: 10px; }}
            h1 {{ font-size: 2em; margin-bottom: 0; }}
            h2.subtitle {{ font-size: 1.2em; color: #888; border: none; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 20px; font-size: 0.9em; }}
            th, td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .final-table th {{ background-color: #27ae60; color: white; }}
            .footer {{ margin-top: 20px; font-size: 0.9em; color: #888; text-align: center; }}
            p {{ margin: 10px 0; }}
            .error {{ color: #e74c3c; }}
            .small-table table {{ width: auto; }}
            .dataframe {{ border-collapse: collapse; width: auto; margin-bottom: 15px; }}
            .dataframe th, .dataframe td {{ padding: 5px 8px; border: 1px solid #ccc; text-align: right; }}
            .dataframe thead th {{ background-color: #f2f2f2; text-align: center; }}
            .calculator-container {{ background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 20px; margin-top: 30px; border-radius: 8px; }}
            .calculator-container input[type="text"], .calculator-container input[type="number"] {{ width: 95%; padding: 8px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px; }}
            .calculator-container button {{ padding: 8px 15px; background-color: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            .calculator-container button:hover {{ background-color: #229954; }}
            #my-turnover-result-container {{ margin-top: 20px; padding: 15px; background-color: #e8f8f5; border: 1px solid #a9dfbf; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏆 Cap Defend V9 포트폴리오</h1>
            <h2 class="subtitle">({portfolio_date} 기준)</h2>
            <p><b>주식 전략 (V3) 상태:</b> {stock_status}</p>
            <p><b>코인 전략 (V4) 상태:</b> {coin_status_today}</p>
            <table class="final-table">
                <thead><tr><th>종목</th><th>자산군</th><th>최종 비중</th></tr></thead>
                <tbody>
                    {tbody_html}
                </tbody>
                <tfoot><tr style="font-weight: bold;"><td colspan="2">총 합계</td><td>{total_weight:.2%}</td></tr></tfoot>
            </table>

            <div class="calculator-container">
                <h3>🧮 총 자산 배분 계산기</h3>
                <p>총 투자 예정 금액(원)을 입력하시면 각 자산에 얼마씩 배분해야 하는지 계산합니다.</p>
                <input type="number" id="total-assets-input" placeholder="총 자산액 (원)" min="0">
                <button id="calculate-total">계산하기</button>
                <div id="total-assets-results" style="margin-top: 15px;"></div>
            </div>

            <div class="calculator-container">
                <h3>🪙 코인 자산 배분 계산기</h3>
                <p>코인에만 투자할 총 금액(원)을 입력하시면 코인과 현금의 배분 금액을 계산합니다.</p>
                <input type="number" id="coin-assets-input" placeholder="코인 총 자산액 (원)" min="0">
                <button id="calculate-coin">계산하기</button>
                <div id="coin-assets-results" style="margin-top: 15px;"></div>
            </div>

            <h2>🔄 코인 포트폴리오 턴오버 분석 (추천 포트폴리오 간)</h2>
            <p>어제({date_yesterday_date})와 오늘({portfolio_date}) 코인 포트폴리오 간의 턴오버 비율: <b>{turnover:.2%}</b></p>

            <hr>

            <div class="calculator-container">
                <h1>🪙 내 포트폴리오 턴오버 계산기</h1>
                <p>현재 보유하고 계신 코인과 현금 보유액을 원화(KRW) 기준으로 입력하시면, 추천 포트폴리오와의 턴오버를 계산해 드립니다.</p>
                
                <h2>1. 내 보유자산 입력</h2>
                <table id="my-portfolio-table">
                    <thead>
                        <tr>
                            <th>자산 (예: BTC, ETH, Cash)</th>
                            <th>보유액 (원)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {my_portfolio_rows}
                    </tbody>
                </table>
                <button id="calculate-my-turnover">내 턴오버 계산하기</button>

                <div id="my-turnover-result-container" style="display:none;">
                    <h2>2. 계산 결과</h2>
                    <div id="my-turnover-result"></div>
                </div>
            </div>

            <hr>
            <h1>📜 상세 실행 로그</h1>
            {global_log_html}
            <h3>오늘 코인 포트폴리오 상세 로그 ({portfolio_date})</h3>
            {log_today_html}
            <h3>어제 코인 포트폴리오 상세 로그 ({date_yesterday_date})</h3>
            {log_yesterday_html}

            <div class="footer">마지막 업데이트: {update_time}</div>
        </div>
        <script>
            const finalPortfolio = {final_portfolio_json};
            const coinStrategyPortfolio = {coin_strategy_json};
            const symbolToTickerMap = {symbol_to_ticker_map_json};
            const assetPrices = {asset_prices_json};

            function formatKRW(num) {{
                return new Intl.NumberFormat('ko-KR').format(num) + ' 원';
            }}

            document.getElementById('calculate-total').addEventListener('click', function() {{
                const totalValue = parseFloat(document.getElementById('total-assets-input').value);
                const resultsDiv = document.getElementById('total-assets-results');
                if (isNaN(totalValue) || totalValue <= 0) {{
                    resultsDiv.innerHTML = '<p style="color:red;">유효한 금액을 입력하세요.</p>';
                    return;
                }}
                
                let tableHtml = '<table class="small-table"><thead><tr><th>종목</th><th>예상 배분 금액</th><th>기준 단가(원)</th><th>예상 수량</th></tr></thead><tbody>';
                const sortedItems = Object.entries(finalPortfolio).sort(([,a],[,b]) => b-a);
                for (const [ticker, weight] of sortedItems) {{
                    const amount = totalValue * weight;
                    let quantity = '-';
                    let priceStr = '-';
                    if (ticker !== 'Cash' && assetPrices[ticker]) {{
                        const price = assetPrices[ticker];
                        priceStr = formatKRW(Math.round(price));
                        const num_units = amount / price;
                        if (num_units < 10) {{
                            quantity = num_units.toFixed(4);
                        }} else {{
                            quantity = num_units.toFixed(2);
                        }}
                    }}
                    tableHtml += `<tr><td>${{ticker}}</td><td>${{formatKRW(Math.round(amount))}}</td><td>${{priceStr}}</td><td>${{quantity}}</td></tr>`;
                }}
                tableHtml += '</tbody></table>';
                resultsDiv.innerHTML = tableHtml;
            }});

            document.getElementById('calculate-coin').addEventListener('click', function() {{
                const totalValue = parseFloat(document.getElementById('coin-assets-input').value);
                const resultsDiv = document.getElementById('coin-assets-results');
                if (isNaN(totalValue) || totalValue <= 0) {{
                    resultsDiv.innerHTML = '<p style="color:red;">유효한 금액을 입력하세요.</p>';
                    return;
                }}

                let tableHtml = '<table class="small-table"><thead><tr><th>자산</th><th>예상 배분 금액</th><th>기준 단가(원)</th><th>예상 수량</th></tr></thead><tbody>';
                const sortedItems = Object.entries(coinStrategyPortfolio).sort(([,a],[,b]) => b-a);
                for (const [ticker, weight] of sortedItems) {{
                    const amount = totalValue * weight;
                    let quantity = '-';
                    let priceStr = '-';
                    if (ticker !== 'Cash' && assetPrices[ticker]) {{
                        const price = assetPrices[ticker];
                        priceStr = formatKRW(Math.round(price));
                        const num_units = amount / price;
                        if (num_units < 10) {{
                            quantity = num_units.toFixed(4);
                        }} else {{
                            quantity = num_units.toFixed(2);
                        }}
                    }}
                    tableHtml += `<tr><td>${{ticker}}</td><td>${{formatKRW(Math.round(amount))}}</td><td>${{priceStr}}</td><td>${{quantity}}</td></tr>`;
                }}
                tableHtml += '</tbody></table>';
                resultsDiv.innerHTML = tableHtml;
            }});

            document.getElementById('calculate-my-turnover').addEventListener('click', function() {{
                const myPortfolio = {{}};
                let totalValue = 0;
                const rows = document.querySelectorAll('#my-portfolio-table tbody tr');
                
                rows.forEach(row => {{
                    const tickerInput = row.querySelector('.ticker-input');
                    const amountInput = row.querySelector('.amount-input');
                    const tickerRaw = tickerInput.value.trim();
                    const amount = parseFloat(amountInput.value);

                    if (tickerRaw && !isNaN(amount) && amount > 0) {{
                        let ticker = tickerRaw;
                        if (ticker.toLowerCase() === 'cash') {{
                            ticker = 'Cash';
                        }} else {{
                            ticker = ticker.toUpperCase();
                            if (symbolToTickerMap[ticker]) {{
                                ticker = symbolToTickerMap[ticker];
                            }}
                        }}
                        myPortfolio[ticker] = (myPortfolio[ticker] || 0) + amount;
                        totalValue += amount;
                    }}
                }});

                if (totalValue === 0) {{
                    alert("유효한 보유자산을 입력해주세요.");
                    return;
                }}

                const myPortfolioWeights = {{}};
                for (const ticker in myPortfolio) {{
                    myPortfolioWeights[ticker] = myPortfolio[ticker] / totalValue;
                }}

                const recommended = coinStrategyPortfolio || {{}};
                const allAssets = new Set([...Object.keys(myPortfolioWeights), ...Object.keys(recommended)]);
                let turnover = 0;

                let resultHtml = '<h3>포트폴리오 비교</h3>';
                resultHtml += '<table class="small-table"><thead><tr><th>자산</th><th>내 비중</th><th>추천 비중</th><th>차이</th></tr></thead><tbody>';

                const sortedAssets = Array.from(allAssets).sort();

                sortedAssets.forEach(asset => {{
                    const myWeight = myPortfolioWeights[asset] || 0;
                    const recommendedWeight = recommended[asset] || 0;
                    const diff = Math.abs(myWeight - recommendedWeight);
                    turnover += diff;

                    resultHtml += `
                        <tr>
                            <td>${{asset}}</td>
                            <td>${{(myWeight * 100).toFixed(2)}}%</td>
                            <td>${{(recommendedWeight * 100).toFixed(2)}}%</td>
                            <td>${{(diff * 100).toFixed(2)}}%</td>
                        </tr>
                    `;
                }});
                
                turnover = turnover / 2;

                resultHtml += '</tbody></table>';
                resultHtml += `<h3>🔄 계산된 턴오버: <strong>${{(turnover * 100).toFixed(2)}}%</strong></h3>`;
                resultHtml += '<p>턴오버는 현재 포트폴리오에서 추천 포트폴리오로 변경하기 위해 매매해야 할 자산의 비율을 의미합니다.</p>';

                document.getElementById('my-turnover-result').innerHTML = resultHtml;
                document.getElementById('my-turnover-result-container').style.display = 'block';
            }});
        </script>
    </body>
    </html>
    '''
    formatted_html = html_template.format(
        portfolio_date=portfolio_date,
        stock_status=stock_status,
        coin_status_today=coin_status_today,
        tbody_html=tbody_html,
        total_weight=total_weight,
        date_yesterday_date=date_yesterday.date(),
        turnover=turnover,
        my_portfolio_rows=''.join(['<tr><td><input type="text" class="ticker-input" placeholder="코인 티커 또는 Cash"></td><td><input type="number" class="amount-input" placeholder="보유액 (원)" min="0"></td></tr>' for _ in range(6)]),
        global_log_html=''.join(global_log),
        log_today_html=''.join(log_today),
        log_yesterday_html=''.join(log_yesterday),
        update_time=update_time,
        final_portfolio_json=final_portfolio_json,
        coin_strategy_json=coin_strategy_json,
        symbol_to_ticker_map_json=symbol_to_ticker_map_json,
        asset_prices_json=asset_prices_json
    )
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(formatted_html)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    global_log = []
    
    current_coin_universe, coin_id_map = get_dynamic_coin_universe(global_log)
    if not current_coin_universe: 
        print("Universe selection failed.")
        sys.exit(1)
    
    # Check BTC Data before downloading everything
    if 'BTC-USD' not in coin_id_map: coin_id_map['BTC-USD'] = 'bitcoin'
    
    tickers_to_download = list(set(OFFENSIVE_STOCK_UNIVERSE + DEFENSIVE_STOCK_UNIVERSE + CANARY_ASSETS + current_coin_universe + ['BTC-USD']))
    download_data(tickers_to_download, global_log, coin_id_map)
    
    print("\n--- 🚀 Step 3: 전략 실행 및 포트폴리오 분석 ---")
    global_log.append("<h2>🚀 Step 3: 전략 실행 및 포트폴리오 분석</h2>")

    all_prices = load_prices(tickers_to_download)
    
    if not all_prices.get('BTC-USD', pd.Series(dtype=float)).empty:
        available_dates = all_prices['BTC-USD'].index.unique().sort_values()
        if len(available_dates) < 3:
            print("Insufficient Data dates.")
            sys.exit(1)
        date_today = available_dates[-1]
        date_yesterday = available_dates[-2]
    else:
        print("Fatal: BTC Data Missing.")
        sys.exit(1)

    # Stock V3 Run
    stock_portfolio, stock_status = run_stock_strategy_v3(global_log, all_prices, date_today)

    # Coin V4 Run (Today & Yesterday for Turnover)
    log_today = []
    coin_portfolio_today, coin_status_today, log_today = run_coin_strategy_v4(current_coin_universe, all_prices, date_today, log_today, is_today=True)
    
    log_yesterday = []
    coin_portfolio_yesterday, _, log_yesterday = run_coin_strategy_v4(current_coin_universe, all_prices, date_yesterday, log_yesterday, is_today=False) # Status mostly irrelevant for turnover check, but function returns it.

    turnover = calculate_turnover(coin_portfolio_yesterday, coin_portfolio_today)

    # Final Weights
    final_portfolio = {}
    for t, w in stock_portfolio.items(): final_portfolio[t] = final_portfolio.get(t, 0) + w * STOCK_RATIO
    for t, w in coin_portfolio_today.items(): final_portfolio[t] = final_portfolio.get(t, 0) + w * COIN_RATIO
    
    # KRW Prices for Calculator
    # Try exchange rate
    try:
        usdt_krw_rate = 1380.0
        resp = requests.get("https://api.frankfurter.app/latest?from=USD&to=KRW", timeout=5)
        if resp.status_code == 200:
            usdt_krw_rate = resp.json()['rates']['KRW']
            print(f"Rate (Forex): {usdt_krw_rate}")
        else:
             # Upbit Fallback
             upbit_rate = pyupbit.get_current_price("KRW-USDT")
             if upbit_rate: usdt_krw_rate = upbit_rate
             print(f"Rate (Upbit): {usdt_krw_rate}")
    except:
        usdt_krw_rate = 1380.0
        print(f"Rate (Fixed): {usdt_krw_rate}")

    asset_prices_krw = {}
    all_assets_port = set(final_portfolio.keys()) | set(coin_portfolio_today.keys())
    
    for asset in all_assets_port:
        if asset == CASH_ASSET: continue
        try:
            if asset.endswith('-USD'):
                sym = asset.replace('-USD', '')
                kp = pyupbit.get_current_price(f"KRW-{sym}")
                if kp: asset_prices_krw[asset] = kp
                else: 
                     p_usd = all_prices[asset].iloc[-1]
                     asset_prices_krw[asset] = p_usd * usdt_krw_rate
            else:
                if asset in all_prices:
                    p_usd = all_prices[asset].iloc[-1]
                    asset_prices_krw[asset] = p_usd * usdt_krw_rate
        except:
            pass

    # Save
    save_portfolio_to_html(global_log, final_portfolio, stock_portfolio, coin_portfolio_today, stock_status, coin_status_today, coin_portfolio_yesterday, turnover, log_yesterday, log_today, date_yesterday, asset_prices_krw)
    
    print(f"\nSaved portfolio_result.html")
