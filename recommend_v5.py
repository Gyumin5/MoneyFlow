import pandas as pd
import numpy as np
import os
import sys
import time
import json
import requests
from datetime import datetime, timezone, timedelta
import pyupbit  # <-- 오류 해결을 위해 추가

# --- 1. 설정 및 상수 (v5.1) ---
DATA_DIR = "./data"
STOCK_RATIO, COIN_RATIO = 0.60, 0.40
COIN_CASH_ASSET = 'Cash'
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN']

OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'EFA', 'QQQ', 'EEM', 'VT', 'VNQ', 'GLD', 'PDBC', 'IEF', 'VEA']
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'GLD', 'DBC']

VT_EEM_CANARY_MA_PERIOD = 200
N_FACTOR_ASSETS = 3
N_SELECTED_COINS = 5

# --- 2. 동적 코인 유니버스 선정 ---
def get_dynamic_coin_universe(log: list) -> list:
    print("\n--- 🛰️ Step 1: 동적 코인 유니버스 선정 시작 (Live API) ---")
    log.append("<h2>🛰️ Step 1: 동적 코인 유니버스 선정 시작 (Live API)</h2>")
    
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    MARKET_CAP_RANK_LIMIT = 20
    MIN_TRADE_VALUE_KRW = 1_000_000_000
    DAYS_TO_CHECK = 31
    headers = {"accept": "application/json"}
    try:
        print(f"\n  - 1. CoinGecko API 호출: 글로벌 시가총액 상위 {MARKET_CAP_RANK_LIMIT}위 코인 조회...")
        log.append(f"<p>  - 1. CoinGecko API 호출: 글로벌 시가총액 상위 {MARKET_CAP_RANK_LIMIT}위 코인 조회...</p>")
        cg_params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': MARKET_CAP_RANK_LIMIT, 'page': 1}
        cg_response = requests.get(COINGECKO_URL, params=cg_params, headers=headers)
        cg_response.raise_for_status()
        cg_data = cg_response.json()
        cg_symbols = {item['symbol'].upper() for item in cg_data}
        
        print("\n  - 2. Upbit 원화마켓 교차 확인 및 유동성 필터링...")
        log.append("<p>  - 2. Upbit 원화마켓 교차 확인 및 유동성 필터링...</p>")
        upbit_krw_tickers_full = pyupbit.get_tickers(fiat="KRW")
        upbit_symbols = {ticker.split('-')[1] for ticker in upbit_krw_tickers_full}
        common_symbols = cg_symbols.intersection(upbit_symbols)
        final_universe = []
        
        print(f"    - 기준: {DAYS_TO_CHECK}일 평균/중간 거래대금 {MIN_TRADE_VALUE_KRW / 1_000_000_000:,.0f}십억 원 이상")
        log.append(f"<p>    - 기준: {DAYS_TO_CHECK}일 평균/중간 거래대금 {MIN_TRADE_VALUE_KRW / 1_000_000_000:,.0f}십억 원 이상</p>")
        for symbol in sorted(list(common_symbols)):
            upbit_ticker = f"KRW-{symbol}"
            df_ohlcv = pyupbit.get_ohlcv(ticker=upbit_ticker, interval="day", count=DAYS_TO_CHECK + 1)
            if df_ohlcv is None or len(df_ohlcv) < DAYS_TO_CHECK: continue
            trade_values = df_ohlcv['value'].iloc[:DAYS_TO_CHECK]
            if trade_values.mean() >= MIN_TRADE_VALUE_KRW and trade_values.median() >= MIN_TRADE_VALUE_KRW:
                if symbol not in STABLECOINS:
                    final_universe.append(f"{symbol}-USD")
                else:
                    print(f"    - 스테이블 코인 제외: {symbol}")
                    log.append(f"<p>    - 스테이블 코인 제외: {symbol}</p>")
            time.sleep(0.2)
    except Exception as e:
        print(f"\n  - [오류] 코인 유니버스 선정 실패: {e}")
        log.append(f"<p class='error'>  - [오류] 코인 유니버스 선정 실패: {e}</p>")
        return []
    
    print(f"\n  -> 최종 선정된 코인 유니버스 ({len(final_universe)}개): {final_universe}")
    log.append(f"<p><b>  -> 최종 선정된 코인 유니버스 ({len(final_universe)}개):</b> {final_universe}</p>")
    print("--- ✅ 동적 코인 유니버스 선정 완료 ---")
    log.append("<h3>✅ 동적 코인 유니버스 선정 완료</h3>")
    return final_universe

# --- 3. 데이터 다운로드 모듈 ---
def download_required_data(tickers: list, log: list):
    print("\n--- 📥 Step 2: 필요 데이터 다운로드 및 업데이트 시작 ---")
    log.append("<h2>📥 Step 2: 필요 데이터 다운로드 및 업데이트 시작</h2>")
    os.makedirs(DATA_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    start_ts, end_ts = int(datetime(2009, 1, 1, tzinfo=timezone.utc).timestamp()), int(datetime.now(timezone.utc).timestamp())
    tickers_to_download = list(set(tickers))
    for ticker in sorted(tickers_to_download):
        if ticker == 'Cash': continue
        filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
        try:
            url, params = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", {"period1": start_ts, "period2": end_ts, "interval": "1d", "includeAdjustedClose": "true"}
            data = session.get(url, params=params, timeout=15).json()['chart']['result'][0]
            df = pd.DataFrame({'Date': pd.to_datetime(data['timestamp'], unit='s').date, 'Adj_Close': data['indicators']['adjclose'][0]['adjclose']}).dropna()
            df.to_csv(filepath, index=False)
            print(f"  - {ticker} 데이터 다운로드/업데이트 완료")
            log.append(f"<p>  - {ticker} 데이터 다운로드/업데이트 완료</p>")
        except Exception as e:
            print(f"  - {ticker} 데이터 다운로드 실패: {e}")
            log.append(f"<p class='error'>  - {ticker} 데이터 다운로드 실패: {e}</p>")
        time.sleep(0.2)
    print("--- ✅ 데이터 준비 완료 ---")
    log.append("<h3>✅ 데이터 준비 완료</h3>")

# --- 4. 계산 헬퍼 및 핵심 전략 구현 (v5.1) ---
def load_price_data(ticker: str) -> pd.Series:
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=['Date'])
        return df.set_index('Date').sort_index()['Adj_Close']
    except Exception:
        return None

def calculate_sma(s: pd.Series, w: int) -> float: 
    if s is None or len(s.dropna()) < w: return np.nan
    return s.rolling(window=w).mean().iloc[-1]

def calculate_return(s: pd.Series, d: int) -> float: 
    if s is None or len(s.dropna()) < d + 1: return np.nan
    if s.iloc[-1 - d] == 0: return -np.inf
    return (s.iloc[-1] / s.iloc[-1 - d]) - 1

def calculate_sharpe_ratio(s: pd.Series, d: int) -> float:
    if s is None or len(s.dropna()) < d + 1: return np.nan
    ret = s.pct_change().iloc[-d:].dropna()
    if ret.empty or ret.std() == 0: return 0.0
    return (ret.mean() / ret.std()) * np.sqrt(252)

def calculate_volatility(s: pd.Series, d: int) -> float:
    if s is None or len(s.dropna()) < d + 1: return np.nan
    ret = s.pct_change().iloc[-d:].dropna()
    if ret.empty: return 0.0
    return ret.std() * np.sqrt(252)

def run_champion_stock_strategy_v5(log: list):
    print("\n--- 📈 주식 포트폴리오 분석 시작 (60%) - Champion VT+EEM v5.1 ---")
    log.append("<h2>📈 주식 포트폴리오 분석 시작 (60%)</h2>")
    vt_prices = load_price_data('VT')
    eem_prices = load_price_data('EEM')
    if vt_prices is None or len(vt_prices.dropna()) < VT_EEM_CANARY_MA_PERIOD or eem_prices is None or len(eem_prices.dropna()) < VT_EEM_CANARY_MA_PERIOD:
        print(f"    - [결과] 🚨 VT/EEM 데이터 부족. 수비 모드로 전환합니다.")
        log.append(f"<p class='error'>    - [결과] 🚨 VT/EEM 데이터 부족. 수비 모드로 전환합니다.</p>")
        return _run_defensive_stock_engine_v5(log), "데이터 부족 (수비 모드)"
    vt_price, eem_price = vt_prices.iloc[-1], eem_prices.iloc[-1]
    vt_sma_200, eem_sma_200 = calculate_sma(vt_prices, VT_EEM_CANARY_MA_PERIOD), calculate_sma(eem_prices, VT_EEM_CANARY_MA_PERIOD)
    
    print(f"    - VT 최신({vt_prices.index[-1].date()}): ${vt_price:,.2f} | 200일 MA: ${vt_sma_200:,.2f}")
    log.append(f"<p>    - VT 최신({vt_prices.index[-1].date()}): ${vt_price:,.2f} | 200일 MA: ${vt_sma_200:,.2f}</p>")
    print(f"    - EEM 최신({eem_prices.index[-1].date()}): ${eem_price:,.2f} | 200일 MA: ${eem_sma_200:,.2f}")
    log.append(f"<p>    - EEM 최신({eem_prices.index[-1].date()}): ${eem_price:,.2f} | 200일 MA: ${eem_sma_200:,.2f}</p>")
    
    if (vt_price > vt_sma_200) and (eem_price > eem_sma_200):
        print(f"    - [결과] ✅ 공격 모드")
        log.append(f"<p><b>    - [결과] ✅ 공격 모드</b></p>")
        return _run_offensive_stock_engine_v5(log), "공격 모드"
    else:
        print(f"    - [결과] 🚨 수비 모드")
        log.append(f"<p><b>    - [결과] 🚨 수비 모드</b></p>")
        return _run_defensive_stock_engine_v5(log), "수비 모드"

def _run_offensive_stock_engine_v5(log: list):
    print("  - 2단계 (공격 모드): 팩터 기반 자산 선정")
    log.append("<p>  - 2단계 (공격 모드): 팩터 기반 자산 선정</p>")
    factor_details = []
    for ticker in OFFENSIVE_STOCK_UNIVERSE:
        p = load_price_data(ticker)
        if p is None or len(p.dropna()) < 253: continue
        ret_63, ret_126, ret_252 = calculate_return(p, 63), calculate_return(p, 126), calculate_return(p, 252)
        sharpe_126 = calculate_sharpe_ratio(p, 126)
        if not any(np.isnan([ret_63, ret_126, ret_252, sharpe_126])) and not any(r == -np.inf for r in [ret_63, ret_126, ret_252]):
            momentum_score = (0.5 * ret_63) + (0.3 * ret_126) + (0.2 * ret_252)
            factor_details.append({'Ticker': ticker, 'Momentum Score': momentum_score, 'Quality (Sharpe)': sharpe_126})
    if not factor_details: return {'SHY': 1.0}
    df = pd.DataFrame(factor_details).set_index('Ticker')
    
    print(f"    - [세부] 공격 모드 팩터 점수:\n{df}")
    log.append(f"<h4>    - [세부] 공격 모드 팩터 점수:</h4>{df.to_html(classes='small-table')}")
    
    top_m = df.sort_values('Momentum Score', ascending=False).index[:N_FACTOR_ASSETS].tolist()
    print(f"    - [세부] 모멘텀 상위 {N_FACTOR_ASSETS}개: {top_m}")
    log.append(f"<p>    - [세부] 모멘텀 상위 {N_FACTOR_ASSETS}개: {top_m}</p>")
    
    top_q = df.sort_values('Quality (Sharpe)', ascending=False).index[:N_FACTOR_ASSETS].tolist()
    print(f"    - [세부] 퀄리티 상위 {N_FACTOR_ASSETS}개: {top_q}")
    log.append(f"<p>    - [세부] 퀄리티 상위 {N_FACTOR_ASSETS}개: {top_q}</p>")
    
    final_assets = sorted(list(set(top_m + top_q)))
    print(f"    - 최종 주식 포트폴리오: {final_assets}")
    log.append(f"<p>    - <b>최종 주식 포트폴리오: {final_assets}</b></p>")
    return {asset: 1.0/len(final_assets) for asset in final_assets} if final_assets else {'SHY': 1.0}

def _run_defensive_stock_engine_v5(log: list):
    print("  - 2단계 (수비 모드): 최적 방어형 자산 선정")
    log.append("<p>  - 2단계 (수비 모드): 최적 방어형 자산 선정</p>")
    momentum_results = []
    for ticker in DEFENSIVE_STOCK_UNIVERSE:
        p = load_price_data(ticker)
        if p is None or len(p.dropna()) < 127: continue
        ret_126 = calculate_return(p, 126)
        if not np.isnan(ret_126) and ret_126 != -np.inf:
            momentum_results.append({'Ticker': ticker, '6m Return': ret_126})
    if not momentum_results: return {'SHY': 1.0}
    df_def = pd.DataFrame(momentum_results).set_index('Ticker')
    
    print(f"    - [세부] 수비 모드 모멘텀 결과:\n{df_def}")
    log.append(f"<h4>    - [세부] 수비 모드 모멘텀 결과:</h4>{df_def.to_html(classes='small-table')}")
    
    positive_momentum_assets = df_def[df_def['6m Return'] > 0]
    if not positive_momentum_assets.empty:
        winner = positive_momentum_assets.sort_values('6m Return', ascending=False).index[0]
        print(f"    - 최종 수비 자산: {winner}")
        log.append(f"<p>    - <b>최종 수비 자산: {winner}</b></p>")
        return {winner: 1.0}
    else:
        print("    - 최종 수비 자산: SHY (모든 자산 6개월 모멘텀 음수)")
        log.append("<p>    - <b>최종 수비 자산: SHY (모든 자산 6개월 모멘텀 음수)</b></p>")
        return {'SHY': 1.0}

def run_crypto_strategy(coin_universe: list, log: list):
    print("\n--- 🪙 코인 포트폴리오 분석 시작 (40%) ---")
    log.append("<h2>🪙 코인 포트폴리오 분석 시작 (40%)</h2>")
    btc = load_price_data('BTC-USD')
    if btc is None or len(btc.dropna()) < 100: return {COIN_CASH_ASSET: 1.0}, "데이터 부족"
    
    if btc.index[-1].date() == datetime.now().date():
        btc_price, btc_sma_100, btc_date_str = btc.iloc[-2], calculate_sma(btc.iloc[:-1], 100), f"전날 종가 {btc.index[-2].date()}"
    else:
        btc_price, btc_sma_100, btc_date_str = btc.iloc[-1], calculate_sma(btc, 100), f"최신 종가 {btc.index[-1].date()}"
    
    print(f"    - BTC 최신({btc_date_str}): ${btc_price:,.2f} | 100일 MA: ${btc_sma_100:,.2f}")
    log.append(f"<p>    - BTC 기준({btc_date_str}): ${btc_price:,.2f} | 100일 MA: ${btc_sma_100:,.2f}</p>")
    
    if btc_price <= btc_sma_100:
        print(f"    - [결과] 🚨 약세장. 코인 비중을 '{COIN_CASH_ASSET}'으로 전환합니다.")
        log.append(f"<p><b>    - [결과] 🚨 약세장. 코인 비중을 '{COIN_CASH_ASSET}'으로 전환합니다.</b></p>")
        return {COIN_CASH_ASSET: 1.0}, "약세장 진입"
    
    print("    - [결과] ✅ 강세장. 코인 투자를 진행합니다.")
    log.append("<p><b>    - [결과] ✅ 강세장. 코인 투자를 진행합니다.</b></p>")
    
    print("    - [세부] 코인별 헬스체크 결과:")
    log.append("<h4>    - [세부] 코인별 헬스체크 결과:</h4>")
    healthy, health_check_logs, health_check_terminal_logs = [], [], []
    for t in coin_universe:
        p = load_price_data(t)
        if p is None or len(p.dropna()) < 64:
            health_check_terminal_logs.append(f"      - {t}: 데이터 부족 (건너김)")
            continue
        if p.index[-1].date() == datetime.now().date():
            current_price, sma_50, ret_63, date_str = p.iloc[-2], calculate_sma(p.iloc[:-1], 50), calculate_return(p.iloc[:-1], 63), f"전날({p.index[-2].date()})"
        else:
            current_price, sma_50, ret_63, date_str = p.iloc[-1], calculate_sma(p, 50), calculate_return(p, 63), f"최신({p.index[-1].date()})"
        
        condition_sma, condition_return = current_price > sma_50, ret_63 > 0
        log_line = f"      - {t}: {date_str} (${current_price:.2f}) > 50일SMA(${sma_50:.2f}) = {condition_sma} | 63일수익률({ret_63:.2%}) > 0 = {condition_return}"
        health_check_terminal_logs.append(log_line)
        if condition_sma and condition_return: healthy.append(t)
    
    print('\n'.join(health_check_terminal_logs))
    log.append(f"<pre>{'<br>'.join(health_check_terminal_logs)}</pre>")

    if not healthy: 
        print("    - [세부] 건강한 코인 없음.")
        log.append("<p>    - [세부] 건강한 코인 없음.</p>")
        return {COIN_CASH_ASSET: 1.0}, "건강한 코인 없음"
    
    print(f"    - [세부] 건강한 코인 ({len(healthy)}개): {healthy}")
    log.append(f"<p>    - [세부] 건강한 코인 ({len(healthy)}개): {healthy}</p>")
    
    ranked_scores = {}
    for t in healthy:
        p = load_price_data(t)
        if p is None or len(p.dropna()) < 253: continue
        series_for_sharpe = p.iloc[:-1] if p.index[-1].date() == datetime.now().date() else p
        score = calculate_sharpe_ratio(series_for_sharpe, 126) + calculate_sharpe_ratio(series_for_sharpe, 252)
        if not np.isnan(score): ranked_scores[t] = score
    if not ranked_scores: 
        print("    - [세부] 랭킹 계산 가능한 코인 없음.")
        log.append("<p>    - [세부] 랭킹 계산 가능한 코인 없음.</p>")
        return {COIN_CASH_ASSET: 1.0}, "랭킹 계산 불가"
    
    print(f"    - [세부] 랭킹 점수: {ranked_scores}")
    log.append(f"<p>    - [세부] 랭킹 점수: {ranked_scores}</p>")
    
    selected = sorted(ranked_scores, key=ranked_scores.get, reverse=True)[:N_SELECTED_COINS]
    print(f"    - [세부] 최종 선택 코인 ({len(selected)}개): {selected}")
    log.append(f"<p>    - [세부] 최종 선택 코인 ({len(selected)}개): {selected}</p>")
    
    vols = {t: calculate_volatility((p.iloc[:-1] if (p := load_price_data(t)).index[-1].date() == datetime.now().date() else p), 90) for t in selected}
    inv_vols = {t: 1/v if v > 0 else 0 for t, v in vols.items()}
    total_inv_vol = sum(inv_vols.values())
    if total_inv_vol == 0: 
        print("    - [세부] 비중 계산 불가 (모든 코인 변동성 0).")
        log.append("<p>    - [세부] 비중 계산 불가 (모든 코인 변동성 0).</p>")
        return {COIN_CASH_ASSET: 1.0}, "비중 계산 불가"
    
    final_weights = {t: v / total_inv_vol for t, v in inv_vols.items()}
    print(f"    - 최종 코인 포트폴리오: {list(final_weights.keys())}")
    log.append(f"<p>    - <b>최종 코인 포트폴리오: {list(final_weights.keys())}</b></p>")
    return final_weights, "강세장 유지"


# --- 5. 결과를 HTML 파일로 저장하는 함수 ---
def save_portfolio_to_html(log_messages, final_portfolio, stock_portfolio, coin_portfolio, stock_status, coin_status):
    filepath = '/home/gmoh/mysite/templates/portfolio_result.html'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst)
    update_time = now_kst.strftime('%Y년 %m월 %d일 %H:%M:%S KST')
    portfolio_date = now_kst.strftime('%Y년 %m월 %d일') # <-- 1. 날짜 제목을 위한 변수 추가

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>자동 포트폴리오 추천</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; line-height: 1.6; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eaecef; padding-bottom: 10px; }}
            h1 {{ font-size: 2em; margin-bottom: 0; }}
            h2.subtitle {{ font-size: 1.2em; color: #888; border: none; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 20px; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .final-table th {{ background-color: #3498db; color: white; }}
            .footer {{ margin-top: 20px; font-size: 0.9em; color: #888; text-align: center; }}
            p {{ margin: 10px 0; }}
            .error {{ color: #e74c3c; }}
            pre {{ background-color: #eee; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-family: 'Courier New', Courier, monospace; }}
            .small-table table {{ width: auto; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏆 최종 v5.1 포트폴리오 추천 🏆</h1>
            <h2 class="subtitle">({portfolio_date} 기준)</h2>
            <p><b>주식 전략 상태:</b> {stock_status}</p>
            <p><b>코인 전략 상태:</b> {coin_status}</p>
            <table class="final-table">
                <thead><tr><th>종목</th><th>자산군</th><th>최종 비중</th></tr></thead>
                <tbody>
    """
    sorted_portfolio = sorted(final_portfolio.items(), key=lambda item: item[1], reverse=True)
    total_weight = 0
    for t, w in sorted_portfolio:
        # 자산군 분류 로직을 더 명확하게 수정
        if t in coin_portfolio:
            asset_class = "현금" if t == COIN_CASH_ASSET else "코인"
        else:
            asset_class = "주식"
        html_content += f"<tr><td>{t}</td><td>{asset_class}</td><td>{w:.2%}</td></tr>"
        total_weight += w
    html_content += f"""
                </tbody>
                <tfoot><tr style="font-weight: bold;"><td colspan="2">총 합계</td><td>{total_weight:.2%}</td></tr></tfoot>
            </table>
            <hr>
            <h1>📜 상세 실행 로그</h1>
            {''.join(log_messages)}
            <div class="footer">마지막 업데이트: {update_time}</div>
        </div>
    </body>
    </html>
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    # HTML 저장은 터미널에 출력하지 않음.
    
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    log_messages = []
    
    current_coin_universe = get_dynamic_coin_universe(log_messages)
    
    tickers_to_download = list(set(OFFENSIVE_STOCK_UNIVERSE + DEFENSIVE_STOCK_UNIVERSE + current_coin_universe + ['BTC-USD', 'SHY']))
    download_required_data(tickers_to_download, log_messages)
    
    print("\n--- 🚀 Step 3: 전략 실행 및 포트폴리오 분석 ---")
    log_messages.append("<h2>🚀 Step 3: 전략 실행 및 포트폴리오 분석</h2>")
    stock_portfolio, stock_status = run_champion_stock_strategy_v5(log_messages)
    coin_portfolio, coin_status = run_crypto_strategy(current_coin_universe, log_messages)
    
    final_portfolio = {}
    for t, w in stock_portfolio.items(): final_portfolio[t] = w * STOCK_RATIO
    for t, w in coin_portfolio.items(): final_portfolio[t] = final_portfolio.get(t, 0) + w * COIN_RATIO
    
    # --- Final Terminal Output ---
    print("\n" + "=" * 60)
    print("               🏆 최종 v5.1 포트폴리오 추천 🏆")
    print("=" * 60)
    print(f"주식 전략 상태: {stock_status}")
    print(f"코인 전략 상태: {coin_status}")
    print("-" * 60)
    print(f"{'종목':<15} | {'자산군':<10} | {'최종 비중':>10}")
    print("-" * 60)
    sorted_portfolio = sorted(final_portfolio.items(), key=lambda item: item[1], reverse=True)
    total_weight = 0
    # --- Final Terminal Output --- 
    for t, w in sorted_portfolio:
        # 자산군 분류 로직을 더 명확하게 수정
        if t in coin_portfolio:
            asset_class = "현금" if t == COIN_CASH_ASSET else "코인"
        else:
            asset_class = "주식"
        print(f" {t:<15} | {asset_class:<10} | {w:>9.2%}")
    ...
    total_weight += w
    print("-" * 60)
    print(f"{'총 합계':<28} | {total_weight:>9.2%}")
    print("=" * 60)

    # --- Save to HTML ---
    save_portfolio_to_html(log_messages, final_portfolio, stock_portfolio, coin_portfolio, stock_status, coin_status)
    print(f"\n웹 결과가 /home/yourusername/mysite/templates/portfolio_result.html 에 저장되었습니다.")
