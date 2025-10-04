import pandas as pd
import numpy as np
import os
import sys
import time
import json
import requests
from datetime import datetime, timezone, timedelta
import pyupbit

# --- 1. 설정 및 상수 (v7) ---
DATA_DIR = "./data"
STOCK_RATIO, COIN_RATIO = 0.60, 0.40
CASH_ASSET = 'Cash'
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN']

# 주식 전략 (v1) 관련 상수
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA', 'VNQ', 'GLD', 'PDBC']
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'GLD', 'PDBC']
CANARY_ASSETS = ['VT', 'EEM']
STOCK_CANARY_MA_PERIOD = 200
N_FACTOR_ASSETS = 3

# 코인 전략 (v3.0) 관련 상수
COIN_CANARY_MA_PERIOD = 50
HEALTH_FILTER_MA_PERIOD = 30
HEALTH_FILTER_RETURN_PERIOD = 21
N_SELECTED_COINS = 5
CORRELATION_WINDOW = 60
VOLATILITY_WINDOW = 90


# --- 2. 동적 코인 유니버스 선정 ---
def get_dynamic_coin_universe(log: list) -> (list, dict):
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
        cg_symbol_to_id_map = {item['symbol'].upper(): item['id'] for item in cg_data}
        cg_symbols = set(cg_symbol_to_id_map.keys())
        
        print("\n  - 2. Upbit 원화마켓 교차 확인 및 유동성 필터링...")
        log.append("<p>  - 2. Upbit 원화마켓 교차 확인 및 유동성 필터링...</p>")
        upbit_krw_tickers_full = pyupbit.get_tickers(fiat="KRW")
        upbit_symbols = {ticker.split('-')[1] for ticker in upbit_krw_tickers_full}
        common_symbols = cg_symbols.intersection(upbit_symbols)
        final_universe = []
        coin_id_map = {}
        
        print(f"    - 기준: {DAYS_TO_CHECK}일 평균/중간 거래대금 {MIN_TRADE_VALUE_KRW / 1_000_000_000:,.0f}십억 원 이상")
        log.append(f"<p>    - 기준: {DAYS_TO_CHECK}일 평균/중간 거래대금 {MIN_TRADE_VALUE_KRW / 1_000_000_000:,.0f}십억 원 이상</p>")
        for symbol in sorted(list(common_symbols)):
            upbit_ticker = f"KRW-{symbol}"
            df_ohlcv = pyupbit.get_ohlcv(ticker=upbit_ticker, interval="day", count=DAYS_TO_CHECK + 1)
            if df_ohlcv is None or len(df_ohlcv) < DAYS_TO_CHECK: continue
            trade_values = df_ohlcv['value'].iloc[:DAYS_TO_CHECK]
            if trade_values.mean() >= MIN_TRADE_VALUE_KRW and trade_values.median() >= MIN_TRADE_VALUE_KRW:
                if symbol not in STABLECOINS:
                    ticker = f"{symbol}-USD"
                    final_universe.append(ticker)
                    if symbol in cg_symbol_to_id_map:
                        coin_id_map[ticker] = cg_symbol_to_id_map[symbol]
                else:
                    print(f"    - 스테이블 코인 제외: {symbol}")
                    log.append(f"<p>    - 스테이블 코인 제외: {symbol}</p>")
            time.sleep(10) # API rate limit
    except Exception as e:
        print(f"\n  - [오류] 코인 유니버스 선정 실패: {e}")
        log.append(f"<p class='error'>  - [오류] 코인 유니버스 선정 실패: {e}</p>")
        return [], {}
    
    print(f"\n  -> 최종 선정된 코인 유니버스 ({len(final_universe)}개): {final_universe}")
    log.append(f"<p><b>  -> 최종 선정된 코인 유니버스 ({len(final_universe)}개):</b> {final_universe}</p>")
    print("--- ✅ 동적 코인 유니버스 선정 완료 ---")
    log.append("<h3>✅ 동적 코인 유니버스 선정 완료</h3>")
    return final_universe, coin_id_map

# --- 3. 데이터 다운로드 모듈 ---
def download_required_data(tickers: list, log: list, coin_id_map: dict):
    print("\n--- 📥 Step 2: 필요 데이터 다운로드 및 업데이트 시작 ---")
    log.append("<h2>📥 Step 2: 필요 데이터 다운로드 및 업데이트 시작</h2>")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    yahoo_session = requests.Session()
    yahoo_session.headers.update({"User-Agent": "Mozilla/5.0"})
    cg_session = requests.Session()
    cg_session.headers.update({"accept": "application/json"})

    tickers_to_download = list(set(tickers))
    for ticker in sorted(tickers_to_download):
        if ticker == 'Cash': continue
        filepath = os.path.join(DATA_DIR, f"{ticker}.csv")

        if ticker in coin_id_map:
            try:
                coingecko_id = coin_id_map[ticker]
                days_to_fetch = 300 
                url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
                params = {'vs_currency': 'usd', 'days': str(days_to_fetch), 'interval': 'daily'}
                
                response = cg_session.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json().get('prices')

                if not data: raise ValueError("CoinGecko API가 가격 데이터를 반환하지 않았습니다.")

                df = pd.DataFrame(data, columns=['timestamp', 'Adj_Close'])
                df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
                df = df[['Date', 'Adj_Close']].dropna()
                df.to_csv(filepath, index=False)
                print(f"  - {ticker} (CoinGecko) 데이터 다운로드/업데이트 완료")
                log.append(f"<p>  - {ticker} (CoinGecko) 데이터 다운로드/업데이트 완료</p>")
            except Exception as e:
                print(f"  - {ticker} (CoinGecko) 데이터 다운로드 실패: {e}")
                log.append(f"<p class='error'>  - {ticker} (CoinGecko) 데이터 다운로드 실패: {e}</p>")
        else:  # 주식
            try:
                start_ts, end_ts = int(datetime(2009, 1, 1, tzinfo=timezone.utc).timestamp()), int(datetime.now(timezone.utc).timestamp())
                url, params = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", {"period1": start_ts, "period2": end_ts, "interval": "1d", "includeAdjustedClose": "true"}
                data = yahoo_session.get(url, params=params, timeout=15).json()['chart']['result'][0]
                df = pd.DataFrame({'Date': pd.to_datetime(data['timestamp'], unit='s').date, 'Adj_Close': data['indicators']['adjclose'][0]['adjclose']}).dropna()
                df.to_csv(filepath, index=False)
                print(f"  - {ticker} 데이터 다운로드/업데이트 완료")
                log.append(f"<p>  - {ticker} 데이터 다운로드/업데이트 완료</p>")
            except Exception as e:
                print(f"  - {ticker} 데이터 다운로드 실패: {e}")
                log.append(f"<p class='error'>  - {ticker} 데이터 다운로드 실패: {e}</p>")
        
        time.sleep(10) # API rate limit
        
    print("--- ✅ 데이터 준비 완료 ---")
    log.append("<h3>✅ 데이터 준비 완료</h3>")

# --- 4. 계산 헬퍼 및 핵심 전략 구현 (v7) ---
def load_price_data(ticker: str) -> pd.Series:
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=['Date'])
        return df.set_index('Date').sort_index()['Adj_Close']
    except Exception:
        return pd.Series(dtype=float)

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
    return ret.mean() / ret.std() if ret.std() != 0 else 0.0

def run_stock_strategy_v1(log: list, all_prices: dict, target_date: pd.Timestamp):
    log.append("<h2>📈 주식 포트폴리오 분석 시작 (60%)</h2>")
    
    # target_date까지의 데이터만 사용하도록 슬라이싱
    prices_sliced = {t: p.loc[:target_date] for t, p in all_prices.items() if t in CANARY_ASSETS and not p.loc[:target_date].empty}

    vt_prices, eem_prices = prices_sliced.get('VT'), prices_sliced.get('EEM')

    if vt_prices.empty or len(vt_prices.dropna()) < STOCK_CANARY_MA_PERIOD or eem_prices.empty or len(eem_prices.dropna()) < STOCK_CANARY_MA_PERIOD:
        log.append(f"<p class='error'>    - [결과] 🚨 VT/EEM 데이터 부족. 수비 모드로 전환합니다.</p>")
        return _run_defensive_stock_engine_v1(log, all_prices, target_date), "데이터 부족 (수비 모드)"

    vt_price, eem_price = vt_prices.iloc[-1], eem_prices.iloc[-1]
    vt_sma, eem_sma = calculate_sma(vt_prices, STOCK_CANARY_MA_PERIOD), calculate_sma(eem_prices, STOCK_CANARY_MA_PERIOD)
    
    log.append(f"<p>    - VT 최신({vt_prices.index[-1].date()}): ${vt_price:,.2f} | {STOCK_CANARY_MA_PERIOD}일 MA: ${vt_sma:,.2f}</p>")
    log.append(f"<p>    - EEM 최신({eem_prices.index[-1].date()}): ${eem_price:,.2f} | {STOCK_CANARY_MA_PERIOD}일 MA: ${eem_sma:,.2f}</p>")
    
    if (vt_price > vt_sma) and (eem_price > eem_sma):
        log.append(f"<p><b>    - [결과] ✅ 공격 모드</b></p>")
        return _run_offensive_stock_engine_v1(log, all_prices, target_date), "공격 모드"
    else:
        log.append(f"<p><b>    - [결과] 🚨 수비 모드</b></p>")
        return _run_defensive_stock_engine_v1(log, all_prices, target_date), "수비 모드"

def _run_offensive_stock_engine_v1(log: list, all_prices: dict, target_date: pd.Timestamp):
    log.append("<h4>  - 2단계 (공격 모드): 팩터 기반 자산 선정</h4>")
    factor_details = []
    for ticker in OFFENSIVE_STOCK_UNIVERSE:
        p = all_prices.get(ticker)
        if p is None or p.loc[:target_date].empty or len(p.loc[:target_date].dropna()) < 253: continue
        p_sliced = p.loc[:target_date]
        ret_63, ret_126, ret_252 = calculate_return(p_sliced, 63), calculate_return(p_sliced, 126), calculate_return(p_sliced, 252)
        sharpe_126 = calculate_sharpe_ratio(p_sliced, 126) * np.sqrt(252) # 연율화
        if not any(np.isnan([ret_63, ret_126, ret_252, sharpe_126])) and not any(r == -np.inf for r in [ret_63, ret_126, ret_252]):
            momentum_score = (0.5 * ret_63) + (0.3 * ret_126) + (0.2 * ret_252)
            factor_details.append({'Ticker': ticker, 'Momentum Score': round(momentum_score, 2), 'Quality (Sharpe)': round(sharpe_126, 2)})
    
    if not factor_details: return {CASH_ASSET: 1.0}
    df = pd.DataFrame(factor_details).set_index('Ticker')
    log.append(f"<h5>    - [세부] 공격 모드 팩터 점수:</h5>{df.to_html(classes='dataframe small-table')}")
    
    top_m = df.sort_values('Momentum Score', ascending=False).index[:N_FACTOR_ASSETS].tolist()
    top_q = df.sort_values('Quality (Sharpe)', ascending=False).index[:N_FACTOR_ASSETS].tolist()
    
    final_assets = sorted(list(set(top_m + top_q)))
    log.append(f"<p>    - <b>최종 주식 포트폴리오: {final_assets}</b></p>")
    return {asset: 1.0/len(final_assets) for asset in final_assets} if final_assets else {CASH_ASSET: 1.0}

def _run_defensive_stock_engine_v1(log: list, all_prices: dict, target_date: pd.Timestamp):
    log.append("<h4>  - 2단계 (수비 모드): 최적 방어형 자산 선정</h4>")
    momentum_results = []
    for ticker in DEFENSIVE_STOCK_UNIVERSE:
        p = all_prices.get(ticker)
        if p is None or p.loc[:target_date].empty or len(p.loc[:target_date].dropna()) < 127: continue
        p_sliced = p.loc[:target_date]
        ret_126 = calculate_return(p_sliced, 126)
        if not np.isnan(ret_126) and ret_126 != -np.inf:
            momentum_results.append({'Ticker': ticker, '6m Return': ret_126})
    
    if not momentum_results: return {CASH_ASSET: 1.0}
    df_def = pd.DataFrame(momentum_results).set_index('Ticker')
    
    positive_momentum_assets = df_def[df_def['6m Return'] > 0]
    if not positive_momentum_assets.empty:
        winner = positive_momentum_assets.sort_values('6m Return', ascending=False).index[0]
        log.append(f"<p>    - <b>최종 수비 자산: {winner}</b></p>")
        return {winner: 1.0}
    else:
        log.append(f"<p>    - <b>최종 수비 자산: {CASH_ASSET} (모든 자산 6개월 모멘텀 음수)</b></p>")
        return {CASH_ASSET: 1.0}

def run_crypto_strategy_v7(coin_universe: list, all_prices: dict, target_date: pd.Timestamp, log_for_date: list) -> (dict, str, list):
    log_for_date.append(f"<h3>코인 포트폴리오 분석 (기준일: {target_date.date()})</h3>")
    
    # target_date까지의 데이터만 사용하도록 슬라이싱
    prices = {t: p.loc[:target_date] for t, p in all_prices.items() if not p.loc[:target_date].empty}

    # 1. 카나리 신호 확인
    log_for_date.append("<h4>1. 카나리 신호 확인</h4>")
    btc = prices.get('BTC-USD')
    if btc is None or len(btc) < COIN_CANARY_MA_PERIOD:
        log_for_date.append(f"<p class='error'>    - [결과] 🚨 BTC 데이터 부족. 코인 비중을 '{CASH_ASSET}'으로 전환합니다.</p>")
        return {CASH_ASSET: 1.0}, "데이터 부족", log_for_date
    
    btc_price = btc.iloc[-1]
    btc_sma = calculate_sma(btc, COIN_CANARY_MA_PERIOD)
    
    log_for_date.append(f"<p>    - BTC 기준(종가 {btc.index[-1].date()}): ${btc_price:,.2f} | {COIN_CANARY_MA_PERIOD}일 MA: ${btc_sma:,.2f}</p>")
    
    if btc_price <= btc_sma:
        log_for_date.append(f"<p><b>    - [결과] 🚨 약세장. 코인 비중을 '{CASH_ASSET}'으로 전환합니다.</b></p>")
        return {CASH_ASSET: 1.0}, "약세장 진입", log_for_date
    
    log_for_date.append("<p><b>    - [결과] ✅ 강세장. 코인 투자를 진행합니다.</b></p>")

    # 2. 헬스 체크
    log_for_date.append("<h4>2. 헬스 체크 결과</h4>")
    health_check_details = []
    healthy_coins = []
    for t in coin_universe:
        p = prices.get(t)
        if p is None or len(p) < HEALTH_FILTER_MA_PERIOD or len(p) < HEALTH_FILTER_RETURN_PERIOD + 1: continue
        
        sma_val = calculate_sma(p, HEALTH_FILTER_MA_PERIOD)
        ret_val = calculate_return(p, HEALTH_FILTER_RETURN_PERIOD)

        sma_pass = p.iloc[-1] > sma_val
        ret_pass = ret_val > 0
        is_healthy = sma_pass and ret_pass
        
        details = {
            "코인": t, "현재가": f"${p.iloc[-1]:,.2f}", f"{HEALTH_FILTER_MA_PERIOD}일 SMA": f"${sma_val:,.2f}", "SMA 통과": "✅" if sma_pass else "❌",
            f"{HEALTH_FILTER_RETURN_PERIOD}일 수익률": f"{ret_val:.2%}", "수익률 통과": "✅" if ret_pass else "❌",
            "최종 결과": "🟢 건강" if is_healthy else "🔴 비건강"
        }
        health_check_details.append(details)
        if is_healthy: healthy_coins.append(t)
        
    if health_check_details:
        log_for_date.append(pd.DataFrame(health_check_details).to_html(classes='dataframe small-table', index=False))
    else:
        log_for_date.append("<p>  - 헬스 체크를 수행할 코인이 없습니다.</p>")

    # 3. 동적 자산 배분
    coin_alloc_ratio = min(len(healthy_coins) * 0.2, 1.0)
    cash_ratio = 1.0 - coin_alloc_ratio
    log_for_date.append(f"<p>    - <b>건강한 코인 수: {len(healthy_coins)}개 -> 코인 투자 비중: {coin_alloc_ratio:.0%}, 현금 비중: {cash_ratio:.0%}</b></p>")

    if not healthy_coins: return {CASH_ASSET: 1.0}, "건강한 코인 없음", log_for_date

    # 4. 코인 선정 (샤프 지수)
    log_for_date.append("<h4>3. 코인 선정 (샤프 지수 랭킹)</h4>")
    sharpe_scores = []
    for t in healthy_coins:
        p = prices.get(t)
        if p is None or len(p) < 253: continue
        sharpe_126 = calculate_sharpe_ratio(p, 126)
        sharpe_252 = calculate_sharpe_ratio(p, 252)
        if not np.isnan(sharpe_126) and not np.isnan(sharpe_252):
            sharpe_scores.append({"코인": t, "126일 샤프": round(sharpe_126, 2), "252일 샤프": round(sharpe_252, 2), "종합 점수": round(sharpe_126 + sharpe_252, 2)})
    
    if not sharpe_scores:
        log_for_date.append("<p>  - 랭킹 계산 가능한 코인이 없습니다. 현금으로 전환합니다.</p>")
        return {CASH_ASSET: 1.0} if cash_ratio == 1.0 else {CASH_ASSET: cash_ratio}, "랭킹 계산 불가", log_for_date

    sharpe_df = pd.DataFrame(sharpe_scores).sort_values("종합 점수", ascending=False)
    log_for_date.append(sharpe_df.to_html(classes='dataframe small-table', index=False))
    
    selected_coins = sharpe_df['코인'].head(N_SELECTED_COINS).tolist()
    log_for_date.append(f"<p>  - <b>상위 {len(selected_coins)}개 코인 선정:</b> {selected_coins}</p>")

    # 5. 비중 결정 (상관관계 조정)
    log_for_date.append("<h4>4. 최종 비중 결정 (상관관계 조정)</h4>")
    
    returns_df = pd.DataFrame({t: prices[t].pct_change() for t in selected_coins}).iloc[-CORRELATION_WINDOW:].dropna()
    weights = {}
    if returns_df.empty or len(returns_df.columns) < 2:
        log_for_date.append("<p>  - 상관관계 계산 불가. 역변동성 가중치 적용.</p>")
        vols = {t: prices[t].pct_change().iloc[-VOLATILITY_WINDOW:].std() for t in selected_coins}
        inv_vols = {t: 1/v if v > 0 else 0 for t, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        weights = {t: v / total_inv_vol for t, v in inv_vols.items()} if total_inv_vol > 0 else {t: 1/len(selected_coins) for t in selected_coins}
    else:
        vols = returns_df.std()
        correlations = returns_df.corr()
        
        diversification_scores = pd.Series({t: 1.0 / (1.0 + correlations.loc[t, [c for c in selected_coins if c != t]].sum()) for t in selected_coins})
        vols[vols == 0] = 1e-10
        
        final_scores = (1 / vols) * diversification_scores
        weights = (final_scores / final_scores.sum()).to_dict()

        weighting_details = []
        for t in selected_coins:
            weighting_details.append({
                "코인": t,
                "역변동성 점수": round(1/vols.get(t, 1e-10), 2),
                "다각화 점수": round(diversification_scores.get(t, 0), 2),
                "최종 점수": round(final_scores.get(t, 0), 2),
                "최종 비중": f"{weights.get(t, 0):.2%}"
            })
        log_for_date.append(pd.DataFrame(weighting_details).to_html(classes='dataframe small-table', index=False))

    final_coin_portfolio = {t: w * coin_alloc_ratio for t, w in weights.items()}
    if cash_ratio > 1e-6: final_coin_portfolio[CASH_ASSET] = cash_ratio
    
    log_for_date.append(f"<p>    - <b>최종 코인 포트폴리오: {list(final_coin_portfolio.keys())}</b></p>")
    return final_coin_portfolio, f"강세장 (동적 비중)", log_for_date

def calculate_turnover(p_yesterday: dict, p_today: dict) -> float:
    all_assets = set(p_yesterday.keys()) | set(p_today.keys())
    turnover = 0.5 * sum(abs(p_today.get(asset, 0) - p_yesterday.get(asset, 0)) for asset in all_assets)
    return turnover

# --- 5. 결과를 HTML 파일로 저장하는 함수 ---
def save_portfolio_to_html(global_log: list, final_portfolio: dict, stock_portfolio: dict, coin_portfolio_today: dict, stock_status: str, coin_status_today: str, portfolio_yesterday_coin_only: dict, portfolio_today_coin_only: dict, turnover: float, log_yesterday: list, log_today: list, date_yesterday: pd.Timestamp):
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
        sorted_portfolio_items = other_items

    tbody_html = ""
    for item in sorted_final_portfolio_items:
        tbody_html += f"<tr><td>{item['종목']}</td><td>{item['자산군']}</td><td>{item['최종 비중']:.2%}</td></tr>"
    
    total_weight = sum(p['최종 비중'] for p in sorted_final_portfolio_items)

    final_portfolio_json = json.dumps({p['종목']: p['최종 비중'] for p in sorted_final_portfolio_items})
    
    # 코인 전략 포트폴리오 (코인 전략 내 비중, 현금 포함) - 오늘
    coin_strategy_portfolio_today_normalized = {}
    if coin_portfolio_today:
        total_coin_weight = sum(coin_portfolio_today.values())
        if total_coin_weight > 0:
            coin_strategy_portfolio_today_normalized = {t: w / total_coin_weight for t, w in coin_portfolio_today.items()}
    coin_strategy_json = json.dumps(coin_strategy_portfolio_today_normalized)

    # 어제 코인 포트폴리오 (코인 전략 내 비중, 현금 포함) - 정규화된 값
    coin_strategy_portfolio_yesterday_normalized = {}
    if portfolio_yesterday_coin_only:
        total_coin_weight_yesterday = sum(portfolio_yesterday_coin_only.values())
        if total_coin_weight_yesterday > 0:
            coin_strategy_portfolio_yesterday_normalized = {t: w / total_coin_weight_yesterday for t, w in portfolio_yesterday_coin_only.items()}

    html_template = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>자동 포트폴리오 추천 (v7)</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; line-height: 1.6; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2, h3, h4, h5 {{ color: #2c3e50; border-bottom: 1px solid #eaecef; padding-bottom: 10px; }}
            h1 {{ font-size: 2em; margin-bottom: 0; }}
            h2.subtitle {{ font-size: 1.2em; color: #888; border: none; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 20px; font-size: 0.9em; }}
            th, td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .final-table th {{ background-color: #3498db; color: white; }}
            .footer {{ margin-top: 20px; font-size: 0.9em; color: #888; text-align: center; }}
            p {{ margin: 10px 0; }}
            .error {{ color: #e74c3c; }}
            .small-table table {{ width: auto; }}
            .dataframe {{ border-collapse: collapse; width: auto; margin-bottom: 15px; }}
            .dataframe th, .dataframe td {{ padding: 5px 8px; border: 1px solid #ccc; text-align: right; }}
            .dataframe thead th {{ background-color: #f2f2f2; text-align: center; }}
            .calculator-container {{ background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 20px; margin-top: 30px; border-radius: 8px; }}
            .calculator-container input[type="number"] {{ width: 200px; padding: 8px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px; }}
            .calculator-container button {{ padding: 8px 15px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            .calculator-container button:hover {{ background-color: #2980b9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏆 최종 v7 포트폴리오 추천 🏆</h1>
            <h2 class="subtitle">
            ({portfolio_date} 기준)</h2>
            <p><b>주식 전략 상태:</b> {stock_status}</p>
            <p><b>코인 전략 상태:</b> {coin_status_today}</p>
            <table class="final-table">
                <thead><tr><th>종목</th><th>자산군</th><th>최종 비중</th></tr></thead>
                <tbody>
                    {tbody_html}
                </tbody>
                <tfoot><tr style="font-weight: bold;"><td colspan="2">총 합계</td><td>{total_weight_str}</td></tr></tfoot>
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

            <h2>🔄 코인 포트폴리오 턴오버 분석</h2>
            <p>어제({yesterday_date})와 오늘({today_date}) 코인 포트폴리오 간의 턴오버 비율: <b>{turnover:.2%}</b></p>

            <hr>
            <h1>📜 상세 실행 로그</h1>
            {global_log_html}
            <h3>오늘 코인 포트폴리오 상세 로그 ({today_date})</h3>
            {log_today_html}
            <h3>어제 코인 포트폴리오 상세 로그 ({yesterday_date})</h3>
            {log_yesterday_html}

            <div class="footer">마지막 업데이트: {update_time}</div>
        </div>
        <script>
            const finalPortfolio = {final_portfolio_json};
            const coinStrategyPortfolio = {coin_strategy_json};

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
                
                let tableHtml = '<table class="small-table"><thead><tr><th>종목</th><th>예상 배분 금액</th></tr></thead><tbody>';
                const sortedItems = Object.entries(finalPortfolio).sort(([,a],[,b]) => b-a);
                for (const [ticker, weight] of sortedItems) {{
                    const amount = totalValue * weight;
                    tableHtml += `<tr><td>${{ticker}}</td><td>${{formatKRW(Math.round(amount))}}</td></tr>`;
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

                let tableHtml = '<table class="small-table"><thead><tr><th>자산</th><th>예상 배분 금액</th></tr></thead><tbody>';
                const sortedItems = Object.entries(coinStrategyPortfolio).sort(([,a],[,b]) => b-a);
                for (const [ticker, weight] of sortedItems) {{
                    const amount = totalValue * weight;
                    tableHtml += `<tr><td>${{ticker}}</td><td>${{formatKRW(Math.round(amount))}}</td></tr>`;
                }}
                tableHtml += '</tbody></table>';
                resultsDiv.innerHTML = tableHtml;
            }});
        </script>
    </body>
    </html>
    """
    html_content = html_template.format(
        portfolio_date=portfolio_date,
        stock_status=stock_status,
        coin_status_today=coin_status_today,
        tbody_html=tbody_html,
        total_weight_str=f"{total_weight:.2%}",
        global_log_html=''.join(global_log),
        log_today_html=''.join(log_today),
        log_yesterday_html=''.join(log_yesterday),
        update_time=update_time,
        final_portfolio_json=final_portfolio_json,
        coin_strategy_json=coin_strategy_json,
        turnover=turnover,
        today_date=portfolio_date,
        yesterday_date=date_yesterday.date() # Corrected line
    )
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    global_log = []
    
    current_coin_universe, coin_id_map = get_dynamic_coin_universe(global_log)
    
    if not current_coin_universe: 
        global_log.append("<p class='error'>코인 유니버스 선정에 실패하여 프로그램을 종료합니다.</p>")
        print("\n" + "".join(global_log))
        exit()

    if 'BTC-USD' not in coin_id_map:
        coin_id_map['BTC-USD'] = 'bitcoin'
    
    tickers_to_download = list(set(OFFENSIVE_STOCK_UNIVERSE + DEFENSIVE_STOCK_UNIVERSE + CANARY_ASSETS + current_coin_universe + ['BTC-USD']))
    download_required_data(tickers_to_download, global_log, coin_id_map)
    
    print("\n--- 🚀 Step 3: 전략 실행 및 포트폴리오 분석 ---")
    global_log.append("<h2>🚀 Step 3: 전략 실행 및 포트폴리오 분석</h2>")

    all_prices = {ticker: load_price_data(ticker) for ticker in tickers_to_download}
    all_prices = {k: v for k, v in all_prices.items() if not v.empty}

    if not all_prices.get('BTC-USD', pd.Series(dtype=float)).empty:
        available_dates = all_prices['BTC-USD'].index.unique().sort_values()
        if len(available_dates) < 2:
            global_log.append("<p class='error'>데이터가 충분하지 않아 어제/오늘 포트폴리오를 계산할 수 없습니다. 종료합니다.</p>")
            print("\n" + "".join(global_log))
            exit()
        date_today = available_dates[-1]
        date_yesterday = available_dates[-2]
    else:
        global_log.append("<p class='error'>BTC 데이터가 없어 날짜를 설정할 수 없습니다. 종료합니다.</p>")
        print("\n" + "".join(global_log))
        exit()

    # 주식 포트폴리오 (오늘 기준)
    stock_portfolio, stock_status = run_stock_strategy_v1(global_log, all_prices, date_today)

    # 코인 포트폴리오 (오늘 기준)
    log_today_coin_calc = []
    coin_portfolio_today, coin_status_today, log_today_coin_calc = run_crypto_strategy_v7(current_coin_universe, all_prices, date_today, log_today_coin_calc)
    
    # 코인 포트폴리오 (어제 기준)
    log_yesterday_coin_calc = []
    coin_portfolio_yesterday, coin_status_yesterday, log_yesterday_coin_calc = run_crypto_strategy_v7(current_coin_universe, all_prices, date_yesterday, log_yesterday_coin_calc)

    # 턴오버 계산
    turnover = calculate_turnover(coin_portfolio_yesterday, coin_portfolio_today)

    final_portfolio = {}
    for t, w in stock_portfolio.items(): final_portfolio[t] = final_portfolio.get(t, 0) + w * STOCK_RATIO
    for t, w in coin_portfolio_today.items(): final_portfolio[t] = final_portfolio.get(t, 0) + w * COIN_RATIO
    
    # --- 최종 터미널 출력 ---
    print("\n" + "=" * 60)
    print("               🏆 최종 v7 포트폴리오 추천 🏆")
    print("=" * 60)
    print(f"주식 전략 상태: {stock_status}")
    print(f"코인 전략 상태 (오늘): {coin_status_today}")
    print(f"코인 전략 상태 (어제): {coin_status_yesterday}")
    print("-" * 60)
    print(f"{'종목':<15} | {'자산군':<10} | {'최종 비중':>10}")
    print("-" * 60)
    
    sorted_final = sorted(final_portfolio.items(), key=lambda item: item[1], reverse=True)
    total_weight = 0
    for t, w in sorted_final:
        asset_class = "현금"
        if t in coin_portfolio_today and t != CASH_ASSET: asset_class = "코인"
        elif t in stock_portfolio and t != CASH_ASSET: asset_class = "주식"
        print(f" {t:<15} | {asset_class:<10} | {w:>9.2%}")
        total_weight += w
    print("-" * 60)
    print(f"{'총 합계':<28} | {total_weight:>9.2%}")
    print("=" * 60)
    print(f"\n🔄 코인 포트폴리오 턴오버 (어제/오늘): {turnover:.2%}")

    save_portfolio_to_html(global_log, final_portfolio, stock_portfolio, coin_portfolio_today, stock_status, coin_status_today, coin_portfolio_yesterday, coin_portfolio_today, turnover, log_yesterday_coin_calc, log_today_coin_calc, date_yesterday)
    print(f"\n웹 결과가 portfolio_result.html 에 저장되었습니다.")