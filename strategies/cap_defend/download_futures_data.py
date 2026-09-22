#!/usr/bin/env python3
"""바이낸스 선물 OHLCV + 펀딩레이트 다운로드.

기본 동작은 증분 업데이트다.
- 기존 CSV가 있으면 마지막 시점 이후만 이어받는다.
- `--full`일 때만 처음부터 다시 받는다.
"""
import requests, pandas as pd, time, os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'futures')
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = 'https://fapi.binance.com'

STABLE = {'USDT', 'USDC', 'DAI', 'TUSD', 'BUSD', 'FDUSD', 'PYUSD', 'USDD',
          'RLUSD', 'USDG', 'EURC', 'XAUT', 'PAXG', 'USDE', 'USD1'}
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'data', 'historical_universe.json')
UNIVERSE_TOP = 40          # 전략이 실제로 보는 후보 깊이
UNIVERSE_FROM = '2020-09-01'   # 백테스트 시작(2020-10-01) 직전부터


def target_symbols():
    """받아야 할 심볼 = 시점별 상위 40 에 한 번이라도 든 코인 중 무기한선물이 있는 것.

    고정 목록으로 두면 조용히 낡는다 — 2026-09 시점에 상위 40 중 스물한 종의
    가격이 없어서 백테스트가 그 코인들을 후보로 보지도 못하는 상태였다.
    목록을 시점별 유니버스에서 만들면 유니버스를 갱신할 때 같이 따라온다.
    """
    import json
    try:
        r = requests.get(f'{BASE_URL}/fapi/v1/exchangeInfo', timeout=30).json()
        perp = {s['symbol'] for s in r['symbols']
                if s.get('contractType') == 'PERPETUAL'
                and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'}
    except Exception as exc:
        raise SystemExit(f'exchangeInfo 조회 실패: {exc}')

    with open(UNIVERSE_FILE) as f:
        months = json.load(f)

    want = set()
    for month, tickers in months.items():
        if month < UNIVERSE_FROM:
            continue
        n = 0
        for t in tickers:
            sym = t.replace('-USD', '')
            if sym in STABLE:
                continue
            n += 1
            want.add(sym)
            if n >= UNIVERSE_TOP:
                break

    # 값이 작은 코인은 계약이 1000배 단위다(1000SHIBUSDT 등). 같은 자산이고
    # 수익률·모멘텀은 배수와 무관하므로 그 이름으로 받는다. 이걸 빼면 시바이누가
    # 64개월, 페페가 28개월 동안 후보에서 통째로 빠진다.
    picked = set()
    for s in want:
        for cand in (s + 'USDT', '1000' + s + 'USDT'):
            if cand in perp:
                picked.add(cand)
                break

    # 이미 받아둔 심볼은 유니버스에서 빠졌어도 계속 갱신한다 — 과거 구간에서 쓰인다.
    existing = {f.split('_')[0] for f in os.listdir(DATA_DIR) if f.endswith('_1h.csv')}
    return sorted(picked | existing)


SYMBOLS = target_symbols()


def download_klines(symbol, interval='1h', start='2019-01-01', limit=1500, start_ts_override=None):
    """바이낸스 선물 klines 다운로드."""
    all_data = []
    start_ts = start_ts_override if start_ts_override is not None else int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    while start_ts < end_ts:
        url = f'{BASE_URL}/fapi/v1/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'limit': limit,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 400:
                # Symbol might not exist
                return None
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  Error {symbol}: {e}")
            time.sleep(5)
            continue

        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1][0] + 1  # next ms after last candle
        time.sleep(0.1)  # rate limit

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=[
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore'
    ])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = df[c].astype(float)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates('Date')
    df = df.set_index('Date').sort_index()
    return df


def download_funding(symbol, start='2019-01-01', start_ts_override=None):
    """바이낸스 선물 펀딩레이트 다운로드."""
    all_data = []
    start_ts = start_ts_override if start_ts_override is not None else int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    while start_ts < end_ts:
        url = f'{BASE_URL}/fapi/v1/fundingRate'
        params = {
            'symbol': symbol,
            'startTime': start_ts,
            'limit': 1000,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 400:
                return None
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  Funding error {symbol}: {e}")
            time.sleep(5)
            continue

        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1]['fundingTime'] + 1
        time.sleep(0.1)

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    df['Date'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    df = df[['Date', 'fundingRate']].drop_duplicates('Date')
    df = df.set_index('Date').sort_index()
    return df


if __name__ == '__main__':
    full_redownload = '--full' in sys.argv

    for sym in SYMBOLS:
        coin = sym.replace('USDT', '')
        fpath_1h = os.path.join(DATA_DIR, f'{sym}_1h.csv')
        existing_df = None
        start_ts = None
        if os.path.exists(fpath_1h) and not full_redownload:
            try:
                existing_df = pd.read_csv(fpath_1h, parse_dates=['Date'], index_col='Date')
                if len(existing_df) > 0:
                    last_dt = existing_df.index[-1]
                    start_ts = int((last_dt + pd.Timedelta(hours=1)).timestamp() * 1000)
            except Exception:
                existing_df = None

        mode = 'full' if full_redownload or existing_df is None else 'append'
        print(f"  {coin}: downloading 1h ({mode})...", end='', flush=True)
        df_new = download_klines(sym, '1h', start_ts_override=start_ts)

        if existing_df is not None and df_new is not None and len(df_new) > 0:
            df = pd.concat([existing_df, df_new]).sort_index()
            df = df[~df.index.duplicated(keep='last')]
        elif existing_df is not None and (df_new is None or len(df_new) == 0):
            df = existing_df
        else:
            df = df_new

        if df is not None and len(df) > 100:
            df.to_csv(fpath_1h)
            if df_new is not None and len(df_new) > 0:
                print(f" +{len(df_new)} -> {len(df)} bars ({df.index[0].date()} ~ {df.index[-1].date()})")
            else:
                print(f" up-to-date ({len(df)} bars, last={df.index[-1]})")
        else:
            print(" FAILED or no data")
            continue

        # Funding
        fpath_f = os.path.join(DATA_DIR, f'{sym}_funding.csv')
        existing_fdf = None
        funding_start_ts = None
        if os.path.exists(fpath_f) and not full_redownload:
            try:
                existing_fdf = pd.read_csv(fpath_f, parse_dates=['Date'], index_col='Date')
                if len(existing_fdf) > 0:
                    last_dt = existing_fdf.index[-1]
                    funding_start_ts = int((last_dt + pd.Timedelta(milliseconds=1)).timestamp() * 1000)
            except Exception:
                existing_fdf = None

        print(f"  {coin}: downloading funding ({mode})...", end='', flush=True)
        fdf_new = download_funding(sym, start_ts_override=funding_start_ts)

        if existing_fdf is not None and fdf_new is not None and len(fdf_new) > 0:
            fdf = pd.concat([existing_fdf, fdf_new]).sort_index()
            fdf = fdf[~fdf.index.duplicated(keep='last')]
        elif existing_fdf is not None and (fdf_new is None or len(fdf_new) == 0):
            fdf = existing_fdf
        else:
            fdf = fdf_new

        if fdf is not None and len(fdf) > 10:
            fdf.to_csv(fpath_f)
            if fdf_new is not None and len(fdf_new) > 0:
                print(f" +{len(fdf_new)} -> {len(fdf)} records")
            else:
                print(f" up-to-date ({len(fdf)} records, last={fdf.index[-1]})")
        else:
            print(" no funding data")

    print("\nDone!")
