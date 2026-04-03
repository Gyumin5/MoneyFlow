#!/usr/bin/env python3
"""바이낸스 선물 OHLCV + 펀딩레이트 다운로드.

기본 동작은 증분 업데이트다.
- 기존 CSV가 있으면 마지막 시점 이후만 이어받는다.
- `--full`일 때만 처음부터 다시 받는다.
"""
import requests, pandas as pd, time, os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'futures')
os.makedirs(DATA_DIR, exist_ok=True)

# Top 40 by market cap that have Binance USDT-M futures
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'TRXUSDT', 'LINKUSDT',
    'DOTUSDT', 'MATICUSDT', 'UNIUSDT', 'NEARUSDT', 'LTCUSDT',
    'BCHUSDT', 'APTUSDT', 'ICPUSDT', 'FILUSDT', 'ATOMUSDT',
    'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'SHIBUSDT', 'PEPEUSDT',
    'XLMUSDT', 'VETUSDT', 'ALGOUSDT', 'FTMUSDT', 'GRTUSDT',
    'AAVEUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'THETAUSDT',
    'EOSUSDT', 'FLOWUSDT', 'CHZUSDT', 'APEUSDT', 'GALAUSDT',
]

BASE_URL = 'https://fapi.binance.com'


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
