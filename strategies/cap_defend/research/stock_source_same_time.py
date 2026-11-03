"""같은 수집 시점에 주식 가격 세 조회 방식이 실제로 다른가.

왜 재나. 2026-11-03 cd-d9 토론이 1번(리포트 캐시를 집행기와 같은 소스로 통일)을
고르면서 즉시 배포를 막았다. 막은 이유는 앞서 잰 "신호가 갈린 날 6일"
(research/stock_price_source_delta.py)이 리포트 대 집행기가 아니라 리포트 대
집 백테스트 캐시를 비교한 값이라는 것이다. 토론이 남긴 열린 물음 셋을 닫는다 —
같은 시점에 두 조회 방식이 실제로 다른가, 차이가 조회 구간에서 오나 계산
방식에서 오나, 현재 리포트 대 실제 집행기 입력의 신호 차이는 몇 회인가.

세 방식을 같은 시점에 받는다.
  A  야후 차트 API 의 adjclose     — 지금 리포트 두 파일이 쓰는 방식
  B  yfinance auto_adjust 전체이력 — 토론이 고른 새 리포트 방식 (2018-01-01~)
  C  집행기가 실제로 부르는 함수   — executor_stock._fetch_strategy_prices 그대로

읽기 전용이다. 라이브도 상태파일도 캐시도 건드리지 않는다. 받은 것은 전부
state/20261103-same-time/ 아래에만 쓴다.

출력
  state/20261103-same-time/raw_<방식>_<티커>.csv
  state/20261103-same-time/result.txt
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAP))
for p in (ROOT, CAP, os.path.join(ROOT, 'trade')):
    if p not in sys.path:
        sys.path.insert(0, p)

from stock_strategy_v25 import (  # noqa: E402
    DEFENSIVE_STOCK_UNIVERSE, OFFENSIVE_STOCK_UNIVERSE, STOCK_CANARY_TICKER,
    compute_defense, compute_offense, eem_canary,
)

OUT_DIR = os.path.join(ROOT, 'state', '20261103-same-time')
TICKERS = sorted(set(OFFENSIVE_STOCK_UNIVERSE) | set(DEFENSIVE_STOCK_UNIVERSE)
                 | {STOCK_CANARY_TICKER})
START = '2018-01-01'


def fetch_a(ticker, session):
    """리포트 현행 — 야후 차트 API 의 adjclose. recommend.py 396~403행과 같은 호출."""
    p2 = int(datetime.now(timezone.utc).timestamp())
    p1 = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": p1, "period2": p2, "interval": "1d",
              "includeAdjustedClose": "true"}
    resp = session.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"status {resp.status_code}")
    res = resp.json()['chart']['result'][0]
    df = pd.DataFrame({
        'Date': pd.to_datetime(res['timestamp'], unit='s').date,
        'Adj_Close': res['indicators']['adjclose'][0]['adjclose'],
    }).dropna().drop_duplicates('Date')
    s = pd.Series(df['Adj_Close'].values,
                  index=pd.to_datetime(df['Date'])).astype(float).sort_index()
    return s


def fetch_b(ticker):
    """제안하는 새 리포트 방식 — yfinance auto_adjust, 같은 시작일부터 전체."""
    df = yf.download(ticker, start=START, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError('빈 응답')
    if isinstance(df.columns, pd.MultiIndex):
        s = df['Close'][ticker]
    else:
        s = df['Close'].squeeze()
    s = pd.Series(s).dropna().astype(float).sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def fetch_c():
    """집행기 입력 그 자체 — executor_stock 의 함수를 그대로 부른다."""
    import executor_stock as ex
    prices = ex._fetch_strategy_prices(TICKERS)
    prices.pop('__date_mismatch__', None)
    out = {}
    for t, s in prices.items():
        s = pd.Series(s).dropna().astype(float).sort_index()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        out[t] = s
    return out


def cmp_pair(name, left, right, lines):
    """두 방식의 공통 확정일에서 값이 얼마나 다른가."""
    lines.append(f"\n--- {name}")
    worst = 0.0
    for t in TICKERS:
        a, b = left.get(t), right.get(t)
        if a is None or b is None:
            lines.append(f"  {t}: 한쪽 없음 (A={a is not None} B={b is not None})")
            continue
        idx = a.index.intersection(b.index)
        if len(idx) == 0:
            lines.append(f"  {t}: 공통 날짜 0")
            continue
        d = (a[idx] - b[idx]).abs() / b[idx].abs().clip(lower=1e-12)
        mx = float(d.max())
        worst = max(worst, mx)
        n_big = int((d > 1e-4).sum())
        lines.append(f"  {t}: 공통 {len(idx)}일 · 최대 상대차 {mx:.3e} · "
                     f"0.01% 초과 {n_big}일 · 구간 {idx[0].date()}~{idx[-1].date()}")
    lines.append(f"  → 전 종목 최대 상대차 {worst:.3e}")
    return worst


def picks_on(prices, day, prev_on):
    """그 날 신호 하나. slice 후 전략 순수함수를 그대로 부른다."""
    sub = {}
    for t, s in prices.items():
        cut = s[s.index <= day]
        if len(cut) > 300:
            sub[t] = cut
    if len(sub) < len(TICKERS):
        return None
    on, _ = eem_canary(sub.get(STOCK_CANARY_TICKER), prev_on)
    if on:
        picks, w, _ = compute_offense(sub)
    else:
        picks, w, _ = compute_defense(sub)
    return on, tuple(sorted(picks)), {k: round(float(v), 6) for k, v in sorted(w.items())}


def signal_scan(name, left, right, lines):
    """공통 확정일 전 구간에서 신호가 갈린 날을 센다."""
    common = None
    for t in TICKERS:
        a, b = left.get(t), right.get(t)
        if a is None or b is None:
            lines.append(f"\n--- {name}: {t} 한쪽 없음 → 신호 대조 생략")
            return
        idx = a.index.intersection(b.index)
        common = idx if common is None else common.intersection(idx)
    days = [d for d in common if d >= pd.Timestamp('2019-01-01')]
    lines.append(f"\n--- {name} 신호 대조: 공통 확정일 {len(days)}일 "
                 f"({days[0].date()}~{days[-1].date()})")
    on_l = on_r = None
    diff_pick = diff_can = 0
    rows = []
    for d in days:
        rl = picks_on(left, d, on_l)
        rr = picks_on(right, d, on_r)
        if rl is None or rr is None:
            continue
        on_l, pl, wl = rl
        on_r, pr, wr = rr
        if on_l != on_r:
            diff_can += 1
            rows.append(f"    {d.date()} 카나리 {on_l} vs {on_r}")
        if pl != pr:
            diff_pick += 1
            rows.append(f"    {d.date()} 종목 {pl} vs {pr}")
    lines.append(f"  종목 갈린 날 {diff_pick}일 · 카나리 갈린 날 {diff_can}일")
    lines.extend(rows[:40])
    if len(rows) > 40:
        lines.append(f"    … 외 {len(rows) - 40}줄")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = [f"같은 시점 주식 가격 세 조회 방식 비교",
             f"수집 시각 {datetime.now().strftime('%Y-%m-%d %H:%M KST')}",
             f"종목 {len(TICKERS)}개: {' '.join(TICKERS)}"]

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    a, b = {}, {}
    for t in TICKERS:
        try:
            a[t] = fetch_a(t, session)
        except Exception as e:                           # noqa: BLE001
            lines.append(f"  A {t} 실패: {type(e).__name__}: {e}")
        try:
            b[t] = fetch_b(t)
        except Exception as e:                           # noqa: BLE001
            lines.append(f"  B {t} 실패: {type(e).__name__}: {e}")
    try:
        c = fetch_c()
    except Exception as e:                               # noqa: BLE001
        lines.append(f"  C 실패: {type(e).__name__}: {e}")
        c = {}

    lines.append(f"\n받은 종목 수 — A {len(a)} · B {len(b)} · C {len(c)}")
    for tag, src in (('A', a), ('B', b), ('C', c)):
        for t, s in src.items():
            s.to_csv(os.path.join(OUT_DIR, f'raw_{tag}_{t}.csv'))
            lines.append(f"  {tag} {t}: {len(s)}행 {s.index[0].date()}~{s.index[-1].date()}")

    cmp_pair('A(현행 리포트) 대 B(새 리포트)', a, b, lines)
    cmp_pair('B(새 리포트) 대 C(집행기 입력)', b, c, lines)
    cmp_pair('A(현행 리포트) 대 C(집행기 입력)', a, c, lines)

    signal_scan('A 대 B', a, b, lines)

    txt = '\n'.join(lines)
    with open(os.path.join(OUT_DIR, 'result.txt'), 'w') as f:
        f.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
