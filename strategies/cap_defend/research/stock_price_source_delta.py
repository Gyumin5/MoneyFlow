"""주식 가격 자료 두 소스가 과거 신호를 며칠이나 바꿨나.

왜 재나. 2026-11-03 신호 대조가 주식에서 불일치로 나왔고 원인이 전략 로직이 아니라
가격 자료였다(history #141). 목표를 적는 쪽(recommend_personal, 야후 차트 adjclose)과
주문을 내는 쪽(executor_stock, yfinance auto_adjust)이 서로 다른 소스를 쓴다.
어느 쪽으로 통일할지 정하려면 그 차이가 과거에 얼마나 컸고 신호를 며칠이나 바꿨는지
알아야 한다.

읽기 전용이다. 라이브도 상태파일도 건드리지 않는다.

입력
  state/20261103-price-source/srv_<티커>.csv   서버 캐시 사본 (Date, Adj_Close)
  집 쪽은 strategies/cap_defend/stock_engine.load_prices 가 보는 그대로

출력
  state/20261103-price-source/delta.txt
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAP))
for p in (ROOT, CAP):
    if p not in sys.path:
        sys.path.insert(0, p)

from stock_strategy_v25 import (  # noqa: E402
    DEFENSIVE_STOCK_UNIVERSE, OFFENSIVE_STOCK_UNIVERSE, STOCK_CANARY_TICKER,
    compute_defense, compute_offense, eem_canary,
)

SRV_DIR = os.path.join(ROOT, 'state', '20261103-price-source')
TICKERS = sorted(set(OFFENSIVE_STOCK_UNIVERSE) | set(DEFENSIVE_STOCK_UNIVERSE)
                 | {STOCK_CANARY_TICKER})


def load_server():
    out = {}
    for t in TICKERS:
        fp = os.path.join(SRV_DIR, f'srv_{t}.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        df['Date'] = pd.to_datetime(df['Date'])
        out[t] = df.set_index('Date')['Adj_Close'].astype(float).sort_index()
    return out


def load_home():
    from stock_engine import load_prices
    raw = load_prices(TICKERS, start='2005-01-01')
    out = {}
    for t, s in raw.items():
        s = pd.Series(s).dropna().sort_index()
        idx = pd.DatetimeIndex(s.index)
        s = s[idx.normalize() == idx]
        out[t] = s[~s.index.duplicated(keep='first')]
    return out


def slice_to(prices, day):
    """그 날까지의 시계열만 남긴다 (전략 함수가 마지막 행을 신호로 쓴다)."""
    out = {}
    for t, s in prices.items():
        cut = s[s.index <= day]
        if len(cut) > 300:
            out[t] = cut
    return out


def picks_on(prices, day, prev_on):
    sub = slice_to(prices, day)
    if len(sub) < len(TICKERS):
        return None
    on, _meta = eem_canary(sub.get(STOCK_CANARY_TICKER), prev_on)
    if on:
        picks, w, _ = compute_offense(sub)
    else:
        picks, w, _ = compute_defense(sub)
    return {'on': bool(on), 'picks': tuple(sorted(picks)),
            'w': {k: round(v, 6) for k, v in w.items()}}


def main():
    srv, home = load_server(), load_home()
    common = sorted(set(srv) & set(home))
    print(f'종목 {len(common)}개: {common}')

    lines = []
    lines.append('== 가격 자체의 차이 (겹치는 날짜만) ==')
    for t in common:
        a, b = srv[t], home[t]
        idx = a.index.intersection(b.index)
        if len(idx) == 0:
            lines.append(f'{t}: 겹치는 날짜 없음')
            continue
        rel = ((a.loc[idx] - b.loc[idx]).abs() / b.loc[idx].abs()).dropna()
        lines.append(
            f'{t}: 겹침 {len(idx)}일 {idx.min().date()}~{idx.max().date()} · '
            f'상대차 중앙 {rel.median()*100:.4f}% · 95분위 {rel.quantile(0.95)*100:.4f}% · '
            f'최대 {rel.max()*100:.4f}%')

    # 신호가 갈린 날 — 두 소스가 다 있는 구간의 매 거래일
    both = None
    for t in common:
        idx = srv[t].index.intersection(home[t].index)
        both = idx if both is None else both.intersection(idx)
    days = [d for d in both if d >= both.min() + pd.Timedelta(days=400)]
    lines.append('')
    lines.append(f'== 신호 대조 == 대상 {len(days)}일 '
                 f'{days[0].date()}~{days[-1].date()}')

    prev_s = prev_h = None
    diff_pick = diff_w = diff_canary = 0
    examples = []
    for d in days:
        rs = picks_on(srv, d, prev_s)
        rh = picks_on(home, d, prev_h)
        if rs is None or rh is None:
            continue
        prev_s, prev_h = rs['on'], rh['on']
        if rs['on'] != rh['on']:
            diff_canary += 1
        if rs['picks'] != rh['picks']:
            diff_pick += 1
            if len(examples) < 12:
                examples.append(f'  {d.date()} 서버 {rs["picks"]} · 집 {rh["picks"]}')
        if rs['w'] != rh['w']:
            diff_w += 1

    n = len(days)
    lines.append(f'카나리가 갈린 날: {diff_canary} / {n} ({diff_canary/n*100:.2f}%)')
    lines.append(f'종목이 갈린 날:   {diff_pick} / {n} ({diff_pick/n*100:.2f}%)')
    lines.append(f'비중이 갈린 날:   {diff_w} / {n} ({diff_w/n*100:.2f}%)')
    if examples:
        lines.append('종목이 갈린 날 보기:')
        lines.extend(examples)

    out = '\n'.join(lines)
    print(out)
    with open(os.path.join(SRV_DIR, 'delta.txt'), 'w') as f:
        f.write(out + '\n')


if __name__ == '__main__':
    main()
