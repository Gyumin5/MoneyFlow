"""집 주식 캐시를 전체 재수집으로 바꾸면 과거가 얼마나 움직이나.

왜 재나. 2026-11-03 에 11:40 신호 대조 경보의 원인이 집 캐시의 증분 갱신으로
밝혀졌다(history #148). 고치는 길은 refresh_backtest_data.refresh_stock 을 전체
교체로 바꾸는 것인데, 그건 집 백테스트 정본 자료를 바꾼다. 운영자가 정하려면
"바꾸면 과거 수치가 얼마나 달라지나" 를 알아야 한다.

무엇을 비교하나.
  현행  strategies/cap_defend/data/stock_cache 에 지금 들어 있는 값 (증분으로 자란 것)
  재수집 state/20261103-same-time/raw_B_<티커>.csv (같은 날 전체를 다시 받은 값)

두 축으로 본다.
  ⑴ 신호  — 날마다 카나리와 고른 종목이 갈리나
  ⑵ 성과  — 주식 슬리브 단독 곡선의 연복리·최대낙폭·칼마·샤프

읽기 전용이다. 캐시도 상태파일도 라이브도 건드리지 않는다. 현행 캐시는 읽기만 하고
재수집 자료는 이미 받아 둔 파일을 쓴다 — 이 스크립트는 네트워크를 쓰지 않는다.

출력
  state/20261103-same-time/rebuild_delta.txt
"""
import os
import sys

import numpy as np
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

SAME = os.path.join(ROOT, 'state', '20261103-same-time')
CACHE = os.path.join(CAP, 'data', 'stock_cache')
TICKERS = sorted(set(OFFENSIVE_STOCK_UNIVERSE) | set(DEFENSIVE_STOCK_UNIVERSE)
                 | {STOCK_CANARY_TICKER})
START = pd.Timestamp('2019-01-01')


def _read(fp):
    d = pd.read_csv(fp)
    c = d.columns
    s = pd.Series(d[c[1]].values, index=pd.to_datetime(d[c[0]])).astype(float)
    s = s.dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep='last')]


def load(which):
    out = {}
    for t in TICKERS:
        fp = (os.path.join(CACHE, f'{t}.csv') if which == 'now'
              else os.path.join(SAME, f'raw_B_{t}.csv'))
        if os.path.exists(fp):
            out[t] = _read(fp)
    return out


def day_target(prices, day, prev_on):
    """그 날의 카나리와 목표 비중. 전략 순수함수를 그대로 부른다."""
    sub = {}
    for t, s in prices.items():
        cut = s[s.index <= day]
        if len(cut) > 300:
            sub[t] = cut
    if len(sub) < len(TICKERS):
        return None
    on, _ = eem_canary(sub.get(STOCK_CANARY_TICKER), prev_on)
    picks, w, _ = (compute_offense(sub) if on else compute_defense(sub))
    return on, tuple(sorted(picks)), {k: float(v) for k, v in w.items()}


def curve(prices, days, tx=0.001):
    """목표 비중을 다음 날 수익률에 적용한 단순 곡선. 두 자료를 같은 규칙으로 잰다."""
    eq, on, prev_w = 1.0, None, {}
    vals, dates = [], []
    for i, d in enumerate(days[:-1]):
        r = day_target(prices, d, on)
        if r is None:
            continue
        on, _picks, w = r
        turn = sum(abs(w.get(k, 0.0) - prev_w.get(k, 0.0))
                   for k in set(w) | set(prev_w)) / 2
        eq *= (1 - turn * tx)
        nxt = days[i + 1]
        ret = 0.0
        for t, wt in w.items():
            s = prices.get(t)
            if s is None or d not in s.index or nxt not in s.index:
                continue
            ret += wt * (float(s[nxt]) / float(s[d]) - 1.0)
        eq *= (1 + ret)
        prev_w = w
        vals.append(eq)
        dates.append(nxt)
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


def stats(eq):
    if len(eq) < 50:
        return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    rr = eq.pct_change().dropna()
    sharpe = float(rr.mean() / rr.std() * np.sqrt(252)) if rr.std() > 0 else 0.0
    cal = cagr / abs(mdd) if mdd < 0 else float('inf')
    return dict(yrs=yrs, cagr=cagr, mdd=mdd, cal=cal, sharpe=sharpe, last=eq.iloc[-1])


def main():
    lines = ['집 주식 캐시를 전체 재수집으로 바꾸면 과거가 얼마나 움직이나',
             '현행 = 증분으로 자란 지금 캐시 · 재수집 = 같은 날 전체를 다시 받은 값',
             '네트워크를 쓰지 않는다 (둘 다 디스크에서 읽는다)']
    now, reb = load('now'), load('rebuild')
    lines.append(f"\n읽은 종목 — 현행 {len(now)} · 재수집 {len(reb)}")
    if len(now) < len(TICKERS) or len(reb) < len(TICKERS):
        lines.append('한쪽이 모자라 비교를 멈춘다')
        print('\n'.join(lines))
        return

    common = None
    for t in TICKERS:
        idx = now[t].index.intersection(reb[t].index)
        common = idx if common is None else common.intersection(idx)
    days = [d for d in common if d >= START]
    lines.append(f"공통 확정일 {len(days)}일 ({days[0].date()}~{days[-1].date()})")

    on_a = on_b = None
    d_can = d_pick = 0
    d_w = []
    rows = []
    for d in days:
        ra, rb = day_target(now, d, on_a), day_target(reb, d, on_b)
        if ra is None or rb is None:
            continue
        on_a, pa, wa = ra
        on_b, pb, wb = rb
        if on_a != on_b:
            d_can += 1
            rows.append(f"    {d.date()} 카나리 {on_a} vs {on_b}")
        if pa != pb:
            d_pick += 1
            rows.append(f"    {d.date()} 종목 {pa} vs {pb}")
        gap = sum(abs(wa.get(k, 0.0) - wb.get(k, 0.0))
                  for k in set(wa) | set(wb)) / 2
        d_w.append(gap)
    lines.append(f"\n[신호] 카나리 갈린 날 {d_can}일 · 종목 갈린 날 {d_pick}일")
    if d_w:
        arr = np.array(d_w)
        lines.append(f"  비중 차이(반회전율) 평균 {arr.mean():.5f} · 최대 {arr.max():.5f} · "
                     f"0.01 초과 {int((arr > 0.01).sum())}일")
    lines.extend(rows[:30])
    if len(rows) > 30:
        lines.append(f"    … 외 {len(rows) - 30}줄")

    lines.append('\n[성과] 같은 규칙으로 잰 주식 슬리브 단독 곡선 (편도 0.1%)')
    sa, sb = stats(curve(now, days)), stats(curve(reb, days))
    if sa and sb:
        lines.append(f"  {'':8} {'연복리':>9} {'최대낙폭':>9} {'칼마':>7} {'샤프':>7} {'누적':>8}")
        for nm, s in (('현행', sa), ('재수집', sb)):
            lines.append(f"  {nm:8} {s['cagr']:>8.2%} {s['mdd']:>9.2%} "
                         f"{s['cal']:>7.2f} {s['sharpe']:>7.2f} {s['last']:>7.3f}x")
        lines.append(f"  차이     {sb['cagr'] - sa['cagr']:>+8.2%} "
                     f"{sb['mdd'] - sa['mdd']:>+9.2%} {sb['cal'] - sa['cal']:>+7.2f} "
                     f"{sb['sharpe'] - sa['sharpe']:>+7.2f} "
                     f"{(sb['last'] / sa['last'] - 1):>+7.2%}")
        lines.append(f"  기간 {sa['yrs']:.2f}년")
    else:
        lines.append('  곡선이 짧아 지표를 못 낸다')

    txt = '\n'.join(lines)
    with open(os.path.join(SAME, 'rebuild_delta.txt'), 'w') as f:
        f.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
