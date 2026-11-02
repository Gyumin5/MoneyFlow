"""비트파이넥스 일봉이 지금 쓰는 바이낸스 일봉을 따라가는지 겹치는 구간에서 잰다.

운영자가 정한 판정 기준이다(2026-11-02 텔레그램 9523): "극단적으로 가면 문제가 생긴다는
가정은 제외하고 동일하게 봉이 따라가면 정상인지를 판단해".

그래서 보는 것은 두 가지다.
  ⑴ 종가의 날짜별 상대차 — 중앙값과 분위수. 두 거래소는 호가가 따로 서므로 완전 일치는
     애초에 없다. 봐야 하는 건 수준이 같은가다.
  ⑵ 일간 수익률의 상관과 차이 — 전략이 보는 것은 가격 수준이 아니라 변화다. 모멘텀·
     이동평균·변동성이 전부 수익률에서 나오므로 여기가 어긋나면 긴 구간을 못 쓴다.

판정에 쓰지 않는 것: 하루치 최대 괴리. 한쪽 거래소의 순간 유동성 공백이 전체를 기각하게
된다 — 운영자가 빼라고 한 "극단" 이 이것이다. 대신 몇 번 났는지를 같이 적는다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bitfinex_vs_binance.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAP))
BFX = os.path.join(ROOT, 'data_research', 'bitfinex_daily')
BIN = os.path.join(ROOT, 'data', 'futures')
OUT = os.path.join(ROOT, 'state', '20261102-bitfinex')

# 비트파이넥스 세 글자 티커 → 우리 심볼
ALIAS = {'DSH': 'DASH', 'IOT': 'IOTA', 'QTM': 'QTUM', 'MNA': 'MANA', 'UDC': 'USDC'}

# 겹치는 구간이 이보다 짧으면 판정하지 않는다 — 표본이 모자란 것을 일치로도 불일치로도 안 읽는다.
MIN_OVERLAP = 200


def _load_bfx(sym):
    p = os.path.join(BFX, f'{sym}USD.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=['Date']).set_index('Date').sort_index()
    return df[~df.index.duplicated(keep='last')]


_OURS = None


def _load_bin(sym):
    """비교 상대는 백테스트가 실제로 보는 일봉이다 — 원시 csv 가 아니라 load_data('D').

    원시 파일을 따로 읽어 비교하면 엔진이 하는 재표본·결측 처리와 어긋나서,
    "자료는 맞는데 엔진이 보는 것은 다른" 상태를 못 잡는다.
    """
    global _OURS
    if _OURS is None:
        sys.path.insert(0, CAP)
        from unified_backtest import load_data
        _OURS, _ = load_data('D')
        print(f"우리 일봉 {len(_OURS)}종 로드", flush=True)
    df = _OURS.get(sym)
    if df is None:
        return None
    out = df.copy()
    if getattr(out.index, 'tz', None) is not None:
        out.index = out.index.tz_localize(None)
    return out[~out.index.duplicated(keep='last')]


def compare(sym):
    a = _load_bfx(sym)
    if a is None:
        return None
    ours = ALIAS.get(sym, sym)
    b = _load_bin(ours)
    if b is None:
        return None
    idx = a.index.intersection(b.index)
    if len(idx) < MIN_OVERLAP:
        return dict(sym=sym, ours=ours, n=len(idx), verdict='표본부족')
    ca = a.loc[idx, 'Close'].astype(float)
    cb = b.loc[idx, 'Close'].astype(float)
    ok = (ca > 0) & (cb > 0)
    ca, cb = ca[ok], cb[ok]
    rel = (ca / cb - 1.0).abs()
    ra = np.diff(np.log(ca.values))
    rb = np.diff(np.log(cb.values))
    corr = float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 else float('nan')
    rdiff = np.abs(ra - rb)
    row = dict(
        sym=sym, ours=ours, n=int(len(ca)),
        first=str(ca.index[0].date()), last=str(ca.index[-1].date()),
        close_med=float(np.median(rel)), close_p95=float(np.quantile(rel, 0.95)),
        close_over_1pct=int((rel > 0.01).sum()),
        ret_corr=corr, ret_med=float(np.median(rdiff)),
        ret_p95=float(np.quantile(rdiff, 0.95)),
    )
    # 수준이 같고(중앙 0.5% 이내) 변화가 같이 움직이면(상관 0.98 이상) 따라간다고 본다.
    row['verdict'] = '따라감' if (row['close_med'] <= 0.005 and corr >= 0.98) else '어긋남'
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    syms = sorted(f[:-7] for f in os.listdir(BFX) if f.endswith('USD.csv'))
    rows = []
    print(f"{'심볼':<6}{'우리':<7}{'겹침':>6} {'종가중앙':>9}{'종가95%':>9}{'1%초과':>7}"
          f"{'수익상관':>9}{'수익95%':>9}  판정")
    for s in syms:
        r = compare(s)
        if r is None:
            continue
        rows.append(r)
        if r.get('verdict') == '표본부족':
            print(f"{s:<6}{r['ours']:<7}{r['n']:>6}  {'겹치는 구간이 짧다':>20}")
            continue
        print(f"{s:<6}{r['ours']:<7}{r['n']:>6} {r['close_med']*100:>8.3f}%"
              f"{r['close_p95']*100:>8.3f}%{r['close_over_1pct']:>7}"
              f"{r['ret_corr']:>9.4f}{r['ret_p95']*100:>8.3f}%  {r['verdict']}")

    judged = [r for r in rows if r.get('verdict') in ('따라감', '어긋남')]
    good = [r for r in judged if r['verdict'] == '따라감']
    print(f"\n판정한 종목 {len(judged)}개 중 따라가는 것 {len(good)}개")
    if judged and len(good) < len(judged):
        print("어긋난 종목:", ', '.join(r['sym'] for r in judged if r['verdict'] == '어긋남'))
    p = os.path.join(OUT, 'overlap_check.json')
    json.dump(rows, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"기록 {os.path.abspath(p)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
