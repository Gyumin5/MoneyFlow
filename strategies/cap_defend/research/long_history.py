"""2017년까지 이어 붙인 일봉을 만든다.

운영자가 정한 것(텔레그램 9520·9521): 알트가 몇 개 없어도 그때 있던 코인만으로
2017년 수준까지 구간을 늘린다. 그때 그것만 있었으니 전략도 그 안에서 돌았을 것이다.

이어 붙이는 규칙은 하나다. 종목마다 우리 자료가 시작하는 날을 경계로 삼아, 그 앞은
비트파이넥스 일봉을 쓰고 그 뒤는 지금 쓰던 자료를 그대로 쓴다. 겹치는 구간은 우리
자료를 남긴다 — 라이브와 같은 봉을 보기 위해서다.

붙이는 종목은 봉 대조를 통과한 것만이다(bitfinex_vs_binance.py, 판정 기준은 운영자가
정한 9523: 종가 중앙 상대차 0.5% 이내 AND 일간 수익률 상관 0.98 이상). 질리카는
상관 0.940 으로 떨어져 뺀다. 비트코인캐시·비트코인에스브이는 겹치는 구간이 200봉이
안 돼 판정을 못 했으므로 역시 안 붙인다.

쓰지 않는 것: 비트파이넥스에만 있고 우리 자료에 없는 종목(오미세고·퀀텀 등). 대조할
상대가 없어 같은 봉인지 확인할 방법이 없다. 2017년 유니버스가 여섯으로 좁아지지만,
확인 안 된 자료로 넓히는 것보다 좁고 확인된 쪽이 낫다.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAP))
BFX = os.path.join(ROOT, 'data_research', 'bitfinex_daily')

if CAP not in sys.path:
    sys.path.insert(0, CAP)

# 우리 심볼 → 비트파이넥스 심볼. 봉 대조를 통과한 24종목만 둔다.
VERIFIED = {
    'ADA': 'ADA', 'BAT': 'BAT', 'BTC': 'BTC', 'CHZ': 'CHZ', 'DASH': 'DSH',
    'DOT': 'DOT', 'EOS': 'EOS', 'ETC': 'ETC', 'ETH': 'ETH', 'FIL': 'FIL',
    'IOTA': 'IOT', 'LTC': 'LTC', 'MANA': 'MNA', 'NEO': 'NEO', 'SOL': 'SOL',
    'TRX': 'TRX', 'UNI': 'UNI', 'XLM': 'XLM', 'XMR': 'XMR', 'XRP': 'XRP',
    'XTZ': 'XTZ', 'YFI': 'YFI', 'ZEC': 'ZEC', 'ZRX': 'ZRX',
}

COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


def _read_bfx(sym):
    p = os.path.join(BFX, f'{sym}USD.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=['Date']).set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df[COLS]


def load_extended(verbose=True):
    """지금 쓰는 일봉에 2017년 앞구간을 덧댄 bars 를 돌려준다. funding 은 그대로."""
    from unified_backtest import load_data
    bars, funding = load_data('D')
    added = []
    for ours, bsym in sorted(VERIFIED.items()):
        cur = bars.get(ours)
        bf = _read_bfx(bsym)
        if cur is None or bf is None or len(cur) == 0:
            continue
        idx = cur.index
        if getattr(idx, 'tz', None) is not None:
            cur = cur.copy()
            cur.index = idx.tz_localize(None)
        head = bf[bf.index < cur.index[0]]
        if len(head) == 0:
            continue
        merged = pd.concat([head[COLS], cur[COLS]])
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        bars[ours] = merged
        added.append((ours, head.index[0].date(), len(head)))
    if verbose:
        print(f"앞구간 덧댄 종목 {len(added)}개")
        for s, d0, n in added:
            print(f"  {s:<6} {d0} 부터 {n}행 추가")
    return bars, funding


def coverage(bars, dates=('2017-01-01', '2017-07-01', '2018-01-01', '2019-01-01',
                          '2020-01-01', '2020-10-01')):
    """날짜마다 봉이 서 있는 종목 수를 센다 — 유니버스가 언제 몇 개인지 본다."""
    out = []
    for d in dates:
        ts = pd.Timestamp(d)
        have = sorted(s for s, df in bars.items()
                      if len(df) and df.index[0] <= ts <= df.index[-1])
        out.append((d, have))
    return out


if __name__ == '__main__':
    bars, _ = load_extended()
    print()
    for d, have in coverage(bars):
        print(f"  {d}  {len(have):>3}종  {' '.join(have)[:120]}")
