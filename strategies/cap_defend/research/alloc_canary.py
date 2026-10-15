"""세 슬리브의 라이브 카나리를 일별 계열로 만든다 — 자산간 리밸런싱 조건 연구의 선행조건.

왜 필요한가. 기존 송금 지연 하니스(bt_alloc_transfer_lag.py)는 카나리를 "최근 20거래일
평균 수익 > 0" 으로 근사한다. 팔 여섯에 똑같이 걸리니 팔 사이 비교에는 안 닿지만,
"카나리가 꺼지는 순간·켜지는 순간" 자체를 트리거로 쓰는 후보를 재려면 근사로는 못 한다 —
재려는 사건이 바로 그 계열의 전이점이기 때문이다.

정의는 라이브 그대로다.
  코인 현물·선물 = BTC 종가 대 SMA42, 히스테리시스 ±1.5%. 판정은 전 거래일 종가로.
    (unified_backtest.py 873-881 줄과 backtest_futures_v25.py 의 같은 블록이 정본)
  주식 = EEM 종가 대 SMA200, 히스테리시스 ±0.5%.
    (bt_stock_coin_v3.precompute 가 이미 그 계열을 돌려주므로 다시 안 짠다)

현물과 선물은 같은 BTC SMA42 규칙이라 계열이 하나다. 둘을 따로 받는 자리를 남겨 둔 것은
앞으로 갈릴 수 있어서가 아니라, 부르는 쪽이 슬리브 이름으로 집게 하려는 것뿐이다.

스스로 돌리면 자기검사를 한다 — 만든 코인 계열을 정본 엔진 trace 와 맞대 본다.
카나리가 꺼진 날은 엔진의 합성 목표가 전액 현금이므로 그 두 집합이 같아야 한다.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

COIN_SMA = 42
COIN_HYST = 0.015


def coin_canary(bars: dict, sma_period: int = COIN_SMA,
                hyst: float = COIN_HYST) -> pd.Series:
    """BTC SMA42 카나리. 값은 그 날짜에 매매가 보는 상태(= 전 거래일 종가로 판정).

    엔진과 같은 히스테리시스다 — 켜져 있으면 아래 띠를 깨야 꺼지고, 꺼져 있으면 위 띠를
    넘어야 켜진다. 첫 판정은 꺼짐에서 시작한다(엔진 prev_canary 초기값과 같다).

    판정할 수 없는 앞쪽 구간(이동평균이 아직 안 찬 날)은 False 로 채우지 않고 아예 뺀다.
    채우면 "꺼져 있다" 로 읽히고, 그러면 자료가 차는 날이 가짜 켜짐 사건이 된다.
    """
    btc = bars['BTC']['Close'] if isinstance(bars['BTC'], pd.DataFrame) else bars['BTC']
    btc = btc.dropna()
    sma = btc.rolling(sma_period).mean()
    out, idx = [], []
    state = False
    for i, d in enumerate(btc.index):
        if i == 0 or np.isnan(sma.iloc[i - 1]) or sma.iloc[i - 1] <= 0:
            continue
        px, s = float(btc.iloc[i - 1]), float(sma.iloc[i - 1])
        state = (not (px < s * (1 - hyst))) if state else (px > s * (1 + hyst))
        out.append(state)
        idx.append(d)
    return pd.Series(out, index=pd.DatetimeIndex(idx), dtype=bool)


def stock_canary() -> pd.Series:
    """EEM SMA200 카나리. bt_stock_coin_v3.precompute 의 계열을 그대로 쓴다.

    주의 — 이 저장소의 EEM 가격은 2016-01-04 부터만 있다. precompute 는 그 앞을 결측으로
    보고 꺼짐을 유지하므로, 그 구간을 그대로 쓰면 2016년 중반에 가짜 켜짐 사건이 하나
    생긴다. 그래서 판정 가능한 첫 날(EEM 첫 유효일 + SMA200 길이) 앞을 잘라낸다.
    """
    from bt_stock_coin_v3 import precompute, CANARY_SMA
    from stock_engine import load_prices, ALL_TICKERS
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    _, _, _, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
    eem = pdf['EEM']
    ok = eem.notna()
    first = ok.idxmax() if bool(ok.any()) else None
    if first is None:
        raise RuntimeError("EEM 가격이 하나도 없다 — 주식 카나리를 만들 수 없다")
    valid = eem.loc[first:].index
    if len(valid) <= CANARY_SMA:
        raise RuntimeError("EEM 가격이 SMA200 을 채우기에 모자란다")
    return canary.loc[valid[CANARY_SMA]:].astype(bool)


def flips(s: pd.Series) -> pd.Series:
    """전이점만. True = 그 날 상태가 바뀌었다."""
    return s.ne(s.shift(1)).fillna(False) & s.index.to_series().ne(s.index[0])


def _selfcheck() -> int:
    """만든 코인 계열을 정본 엔진 trace 와 맞대 본다."""
    import bt_coin_param_resweep as R
    from unified_backtest import load_data, run as bt_run

    bars, funding = load_data("D")
    can = coin_canary(bars)

    trace: list = []
    kw = dict(R.SPOT_BASE)
    bt_run(bars, funding, interval="D", asset_type="spot", leverage=1.0,
           tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
           vol_mode="daily", phase_offset_bars=R.PH_SPOT[0],
           post_flip_delay=5, selection="greedy", stop_kind="none", stop_pct=0.0,
           dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7,
           crash_threshold=-99.0, health_mode="mom2vol", _trace=trace, **kw)

    rows = []
    for t in trace:
        d = pd.Timestamp(t['date'])
        if d not in can.index:
            continue
        tg = t['target']
        risk = sum(v for k, v in tg.items() if k.upper() != 'CASH')
        rows.append((d, bool(can.loc[d]), risk))
    df = pd.DataFrame(rows, columns=['date', 'canary', 'risk_w']).set_index('date')

    off = df[~df.canary]
    on = df[df.canary]
    off_allcash = int((off.risk_w <= 1e-9).sum())
    on_allcash = int((on.risk_w <= 1e-9).sum())

    print(f"대조 일수 {len(df)}  (카나리 켜짐 {len(on)} · 꺼짐 {len(off)})")
    print(f"  꺼진 날 중 엔진 목표가 전액 현금: {off_allcash}/{len(off)}")
    print(f"  켜진 날 중 엔진 목표가 전액 현금: {on_allcash}/{len(on)}"
          f"   (건강한 코인이 없으면 켜져 있어도 현금이라 0 이 아니어도 된다)")
    print(f"  전이 횟수 — 코인 {int(flips(can).sum())}")
    print(f"  코인 계열 범위 {can.index[0].date()} ~ {can.index[-1].date()} ({len(can)}일)")

    st = stock_canary()
    print(f"주식 계열 범위 {st.index[0].date()} ~ {st.index[-1].date()} ({len(st)}일)"
          f"  켜짐 {int(st.sum())} · 꺼짐 {int((~st).sum())} · 전이 {int(flips(st).sum())}")
    yearly = st.groupby(st.index.year).mean().round(2)
    print("  연도별 켜짐 비율: " + " ".join(f"{y}:{v}" for y, v in yearly.items()))

    bad = len(off) - off_allcash
    if bad:
        print(f"  !! 꺼졌는데 위험자산을 들고 있는 날 {bad} 건 — 계열이 엔진과 다르다")
        print(off[off.risk_w > 1e-9].head(10))
    else:
        print("  통과: 꺼진 날은 예외 없이 전액 현금.")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(_selfcheck())
