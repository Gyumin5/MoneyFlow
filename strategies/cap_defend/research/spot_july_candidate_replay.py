"""2026년 7월 라이브 유니버스에서 후보 넷이 언제 들어갔을지, 그 차이가 얼마인지.

백테스트 측정(spot_flip_retry_measure.py)은 백테스트 유니버스 위에서 잰다. 그런데 이번
사고는 라이브 유니버스(업비트 원화 + 거래대금 문턱으로 16종)에서만 생겼다. 그래서 같은
후보를 라이브 유니버스에 직접 걸어 진입일과 그 구간 수익을 따로 잰다.

가정 둘을 밝힌다.
  · 신호봉은 UTC 완료 일봉이고 체결은 그 다음 봉 종가로 본다(라이브 09:05 KST 실행 근사).
  · 유니버스는 7월 내내 16종 고정으로 둔다 — recommend_personal.log 가 7월 매일 같은 16종을
    적고 있다. 순서는 그 로그의 시총 순위 순서다.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = "/home/gmoh/mon/251229"
UNIV = ["BTC", "ETH", "XRP", "SOL", "TRX", "DOGE", "XLM", "LINK", "ADA", "BCH",
        "SUI", "HBAR", "AVAX", "NEAR", "SHIB", "UNI"]
VOL_TH, VOL_LB = 0.05, 90
CAP, NPICK = 1 / 3, 3
FLIP = "2026-07-14"      # 카나리 켜짐이 확인된 완료봉
OFF = "2026-08-14"       # 다음 카나리 꺼짐 완료봉
LIVE_ENTRY = "2026-08-01"  # 라이브가 실제로 산 날의 신호봉(체결 08-02)


def daily(sym):
    p = os.path.join(ROOT, f"data/futures/{sym}USDT_1h.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    s = df["Close"] if "Close" in df.columns else df.iloc[:, -1]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.resample("1D").last().dropna()


SER = {s: daily(s) for s in UNIV}
SER = {k: v for k, v in SER.items() if v is not None}


def mom(c, n):
    return c[-1] / c[-n - 1] - 1


def passers(bar: str, ms: int) -> list[str]:
    """그 완료봉에서 건강도 통과 코인을 시총 순서로."""
    out = []
    t = pd.Timestamp(bar, tz="UTC")
    for s in UNIV:
        v = SER.get(s)
        if v is None:
            continue
        x = v[v.index <= t]
        if len(x) < 140 or x.index[-1] != t:
            continue
        c = x.values.astype(float)
        vol = float(np.std(np.diff(np.log(c[-VOL_LB - 1:])), ddof=1))
        if mom(c, ms) > 0 and mom(c, 127) > 0 and vol <= VOL_TH:
            out.append(s)
    return out


def picks(bar: str, ms: int) -> list[str]:
    """상위 3종 + greedy 흡수(앞 코인의 단기 모멘텀이 더 크면 뒤를 버린다)."""
    p = passers(bar, ms)[:NPICK]
    t = pd.Timestamp(bar, tz="UTC")
    for i in range(len(p) - 1, 0, -1):
        ca = SER[p[i - 1]][SER[p[i - 1]].index <= t].values.astype(float)
        cb = SER[p[i]][SER[p[i]].index <= t].values.astype(float)
        if mom(ca, ms) >= mom(cb, ms):
            p.pop(i)
    return p


def fill_date(sig: str):
    """신호봉 다음 봉(체결 가정)."""
    idx = SER["BTC"].index
    t = pd.Timestamp(sig, tz="UTC")
    nxt = idx[idx > t]
    return nxt[0] if len(nxt) else None


def _ret_from(p: list[str], sig: str, end: str) -> float:
    d0 = fill_date(sig)
    d1 = pd.Timestamp(end, tz="UTC")
    if d0 is None or not p:
        return 0.0
    w = min(1 / len(p), CAP)
    tot = 0.0
    for s in p:
        v = SER[s]
        a = v[v.index >= d0]
        b = v[v.index <= d1]
        if len(a) == 0 or len(b) == 0:
            continue
        tot += w * (float(b.iloc[-1]) / float(a.iloc[0]) - 1)
    return tot


def main():
    bars = [str(d.date()) for d in SER["BTC"].index
            if pd.Timestamp(FLIP, tz="UTC") <= d <= pd.Timestamp("2026-08-05", tz="UTC")]

    print("라이브 유니버스 16종 · 7월 통과 코인 (완료봉 기준)")
    print(f"{'완료봉':<12}{'단기20일 통과':<34}{'단기18일 통과'}")
    first20 = first18 = None
    for b in bars:
        p20, p18 = passers(b, 20), passers(b, 18)
        if p20 and first20 is None:
            first20 = b
        if p18 and first18 is None:
            first18 = b
        print(f"{b:<12}{str(p20):<34}{p18}")

    print(f"\n전환봉 {FLIP} 통과 — 단기20일 {len(passers(FLIP,20))}종 · "
          f"단기18일 {len(passers(FLIP,18))}종")
    print(f"처음 통과가 생긴 봉 — 단기20일 {first20} · 단기18일 {first18}")

    # 후보별 신호봉
    nxt = bars[1] if len(bars) > 1 else None
    pfd_bar = bars[5] if len(bars) > 5 else None
    arms = {
        "현행(라이브)": LIVE_ENTRY,
        "가-다음봉": nxt if passers(nxt, 20) else None,
        "나-생길때까지": first20,
        "다-단기18일": FLIP if passers(FLIP, 18) else first18,
        "라-전환후5봉(채택BT)": pfd_bar if passers(pfd_bar, 20) else None,
    }

    print(f"\n후보별 진입과 {OFF} 까지의 슬리브 수익 (체결은 신호봉 다음 봉 종가)")
    print(f"{'팔':<22}{'신호봉':<12}{'체결봉':<12}{'종목':<22}{'수익':>8}{'현행대비':>10}")
    base = None
    rows = []
    for name, sig in arms.items():
        if sig is None:
            rows.append((name, "없음", "", [], float("nan")))
            continue
        ms = 18 if name.startswith("다-") else 20
        p = picks(sig, ms)
        r = _ret_from(p, sig, OFF)
        f = fill_date(sig)
        rows.append((name, sig, str(f.date()) if f is not None else "", p, r))
        if name.startswith("현행"):
            base = r
    for name, sig, f, p, r in rows:
        gap = "" if (base is None or np.isnan(r)) else f"{(r-base)*100:+9.2f}%"
        rr = "    -" if np.isnan(r) else f"{r*100:+7.2f}%"
        print(f"{name:<22}{sig:<12}{f:<12}{str(p):<22}{rr}{gap:>10}")

    live_p = picks(LIVE_ENTRY, 20)
    live_full = _ret_from(live_p, LIVE_ENTRY, OFF)
    print(f"\n라이브는 스냅샷 일곱 중 하나만 발화해 실제 비중이 (1/3)/7 = 4.76% 였다.")
    print(f"  같은 구간 실제 기여 {live_full/7*100:+.2f}% (위 표의 현행은 다른 팔과 견주려고 "
          f"전량 진입으로 가정한 값이다)")
    print("\n주의: 이것은 7월 한 사건의 값이다. 한 사건으로 전략을 바꾸지 않는다 —")
    print("      채택 판정은 5.8년 합성 순위합(spot_flip_retry_measure.py)이 한다.")


if __name__ == "__main__":
    main()
