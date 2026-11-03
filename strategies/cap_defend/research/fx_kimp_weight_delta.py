"""환율과 김치 프리미엄이 자산간 비중에 얼마나 들어가는가 (2026-11-03 운영자 물음 9868·9873).

백테스트는 세 슬리브를 전부 달러로 재고 라이브는 전부 원화로 잰다. 그 차이가 비중과
리밸런싱 트리거를 얼마나 움직이는지 과거 경로로 직접 잰다.

원화 환산 모형 (운영자 9873 이 정한 모양 그대로)
  주식   = V_usd  × FX
  현물   = V_usd  × FX × (1 + 코인김프)    업비트가 원화 시장이라 코인 김프가 들어간다
  선물   = V_usdt × FX × (1 + USDT김프)    라이브가 업비트 USDT 실거래가로 환산한다

환율 FX 는 세 항에 공통이라 비중에서 약분된다 — 이 스크립트가 그걸 수치로 보인다.
환율이 바뀌는 것은 비중이 아니라 원화 기준 총수익이다.

돌리는 법
  cd /home/gmoh/mon/251229
  python3 strategies/cap_defend/research/fx_kimp_weight_delta.py            # 자재 있으면 재사용
  python3 strategies/cap_defend/research/fx_kimp_weight_delta.py --refetch  # 다시 받는다

자재는 state/20261103-fx-kimchi/ 에 남는다(git 미추적). 슬리브 곡선은 합동 격자가 캐시해
둔 것을 읽는다 — 여기서 백테스트를 다시 돌리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = "/home/gmoh/mon/251229"
EQDIR = os.path.join(ROOT, "state/20261015-allocrebal/eq")
OUT = os.path.join(ROOT, "state/20261103-fx-kimchi")
ANCHOR = "a0"          # 합동 격자의 앵커 자리 하나. 비중 차이는 앵커에 거의 안 걸린다
WEIGHTS = np.array([0.60, 0.20, 0.20])
T1 = 0.25              # 반회전율 문턱
T3U = 0.20             # 상대부족 문턱
START = "2020-10-01"


def upbit_daily(market: str, pages: int = 14) -> pd.Series:
    """업비트 원화 일봉 종가. 한 번에 200개라 뒤로 거슬러 올라가며 모은다."""
    rows, to = [], time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime())
    for _ in range(pages):
        url = f"https://api.upbit.com/v1/candles/days?market={market}&count=200&to={to}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        got = json.loads(urllib.request.urlopen(req, timeout=20).read().decode())
        if not got:
            break
        rows += got
        to = got[-1]["candle_date_time_utc"] + "Z"
        time.sleep(0.2)
    df = pd.DataFrame(
        [(r["candle_date_time_utc"][:10], r["trade_price"]) for r in rows],
        columns=["Date", "px"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.drop_duplicates("Date").set_index("Date").sort_index()["px"]


def load_fx(refetch: bool) -> pd.Series:
    path = os.path.join(OUT, "usdkrw.csv")
    if os.path.exists(path) and not refetch:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    import yfinance as yf
    raw = yf.download("KRW=X", start="2020-09-01", progress=False, auto_adjust=False)["Close"]
    if hasattr(raw, "columns"):
        raw = raw.iloc[:, 0]
    s = raw.dropna()
    s.to_csv(path)
    return s


def load_premiums(fx: pd.Series, refetch: bool) -> tuple[pd.Series, pd.Series]:
    """(코인 김프, USDT 김프) 를 비율로 돌려준다.

    코인 김프는 업비트 원화 BTC 를 글로벌 달러 BTC × 환율과 견준다. USDT 김프는
    업비트 KRW-USDT 를 환율과 바로 견준다 — 업비트 USDT 원화 시장이 늦게 열려서
    그 이전 구간은 호출부가 코인 김프로 메운다.
    """
    cpath, upath = os.path.join(OUT, "kimp_btc.csv"), os.path.join(OUT, "kimp_usdt.csv")
    if os.path.exists(cpath) and os.path.exists(upath) and not refetch:
        kc = pd.read_csv(cpath, index_col=0, parse_dates=True)["kimp"] / 100.0
        ku = pd.read_csv(upath, index_col=0, parse_dates=True)["kimp_usdt"] / 100.0
        return kc, ku

    glob = pd.read_csv(os.path.join(ROOT, "data/BTC-USD.csv"), parse_dates=["Date"], index_col="Date")
    col = next(c for c in ("Adj_Close", "Adj Close", "Close") if c in glob.columns)

    btc = pd.concat([upbit_daily("KRW-BTC").rename("krw"), glob[col].rename("usd"), fx.rename("fx")], axis=1).dropna()
    btc = btc[btc.index >= START]
    btc["kimp"] = (btc["krw"] / (btc["usd"] * btc["fx"]) - 1) * 100
    btc.to_csv(cpath)

    usdt = pd.concat([upbit_daily("KRW-USDT").rename("usdt_krw"), fx.rename("fx")], axis=1).dropna()
    usdt = usdt[usdt.index >= START]
    usdt["kimp_usdt"] = (usdt["usdt_krw"] / usdt["fx"] - 1) * 100
    usdt.to_csv(upath)
    return btc["kimp"] / 100.0, usdt["kimp_usdt"] / 100.0


def sleeve_curves() -> pd.DataFrame:
    cols = {}
    for kind in ("stock", "spot", "fut"):
        path = os.path.join(EQDIR, f"{kind}_{ANCHOR}_x1.csv")
        if not os.path.exists(path):
            raise SystemExit(f"슬리브 곡선 캐시가 없다: {path}\n먼저 합동 격자를 한 번 돌려 캐시를 만든다.")
        cols[kind] = pd.read_csv(path, index_col=0, parse_dates=True)["eq"]
    return pd.DataFrame(cols).dropna()


def weight_path(eq: pd.DataFrame, kc=None, ku=None, fx=None) -> np.ndarray:
    """초기 60/20/20 으로 사서 그대로 둔 경우의 비중 경로 (3, n).

    리밸런싱을 안 거는 이유는 두 모형의 차이만 보려는 것이기 때문이다. 발화일 수
    자체는 이 경로에서 과장되고, 비교에 쓰는 것은 두 경로의 차이뿐이다.
    """
    v = np.stack([
        eq["stock"].values / eq["stock"].values[0] * WEIGHTS[0],
        eq["spot"].values / eq["spot"].values[0] * WEIGHTS[1],
        eq["fut"].values / eq["fut"].values[0] * WEIGHTS[2],
    ])
    if kc is not None:
        v[1] = v[1] * (1 + kc.values)
    if ku is not None:
        v[2] = v[2] * (1 + ku.values)
    if fx is not None:
        v = v * fx.values            # 세 항에 공통이라 아래 정규화에서 약분된다
    return v / v.sum(axis=0)


def fire_days(path: np.ndarray) -> tuple[int, int]:
    ht = np.abs(path - WEIGHTS[:, None]).sum(axis=0) / 2
    t3u = ((WEIGHTS[:, None] - path) / WEIGHTS[:, None]).max(axis=0)
    return int((ht >= T1).sum()), int((t3u >= T3U).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true", help="환율·김프를 다시 받는다")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    fx = load_fx(args.refetch)
    kc_raw, ku_raw = load_premiums(fx, args.refetch)
    eq = sleeve_curves()

    kc = kc_raw.reindex(eq.index).ffill()
    ku = ku_raw.reindex(eq.index).ffill()
    ku = ku.fillna(kc)                      # USDT 시장이 열리기 전 구간은 코인 김프로 메운다
    keep = kc.notna() & fx.reindex(eq.index).ffill().notna()
    eq, kc, ku = eq[keep], kc[keep], ku[keep]
    fxa = fx.reindex(eq.index).ffill()

    base = weight_path(eq)
    only_fx = weight_path(eq, fx=fxa)
    oper = weight_path(eq, kc=kc, ku=ku)

    print(f"구간 {eq.index[0].date()} ~ {eq.index[-1].date()} · {len(eq)}일 · 앵커 {ANCHOR}")
    print(f"환율만 더했을 때 비중 최대 차이 {np.abs(only_fx - base).max():.3e}  (약분되면 0)")
    print(f"김프 반영(주식 환율만·현물 코인김프·선물 USDT김프) 비중 최대 차이 "
          f"{np.abs(oper - base).max() * 100:.3f} pp")
    d = np.abs(oper - base).max(axis=0) * 100
    print(f"날짜별 차이 — 중앙 {np.median(d):.3f} pp · 95%ile {np.percentile(d, 95):.3f} pp · 최대 {d.max():.3f} pp")
    for name, p in (("달러 기준", base), ("김프 반영", oper)):
        a, b = fire_days(p)
        print(f"{name} — 반회전율 {T1*100:.0f}pp 이상 {a}일 · 상대부족 {T3U*100:.0f}% 이상 {b}일")

    spread = (kc_raw - ku_raw).dropna() * 100
    if len(spread):
        print(f"코인김프 - USDT김프 {len(spread)}일 — 평균 {spread.mean():.2f}% "
              f"중앙 {spread.median():.2f}% 범위 {spread.min():.2f} ~ {spread.max():.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
