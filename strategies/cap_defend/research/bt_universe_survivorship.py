"""기준선 하락이 정말 생존편향인지 가른다 (2026-09-22).

지금까지 확인된 사실은 하나뿐이다. 고정 40종 심볼표를 풀었더니 코인 현물 칼마가
4.34 에서 2.17 로 내려갔다. 생존편향은 그걸 설명하는 후보 가설이고, 아직 증명이 아니다.
경쟁 가설도 있다 — 새로 들어온 코인이 상장 초기라 변동이 컸을 수도, 1000배 단위 계약
이음매가 가격을 틀었을 수도, 새 데이터 품질이 나쁠 수도 있다.

가르는 시험 넷. 생존편향 가설이 참이면 넷 다 같은 방향을 가리켜야 한다.

  [1] 시간순 시험 — 이게 핵심이다. 고정 목록은 2023년 무렵에 "그때 상위 40 이면서
      선물이 있는 것" 으로 손으로 적혔다. 생존편향의 크기는 목록을 적은 시점에서
      멀수록 커야 한다. 2021년은 백테스트가 이 년 뒤 생존자를 미리 알고 고른 셈이고,
      2025년은 목록 시점과 거의 같아 미리 알 것이 없다. 그러니 옛 목록의 이득이
      초기에 크고 최근으로 갈수록 0 에 수렴해야 한다.
      반증: 최근 구간에도 이득이 그대로거나 더 크면 생존편향이 아니다.

  [2] 1000배 계약 시험 — 시바이누·페페·봉크·이캐시 넷을 빼고 다시 잰다.
      격차가 그대로 남으면 이음매 탓이 아니다.

  [3] 선정 시험 — 새로 들어온 코인을 실제로 고른 날의 다음날 수익이
      옛 목록 코인을 고른 날보다 나빠야 한다. 그게 옛 백테스트가 모르고 피해 간 손실이다.
      반증: 새 코인 쪽이 더 좋거나 같으면, 하락의 원인은 선정이 아니라 다른 데 있다.

  [4] 상장 직후 시험 — 새 코인 선정이 그 코인 데이터 시작일 근처에 몰려 있으면
      원인은 생존편향이 아니라 상장 초기 변동이다.

순위파일 축은 기준선 분해에서 칼마 0.08 밖에 안 움직였으므로 여기서는 새 파일로 고정한다.
전략 파라미터는 한 값도 안 바꾼다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_universe_survivorship.py
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
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from bt_universe_baseline import OLD_SYMBOLS, NEW_MCAP_FILE, apply_state  # noqa: E402

START = "2020-10-01"
END = os.environ.get("SURV_END", "2026-09-18")

# 1000배 단위 계약으로 받은 것들. 엔진은 접두를 떼어 같은 자산으로 잇는다.
THOUSANDX = {"SHIB", "PEPE", "BONK", "XEC"}


def yearly(eq: pd.Series) -> dict:
    """연도별 수익률과 그 해 안에서의 최대낙폭."""
    eq = eq.dropna()
    out = {}
    for y, seg in eq.groupby(eq.index.year):
        out[y] = dict(ret=(seg.iloc[-1] / seg.iloc[0] - 1) * 100,
                      mdd=float((seg / seg.cummax() - 1).min()) * 100,
                      days=len(seg))
    return out


def main() -> None:
    import bt_fut_rebal_band as base
    import unified_backtest as ub
    import backtest_futures_v25 as fv25

    base.START, base.END = START, END

    all_symbols = sorted(f[:-7] for f in os.listdir(ub.DATA_DIR)
                         if f.endswith("_1h.csv"))
    old_coins = {s[:-4] for s in OLD_SYMBOLS}

    def coin_of(sym: str) -> str:
        c = sym[:-4]
        return c[4:] if c.startswith("1000") and len(c) > 4 else c

    no_thousand = [s for s in all_symbols if coin_of(s) not in THOUSANDX]

    print(f"기간 {START} ~ {END}")
    print(f"심볼 — 옛 {len(OLD_SYMBOLS)}종 / 전체 {len(all_symbols)}종 / "
          f"1000배계약 제외 {len(no_thousand)}종")

    arms = [
        ("옛 40종", OLD_SYMBOLS),
        ("전체 (지금)", all_symbols),
        ("전체 − 1000배계약 4종", no_thousand),
    ]

    eq_sp, eq_fu = {}, {}
    for label, syms in arms:
        print(f"\n[{label}] 계산 중...", flush=True)
        apply_state(ub, syms, NEW_MCAP_FILE)
        apply_state(fv25, syms, NEW_MCAP_FILE)
        eq_sp[label] = base.run_spot_live()
        eq_fu[label] = base.run_fut(0.05)["_equity"]

    print("\n" + "=" * 78)
    print("[2] 1000배 계약 시험 — 넷을 빼도 격차가 남나")
    print("=" * 78)
    for sleeve, eqs in (("코인현물", eq_sp), ("코인선물", eq_fu)):
        print(f"\n{sleeve}")
        print(f"{'팔':<26}{'CAGR':>9}{'MDD':>9}{'Sharpe':>8}{'Calmar':>8}")
        for label, _ in arms:
            m = base.metrics(eqs[label])
            print(f"{label:<26}{m['CAGR']:>8.1f}%{m['MDD']:>8.1f}%"
                  f"{m['Sharpe']:>8.2f}{m['Cal']:>8.2f}")

    print("\n" + "=" * 78)
    print("[1] 시간순 시험 — 옛 목록의 이득이 초기에 크고 최근에 0 으로 가나")
    print("=" * 78)
    print("  이득 = 옛 40종 수익률 − 전체 수익률. 생존편향이면 위에서 아래로 줄어야 한다.")
    for sleeve, eqs in (("코인현물", eq_sp), ("코인선물", eq_fu)):
        yo = yearly(eqs["옛 40종"])
        yn = yearly(eqs["전체 (지금)"])
        print(f"\n{sleeve}")
        print(f"{'연도':<8}{'옛 40종':>12}{'전체':>12}{'옛 목록 이득':>14}"
              f"{'옛 MDD':>10}{'전체 MDD':>10}")
        for y in sorted(set(yo) & set(yn)):
            a, b = yo[y], yn[y]
            print(f"{y:<8}{a['ret']:>11.1f}%{b['ret']:>11.1f}%"
                  f"{a['ret']-b['ret']:>+13.1f}%p{a['mdd']:>9.1f}%{b['mdd']:>9.1f}%")

    # ── 선정 수준 분석: 전체 심볼 팔의 현물 trace 를 쓴다 ──
    print("\n" + "=" * 78)
    print("[3][4] 선정 시험 — 새로 들어온 코인을 고른 날은 어땠나")
    print("=" * 78)
    apply_state(ub, all_symbols, NEW_MCAP_FILE)
    apply_state(fv25, all_symbols, NEW_MCAP_FILE)
    bars, _ = ub.load_data("D")

    trace: list = []
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    ub.run(bars, {}, interval="D", asset_type="spot", leverage=1.0, tx_cost=0.004,
           start_date=START, end_date=END,
           sma_bars=42, mom_short_bars=20, mom_long_bars=127,
           vol_threshold=0.05, vol_mode="daily",
           n_snapshots=7, snap_interval_bars=217,
           canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
           universe_size=3, cap=1 / 3, selection="greedy",
           stop_kind="none", stop_pct=0.0,
           dd_lookback=60, dd_threshold=-99.0,
           bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
           health_mode="mom2vol", _trace=trace)

    first_bar = {c: df.index[0] for c, df in bars.items() if len(df)}
    nxt = {}   # (coin, date) -> 다음 봉 수익률
    for c, df in bars.items():
        r = df["Close"].pct_change().shift(-1)   # 그날 보유 → 다음 봉 수익
        nxt[c] = r

    rows = []
    for item in trace:
        d = item["date"]
        for coin, w in item["target"].items():
            if coin in ("CASH", "Cash") or w <= 0:
                continue
            s = nxt.get(coin)
            if s is None or d not in s.index:
                continue
            fr = s.at[d]
            if not np.isfinite(fr):
                continue
            rows.append(dict(date=d, coin=coin, w=w, ret=fr,
                             새코인=coin not in old_coins,
                             상장경과=(d - first_bar[coin]).days))
    df = pd.DataFrame(rows)
    if df.empty:
        print("  보유 기록이 없다 — 확인 필요.")
        return

    print(f"\n보유 코인-일 {len(df)}건 (현물, 전체 심볼 팔)")
    print(f"{'구분':<16}{'코인-일':>10}{'비중합':>10}{'다음날 평균':>13}{'다음날 중앙':>13}{'승률':>9}")
    for flag, name in ((False, "옛 40종 코인"), (True, "새로 들어온 코인")):
        g = df[df["새코인"] == flag]
        if g.empty:
            continue
        print(f"{name:<16}{len(g):>10}{g.w.sum():>10.1f}{g.ret.mean()*100:>12.3f}%"
              f"{g.ret.median()*100:>12.3f}%{(g.ret > 0).mean()*100:>8.1f}%")

    print("\n연도별 — 새 코인이 전체 보유 코인-일에서 차지한 몫과 그날 성과")
    print(f"{'연도':<8}{'새코인 몫':>11}{'새코인 평균':>13}{'옛코인 평균':>13}{'차이':>12}")
    for y, g in df.groupby(df.date.dt.year):
        a = g[g["새코인"]]
        b = g[~g["새코인"]]
        share = len(a) / len(g) * 100
        ma = a.ret.mean() * 100 if len(a) else float("nan")
        mb = b.ret.mean() * 100 if len(b) else float("nan")
        print(f"{y:<8}{share:>10.1f}%{ma:>12.3f}%{mb:>12.3f}%{ma-mb:>+11.3f}%p")

    print("\n[4] 상장 직후 시험 — 고른 날이 그 코인 데이터 시작에서 얼마나 떨어져 있나")
    print(f"{'구분':<16}{'중앙':>10}{'10분위':>10}{'25분위':>10}{'90일 이내 비율':>16}")
    for flag, name in ((False, "옛 40종 코인"), (True, "새로 들어온 코인")):
        g = df[df["새코인"] == flag]
        if g.empty:
            continue
        q = g["상장경과"]
        print(f"{name:<16}{q.median():>9.0f}일{q.quantile(0.10):>9.0f}일"
              f"{q.quantile(0.25):>9.0f}일{(q <= 90).mean()*100:>15.1f}%")

    print("\n새로 들어온 코인 중 실제로 뽑힌 것 상위 (코인-일)")
    top = df[df["새코인"]].groupby("coin").agg(
        코인일=("ret", "size"), 다음날평균=("ret", "mean"))
    top["다음날평균"] *= 100
    for c, r in top.sort_values("코인일", ascending=False).head(12).iterrows():
        print(f"    {c:<8}{int(r.코인일):>6}일   다음날 평균 {r.다음날평균:>+7.3f}%")

    print("\n판정은 보고서에서 한다. 넷이 같은 방향을 가리킬 때만 생존편향이라고 쓴다.")


if __name__ == "__main__":
    main()
