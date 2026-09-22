"""상장 초기 시험을 다시 만든다 — 앞의 시험이 무효였다 (2026-09-22).

bt_universe_survivorship.py 의 [4] 는 "고른 날이 그 코인 데이터 시작에서 90일 이내인가" 를
물었고 양쪽 다 0.0% 가 나왔다. 그건 증거가 아니다. 엔진이 후보로 올리는 조건 자체가
127봉 이상이기 때문이다 (unified_backtest.py:551,562 — min_bars = max(모멘텀단기 20,
모멘텀장기 127, 변동성 90, SMA 42) 이고 그보다 짧으면 continue). 즉 90일 이내 선정은
전략상 일어날 수 없고, 그 시험은 어떤 자료를 넣어도 0.0% 를 뱉는다.

그래서 기준선을 바닥(127봉)으로 옮겨 다시 묻는다. 상장 초기 변동이 원인이라면 새 코인
선정이 후보 자격을 얻은 직후, 즉 바닥 + 얼마 안 되는 구간에 몰려 있어야 한다.
반증: 새 코인 선정이 옛 코인과 비슷하게 바닥에서 멀리 흩어져 있으면 상장 초기 변동이
원인이 아니다.

같이 재는 것: 바닥 근처 구간(바닥+90일 이내)에서 고른 날의 다음날 수익. 그 구간이
실제로 더 나빴는지를 봐야 "초기 변동이 손실을 만들었다" 가 성립한다.

전략 파라미터는 한 값도 안 바꾼다. 심볼표는 지금 것(전체), 순위파일은 새 것으로 고정.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_universe_listing_floor.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from bt_universe_baseline import OLD_SYMBOLS, NEW_MCAP_FILE, apply_state  # noqa: E402

START = "2020-10-01"
END = os.environ.get("SURV_END", "2026-09-18")

# 엔진이 후보로 올리는 최소 봉수. unified_backtest._compute_weights 의 min_bars 와 같은 식.
FLOOR = max(20, 127, 90, 42)


def main() -> None:
    import unified_backtest as ub
    import backtest_futures_v25 as fv25

    all_symbols = sorted(f[:-7] for f in os.listdir(ub.DATA_DIR) if f.endswith("_1h.csv"))
    old_coins = {s[:-4] for s in OLD_SYMBOLS}

    apply_state(ub, all_symbols, NEW_MCAP_FILE)
    apply_state(fv25, all_symbols, NEW_MCAP_FILE)
    bars, _ = ub.load_data("D")

    print(f"기간 {START} ~ {END} · 심볼 {len(all_symbols)}종")
    print(f"후보 자격 바닥 = {FLOOR}봉 (모멘텀장기 127 이 지배). 이보다 짧으면 선정 자체가 불가능하다.")

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
    nxt = {c: df["Close"].pct_change().shift(-1) for c, df in bars.items()}

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
            # 그 코인 자신의 봉 수 기준 경과 (달력일이 아니라 자료 봉수)
            df_c = bars[coin]
            ci = df_c.index.get_indexer([d], method="ffill")[0]
            rows.append(dict(date=d, coin=coin, ret=fr,
                             새코인=coin not in old_coins,
                             경과봉=int(ci), 바닥초과=int(ci) - FLOOR,
                             경과일=(d - first_bar[coin]).days))
    df = pd.DataFrame(rows)
    if df.empty:
        print("  보유 기록이 없다 — 확인 필요.")
        return

    print(f"\n보유 코인-일 {len(df)}건")
    print("\n[가] 후보 자격 바닥에서 얼마나 떨어져 골랐나 (경과봉 − 127)")
    print(f"{'구분':<16}{'최소':>8}{'10분위':>9}{'25분위':>9}{'중앙':>9}{'바닥+90 이내':>14}")
    for flag, name in ((False, "옛 40종 코인"), (True, "새로 들어온 코인")):
        g = df[df["새코인"] == flag]
        if g.empty:
            continue
        q = g["바닥초과"]
        print(f"{name:<16}{q.min():>7.0f}봉{q.quantile(0.10):>8.0f}봉{q.quantile(0.25):>8.0f}봉"
              f"{q.median():>8.0f}봉{(q <= 90).mean()*100:>13.1f}%")

    print("\n[나] 구간별 코인-일 수와 그날의 다음날 수익")
    bins = [(0, 90, "바닥 직후 (127~217봉)"), (91, 180, "그 다음 (218~307봉)"),
            (181, 365, "308~492봉"), (366, 10 ** 9, "493봉 이상")]
    print(f"{'구간':<24}{'옛 코인-일':>11}{'옛 평균':>10}{'새 코인-일':>11}{'새 평균':>10}")
    for lo, hi, label in bins:
        sel = df[(df["바닥초과"] >= lo) & (df["바닥초과"] <= hi)]
        o = sel[~sel["새코인"]]
        n = sel[sel["새코인"]]
        mo = f"{o.ret.mean()*100:>9.3f}%" if len(o) else f"{'-':>10}"
        mn = f"{n.ret.mean()*100:>9.3f}%" if len(n) else f"{'-':>10}"
        print(f"{label:<24}{len(o):>11}{mo}{len(n):>11}{mn}")

    print("\n[다] 새 코인 손실이 바닥 근처에 몰려 있나 — 새 코인만 구간별로")
    n_all = df[df["새코인"]]
    print(f"{'구간':<24}{'코인-일':>9}{'평균':>10}{'중앙':>10}{'승률':>9}")
    for lo, hi, label in bins:
        g = n_all[(n_all["바닥초과"] >= lo) & (n_all["바닥초과"] <= hi)]
        if g.empty:
            print(f"{label:<24}{0:>9}{'-':>10}{'-':>10}{'-':>9}")
            continue
        print(f"{label:<24}{len(g):>9}{g.ret.mean()*100:>9.3f}%{g.ret.median()*100:>9.3f}%"
              f"{(g.ret > 0).mean()*100:>8.1f}%")


if __name__ == "__main__":
    main()
