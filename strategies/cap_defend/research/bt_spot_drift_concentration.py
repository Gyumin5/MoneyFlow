"""드리프트를 느슨하게 하면 얼마나 쏠리나 (2026-09-25).

계기: 문턱 재측정(bt_spot_drift_buffer.py)에서 0.15·0.20·끔이 두 비용 조건 모두 현행 0.10 을
이겼고 사건 의존 검사도 통과했다. 그런데 라이브 변경은 하지 않았다. 이유가 하나 남아서다 —
목표 비중 상한 1/3 은 앵커 사이 실제 보유 비중의 상한이 아니다. 드리프트가 비중을 되맞추는
유일한 경로이므로, 그걸 느슨하게 하면 한 코인이 커진 채로 오래 간다. 칼마와 최대낙폭은
실제로 일어난 하락만 벌하므로 '일어나지 않은 개별 코인 급락' 위험을 못 잡는다.

그래서 문턱별로 셋을 센다.
  1. 발화 횟수 — 앵커에서만 오는 것과 드리프트가 더하는 것을 가른다(끔 팔이 앵커 전용 값이다).
  2. 실제 보유 비중 — 코인 하나의 최대 비중, 상한 1/3 을 넘긴 날수와 연속 날수.
  3. 쏠린 상태의 손실 — 최대 비중이 1/3 을 넘은 날의 1일·7일 수익 분포를 안 넘은 날과 비교.

엔진은 한 줄도 안 고친다. 실제 비중은 엔진이 원래 내주는 trace(체결 목표와 발화 여부)와
일봉 종가로 재구성한다 — 체결일 d 의 목표 w 를 시작점으로 두고 이후 각 코인 가치를
w_i × P_i(t)/P_i(d) 로 굴리고 현금은 그대로 둔다. 현물은 레버리지가 없어 이 재구성이 곧
보유 비중이다. 체결가는 엔진이 t 봉 시가를 쓰는데 여기서는 종가로 두므로 체결일 당일 비중에
아주 작은 오차가 있고, 그날은 목표와 같으므로 판정(1/3 초과)에 영향이 없다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_spot_drift_concentration.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

ARMS: list[tuple[str, float]] = [
    ("현행 0.10", 0.10),
    ("0.15", 0.15),
    ("0.20", 0.20),
    ("끔", 9.99),
]
CAP_W = 1.0 / 3.0
TOL = 1e-6


def run_trace(thr: float, anchor: int) -> tuple[list[dict], pd.Series]:
    """현물 슬리브를 돌리고 trace 와 자산곡선을 받는다."""
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    kw = dict(R.SPOT_BASE)
    kw["drift_threshold"] = thr
    tr: list[dict] = []
    m = bt_run(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
               vol_mode="daily", phase_offset_bars=R.PH_SPOT[anchor],
               post_flip_delay=5, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", _trace=tr, **kw)
    return tr, m.get("_equity")


def actual_weights(trace: list[dict]) -> pd.DataFrame:
    """체결 목표와 종가로 일별 실제 보유 비중을 재구성한다."""
    dates = [t['date'] for t in trace]
    cols: dict[str, dict] = {}
    cur: dict[str, float] = {}      # 코인별 가치(체결 시점 1.0 기준)
    base: dict[str, float] = {}     # 체결 시점 종가
    cash = 1.0

    def price(coin: str, date) -> float:
        df = R._bars.get(coin)
        if df is None:
            return 0.0
        ci = df.index.get_indexer([date], method='ffill')[0]
        return float(df['Close'].iloc[ci]) if ci >= 0 else 0.0

    for rec in trace:
        d = rec['date']
        if rec['rebal']:
            tot = cash + sum(cur.values())
            cur, base = {}, {}
            tgt = rec['target']
            cash = tot * float(tgt.get('CASH', 0.0) + tgt.get('Cash', 0.0))
            for c, w in tgt.items():
                if c in ('CASH', 'Cash') or w <= 0:
                    continue
                p = price(c, d)
                if p <= 0:
                    cash += tot * float(w)
                    continue
                cur[c] = tot * float(w)
                base[c] = p
        # 가격을 반영한 현재 가치
        vals = {}
        for c in cur:
            p = price(c, d)
            vals[c] = cur[c] * (p / base[c]) if (p > 0 and base[c] > 0) else cur[c]
        tot = cash + sum(vals.values())
        cols[d] = {c: (v / tot if tot > 0 else 0.0) for c, v in vals.items()}
        cols[d]['CASH'] = cash / tot if tot > 0 else 1.0
        # 다음 봉을 위해 가치를 갱신하고 기준가를 오늘로 옮긴다
        for c in list(cur):
            p = price(c, d)
            if p > 0 and base[c] > 0:
                cur[c] = cur[c] * (p / base[c])
                base[c] = p
    return pd.DataFrame(cols).T.fillna(0.0).reindex(dates)


def summarize(name: str, w: pd.DataFrame, eq: pd.Series, fires: int,
              anchor_fires: int) -> dict:
    coin = w.drop(columns=['CASH'], errors='ignore')
    mx = coin.max(axis=1)
    held = coin.sum(axis=1)
    inv = held > 0.01                      # 투자 상태인 날만 본다
    over = inv & (mx > CAP_W + 0.005)      # 상한 1/3 을 넘긴 날
    n_inv = int(inv.sum())

    streak = best = 0
    for v in over.values:
        streak = streak + 1 if v else 0
        best = max(best, streak)

    r1 = eq.pct_change().reindex(w.index)
    r7 = (eq / eq.shift(7) - 1).reindex(w.index)

    def q(s: pd.Series, mask: pd.Series) -> tuple[float, float]:
        x = s[mask].dropna()
        if not len(x):
            return float('nan'), float('nan')
        return float(np.percentile(x, 5)), float(x.min())

    p1o, m1o = q(r1, over)
    p1u, m1u = q(r1, inv & ~over)
    p7o, m7o = q(r7, over)
    p7u, m7u = q(r7, inv & ~over)

    return dict(
        name=name, fires=fires, drift_fires=max(0, fires - anchor_fires),
        n_inv=n_inv, over=int(over.sum()),
        over_share=(over.sum() / n_inv if n_inv else float('nan')),
        streak=best,
        mx_med=float(mx[inv].median()) if n_inv else float('nan'),
        mx_p95=float(np.percentile(mx[inv], 95)) if n_inv else float('nan'),
        mx_max=float(mx[inv].max()) if n_inv else float('nan'),
        d40=int((inv & (mx > 0.40)).sum()), d50=int((inv & (mx > 0.50)).sum()),
        p1o=p1o, m1o=m1o, p1u=p1u, m1u=m1u, p7o=p7o, m7o=m7o, p7u=p7u, m7u=m7u,
    )


def main() -> None:
    t0 = time.time()
    from unified_backtest import load_data
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)\n", flush=True)

    rows_all: dict[str, list[dict]] = {n: [] for n, _ in ARMS}
    for a in R.ANCHORS:
        anchor_fires = None
        cache: dict[str, tuple[int, pd.DataFrame, pd.Series]] = {}
        for name, thr in ARMS:
            tr, eq = run_trace(thr, a)
            fires = sum(1 for x in tr if x['rebal'])
            cache[name] = (fires, actual_weights(tr), eq)
            if name == "끔":
                anchor_fires = fires
        for name, _ in ARMS:
            fires, w, eq = cache[name]
            rows_all[name].append(summarize(name, w, eq, fires, anchor_fires or 0))
        print(f"  앵커 {a} 완료 ({time.time()-t0:.0f}초)", flush=True)

    print("\n=== 발화 구성 (앵커 7곳 평균, 5.9년 누적) ===")
    print(f"{'문턱':<10}{'총발화':>8}{'앵커':>8}{'드리프트':>10}")
    for name, _ in ARMS:
        rs = rows_all[name]
        print(f"{name:<10}{np.mean([r['fires'] for r in rs]):>8.1f}"
              f"{np.mean([r['fires'] - r['drift_fires'] for r in rs]):>8.1f}"
              f"{np.mean([r['drift_fires'] for r in rs]):>10.1f}")

    print("\n=== 코인 하나의 최대 보유 비중 (투자 상태인 날만, 앵커 7곳 평균) ===")
    print(f"{'문턱':<10}{'투자일':>8}{'중앙':>8}{'95분위':>8}{'최대':>8}"
          f"{'1/3초과일':>10}{'비율':>7}{'최장연속':>9}{'40%초과':>8}{'50%초과':>8}")
    for name, _ in ARMS:
        rs = rows_all[name]
        print(f"{name:<10}{np.mean([r['n_inv'] for r in rs]):>8.0f}"
              f"{np.mean([r['mx_med'] for r in rs]):>8.3f}"
              f"{np.mean([r['mx_p95'] for r in rs]):>8.3f}"
              f"{np.mean([r['mx_max'] for r in rs]):>8.3f}"
              f"{np.mean([r['over'] for r in rs]):>10.1f}"
              f"{np.mean([r['over_share'] for r in rs]):>7.1%}"
              f"{np.mean([r['streak'] for r in rs]):>9.1f}"
              f"{np.mean([r['d40'] for r in rs]):>8.1f}"
              f"{np.mean([r['d50'] for r in rs]):>8.1f}")

    print("\n=== 쏠린 날의 손실 (슬리브 수익, 앵커 7곳 평균) ===")
    print(f"{'문턱':<10}{'1일 5%분위':>11}{'1일 최악':>10}{'7일 5%분위':>11}{'7일 최악':>10}"
          f"{'  (1/3 이하 날: 1일 5%분위 · 최악)':<0}")
    for name, _ in ARMS:
        rs = rows_all[name]
        def m(k): return np.nanmean([r[k] for r in rs])
        print(f"{name:<10}{m('p1o'):>11.2%}{m('m1o'):>10.2%}"
              f"{m('p7o'):>11.2%}{m('m7o'):>10.2%}"
              f"    {m('p1u'):>8.2%} · {m('m1u'):>8.2%}")

    print("\n=== 앵커별 1/3 초과 날수 ===")
    for name, _ in ARMS:
        rs = rows_all[name]
        print(f"  {name:<10}" + " ".join(f"{r['over']:>5d}" for r in rs))


if __name__ == '__main__':
    main()
