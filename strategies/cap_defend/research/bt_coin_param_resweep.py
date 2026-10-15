"""넓어진 유니버스 위에서 코인 결정값 전부를 다시 잰다 (2026-09-22).

전제가 바뀌었다. 백테스트 심볼표가 38종에서 102종으로 고쳐졌고(커밋 64843f3),
지금 쓰는 값들은 전부 좁은 후보군에서 고른 것이다. 그래서 축마다 현행을 가운데 두고
인접값을 같이 돌려 무엇이 달라지는지 본다.

수용력 진단(bt_universe_capacity.py)에서 나온 것 둘이 무엇을 재야 하는지를 정했다.
  · 상한 3종은 죽은 축이 아니다 — 건강도 4종 이상 통과한 날이 862일(39.6%).
  · 변동성 문턱 0.03~0.04 를 기각했던 근거("후보가 6종으로 줄어 현금화")가 무너졌다.
    넓어진 후보군에서 0.04 는 평균 3.78종이 통과한다.
의도적 재현이다 — 옛 기각은 좁은 후보군에서 낸 것이라 근거가 무효가 됐다.

재는 법:
  · 축 하나에 한 값만 바꾼다. 공유 파라미터(변동성 문턱·모멘텀 기간·카나리)는
    현물·선물 양쪽에 같이 건다. 슬리브 고유 파라미터는 그쪽만 건다.
  · 앵커는 7곳이다(ANCHORS_N=11 이면 11곳 — 뒤에 덧붙인 넷을 더 쓴다).
    현물 위상은 31의 배수를 피하고(스냅 간격 217 = 7 × 31 이라
    31의 배수는 스냅 집합을 자기 자신으로 옮겨 같은 포트폴리오가 된다),
    선물 위상은 19의 배수를 피한다(95 = 5 × 19). 주식은 앵커 0~6.
  · 결정층은 합성 60/25/15 윈도우 순위합이고, 앵커 0 하나가 아니라 앵커 7곳
    각각에서 낸 뒤 평균·중앙·최악을 같이 적는다. 둘이 어긋나면 어긋난 채로 적는다.
  · 비용 5배 스트레스를 같은 축에 붙인다.

중간 결과는 state/20260922-resweep/eq/*.csv 에 남고 재시작하면 이어 간다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_coin_param_resweep.py
      SHARD=0/6 … SHARD=5/6 으로 여섯 벌을 같이 돌리면 계산만 나눠 한다(보고는 건너뜀).
      다 모이면 SHARD 없이 한 번 더 돌려 보고만 받는다 — 끝난 조합은 건너뛴다.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

OUT = "/home/gmoh/mon/251229/state/20260922-resweep"
EQDIR = os.path.join(OUT, "eq")

START = "2020-10-01"
END = "2026-09-18"
WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]

TX_SPOT, TX_FUT = 0.004, 0.0006
STRESS = 5.0

# 위상: 현물은 31의 배수, 선물은 19의 배수를 피한다(스냅 집합이 자기 자신으로 옮겨감).
# 앞 일곱이 기본값이고 뒤 넷은 ANCHORS_N=11 일 때만 쓴다. 뒤에 덧붙이는 까닭은
# 앵커 번호가 체크포인트 파일 이름(a0~a6)이라, 사이에 끼워 넣으면 이미 계산된
# 곡선이 다른 위상을 가리키게 되기 때문이다. 넷은 기존 일곱의 빈 칸을 메운다
# (현물 주기 31 에 2·6·15·24, 선물 주기 19 에 2·6·10·17).
PH_SPOT = [0, 4, 9, 13, 18, 22, 27] + [2, 6, 15, 24]
PH_FUT = [0, 3, 5, 8, 11, 14, 16] + [2, 6, 10, 17]
ANCHORS = list(range(int(os.environ.get("ANCHORS_N", 7))))

CUR = "현행"

SPOT_BASE = dict(sma_bars=42, mom_short_bars=20, mom_long_bars=127,
                 vol_threshold=0.05, n_snapshots=7, snap_interval_bars=217,
                 canary_hyst=0.015, drift_threshold=0.10, universe_size=3, cap=1 / 3)
FUT_BASE = dict(sma_days=42, mom_short_days=18, mom_long_days=127,
                vol_threshold=0.05, n_snapshots=5, snap_interval_bars=95,
                canary_hyst=0.015, drift_threshold=0.015, universe_size=3, cap=1 / 3)


def _base_tag() -> str:
    """기준선 지문. 캐시 이름에 넣어 기준선이 바뀌면 옛 곡선을 안 읽게 한다.

    기준선 값을 고치면 '현행' 팔의 곡선이 달라지는데 파일명이 같으면 조용히 섞인다
    (2026-10-15 history #50 과 같은 기제). 지문이 다르면 파일명이 달라져 새로 계산한다.
    """
    import hashlib
    raw = repr(sorted(SPOT_BASE.items())) + repr(sorted(FUT_BASE.items()))
    return hashlib.sha1(raw.encode()).hexdigest()[:6]


def _topn(n: int) -> tuple[dict, dict]:
    """상위 N 종 균등. 상한을 같이 옮기지 않으면 N 을 늘려도 아무 일도 안 난다."""
    return dict(universe_size=n, cap=1 / n), dict(universe_size=n, cap=1 / n)


# 축 = [(이름, 현물 덮어쓰기, 선물 덮어쓰기)]. 현행은 빈 덮어쓰기로 둔다.
AXES: dict[str, list[tuple[str, dict, dict]]] = {
    "상위 종수(상한 1/N 동반)": [
        ("상위 2종", *_topn(2)), (CUR, {}, {}),
        ("상위 4종", *_topn(4)), ("상위 5종", *_topn(5))],
    "변동성 문턱": [
        ("문턱 0.03", dict(vol_threshold=0.03), dict(vol_threshold=0.03)),
        ("문턱 0.04", dict(vol_threshold=0.04), dict(vol_threshold=0.04)),
        (CUR, {}, {}),
        ("문턱 0.06", dict(vol_threshold=0.06), dict(vol_threshold=0.06)),
        ("문턱 0.07", dict(vol_threshold=0.07), dict(vol_threshold=0.07))],
    "장기 모멘텀 기간": [
        ("장기 90일", dict(mom_long_bars=90), dict(mom_long_days=90)),
        ("장기 110일", dict(mom_long_bars=110), dict(mom_long_days=110)),
        (CUR, {}, {}),
        ("장기 150일", dict(mom_long_bars=150), dict(mom_long_days=150)),
        ("장기 180일", dict(mom_long_bars=180), dict(mom_long_days=180))],
    "단기 모멘텀 기간": [
        ("단기 -6일", dict(mom_short_bars=14), dict(mom_short_days=12)),
        (CUR, {}, {}),
        ("단기 +6일", dict(mom_short_bars=26), dict(mom_short_days=24)),
        ("단기 +12일", dict(mom_short_bars=32), dict(mom_short_days=30))],
    "카나리 이동평균": [
        ("30일선", dict(sma_bars=30), dict(sma_days=30)),
        (CUR, {}, {}),
        ("55일선", dict(sma_bars=55), dict(sma_days=55)),
        ("70일선", dict(sma_bars=70), dict(sma_days=70))],
    "카나리 히스테리시스": [
        ("±0.5%", dict(canary_hyst=0.005), dict(canary_hyst=0.005)),
        (CUR, {}, {}),
        ("±2.5%", dict(canary_hyst=0.025), dict(canary_hyst=0.025)),
        ("±4.0%", dict(canary_hyst=0.040), dict(canary_hyst=0.040))],
    "현물 드리프트 문턱": [
        ("현물 0.06", dict(drift_threshold=0.06), {}),
        (CUR, {}, {}),
        ("현물 0.15", dict(drift_threshold=0.15), {}),
        ("현물 0.20", dict(drift_threshold=0.20), {})],
    # 현행이 0.015 다. 옛 기준선(0.03)을 이웃 하나로 내려 두고 더 조인 쪽을 하나 넣는다.
    "선물 드리프트 문턱": [
        ("선물 0.008", {}, dict(drift_threshold=0.008)),
        (CUR, {}, {}),
        ("선물 0.03", {}, dict(drift_threshold=0.03)),
        ("선물 0.05", {}, dict(drift_threshold=0.05)),
        ("선물 0.08", {}, dict(drift_threshold=0.08))],
    "현물 스냅 간격": [
        ("현물 155봉", dict(snap_interval_bars=155), {}),
        (CUR, {}, {}),
        ("현물 280봉", dict(snap_interval_bars=280), {}),
        ("현물 340봉", dict(snap_interval_bars=340), {})],
    "선물 스냅 간격": [
        ("선물 65봉", {}, dict(snap_interval_bars=65)),
        (CUR, {}, {}),
        ("선물 125봉", {}, dict(snap_interval_bars=125)),
        ("선물 160봉", {}, dict(snap_interval_bars=160))],
}

_bars = None
_funding = None
_k2 = None


def _key(name: str, anchor: int, stress: bool) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace("±", "pm")
    return f"{safe}__a{anchor}{'__s5' if stress else ''}__b{_base_tag()}"


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(os.path.join(EQDIR, f"{kind}__{key}.csv"), header=["eq"])


def _load(kind: str, key: str) -> pd.Series | None:
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def run_spot(over: dict, phase: int, stress: bool) -> pd.Series:
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    kw = dict(SPOT_BASE)
    kw.update(over)
    m = bt_run(_bars, _funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=TX_SPOT * (STRESS if stress else 1.0),
               start_date=START, end_date=END,
               vol_mode="daily", phase_offset_bars=phase,
               post_flip_delay=5, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", **kw)
    return m.get("_equity")


def run_fut(over: dict, phase: int, stress: bool) -> pd.Series:
    from backtest_futures_v25 import run as fbt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    kw = dict(FUT_BASE)
    kw.update(over)
    m = fbt_run(_bars, _funding, interval="D", leverage=_k2,
                tx_cost=TX_FUT * (STRESS if stress else 1.0), maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=5,
                phase_offset_bars=phase, health_mode="mom2vol",
                start_date=START, end_date=END, **kw)
    return m.get("_equity")


def run_stock(anchor: int, stress: bool) -> pd.Series:
    """주식 슬리브는 이 스윕의 축이 아니다 — 합성용 고정 팔이고 앵커만 따라 움직인다."""
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
    return run_multi_3mom(pdf, ranked, mom_off, mom_def, canary,
                          pd.Timestamp(START), pd.Timestamp(END), anchor=anchor,
                          drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
                          snap_int=69, n_snaps=3)


def build_alloc(eq_st, eq_sp, eq_fu, w=(0.60, 0.25, 0.15)) -> pd.Series:
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    r = (w[0] * eq_st.loc[common].pct_change().fillna(0)
         + w[1] * eq_sp.loc[common].pct_change().fillna(0)
         + w[2] * eq_fu.loc[common].pct_change().fillna(0))
    return (1 + r).cumprod()


def metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    rets = eq.pct_change().dropna()
    sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    return dict(CAGR=cagr * 100, MDD=mdd * 100, Sharpe=sh,
                Cal=cagr / abs(mdd) if mdd < 0 else 0.0)


def window_rs(eq_dict: dict[str, pd.Series]):
    """윈도우 순위합. 낮은 평균순위가 우세. 동률은 평균순위로 준다."""
    common = None
    for s in eq_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    common = sorted(common)
    sums: dict[str, float] = defaultdict(float)
    wins: dict[str, int] = defaultdict(int)
    n = 0
    for size in WIN_SIZES:
        for stride in STRIDES:
            for i in range(0, len(common) - size, stride):
                d0, d1 = common[i], common[i + size - 1]
                cals = {}
                bad = False
                for k, s in eq_dict.items():
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30:
                        bad = True
                        break
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0:
                        bad = True
                        break
                    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                    mdd = float((seg / seg.cummax() - 1).min())
                    cals[k] = cagr / abs(mdd) if mdd < 0 else 0.0
                if bad:
                    continue
                order = sorted(cals.items(), key=lambda x: -x[1])
                # 동률은 평균순위
                i0 = 0
                while i0 < len(order):
                    i1 = i0
                    while i1 + 1 < len(order) and order[i1 + 1][1] == order[i0][1]:
                        i1 += 1
                    avg_rank = (i0 + i1) / 2 + 1
                    for j in range(i0, i1 + 1):
                        sums[order[j][0]] += avg_rank
                    i0 = i1 + 1
                wins[order[0][0]] += 1
                n += 1
    return sums, wins, n


def ensure(name: str, sp_over: dict, fu_over: dict, anchor: int, stress: bool) -> None:
    k = _key(name, anchor, stress)
    if _load("sp", k) is None:
        _save("sp", k, run_spot(sp_over, PH_SPOT[anchor], stress))
    if _load("fu", k) is None:
        _save("fu", k, run_fut(fu_over, PH_FUT[anchor], stress))


def report(axis: str, arms: list[tuple[str, dict, dict]], stress: bool,
           stock_eq: dict[int, pd.Series]) -> None:
    names = [a[0] for a in arms]
    per_anchor_rank: dict[str, list[float]] = {n: [] for n in names}
    per_anchor_cal: dict[str, list[float]] = {n: [] for n in names}
    sleeve_sp: dict[str, list[float]] = {n: [] for n in names}
    sleeve_fu: dict[str, list[float]] = {n: [] for n in names}
    comp_mdd: dict[str, list[float]] = {n: [] for n in names}

    for a in ANCHORS:
        comps = {}
        for name in names:
            k = _key(name, a, stress)
            sp, fu = _load("sp", k), _load("fu", k)
            if sp is None or fu is None:
                continue
            comps[name] = build_alloc(stock_eq[a], sp, fu)
            sleeve_sp[name].append(metrics(sp)["Cal"])
            sleeve_fu[name].append(metrics(fu)["Cal"])
            m = metrics(comps[name])
            per_anchor_cal[name].append(m["Cal"])
            comp_mdd[name].append(m["MDD"])
        if len(comps) < 2:
            continue
        sums, _, n = window_rs(comps)
        for name in comps:
            per_anchor_rank[name].append(sums[name] / max(n, 1))

    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== {axis} · {tag} · 앵커 {len(ANCHORS)}곳 ===")
    print(f"{'팔':<16}{'순위 평균':>10}{'중앙':>7}{'최악':>7}{'현행대비':>9}"
          f"{'합성Cal':>9}{'합성MDD':>9}{'현물Cal':>9}{'선물Cal':>9}")
    base = per_anchor_rank.get(CUR, [])
    base_mean = float(np.mean(base)) if base else float("nan")
    rows = []
    for name in names:
        r = per_anchor_rank[name]
        if not r:
            continue
        rows.append((float(np.mean(r)), name, r))
    for mean_r, name, r in sorted(rows):
        gap = mean_r - base_mean
        mark = "  <- 현행" if name == CUR else ""
        print(f"{name:<16}{mean_r:>10.3f}{float(np.median(r)):>7.3f}{max(r):>7.3f}"
              f"{gap:>+9.3f}{float(np.mean(per_anchor_cal[name])):>9.2f}"
              f"{float(np.mean(comp_mdd[name])):>9.2f}"
              f"{float(np.mean(sleeve_sp[name])):>9.2f}"
              f"{float(np.mean(sleeve_fu[name])):>9.2f}{mark}")
    if base:
        print("  앵커별 순위(현행 기준 차이) — 낮을수록 우세, 음수면 그 팔이 현행보다 낫다")
        for mean_r, name, r in sorted(rows):
            if name == CUR:
                continue
            diffs = [x - y for x, y in zip(r, base)]
            cells = " ".join(f"{d:+6.2f}" for d in diffs)
            wins = sum(1 for d in diffs if d < 0)
            print(f"    {name:<14} {cells}   이긴 앵커 {wins}/{len(diffs)}")


def main() -> None:
    global _bars, _funding, _k2
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...")
    _bars, _funding = load_data("D")
    _k2 = build_K2_signal(_bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                          btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                          l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(_bars)}종 ({time.time()-t0:.0f}초)")

    print("주식 슬리브(합성용 고정 팔) 앵커별 계산 중...")
    stock_eq = {}
    for a in ANCHORS:
        p = os.path.join(EQDIR, f"st__a{a}.csv")
        if os.path.exists(p):
            stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = run_stock(a, False)
            stock_eq[a].to_csv(p, header=["eq"])
    print(f"  완료 ({time.time()-t0:.0f}초)")

    todo = []
    for axis, arms in AXES.items():
        for name, sp_over, fu_over in arms:
            for a in ANCHORS:
                for stress in (False, True):
                    todo.append((name, sp_over, fu_over, a, stress))
    # 현행은 축마다 겹치므로 한 번만
    seen = set()
    uniq = []
    for item in todo:
        kk = (item[0], item[3], item[4])
        if kk in seen:
            continue
        seen.add(kk)
        uniq.append(item)
    print(f"\n계산할 조합 {len(uniq)}건 (현행 중복 제거 후)")

    # 쪼개 돌리기. SHARD=i/n 이면 번호가 i 인 조합만 계산하고 보고는 건너뛴다.
    # 조합마다 파일 이름이 달라서 쪼갠 몫끼리 같은 파일을 건드리지 않는다 —
    # 그래서 여러 벌을 동시에 돌려도 서로의 결과를 덮거나 반쯤 쓴 파일을 읽지 않는다.
    shard = os.environ.get("SHARD", "")
    if shard:
        si, sn = (int(x) for x in shard.split("/"))
        uniq = [item for i, item in enumerate(uniq) if i % sn == si]
        print(f"  이 몫 {si}/{sn} = {len(uniq)}건")

    done = 0
    for name, sp_over, fu_over, a, stress in uniq:
        k = _key(name, a, stress)
        if _load("sp", k) is not None and _load("fu", k) is not None:
            done += 1
            continue
        ensure(name, sp_over, fu_over, a, stress)
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(uniq)} ({time.time()-t0:.0f}초)")
    print(f"  전부 완료 ({time.time()-t0:.0f}초)")

    if shard:
        print("쪼갠 몫이라 보고는 건너뛴다 — 전부 모인 뒤에 보고만 따로 돌려라.")
        return

    for axis, arms in AXES.items():
        report(axis, arms, False, stock_eq)
        report(axis, arms, True, stock_eq)

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
