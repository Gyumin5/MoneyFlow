"""주식 슬리브 결정값을 축마다 다시 잰다 (2026-09-25).

코인은 2026-09-22 에 열 축을 다시 쟀는데(bt_coin_param_resweep.py) 주식은 그때
합성용 고정 팔로만 들어가 있었다. 그래서 "바꿀 게 없다" 는 판정의 근거가 코인·선물보다
약하다. 이 스윕이 그 자리를 메운다.

재는 법은 코인 쪽과 같다.
  · 축 하나에 한 값만 바꾼다. 현행을 가운데 두고 인접값을 같이 돌린다.
  · 앵커는 7곳(주식 앵커 0~6). 합성 60/25/15 윈도우 순위합을 앵커마다 낸 뒤
    평균·중앙·최악을 같이 적는다.
  · 비용 5배 스트레스를 같은 축에 붙인다. 주식 슬리브 비용은 bt_stock_mom3.TX 다 —
    bt_coin_param_resweep.run_stock 은 stress 인자를 받기만 하고 쓰지 않으므로
    (주식이 그 스윕의 축이 아니었다) 여기서 직접 건다.
  · 현물·선물은 현행 고정이고 이미 계산된 곡선을 그대로 쓴다
    (state/20260925-spotdrift-buf/eq, 현금버퍼 1% 교정 후 판본).

주의할 성질 둘.
  · 스태거는 snap_int // n_snaps 로 파생된다. 스냅 개수를 바꾸면 스태거도 같이 바뀐다
    (현행 69//3 = 23). 자산군끼리 스냅 개수를 서로소로 두는 규약은 현물 7 · 선물 5 라
    주식 2·3·4 는 전부 통과하지만, 스태거가 소수가 아니게 되는 값은 따로 적는다.
  · 3중 모멘텀 기간(ms/mid/ml)은 선정 점수가 아니라 필터에만 쓰인다. Z 점수 랭킹은
    63·126·252 가중 고정이라 이 축을 움직여도 랭킹 자체는 안 바뀐다.
  · 이 엔진에서 드리프트 문턱은 유일한 체결 관문이다(bt_stock_mom3.py 103행,
    ht >= drift_thr 일 때만 holdings 를 갱신). 앵커일에 스냅 목표가 바뀌어도 그
    변화가 문턱을 넘어야 실제로 체결된다. 그래서 "문턱 끔" 팔은 종목 교체를 앵커로만
    제한하는 것이 아니라 6년 내내 전액 현금으로 두는 퇴화 팔이라 넣지 않는다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_stock_param_resweep.py
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

# 현물·선물 현행 곡선은 여기서 읽는다(현금버퍼 1% 교정 후 판본).
COIN_EQ = "/home/gmoh/mon/251229/state/20260925-spotdrift-buf/eq"
OUT = "/home/gmoh/mon/251229/state/20260925-stocksweep"
EQDIR = os.path.join(OUT, "eq")

CUR = R.CUR
ANCHORS = R.ANCHORS

# 현행 주식 파라미터 (executor_stock.py / stock_strategy_v25.py 정합)
BASE = dict(drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
            snap_int=69, n_snaps=3, canary_sma=200, canary_hyst=0.005)

# 축 = [(이름, 덮어쓰기)]. 현행은 빈 덮어쓰기.
AXES: dict[str, list[tuple[str, dict]]] = {
    "드리프트 문턱": [
        ("문턱 0.03", dict(drift_thr=0.03)), (CUR, {}),
        ("문턱 0.08", dict(drift_thr=0.08)), ("문턱 0.10", dict(drift_thr=0.10)),
        ("문턱 0.15", dict(drift_thr=0.15))],
    "스냅 간격": [
        ("간격 60일", dict(snap_int=60)), ("간격 63일", dict(snap_int=63)),
        (CUR, {}),
        ("간격 75일", dict(snap_int=75)), ("간격 84일", dict(snap_int=84))],
    "스냅 개수": [
        ("스냅 2개", dict(n_snaps=2)), (CUR, {}), ("스냅 4개", dict(n_snaps=4))],
    "단기 모멘텀": [
        ("단기 20일", dict(ms=20)), (CUR, {}),
        ("단기 40일", dict(ms=40)), ("단기 50일", dict(ms=50))],
    "중기 모멘텀": [
        ("중기 60일", dict(mid=60)), (CUR, {}), ("중기 90일", dict(mid=90))],
    "장기 모멘텀": [
        ("장기 200일", dict(ml=200)), (CUR, {}), ("장기 260일", dict(ml=260))],
    "카나리 이동평균": [
        ("카나리 150일", dict(canary_sma=150)), (CUR, {}),
        ("카나리 250일", dict(canary_sma=250))],
    "카나리 히스테리시스": [
        ("히스 0%", dict(canary_hyst=0.0)), (CUR, {}),
        ("히스 1.5%", dict(canary_hyst=0.015))],
    "계좌 현금버퍼(운영자 결정 영역)": [
        ("버퍼 3%", dict(cash_buf=0.03)), ("버퍼 5%", dict(cash_buf=0.05)),
        (CUR, {}), ("버퍼 10%", dict(cash_buf=0.10))],
}

_pdf: pd.DataFrame | None = None
_pre: dict[tuple, tuple] = {}


def _key(name: str, anchor: int, stress: bool) -> str:
    return f"{name.replace(' ', '_')}__a{anchor}" + ("__s5" if stress else "")


def _load_st(k: str):
    p = os.path.join(EQDIR, f"st__{k}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)["eq"]


def _save_st(k: str, s: pd.Series) -> None:
    s.to_csv(os.path.join(EQDIR, f"st__{k}.csv"), header=["eq"])


def _prices() -> pd.DataFrame:
    global _pdf
    if _pdf is None:
        from stock_engine import load_prices, ALL_TICKERS
        import bt_stock_coin_v3 as bcv3
        bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
        pm = load_prices(ALL_TICKERS, start="2005-01-01")
        # bt_coin_param_resweep.run_stock 과 한 글자도 다르면 안 된다 — 스냅 위상이
        # 봉 번호(i % snap_int)로 도는데 중복행·장중 타임스탬프가 남으면 봉 수가
        # 두 배가 되어 같은 파라미터가 다른 전략이 된다(실측 2814봉 대 1498봉).
        _pdf = pd.DataFrame(pm)
        _pdf = _pdf[~_pdf.index.duplicated(keep="first")].sort_index()
        _pdf = _pdf[_pdf.index.normalize() == _pdf.index]
    return _pdf


def _precomputed(p: dict):
    """(랭킹·모멘텀·카나리)는 (ms,mid,ml,카나리 2종)에만 의존한다 — 재사용한다."""
    kk = (p["ms"], p["mid"], p["ml"], p["canary_sma"], p["canary_hyst"])
    if kk not in _pre:
        import bt_stock_coin_v3 as bcv3
        bcv3.CANARY_SMA = p["canary_sma"]
        bcv3.CANARY_HYST = p["canary_hyst"]
        _pre[kk] = bcv3.precompute(_prices(), [p["ms"], p["mid"], p["ml"]],
                                   [42, 63, 126])
        bcv3.CANARY_SMA = BASE["canary_sma"]
        bcv3.CANARY_HYST = BASE["canary_hyst"]
    return _pre[kk]


def run_stock(over: dict, anchor: int, stress: bool) -> pd.Series:
    import bt_stock_mom3 as m3
    import bt_stock_coin_v3 as bcv3
    p = dict(BASE)
    p.update(over)
    ranked, mom_off, mom_def, canary = _precomputed(p)
    m3.TX = bcv3.TX * (R.STRESS if stress else 1.0)
    try:
        return m3.run_multi_3mom(
            _prices(), ranked, mom_off, mom_def, canary,
            pd.Timestamp(R.START), pd.Timestamp(R.END), anchor=anchor,
            drift_thr=p["drift_thr"], cash_buf=p["cash_buf"],
            ms=p["ms"], mid=p["mid"], ml=p["ml"],
            snap_int=p["snap_int"], n_snaps=p["n_snaps"])
    finally:
        m3.TX = bcv3.TX


def report(axis: str, arms: list[tuple[str, dict]], stress: bool,
           coin: dict[tuple, tuple]) -> None:
    names = [a[0] for a in arms]
    rank: dict[str, list[float]] = {n: [] for n in names}
    cal: dict[str, list[float]] = {n: [] for n in names}
    mdd: dict[str, list[float]] = {n: [] for n in names}
    st_cal: dict[str, list[float]] = {n: [] for n in names}
    st_mdd: dict[str, list[float]] = {n: [] for n in names}

    for a in ANCHORS:
        sp, fu = coin[(a, stress)]
        comps = {}
        for name in names:
            eq = _load_st(_key(name, a, stress))
            if eq is None:
                continue
            comps[name] = R.build_alloc(eq, sp, fu)
            ms_ = R.metrics(eq)
            st_cal[name].append(ms_["Cal"])
            st_mdd[name].append(ms_["MDD"])
            mc = R.metrics(comps[name])
            cal[name].append(mc["Cal"])
            mdd[name].append(mc["MDD"])
        if len(comps) < 2:
            continue
        sums, _, n = R.window_rs(comps)
        for name in comps:
            rank[name].append(sums[name] / max(n, 1))

    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== {axis} · {tag} · 앵커 {len(ANCHORS)}곳 ===")
    print(f"{'팔':<16}{'순위 평균':>10}{'중앙':>7}{'최악':>7}{'현행대비':>9}"
          f"{'합성Cal':>9}{'합성MDD':>9}{'주식Cal':>9}{'주식MDD':>9}")
    base = rank.get(CUR, [])
    base_mean = float(np.mean(base)) if base else float("nan")
    rows = [(float(np.mean(rank[n])), n, rank[n]) for n in names if rank[n]]
    for mean_r, name, r in sorted(rows):
        mark = "  <- 현행" if name == CUR else ""
        print(f"{name:<16}{mean_r:>10.3f}{float(np.median(r)):>7.3f}{max(r):>7.3f}"
              f"{mean_r - base_mean:>+9.3f}{float(np.mean(cal[name])):>9.2f}"
              f"{float(np.mean(mdd[name])):>9.2f}{float(np.mean(st_cal[name])):>9.2f}"
              f"{float(np.mean(st_mdd[name])):>9.2f}{mark}")
    if base:
        print("  앵커별 순위(현행 대비) — 음수면 그 팔이 현행보다 낫다")
        for mean_r, name, r in sorted(rows):
            if name == CUR:
                continue
            d = [x - y for x, y in zip(r, base)]
            print(f"    {name:<14} {' '.join(f'{v:+6.2f}' for v in d)}"
                  f"   이긴 앵커 {sum(1 for v in d if v < 0)}/{len(d)}")


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()

    coin: dict[tuple, tuple] = {}
    for a in ANCHORS:
        for stress in (False, True):
            suf = "__s5" if stress else ""
            sp = pd.read_csv(f"{COIN_EQ}/sp__현행__a{a}{suf}.csv",
                             index_col=0, parse_dates=True)["eq"]
            fu = pd.read_csv(f"{COIN_EQ}/fu__현행__a{a}{suf}.csv",
                             index_col=0, parse_dates=True)["eq"]
            coin[(a, stress)] = (sp, fu)
    print(f"코인 현행 곡선 {len(coin)}쌍 로드")

    todo, seen = [], set()
    for arms in AXES.values():
        for name, over in arms:
            for a in ANCHORS:
                for stress in (False, True):
                    kk = (name, a, stress)
                    if kk in seen:
                        continue
                    seen.add(kk)
                    todo.append((name, over, a, stress))
    print(f"계산할 조합 {len(todo)}건")

    for i, (name, over, a, stress) in enumerate(todo, 1):
        k = _key(name, a, stress)
        if _load_st(k) is not None:
            continue
        eq = run_stock(over, a, stress)
        if eq is None:
            print(f"  {name} a{a} 실패 — 건너뜀")
            continue
        _save_st(k, eq)
        if i % 40 == 0:
            print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}초)")
    print(f"  전부 완료 ({time.time()-t0:.0f}초)")

    for axis, arms in AXES.items():
        report(axis, arms, False, coin)
        report(axis, arms, True, coin)
    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
