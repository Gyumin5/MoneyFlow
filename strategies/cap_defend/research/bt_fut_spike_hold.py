"""전일 급등 직후 배수 상향을 억제하는 규칙 (2026-09-24).

계기: 운영자 물음 — "하루는 폭등했다가 다음날 폭락하면 배수가 한 박자씩 늦게 따라가서
조금 오르고 많이 떨어지는 게 반복되는데, 감수하는 수밖에 없나".

배수는 전일 종가로 정한다. 어제 오른 걸 보고 오늘 올리므로, 급등 다음날 되돌리는 구간에서는
늘 높은 배수로 맞는다. 여기서 재는 것은 그 한 자리를 직접 겨냥한 규칙이다 —
전일 상승폭이 크면 그날은 배수를 올리지 않는다.

이미 재고 기각한 것과 다른 축이다(그래서 돈다):
  · 변동성 목표(2026-08-31 C) — 20일 실현변동성 수준으로 배수를 정한다. 수준이지 하루 반전이 아니다.
  · 추세이탈 1배(2026-08-31 A) — 단기선·장기선 아래일 때. 급등은 오히려 선 위다.
  · 매일 L 재조정(2026-08-21) — 적용 시점 축이지 조건 축이 아니다.
  · K2 현행 — 7일선 이격이라 하루짜리 반전을 보지 않는다.

두 설계. 둘 다 현행 L 위에 얹는 상한이라 올리는 방향은 없다.
  D 상향 보류: 전일 상승 >= thr 이면 그날은 직전 배수를 넘겨 올리지 않는다(내리는 건 허용).
               되돌림이 하루짜리라는 가정에 정확히 대응한다.
  E 한 칸 감속: 전일 상승 >= thr 이면 그날 배수를 한 칸 내린다(하한 2배).
               보류보다 센 형태이고, 급등 자체를 위험 신호로 읽는다.

축 하나만 바꾼다. 선정·헬스·카나리·스냅·드리프트 0.015·체결밴드 5%·비용 전부 고정.
정본 엔진(backtest_futures_v25.py)은 건드리지 않는다 — leverage 인자로만 주입한다.

판정 기준 (결과 보기 전 고정):
  1. 결정층은 합성 60/25/15 윈도우 순위합이고, 앵커 0 하나가 아니라 앵커 7곳에서 각각 내어
     평균·중앙·최악을 같이 적는다. 슬리브만 좋고 합성이 반대면 채택하지 않는다.
  2. 문턱 스윕에 플래토가 있어야 한다. 단조면(끝값이 최고) 꺾이는 지점이 구간 밖이라는 뜻이라
     채택 근거로 쓰지 않는다.
  3. 비용 5배에서 순서가 뒤집히면 현행 유지로 판정한다.
  4. 규칙이 실제로 배수를 바꾼 코인-일 비율을 먼저 본다. 0 이면 그 팔은 현행과 같고,
     동률을 개선으로 읽지 않는다.

중간 결과는 state/20260924-spike/eq/*.csv 에 남고 재시작하면 이어 간다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_spike_hold.py
     ARMS=quick 로 현행 + 각 설계 대표값 하나만 돌린다(smoke).
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

OUT = "/home/gmoh/mon/251229/state/20260924-spike"
EQDIR = os.path.join(OUT, "eq")

START = "2020-10-01"
END = "2026-09-18"
WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]

TX_SPOT, TX_FUT = 0.004, 0.0006
STRESS = 5.0

# 위상: 현물은 31의 배수, 선물은 19의 배수를 피한다(스냅 집합이 자기 자신으로 옮겨감).
PH_SPOT = [0, 4, 9, 13, 18, 22, 27]
PH_FUT = [0, 3, 5, 8, 11, 14, 16]
ANCHORS = list(range(int(os.environ.get('ANCHORS_N', len(PH_SPOT)))))

CUR = "현행"
L_MIN, L_MAX = 2.0, 4.0
THRS = [0.05, 0.10, 0.15, 0.20]

SPOT_BASE = dict(sma_bars=42, mom_short_bars=20, mom_long_bars=127,
                 vol_threshold=0.05, n_snapshots=7, snap_interval_bars=217,
                 canary_hyst=0.015, drift_threshold=0.10, universe_size=3, cap=1 / 3)
FUT_BASE = dict(sma_days=42, mom_short_days=18, mom_long_days=127,
                vol_threshold=0.05, n_snapshots=5, snap_interval_bars=95,
                canary_hyst=0.015, drift_threshold=0.015, universe_size=3, cap=1 / 3)

_bars = None
_funding = None
_k2 = None


# ────────────────────────── 레버리지 시그널 ──────────────────────────

def build_spike_lev(base_k2: dict, bars: dict, mode: str, thr: float) -> dict:
    """현행 L 위에 '전일 급등이면 올리지 않는다' 를 얹는다.

    base_k2 는 이미 shift(1) 된 값이다(오늘 배수 = 어제까지의 신호).
    급등 여부도 같은 시점 정합으로 shift(1) 한다 — spike(t)=True 는 "어제 닫힌 봉이 급등".
    """
    out = {}
    for coin, lev in base_k2.items():
        df = bars.get(coin)
        if df is None:
            out[coin] = lev
            continue
        close = pd.Series(df['Close'].values, index=df.index)
        spike = (close.pct_change() >= thr).shift(1).fillna(False)
        idx = lev.index.intersection(spike.index)
        lv = lev.loc[idx].values.astype(float)
        sp = spike.loc[idx].values.astype(bool)

        if mode == 'E':
            adj = np.where(sp, np.maximum(lv - 1.0, L_MIN), lv)
        else:  # D — 직전 적용 배수를 넘겨 올리지 않는다
            adj = np.empty_like(lv)
            prev = lv[0] if len(lv) else L_MIN
            for i in range(len(lv)):
                v = lv[i]
                if sp[i]:
                    v = min(v, prev)
                adj[i] = v
                prev = v
        out[coin] = pd.Series(adj, index=idx)
    return out


def arm_specs() -> list[tuple[str, str | None, float | None]]:
    quick = os.environ.get('ARMS', 'full') == 'quick'
    thrs = THRS[1:2] if quick else THRS
    specs: list[tuple[str, str | None, float | None]] = [(CUR, None, None)]
    for t in thrs:
        specs.append((f"D 보류 {t:.2f}", 'D', t))
    for t in thrs:
        specs.append((f"E 감속 {t:.2f}", 'E', t))
    return specs


def lev_profile(base: dict, arm: dict) -> tuple[float, float, dict]:
    """발화율(급등으로 규칙이 걸린 비율)과 실제로 배수가 내려간 비율."""
    b, a = [], []
    for coin in arm:
        if coin not in base:
            continue
        idx = base[coin].index.intersection(arm[coin].index)
        b.append(base[coin].loc[idx].loc[START:END].values)
        a.append(arm[coin].loc[idx].loc[START:END].values)
    if not b:
        return 0.0, 0.0, {}
    bv, av = np.concatenate(b), np.concatenate(a)
    ok = ~(np.isnan(bv) | np.isnan(av))
    bv, av = bv[ok], av[ok]
    changed = float((av < bv).mean()) if len(bv) else 0.0
    dist = {L: float((av == L).mean()) for L in (1.0, 2.0, 3.0, 4.0)}
    return changed, float(np.mean(bv - av)) if len(bv) else 0.0, dist


# ────────────────────────── 실행 ──────────────────────────

def _key(name: str, anchor: int, stress: bool) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace(".", "p")
    return f"{safe}__a{anchor}{'__s5' if stress else ''}"


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(os.path.join(EQDIR, f"{kind}__{key}.csv"), header=["eq"])


def _load(kind: str, key: str) -> pd.Series | None:
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def run_spot(phase: int, stress: bool) -> pd.Series:
    """현물 슬리브는 이 축과 무관(무레버리지) — 앵커·비용만 따라 움직이는 고정 팔."""
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    m = bt_run(_bars, _funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=TX_SPOT * (STRESS if stress else 1.0),
               start_date=START, end_date=END,
               vol_mode="daily", phase_offset_bars=phase,
               post_flip_delay=5, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", **SPOT_BASE)
    return m.get("_equity")


def run_fut(lev: dict, phase: int, stress: bool) -> pd.Series:
    from backtest_futures_v25 import run as fbt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    m = fbt_run(_bars, _funding, interval="D", leverage=lev,
                tx_cost=TX_FUT * (STRESS if stress else 1.0), maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=5,
                phase_offset_bars=phase, health_mode="mom2vol",
                start_date=START, end_date=END, **FUT_BASE)
    return m.get("_equity")


def run_stock(anchor: int) -> pd.Series:
    """주식 슬리브도 이 축과 무관 — 합성용 고정 팔이고 앵커만 따라 움직인다."""
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
                cals, bad = {}, False
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


def spike_stats(bars: dict, thr: float) -> tuple[dict, dict, int]:
    """급등 다음날이 실제로 더 나쁜가 — 규칙의 전제 자체를 먼저 잰다.

    평균은 꼬리 몇 개에 끌려가므로 중앙값과 하락 꼬리 확률을 같이 낸다.
    """
    after, allv = [], []
    for _coin, df in bars.items():
        c = pd.Series(df['Close'].values, index=df.index).loc[START:END]
        r = c.pct_change().values
        sp = np.zeros(len(r), dtype=bool)
        sp[1:] = r[:-1] >= thr  # 전일 급등
        ok = ~np.isnan(r)
        allv.append(r[ok])
        after.append(r[ok & sp])
    a = np.concatenate(after) if after else np.array([])
    b = np.concatenate(allv) if allv else np.array([])

    def d(x):
        if not len(x):
            return dict(mean=float('nan'), med=float('nan'), p5=float('nan'))
        return dict(mean=float(x.mean()), med=float(np.median(x)),
                    p5=float((x <= -0.05).mean()))
    return d(a), d(b), int(len(a))


def report(arms, stress: bool, stock_eq: dict[int, pd.Series]) -> None:
    names = [n for n, _, _ in arms]
    per_rank: dict[str, list[float]] = {n: [] for n in names}
    per_cal: dict[str, list[float]] = {n: [] for n in names}
    per_mdd: dict[str, list[float]] = {n: [] for n in names}
    for a in ANCHORS:
        allocs = {}
        for name, _m, _t in arms:
            k = _key(name, a, stress)
            eq_sp = _load("sp", _key(CUR, a, stress))
            eq_fu = _load("fu", k)
            if eq_sp is None or eq_fu is None:
                continue
            alloc = build_alloc(stock_eq[a], eq_sp, eq_fu)
            allocs[name] = alloc
            mm = metrics(alloc)
            per_cal[name].append(mm["Cal"])
            per_mdd[name].append(mm["MDD"])
        if len(allocs) == len(names):
            sums, _wins, n = window_rs(allocs)
            for name in names:
                per_rank[name].append(sums[name] / max(n, 1))

    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== 합성 60/25/15 · {tag} · 앵커 {len(ANCHORS)}곳 ===")
    print(f"  {'팔':<14s}{'평균순위':>9s}{'중앙':>8s}{'최악':>8s}{'현행대비':>10s}"
          f"{'칼마평균':>10s}{'낙폭평균':>10s}{'승앵커':>7s}")
    base = per_rank.get(CUR, [])
    for name in sorted(names, key=lambda x: float(np.mean(per_rank[x])) if per_rank[x] else 9):
        r = per_rank[name]
        if not r:
            print(f"  {name:<14s}  (자료 없음)")
            continue
        wins = sum(1 for i in range(len(r)) if base and r[i] < base[i])
        gap = float(np.mean(r)) - (float(np.mean(base)) if base else 0.0)
        print(f"  {name:<14s}{float(np.mean(r)):>9.3f}{float(np.median(r)):>8.3f}"
              f"{float(np.max(r)):>8.3f}{gap:>+10.3f}"
              f"{float(np.mean(per_cal[name])):>10.2f}"
              f"{float(np.mean(per_mdd[name])):>10.2f}{wins:>5d}/{len(r)}")


def main() -> None:
    global _bars, _funding, _k2
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    _bars, _funding = load_data("D")
    _k2 = build_K2_signal(_bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                          btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                          l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(_bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    specs = arm_specs()
    print(f"\n기간 {START} ~ {END} | 축: 전일 급등 시 배수 상향 억제 하나")
    print(f"팔 {len(specs)}개 · 앵커 {len(ANCHORS)}곳 · 비용 2조건 "
          f"= 선물 {len(specs)*len(ANCHORS)*2}런\n", flush=True)

    print("[전제 점검 — 급등 다음날 수익률이 실제로 나쁜가 · 전 유니버스 코인-일]")
    print(f"  {'문턱':>6s}{'익일평균':>10s}{'익일중앙':>10s}{'익일-5%확률':>12s}"
          f"{'전체평균':>10s}{'전체중앙':>10s}{'전체-5%확률':>12s}{'표본':>8s}")
    for thr in THRS:
        aa, bb, cnt = spike_stats(_bars, thr)
        print(f"  {thr:>6.2f}{aa['mean']:>+9.3%}{aa['med']:>+10.3%}{aa['p5']:>12.1%}"
              f"{bb['mean']:>+9.3%}{bb['med']:>+10.3%}{bb['p5']:>12.1%}{cnt:>8d}", flush=True)

    levs = {}
    print("\n[규칙이 실제로 배수를 바꾼 비율 — 코인-일]")
    print(f"  {'팔':<14s}{'내려간 비율':>12s}{'평균 감소':>10s}"
          f"{'2배':>8s}{'3배':>8s}{'4배':>8s}")
    for name, mode, thr in specs:
        lev = _k2 if mode is None else build_spike_lev(_k2, _bars, mode, thr)
        levs[name] = lev
        ch, avg, dist = lev_profile(_k2, lev)
        print(f"  {name:<14s}{ch:>11.2%}{avg:>10.3f}"
              f"{dist.get(2.0,0):>8.1%}{dist.get(3.0,0):>8.1%}{dist.get(4.0,0):>8.1%}",
              flush=True)

    print("\n주식 슬리브(합성용 고정 팔) 앵커별 계산 중...", flush=True)
    stock_eq = {}
    for a in ANCHORS:
        k = _key("주식", a, False)
        eq = _load("st", k)
        if eq is None:
            eq = run_stock(a)
            _save("st", k, eq)
        stock_eq[a] = eq
    print(f"  완료 ({time.time()-t0:.0f}초)", flush=True)

    total = len(specs) * len(ANCHORS) * 2
    done = 0
    for stress in (False, True):
        for a in ANCHORS:
            kc = _key(CUR, a, stress)
            if _load("sp", kc) is None:
                _save("sp", kc, run_spot(PH_SPOT[a], stress))
            for name, _m, _t in specs:
                k = _key(name, a, stress)
                done += 1
                if _load("fu", k) is not None:
                    continue
                t1 = time.time()
                _save("fu", k, run_fut(levs[name], PH_FUT[a], stress))
                print(f"  [{done}/{total}] {name} a{a}"
                      f"{' s5' if stress else ''} {time.time()-t1:.0f}초", flush=True)

    print("\n[슬리브 단독 — 앵커 0, 기본 비용]")
    print(f"  {'팔':<14s}{'칼마':>7s}{'연수익':>9s}{'낙폭':>9s}{'최종':>9s}")
    for name, _m, _t in specs:
        eq = _load("fu", _key(name, 0, False))
        if eq is None:
            continue
        mm = metrics(eq)
        print(f"  {name:<14s}{mm['Cal']:>7.2f}{mm['CAGR']:>+8.1f}%"
              f"{mm['MDD']:>+8.1f}%{eq.iloc[-1]/eq.iloc[0]:>8.1f}x")

    for stress in (False, True):
        report(specs, stress, stock_eq)

    print(f"\n소요 {time.time()-t0:.0f}초")


if __name__ == '__main__':
    main()
