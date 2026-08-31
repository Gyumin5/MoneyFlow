"""선물 동적 레버리지에 1배를 넣는 세 가지 설계 비교 (2026-08-31).

계기: 사용자가 "선물 동적 배수를 1배까지 포함하는 전략을 구상해서 테스트해봐" 라고 지시.

이미 한 번 재고 기각한 것이 있다 — 2026-06-02, K2 의 가장 낮은 칸(L2)을 그대로 1배로 내리는
안(bt_k2_l1_downside.py, thr_low = 1.000 / 0.975 / 0.950). 결과는 수익만 줄고 낙폭은 그대로여서
합성 Calmar 2.98 → 2.63. 그래서 그 형태는 다시 돌리지 않는다. 여기서 재는 셋은 전부 그때
안 본 축이다.

  A 추세이탈 1배 : 단기선(SMA7) 아래 AND 장기선(SMA42) 아래일 때만 1배.
                   옛 안은 단기선 하나만 봤다 — 눌림목과 추세이탈을 구분하지 못했다.
                   장기선 이탈 깊이(thr)가 스윕 축이다.
  B 재진입 램프  : 카나리가 꺼졌다 켜지는 날 1배로 진입하고 N봉 뒤 정상 배수로 올린다.
                   바닥을 잘못 잡았을 때 첫 구간의 손실만 줄이는 설계다. N 이 스윕 축.
  C 변동성 목표  : 목표 일변동성 t 를 정하고 L = clip(floor(t / 실현변동성20일), 1, 4).
                   1배는 규칙의 결과지 별도 칸이 아니다. t 가 스윕 축.

세 안 모두 기존 L 위에 상한을 하나 더 얹는 형태다(L = min(BTC_cap, K2, 새 상한)).
따라서 배수를 올리는 방향은 없고 내리는 방향만 있다 — 상방 L5/L6 는 2026-06-02 에 기각됐다.

축 하나만 바꾼다. 선정·헬스·카나리·스냅·드리프트 0.03·체결밴드 5%·비용 전부 고정.
정본 엔진(backtest_futures_v25.py)은 건드리지 않는다 — leverage 인자로만 주입한다.

판정 기준 (결과 보기 전 고정):
  1. 결정층은 합성 60/25/15 다. 슬리브만 좋고 합성이 반대면 채택하지 않는다.
  2. 파라미터 스윕에 플래토가 있어야 한다. 단조 개선(끝값이 최고)이면 꺾이는 지점이
     구간 밖이라는 뜻이라 채택 근거로 쓰지 않는다.
  3. 비용 5배에서 순서가 뒤집히면 현행 유지로 판정한다.
  4. 1배가 실제로 발화한 비율을 먼저 본다. 발화 0 이면 그 팔은 무효(현행과 동일)이고,
     동률을 개선으로 읽지 않는다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_lev_one.py
     ARMS=quick 로 baseline + 각 안 대표값 하나씩만 돌릴 수 있다(smoke).
출력은 state/<job>/ 로 리다이렉트한다. 원시 로그는 리포트가 아니다.
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

import bt_fut_rebal_band as base                                       # noqa: E402
from bt_fut_rebal_band import window_rs, metrics, build_alloc          # noqa: E402

START = '2020-10-01'
END = '2026-08-25'
BASE_TX = 0.0006
STRESS_TX = 0.003            # 5x

SMA_FAST = 7                 # K2 단기선 (현행)
SMA_SLOW = 42                # 카나리·추세 기준선 (현행)
K2_HYST = 0.025
BTC_CAP_MID, BTC_CAP_MAX, BTC_CAP_SMA = 1.015, 1.05, 42
L_MIN, L_MID, L_MAX = 2.0, 3.0, 4.0
CANARY_HYST = 0.015

A_THRS = [1.00, 0.98, 0.95]  # 장기선 이탈 깊이
B_BARS = [5, 10, 20]         # 램프 길이(봉)
C_TARGETS = [0.08, 0.10, 0.12]   # 목표 일변동성
C_VOL_WIN = 20


# ────────────────────────── 시그널 구성요소 ──────────────────────────

def _btc_cap(bars):
    c = pd.Series(bars['BTC']['Close'].values, index=bars['BTC'].index)
    ratio = c / c.rolling(BTC_CAP_SMA).mean()
    return pd.Series(np.where(ratio > BTC_CAP_MAX, L_MAX,
                     np.where(ratio > BTC_CAP_MID, L_MID, L_MIN)),
                     index=ratio.index).shift(1).ffill().fillna(L_MIN)


def _k2_base(close):
    ratio = close / close.rolling(SMA_FAST).mean()
    thr_max, thr_mid = 1.0 + K2_HYST * 3, 1.0 + K2_HYST
    return pd.Series(np.where(ratio > thr_max, L_MAX,
                     np.where(ratio > thr_mid, L_MID, L_MIN)), index=close.index)


def _canary_state(bars):
    """엔진과 같은 정의: BTC 종가 vs SMA42, ±1.5% 히스테리시스."""
    c = pd.Series(bars['BTC']['Close'].values, index=bars['BTC'].index)
    sma = c.rolling(SMA_SLOW).mean()
    on, out = False, []
    for v, s in zip(c.values, sma.values):
        if np.isnan(s):
            on = False
        elif on:
            on = not (v < s * (1 - CANARY_HYST))
        else:
            on = v > s * (1 + CANARY_HYST)
        out.append(on)
    return pd.Series(out, index=c.index)


def build_lev(bars, arm=None, param=None):
    """arm=None 이면 현행(baseline). 아니면 현행 위에 상한을 하나 더 얹는다."""
    cap = _btc_cap(bars)

    extra = None
    if arm == 'B':
        canary = _canary_state(bars)
        flip_on = canary & ~canary.shift(1, fill_value=False)
        last = None
        vals = []
        for i, (d, f) in enumerate(zip(canary.index, flip_on.values)):
            if f:
                last = i
            vals.append(np.nan if last is None else i - last)
        bars_since = pd.Series(vals, index=canary.index)
        ramp = pd.Series(np.where(bars_since < param, 1.0, L_MAX),
                         index=canary.index).shift(1).ffill().fillna(L_MAX)
        extra = {'__all__': ramp}

    out = {}
    for coin, df in bars.items():
        close = pd.Series(df['Close'].values, index=df.index)
        lev = _k2_base(close)

        if arm == 'A':
            slow_break = close < close.rolling(SMA_SLOW).mean() * param
            fast_break = close <= close.rolling(SMA_FAST).mean()
            lev = pd.Series(np.where(slow_break & fast_break, 1.0, lev.values),
                            index=close.index)
        elif arm == 'C':
            vol = close.pct_change().rolling(C_VOL_WIN).std()
            vcap = np.floor(param / vol.replace(0, np.nan))
            vcap = vcap.clip(lower=1.0, upper=L_MAX).fillna(L_MAX)
            lev = pd.Series(np.minimum(lev.values, vcap.values), index=close.index)

        lev = lev.shift(1).ffill().fillna(L_MIN)
        idx = lev.index.intersection(cap.index)
        merged = np.minimum(lev.loc[idx].values, cap.loc[idx].values)
        if extra is not None:
            r = extra['__all__'].reindex(idx).ffill().fillna(L_MAX)
            merged = np.minimum(merged, r.values)
        out[coin] = pd.Series(merged, index=idx)
    return out


def lev_profile(k2):
    """코인-일 기준 L 분포. 1배가 실제로 몇 % 나오는지 본다."""
    allv = np.concatenate([s.loc[START:END].values for s in k2.values()])
    allv = allv[~np.isnan(allv)]
    return {L: float((allv == L).mean()) for L in (1.0, 2.0, 3.0, 4.0)}, len(allv)


# ────────────────────────── 실행 ──────────────────────────

def run_fut_lev(k2, tx_cost=BASE_TX):
    """라이브 V25 선물 sleeve 설정 그대로. 레버리지 시그널만 바꾼다."""
    from backtest_futures_v25 import run as fbt_run
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    return fbt_run(BARS, FUNDING, interval='D', leverage=k2, universe_size=3, cap=1 / 3,
                   tx_cost=tx_cost, maint_rate=0.004,
                   sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                   canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
                   health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                   n_snapshots=5, snap_interval_bars=95,
                   start_date=START, end_date=END)


def arm_specs():
    quick = os.environ.get('ARMS', 'full') == 'quick'
    specs = [('현행 L2~4', None, None)]
    for t in (A_THRS[:1] if quick else A_THRS):
        specs.append((f'A 추세이탈 {t:.2f}', 'A', t))
    for n in (B_BARS[:1] if quick else B_BARS):
        specs.append((f'B 램프 {n}봉', 'B', n))
    for t in (C_TARGETS[:1] if quick else C_TARGETS):
        specs.append((f'C 변동성 {t:.2f}', 'C', t))
    return specs


def main():
    t0 = time.time()
    global BARS, FUNDING
    from unified_backtest import load_data
    BARS, FUNDING = load_data('D')

    specs = arm_specs()
    print(f"기간 {START} ~ {END} | 축: 동적 L 하한(1배 도입 방식) 하나")
    print(f"팔 {len(specs)}개 | tx {BASE_TX} (stress {STRESS_TX})\n")

    print("[레버리지 분포 — 코인-일 기준]")
    print(f"  {'arm':<16s} {'1배':>7s} {'2배':>7s} {'3배':>7s} {'4배':>7s}")
    levs = {}
    for tag, arm, param in specs:
        k2 = build_lev(BARS, arm, param)
        levs[tag] = k2
        prof, n = lev_profile(k2)
        print(f"  {tag:<16s} {prof[1.0]:>6.1%} {prof[2.0]:>6.1%} {prof[3.0]:>6.1%} "
              f"{prof[4.0]:>6.1%}   (코인-일 {n})")

    print(f"\n[sleeve 단독 — tx {BASE_TX*100:.2f}%]")
    print(f"  {'arm':<16s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} "
          f"{'체결':>6s} {'Rebal':>6s} {'청산':>4s} {'최종':>9s} {'초':>5s}")
    res = {}
    for tag, _arm, _p in specs:
        t1 = time.time()
        m = run_fut_lev(levs[tag])
        res[tag] = m
        eq = m['_equity']
        print(f"  {tag:<16s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>6d} {m['Rebal']:>6d} {m['Liq']:>4d} "
              f"{eq.iloc[-1]/eq.iloc[0]:>8.1f}x {time.time()-t1:>5.0f}", flush=True)

    eqs = {k: v['_equity'] for k, v in res.items()}
    print("\n[연도별 수익률 / MDD]")
    keys = list(eqs)
    print("  year   " + "".join(f"{k:>22s}" for k in keys))
    for year in sorted({d.year for d in list(eqs.values())[0].index}):
        cells = []
        for k in keys:
            seg = eqs[k][eqs[k].index.year == year].dropna()
            if len(seg) < 30:
                cells.append(f"{'-':>22s}")
                continue
            r = seg.iloc[-1] / seg.iloc[0] - 1
            mdd = float((seg / seg.cummax() - 1).min())
            cells.append(f"{r:>+11.1%}/{mdd:>+9.1%}")
        print(f"  {year:<6d} " + "".join(cells))

    print("\n[윈도우 rank-sum — sleeve, 기본비용]")
    sums, wins, n = window_rs(eqs)
    for k in sorted(sums, key=lambda k: sums[k]):
        print(f"  {k:<16s} avg_rank {sums[k]/n:.3f}  wins {wins[k]:>4d}/{n} "
              f"({wins[k]/n*100:.0f}%)")
    print(f"  windows n={n}")

    print(f"\n[비용 stress — tx {STRESS_TX*100:.2f}% (5x)]")
    seqs = {}
    print(f"  {'arm':<16s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'체결':>6s}")
    for tag, _arm, _p in specs:
        m = run_fut_lev(levs[tag], tx_cost=STRESS_TX)
        seqs[tag] = m['_equity']
        print(f"  {tag:<16s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Trades']:>6d}", flush=True)
    ssums, swins, sn = window_rs(seqs)
    print("  [rank-sum — stress]")
    for k in sorted(ssums, key=lambda k: ssums[k]):
        print(f"    {k:<16s} avg_rank {ssums[k]/sn:.3f}  wins {swins[k]:>4d}/{sn}")

    print("\n[자산배분 60/25/15 합성 — 선물만 교체] (결정층)")
    base.START, base.END = START, END
    eq_sp = base.run_spot_live()
    eq_st = base.run_stock_live()
    rows = {}
    for k, eq_fu in eqs.items():
        a = build_alloc(eq_st, eq_sp, eq_fu)
        rows[k] = (a, metrics(a))
    for k, (_a, mm) in rows.items():
        print(f"  {k:<16s} Cal {mm['Cal']:.2f} | CAGR {mm['CAGR']:+.1f}% | "
              f"MDD {mm['MDD']:+.1f}% | Sharpe {mm['Sharpe']:.2f}")
    asums, awins, an = window_rs({k: v[0] for k, v in rows.items()})
    print("  [rank-sum — 합성]")
    for k in sorted(asums, key=lambda k: asums[k]):
        print(f"    {k:<16s} avg_rank {asums[k]/an:.3f}  wins {awins[k]}/{an}")

    print("\n[판정 보조]")
    b = specs[0][0]
    for label, group in (('A', A_THRS), ('B', B_BARS), ('C', C_TARGETS)):
        tags = [t for t, a, _ in specs if a == label]
        if len(tags) < 2:
            continue
        seq = [asums[t] / an for t in tags]
        mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)) or \
               all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
        bestt = tags[int(np.argmin(seq))]
        print(f"  {label}: 합성 최상위 {bestt} (avg_rank {min(seq):.3f}, 현행 "
              f"{asums[b]/an:.3f}) | 단조 {'예 → 플래토 미확인' if mono else '아니오'}")
    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
