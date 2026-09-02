"""선물 동적 배수에 펀딩비를 넣는 세 설계 비교 (2026-09-02).

계기: "선물 동적 배수에 변동성도 넣으면 좋아질까, 아니면 상승·하락을 판별하는 팩터가 있으면
그런 것도" 라는 지시. 변동성과 추세이탈은 2026-08-31 에 이미 재고 기각했다
(bt_fut_lev_one.py — C 변동성 목표, A 추세이탈). 그래서 여기서는 현행 세 슬리브가
쓰지 않는 유일한 독립 입력인 펀딩비만 잰다.

펀딩비를 고른 이유는 입력의 성격이 다르기 때문이다. 카나리·헬스·K2 는 전부 가격에서 나온
파생값이라 서로 겹친다. 펀딩비는 포지션 쏠림을 직접 재는 값이고, 선물 BT 에는 이미 코인별
실제 펀딩 데이터가 비용으로 들어가 있다(보유구간 합산 자본 차감). 신호로 쓰는 것만 새롭다.

사전 실사(fund_profile.py, state/fut-funding/profile.log):
  - 38종 전부 펀딩 파일 있음. 결측 코인 없음.
  - 일 합계 펀딩비 중앙값 +0.000242(연 8.8%), q90 +0.000615, q99 +0.004185. 음수 23.1%.
  - 7일 롤링 평균 분위수: q10 -0.000252 / q25 +0.000017 / q50 +0.000180 /
    q90 +0.000714 / q95 +0.001333 / q99 +0.003922.
  - 방향: 펀딩 상위 10% 구간의 다음날 수익 +0.83%, 하위 10% +0.26%, 음수구간 +0.17%,
    전체 +0.15%. 즉 무조건부로는 펀딩이 높을수록 다음날이 좋다.
    ※ 이건 인과가 아니다. 펀딩은 가격이 오른 뒤에 오르므로 모멘텀을 다시 쓴 것에 가깝다.
      그래서 이 관측은 문턱을 데이터에서 뽑는 데만 쓰고 채택 근거로 쓰지 않는다.

문턱은 위 분위수에서 그대로 가져왔다 — 임의로 고르지 않았다.

  F 과열 상한  : 7일 펀딩 평균이 문턱보다 높으면(롱 쏠림) 배수를 2배로 내린다.
                 가설은 쏠림이 곧 되돌림 위험이라는 것. 문턱 q90 / q95 / q99.
  G 저조 상한  : 7일 펀딩 평균이 문턱보다 낮으면(숏 쏠림·무관심) 배수를 2배로 내린다.
                 가설은 F 와 정반대 — 쏠림이 곧 추세 확인이라는 것. 문턱 q10 / q25 / q50.
  H 시장 상한  : 유니버스 전체 펀딩 평균이 문턱보다 높으면 전 코인 배수를 2배로 내린다.
                 코인별이 아니라 시장 레짐으로 보는 층위 차이를 잰다. 문턱 0.0003/0.0005/0.0008.

셋 다 기존 배수 위에 상한을 하나 더 얹는 형태다(L = min(BTC_cap, K2, 새 상한)).
배수를 올리는 방향은 넣지 않았다 — 상방 L5/L6 는 2026-06-02 에 기각됐고 재론 금지다.
1배 칸도 넣지 않았다 — 2026-08-31 에 세 설계 전부 기각됐다. 하한은 현행 2배 그대로다.

축 하나만 바꾼다. 선정·헬스·카나리·스냅·드리프트 0.03·체결밴드 5%·비용 전부 고정.
정본 엔진(backtest_futures_v25.py)은 건드리지 않는다 — leverage 인자로만 주입한다.

판정 기준 (결과 보기 전 고정):
  1. 결정층은 합성 60/25/15 다. 슬리브만 좋고 합성이 반대면 채택하지 않는다.
  2. 파라미터 스윕에 플래토가 있어야 한다. 끝값이 최고면 꺾이는 지점이 구간 밖이라는
     뜻이라 채택 근거로 쓰지 않는다.
  3. 거래비용 5배에서 순서가 뒤집히면 현행 유지로 판정한다.
  4. 상한이 실제로 문 비율을 먼저 본다. 0 이면 그 팔은 무효(현행과 동일)이고 동률을
     개선으로 읽지 않는다.
  5. F 와 G 는 같은 입력에 부호가 반대인 가설이다. 둘 다 현행보다 좋게 나오면 그건
     펀딩비가 정보를 준 게 아니라 배수를 그냥 낮춘 효과다(위험을 덜 진 만큼 낙폭이 준 것).
     그 경우 둘 다 기각한다. 개선폭이 상한이 문 비율에 비례해 커지는지도 같이 본다 —
     비례하면 정보가 아니라 위험축소다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_funding.py
     ARMS=quick 로 baseline + 각 안 대표값 하나씩만 (smoke).
출력은 state/fut-funding/ 로 리다이렉트한다. 원시 로그는 리포트가 아니다.
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
K2_HYST = 0.025
BTC_CAP_MID, BTC_CAP_MAX, BTC_CAP_SMA = 1.015, 1.05, 42
L_MIN, L_MID, L_MAX = 2.0, 3.0, 4.0

FUND_WIN = 7                 # 펀딩 롤링 창(일)
CAP_TO = L_MIN               # 상한이 물면 내려가는 값 (1배 칸은 08-31 기각)

F_THRS = [0.000714, 0.001333, 0.003922]     # q90 / q95 / q99
G_THRS = [-0.000252, 0.000017, 0.000180]    # q10 / q25 / q50
H_THRS = [0.0003, 0.0005, 0.0008]           # 시장 평균, 절대값


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


def _daily_funding(fs):
    """8시간 정산 3회를 하루 합으로. 인덱스 자정 정규화."""
    s = pd.Series(fs.values, index=pd.DatetimeIndex(fs.index).normalize())
    return s.groupby(level=0).sum().sort_index()


def _fund_series(coin, idx, funding):
    """코인 하나의 펀딩 신호. 롤링 평균 뒤 shift(1) — 전일까지의 정보만."""
    f = funding.get(coin)
    if f is None or len(f) == 0:
        return pd.Series(np.nan, index=idx)
    d = _daily_funding(f).rolling(FUND_WIN).mean().shift(1)
    key = pd.DatetimeIndex(idx).normalize()
    return pd.Series(d.reindex(key, method='ffill').values, index=idx)


def _market_fund(bars, funding):
    """유니버스 전체 펀딩 평균(코인 균등). 시장 쏠림 게이지."""
    cols = {}
    for c, f in funding.items():
        if c in bars and len(f):
            cols[c] = _daily_funding(f)
    if not cols:
        return None
    m = pd.DataFrame(cols).mean(axis=1)
    return m.rolling(FUND_WIN).mean().shift(1)


def build_lev(bars, funding, arm=None, param=None):
    """arm=None 이면 현행(baseline). 아니면 현행 위에 상한을 하나 더 얹는다."""
    cap = _btc_cap(bars)

    mkt = None
    if arm == 'H':
        mkt = _market_fund(bars, funding)

    out = {}
    for coin, df in bars.items():
        close = pd.Series(df['Close'].values, index=df.index)
        lev = _k2_base(close)

        if arm in ('F', 'G'):
            fs = _fund_series(coin, close.index, funding)
            hit = (fs > param) if arm == 'F' else (fs < param)
            hit = hit.fillna(False)
            lev = pd.Series(np.where(hit.values, np.minimum(lev.values, CAP_TO),
                                     lev.values), index=close.index)

        lev = lev.shift(1).ffill().fillna(L_MIN)
        idx = lev.index.intersection(cap.index)
        merged = np.minimum(lev.loc[idx].values, cap.loc[idx].values)

        if arm == 'H' and mkt is not None:
            key = pd.DatetimeIndex(idx).normalize()
            mv = pd.Series(mkt.reindex(key, method='ffill').values, index=idx)
            hit = (mv > param).fillna(False)
            merged = np.where(hit.values, np.minimum(merged, CAP_TO), merged)

        out[coin] = pd.Series(merged, index=idx)
    return out


def lev_profile(k2):
    allv = np.concatenate([s.loc[START:END].values for s in k2.values()])
    allv = allv[~np.isnan(allv)]
    return {L: float((allv == L).mean()) for L in (2.0, 3.0, 4.0)}, len(allv)


def bind_rate(k2, k2_base_):
    """상한이 실제로 배수를 내린 코인-일 비율. 발화 0 이면 그 팔은 무효."""
    lo = tot = 0
    for c, s in k2.items():
        a = s.loc[START:END]
        b = k2_base_[c].reindex(a.index)
        ok = a.notna() & b.notna()
        lo += int((a[ok] < b[ok]).sum())
        tot += int(ok.sum())
    return lo / tot if tot else 0.0


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
    for i, t in enumerate(F_THRS[:1] if quick else F_THRS):
        specs.append((f'F 과열 q{[90, 95, 99][i]}', 'F', t))
    for i, t in enumerate(G_THRS[:1] if quick else G_THRS):
        specs.append((f'G 저조 q{[10, 25, 50][i]}', 'G', t))
    for t in (H_THRS[:1] if quick else H_THRS):
        specs.append((f'H 시장 {t:.4f}', 'H', t))
    return specs


def main():
    t0 = time.time()
    global BARS, FUNDING
    from unified_backtest import load_data
    BARS, FUNDING = load_data('D')

    specs = arm_specs()
    print(f"기간 {START} ~ {END} | 축: 펀딩비를 배수 상한으로 쓰는 방식 하나")
    print(f"팔 {len(specs)}개 | 롤링 {FUND_WIN}일 | 상한값 {CAP_TO:.0f}배 | "
          f"tx {BASE_TX} (stress {STRESS_TX})\n")

    levs = {}
    for tag, arm, param in specs:
        levs[tag] = build_lev(BARS, FUNDING, arm, param)
    base_lev = levs[specs[0][0]]

    print("[레버리지 분포 + 상한 발화율 — 코인-일 기준]")
    print(f"  {'arm':<16s} {'2배':>7s} {'3배':>7s} {'4배':>7s} {'상한발화':>8s}")
    for tag, _a, _p in specs:
        prof, n = lev_profile(levs[tag])
        br = bind_rate(levs[tag], base_lev)
        print(f"  {tag:<16s} {prof[2.0]:>6.1%} {prof[3.0]:>6.1%} {prof[4.0]:>6.1%} "
              f"{br:>7.1%}   (코인-일 {n})")

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
    for label, group in (('F', F_THRS), ('G', G_THRS), ('H', H_THRS)):
        tags = [t for t, a, _ in specs if a == label]
        if len(tags) < 2:
            continue
        seq = [asums[t] / an for t in tags]
        mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)) or \
               all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
        bestt = tags[int(np.argmin(seq))]
        print(f"  {label}: 합성 최상위 {bestt} (avg_rank {min(seq):.3f}, 현행 "
              f"{asums[b]/an:.3f}) | 단조 {'예 → 플래토 미확인' if mono else '아니오'}")

    print("\n  [부호 대칭 검사 — 기준 5]")
    fbest = min(asums[t] / an for t, a, _ in specs if a == 'F')
    gbest = min(asums[t] / an for t, a, _ in specs if a == 'G')
    bl = asums[b] / an
    both = fbest < bl and gbest < bl
    print(f"    F 최상 {fbest:.3f} / G 최상 {gbest:.3f} / 현행 {bl:.3f}")
    print(f"    둘 다 현행보다 우세: {'예 → 정보가 아니라 위험축소, 둘 다 기각' if both else '아니오'}")

    print("\n  [발화율 대비 개선폭 — 비례하면 정보가 아니라 위험축소]")
    for tag, arm, _p in specs:
        if arm is None:
            continue
        br = bind_rate(levs[tag], base_lev)
        print(f"    {tag:<16s} 발화 {br:>6.1%}  합성 avg_rank {asums[tag]/an:>6.3f}  "
              f"Cal {rows[tag][1]['Cal']:>5.2f}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
