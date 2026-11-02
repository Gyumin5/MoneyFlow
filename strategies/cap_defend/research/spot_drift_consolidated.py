"""드리프트 문턱 — 흩어진 측정을 한 실행으로 묶고 판정선을 미리 못 박는다.

토론(run-20261102T033100Z, 1라운드 합의)이 지정한 설계를 그대로 옮겼다. 앞선 측정은
여러 실행에 흩어져 있고 가드 설정이 라이브와 어긋난 채였다 — 둘을 한 번에 닫는다.

## 왜 가드 축이 필요한가

채택 BT 호출부(trade/v24_shadow_today.py:45)는 dd_lookback·bl_drop·crash_threshold 를
안 넘긴다. 그러면 엔진 기본값이 걸린다 — dd_lookback=60, dd_threshold=-0.25,
bl_drop=-0.15, crash_threshold=-0.10 (unified_backtest.py:247~260). 즉 가드가 켜진다.
그런데 라이브 현물 엔진(trade/coin_live_engine.py)에는 그런 가드가 아예 없다.
CLAUDE.md 의 "코인 현물 V24 — 전략 가드 없음" 이 정본이고, BT↔라이브 선정 패리티
하니스도 가드를 명시적으로 껐다(dd_lookback=0, bl_drop=0, crash_threshold=-10).

앞선 드리프트 측정은 전부 가드 켜진 쪽이다. 드리프트와 가드는 둘 다 "앵커 사이에
포지션을 건드리는 장치" 라서 서로 가려질 수 있다 — 가드가 이미 뺀 자리를 드리프트가
다시 빼면 드리프트의 몫이 작아 보인다. 그래서 두 군을 나란히 돌린다.

## 격자

구성 여섯 (문턱, 리필모드):
  (0.10, refill)  ← 지금 라이브
  (0.10, off)     ← 리필 효과 분리
  (0.30, refill)
  (0.40, refill)  ← 단일 우선 후보
  (0.50, refill)
  (0.00, refill)  ← 발화 비활성 대조군. 드리프트가 0 이면 리필 경로 자체가 안 돈다.

가드 둘 × 비용 둘(0, 0.004) × 앵커 일곱 = 6 × 2 × 2 × 7 = 168칸.

## 사전 판정선 (측정 전에 적는다 — 비용 0.004, 무가드군, 0.40 대 0.10·refill)

① 두 출처 구간 각각에서 평균 누적 로그수익 차이가 양수이고 앵커 일곱 중 다섯 이상 양수.
② 두 구간 각각에서 길이 동일가중 창 순위합이 1.5 미만이고 창별 승수가 패수보다 많다.
③ 전 구간 평균 누적 로그수익 차이가 0.15 이상이며, 후보 우위가 가장 큰 열 날짜를 빼도
   원래 차이의 절반 이상이 남는다.
④ 어느 앵커·출처 구간에서도 최대낙폭 악화가 2%포인트 이하이고, 최악 504봉 창의 악화가
   5%포인트 이하다.
⑤ 각 앵커의 단일 종목 목표비중 95백분위가 0.55 이하다.

다섯을 다 통과해야 모의운영 후보로 올린다. 하나라도 걸리면 라이브 0.10 을 그대로 둔다.
가드군 재현 검사: 가드 켠 군의 0.10·0.40 칼마가 앞선 측정(1.561 / 2.237, 40bp, 앵커 7
평균)과 소수 셋째 자리에서 맞아야 한다. 안 맞으면 원인을 찾을 때까지 판정을 멈춘다.

리필 효과와 드리프트 효과를 칼마 차이의 단순 차감으로 분해하지 않는다 — 두 경로가
같은 분기를 공유해서 가법이 아니다.

실행: python3 spot_drift_consolidated.py
      python3 spot_drift_consolidated.py --summary
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import unified_backtest as ub  # noqa: E402

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-drift-consolidated')
START = '2017-01-01'
END = '2026-11-01'
PHASES = [0, 4, 8, 12, 16, 20, 24]
SPLIT_DATE = pd.Timestamp('2020-10-01')   # 자료 접합 경계

BASE = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=7, universe_size=3, cap=1 / 3,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=217,
)

# (이름, 문턱, 리필모드)
CONFIGS = [
    ('d10-refill', 0.10, 'refill'),
    ('d10-off',    0.10, 'off'),
    ('d30-refill', 0.30, 'refill'),
    ('d40-refill', 0.40, 'refill'),
    ('d50-refill', 0.50, 'refill'),
    ('d00-refill', 0.00, 'refill'),
]
# 가드군: 엔진 기본값(켬) 대 라이브 정합(끔)
GUARDS = {
    'on':  {},
    'off': dict(dd_lookback=0, bl_drop=0, crash_threshold=-10),
}
COSTS = [0.0, 0.004]

WINDOWS = [(504, 63), (756, 126), (1008, 252)]
KEYS = ('cal', 'sharpe', 'cagr', 'mdd')
EPS = 1e-12

CAND, REF = 'd40-refill', 'd10-refill'


def cell_key(cfg, guard, cost, ph):
    return f"{cfg}|g{guard}|c{int(round(cost*10000))}|p{ph}"


def main():
    os.makedirs(OUT, exist_ok=True)
    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    jpath = os.path.join(OUT, 'cells.jsonl')
    npath = os.path.join(OUT, 'curves.npz')
    done, curves = set(), {}
    if os.path.exists(jpath):
        for line in open(jpath, encoding='utf-8'):
            try:
                done.add(json.loads(line)['key'])
            except Exception:
                pass
    if os.path.exists(npath):
        z = np.load(npath, allow_pickle=True)
        curves = {k: z[k] for k in z.files}
    total = len(CONFIGS) * len(GUARDS) * len(COSTS) * len(PHASES)
    print(f"이미 끝난 칸 {len(done)}개 / 전체 {total}", flush=True)

    t0, n = time.time(), 0
    dates_saved = os.path.exists(os.path.join(OUT, 'dates.npy'))
    with open(jpath, 'a', encoding='utf-8') as f:
        for cfg, drift, mode in CONFIGS:
            for guard, gkw in GUARDS.items():
                for cost in COSTS:
                    for ph in PHASES:
                        key = cell_key(cfg, guard, cost, ph)
                        if key in done:
                            continue
                        os.environ['DRIFT_HEALTH_MODE'] = mode
                        kw = dict(BASE, drift_threshold=drift, tx_cost=cost,
                                  phase_offset_bars=ph, **gkw)
                        tr = []
                        m = ub.run(bars, fund, start_date=START, end_date=END,
                                   _trace=tr, **kw)
                        eq = m.pop('_equity')
                        if not dates_saved:
                            np.save(os.path.join(OUT, 'dates.npy'),
                                    eq.index.values)
                            dates_saved = True
                        curves[key] = np.asarray(eq.values, dtype=float)

                        # 목표비중 집중도 — 현금 키를 뺀 단일 종목 최대값
                        tops = []
                        for t in tr:
                            w = [v for k, v in t['target'].items()
                                 if k not in ('CASH', 'Cash')]
                            tops.append(max(w) if w else 0.0)
                        tops = np.asarray(tops, dtype=float)

                        row = dict(key=key, cfg=cfg, drift=drift, mode=mode,
                                   guard=guard, cost=cost, phase=ph,
                                   **{k: m.get(k) for k in
                                      ('CAGR', 'MDD', 'Sharpe', 'Cal',
                                       'Trades', 'Rebal')},
                                   wmax=float(tops.max()),
                                   wp95=float(np.percentile(tops, 95)),
                                   wover50=float((tops > 0.5).mean()))
                        f.write(json.dumps(row, ensure_ascii=False,
                                           default=str) + '\n')
                        f.flush()
                        n += 1
                        if n % 14 == 0:
                            np.savez_compressed(npath, **curves)
                            print(f"  {n}칸 ({(time.time()-t0)/60:.1f}분) "
                                  f"최근 {key}", flush=True)
    np.savez_compressed(npath, **curves)
    print(f"전체 끝 ({(time.time()-t0)/60:.1f}분)", flush=True)
    summary()
    return 0


# ─────────────────────────── 판정 ───────────────────────────

def win_metrics(seg):
    if len(seg) < 2 or seg[0] <= 0:
        return None
    r = np.diff(seg) / seg[:-1]
    if r.std() == 0:
        return None
    yrs = len(seg) / 365.25
    cagr = (seg[-1] / seg[0]) ** (1 / yrs) - 1
    mdd = float((seg / np.maximum.accumulate(seg) - 1).min())
    return dict(cal=cagr / abs(mdd) if mdd else 0.0,
                sharpe=r.mean() / r.std() * np.sqrt(365.25),
                cagr=cagr, mdd=mdd)


def verdict(ma, mb):
    better = sum(1 for k in KEYS if ma[k] > mb[k] + EPS)
    worse = sum(1 for k in KEYS if ma[k] < mb[k] - EPS)
    return 1 if better > worse else (-1 if better < worse else 0)


def rank_pair(ma, mb):
    s = 0.0
    for k in KEYS:
        if ma[k] > mb[k] + EPS:
            s += 1.0
        elif ma[k] < mb[k] - EPS:
            s += 2.0
        else:
            s += 1.5
    return s / len(KEYS)


def worst_window_mdd(seg_list):
    out = []
    for seg in seg_list:
        if len(seg) < 2:
            continue
        out.append(float((seg / np.maximum.accumulate(seg) - 1).min()))
    return min(out) if out else 0.0


def summary():
    jpath = os.path.join(OUT, 'cells.jsonl')
    rows = [json.loads(l) for l in open(jpath, encoding='utf-8')]
    z = np.load(os.path.join(OUT, 'curves.npz'), allow_pickle=True)
    curves = {k: z[k] for k in z.files}
    dates = pd.DatetimeIndex(np.load(os.path.join(OUT, 'dates.npy'),
                                     allow_pickle=True))
    bi = int(np.searchsorted(dates, SPLIT_DATE))
    print(f"곡선 {len(dates)}봉 {dates[0].date()}~{dates[-1].date()}, "
          f"경계 인덱스 {bi} ({SPLIT_DATE.date()})\n")

    # ── 표 1. 구성 × 가드 × 비용 평균 ──
    agg = {}
    for r in rows:
        agg.setdefault((r['cfg'], r['guard'], r['cost']), []).append(r)
    print("전 구간 평균 (앵커 7곳)")
    print(f"  {'구성':<12}{'가드':>5}{'비용':>7}{'칼마':>8}{'연복리':>9}"
          f"{'낙폭':>9}{'샤프':>8}{'리밸':>8}{'거래':>8}{'비중95':>8}")
    for cfg, _d, _m in CONFIGS:
        for guard in GUARDS:
            for cost in COSTS:
                g = agg.get((cfg, guard, cost), [])
                if len(g) < len(PHASES):
                    continue
                f = lambda k: float(np.mean([x[k] for x in g]))  # noqa: E731
                print(f"  {cfg:<12}{guard:>5}{cost*10000:6.0f}bp"
                      f"{f('Cal'):8.3f}{f('CAGR'):9.4f}{f('MDD'):9.4f}"
                      f"{f('Sharpe'):8.3f}{f('Rebal'):8.1f}{f('Trades'):8.1f}"
                      f"{f('wp95'):8.3f}")

    # ── 가드군 재현 검사 ──
    print("\n가드 켠 군 재현 검사 (앞선 측정 40bp 앵커평균: 0.10=1.561, 0.40=2.237)")
    for cfg, want in (('d10-refill', 1.561), ('d40-refill', 2.237)):
        g = agg.get((cfg, 'on', 0.004), [])
        if len(g) < len(PHASES):
            print(f"  {cfg}: 미완")
            continue
        got = float(np.mean([x['Cal'] for x in g]))
        ok = abs(got - want) < 0.0015
        print(f"  {cfg}: {got:.3f} (기대 {want:.3f}) "
              f"{'일치' if ok else '불일치 — 판정 중단'}")

    # ── 판정선 ①~⑤ (무가드, 40bp, 0.40 대 0.10·refill) ──
    print(f"\n사전 판정선 (무가드 · 40bp · {CAND} 대 {REF})")
    ok = {}
    cand = {p: curves.get(cell_key(CAND, 'off', 0.004, p)) for p in PHASES}
    ref = {p: curves.get(cell_key(REF, 'off', 0.004, p)) for p in PHASES}
    if any(v is None for v in list(cand.values()) + list(ref.values())):
        print("  곡선 미완 — 판정 보류")
        return

    def logdiff(a, b, s, e):
        return float(np.log(a[e - 1] / a[s]) - np.log(b[e - 1] / b[s]))

    # ① 출처 구간별 누적 로그수익 차이
    segs = {'옛자료': (0, bi), '새자료': (bi, len(dates))}
    line1 = []
    for nm, (s, e) in segs.items():
        d = [logdiff(cand[p], ref[p], s, e) for p in PHASES]
        pos = sum(1 for x in d if x > 0)
        line1.append(np.mean(d) > 0 and pos >= 5)
        print(f"  ① {nm}: 평균 {np.mean(d):+.4f}, 양수 앵커 {pos}/7")
    ok['①'] = all(line1)

    # ② 출처 구간별 창 순위합
    line2 = []
    for nm, (s, e) in segs.items():
        per_len, wins, losses = [], 0, 0
        for wlen, step in WINDOWS:
            rs = []
            for p in PHASES:
                a, b = cand[p], ref[p]
                for t in range(s, e - wlen + 1, step):
                    ma, mb = win_metrics(a[t:t + wlen]), win_metrics(b[t:t + wlen])
                    if not (ma and mb):
                        continue
                    rs.append(rank_pair(ma, mb))
                    v = verdict(ma, mb)
                    wins += v > 0
                    losses += v < 0
            if rs:
                per_len.append(float(np.mean(rs)))
        rk = float(np.mean(per_len)) if per_len else 9.9
        line2.append(rk < 1.5 and wins > losses)
        print(f"  ② {nm}: 순위합 {rk:.3f}, 승 {wins} 패 {losses}")
    ok['②'] = all(line2)

    # ③ 전 구간 차이 + 상위 열 날짜 제외 잔여
    tot, resid = [], []
    for p in PHASES:
        a, b = cand[p], ref[p]
        la = np.diff(np.log(a))
        lb = np.diff(np.log(b))
        d = la - lb
        tot.append(float(d.sum()))
        idx = np.argsort(d)[-10:]
        resid.append(float(d.sum() - d[idx].sum()))
    m_tot, m_res = float(np.mean(tot)), float(np.mean(resid))
    ok['③'] = m_tot >= 0.15 and m_res >= m_tot / 2
    print(f"  ③ 전 구간 평균 차이 {m_tot:+.4f}, 상위 10일 제외 잔여 {m_res:+.4f}"
          f" (기준 {m_tot/2:+.4f})")

    # ④ 낙폭 악화
    worst_full, worst_w = [], []
    for nm, (s, e) in segs.items():
        for p in PHASES:
            a, b = cand[p][s:e], ref[p][s:e]
            da = float((a / np.maximum.accumulate(a) - 1).min())
            db = float((b / np.maximum.accumulate(b) - 1).min())
            worst_full.append(db - da)          # 양수면 후보가 더 깊다
            wa = worst_window_mdd([a[t:t + 504] for t in range(0, len(a) - 504 + 1, 63)])
            wb = worst_window_mdd([b[t:t + 504] for t in range(0, len(b) - 504 + 1, 63)])
            worst_w.append(wb - wa)
    ok['④'] = max(worst_full) <= 0.02 and max(worst_w) <= 0.05
    print(f"  ④ 최대낙폭 최악 악화 {max(worst_full):+.4f} (기준 0.0200), "
          f"최악 504봉 악화 {max(worst_w):+.4f} (기준 0.0500)")

    # ⑤ 집중도
    p95 = [r['wp95'] for r in rows
           if r['cfg'] == CAND and r['guard'] == 'off' and r['cost'] == 0.004]
    ok['⑤'] = bool(p95) and max(p95) <= 0.55
    print(f"  ⑤ 단일 종목 목표비중 95백분위 최대 {max(p95) if p95 else 0:.3f}"
          f" (기준 0.550)")

    print("\n  " + "  ".join(f"{k}{'통과' if v else '탈락'}" for k, v in ok.items()))
    print(f"  판정: {'다섯 모두 통과 — 모의운영 후보로 올린다' if all(ok.values()) else '하나 이상 탈락 — 라이브 0.10 을 그대로 둔다'}")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
        raise SystemExit(0)
    raise SystemExit(main())
