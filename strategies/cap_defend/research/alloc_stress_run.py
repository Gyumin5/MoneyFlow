#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실행 검증 스트레스 — 사전등록 research/alloc_stress_prereg.md 의 팔 일곱을 돈다.

본 채점 여섯 기준을 통과한 후보만 받는다(state/.../pass_idx.npy). 재는 것은 성적의
수준이 아니라 순서다 — 스트레스 팔에서 칼마가 떨어지는 것은 당연하고, 떨어뜨리는
근거는 "같은 팔의 현행보다 못해지는 것" 이다. 그래서 팔마다 현행 기준선을 그 팔의
가정 아래에서 다시 센다.

통과 기준 넷(㉮ 칼마 현행 이상 · ㉯ 낙폭 악화 1pp 이하 · ㉰ 상위 스무 후보 순서의
스피어만 0.5 이상 · ㉱ 본 채점 1등이 전 팔 상위 10%)은 사전등록 문서가 정본이고
여기서 옮기지 않는다.

실행: python3 alloc_stress_run.py [--anchor N] [--limit N] [--idx <npy>]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import alloc_grid_run as RUN          # noqa: E402
import alloc_grid_space as G          # noqa: E402
import alloc_ledger_sim as L          # noqa: E402
import alloc_trigger_space as S       # noqa: E402

OUT = RUN.OUT
ANCHORS = [0, 1, 2, 3, 4, 5, 6]
COSTS = ['base', 'x5']
MISS_SEEDS = [0, 1, 2, 3, 4]
MISS_P = 0.05                      # 사전등록한 가정 — 측정값이 아니다


def arms():
    """(이름, simulate_ledger 인자, 놓침 확률) — 놓침 팔은 씨앗마다 한 번씩 돈다."""
    out = [("기준", dict(), None),
           ("지연0", dict(lag_bd=0), None),
           ("지연10", dict(lag_bd=10), None),
           ("수락지연3", dict(accept_bd=3), None),
           ("수락지연5", dict(accept_bd=5), None)]
    for s in MISS_SEEDS:
        out.append((f"놓침1배_s{s}", dict(), (MISS_P, s)))
    for s in MISS_SEEDS:
        out.append((f"놓침3배_s{s}", dict(), (3 * MISS_P, s)))
    return out


def miss_vector(n, p, seed):
    """놓치는 날은 후보마다 따로 뽑지 않는다 — 공통 충격이라야 짝지은 비교가 된다."""
    return np.random.default_rng(10_000 + seed).random(n) < p


def cell_ctx(anchor, stress):
    eq_st, eq_sp, eq_fu = RUN.sleeve_curves(anchor, stress)
    axis, Rm, is_bd = L.build_ledger(eq_st, eq_sp, eq_fu)
    tab = L.bd_tables(axis, is_bd)
    g = L.gate_series_ledger(axis)
    gate = np.column_stack([np.asarray(g[s].values, dtype=bool) for s in S.SLEEVES])
    EV, _, _ = L.event_matrix(S.build_E(g), axis)
    month_id = np.asarray(pd.DatetimeIndex(axis).year * 12
                          + pd.DatetimeIndex(axis).month, dtype=np.int64)
    return (axis, Rm, is_bd, tab, gate, EV, month_id,
            RUN.window_bounds(axis, is_bd), (axis[-1] - axis[0]).days / 365.25)


def score(ctx, spec, kw):
    """한 팔에서 후보 전체의 순위·성적과 그 팔의 현행 기준선을 같이 낸다."""
    axis, Rm, is_bd, tab, gate, EV, month_id, bounds, yrs = ctx
    pv_b, inf_b = L.simulate_ledger(Rm, is_bd, gate, EV, RUN.control_spec("현행"),
                                    tab, month_id=month_id, **kw)
    cal_b, _, _, _ = RUN.window_stats(pv_b, axis, bounds)
    fs_b = RUN.full_stats(pv_b, axis)
    base = dict(Cal=float(fs_b['Cal'][0]), MDD=float(fs_b['MDD'][0]),
                Sharpe=float(fs_b['Sharpe'][0]), CAGR=float(fs_b['CAGR'][0]),
                ht95=float(inf_b['ht_p95'][0]), maxht=float(inf_b['max_ht'][0]),
                per_yr=float(inf_b['dones'][0]) / yrs,
                multi=int(inf_b['multi_months'][0]))

    n_all = len(spec['e_idx'])
    acc = {k: np.empty(n_all, dtype=np.float64) for k in
           ('rank', 'Cal', 'MDD', 'Sharpe', 'CAGR', 'ht95', 'maxht', 'per_yr')}
    for s0 in range(0, n_all, RUN.CHUNK):
        s1 = min(s0 + RUN.CHUNK, n_all)
        sub = {k: v[s0:s1] for k, v in spec.items()}
        pv, inf = L.simulate_ledger(Rm, is_bd, gate, EV, sub, tab,
                                    month_id=month_id, **kw)
        cal, _, _, _ = RUN.window_stats(pv, axis, bounds)
        rk = np.where(cal > cal_b[:, :1], 1.0, np.where(cal < cal_b[:, :1], 2.0, 1.5))
        acc['rank'][s0:s1] = rk.mean(axis=0)
        fs = RUN.full_stats(pv, axis)
        for k in ('Cal', 'MDD', 'Sharpe', 'CAGR'):
            acc[k][s0:s1] = fs[k]
        acc['ht95'][s0:s1] = inf['ht_p95']
        acc['maxht'][s0:s1] = inf['max_ht']
        acc['per_yr'][s0:s1] = inf['dones'] / yrs
    return acc, base


def spearman(a, b):
    """스피어만 — 동점은 평균 순위로. 사전등록 ㉰ 가 쓰는 값이다."""
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', type=int, default=-1, help='하나만 돌릴 때')
    ap.add_argument('--cost', choices=['base', 'x5', 'both'], default='both')
    ap.add_argument('--e', type=int, default=113)
    ap.add_argument('--idx', default=os.path.join(OUT, "pass_idx.npy"))
    ap.add_argument('--limit', type=int, default=0, help='장치 점검용 — 앞에서 N개만')
    a = ap.parse_args()

    if not os.path.exists(a.idx):
        print(f"통과 색인이 없다: {a.idx} — 먼저 alloc_grid_agg.py 를 돌린다.")
        return 1
    idx = np.load(a.idx).astype(np.int64)
    if a.limit:
        idx = idx[:a.limit]
    if idx.size == 0:
        print("통과한 후보가 0개다 — 스트레스는 잴 것이 없고 결론은 현행 유지다.")
        return 0

    full, _ = G.build(a.e, 0)
    spec = {k: v[idx] for k, v in full.items()}
    anchors = ANCHORS if a.anchor < 0 else [a.anchor]
    costs = COSTS if a.cost == 'both' else [a.cost]
    arm_list = arms()
    print(f"후보 {len(idx):,}개  팔 {len(arm_list)}개  칸 {len(anchors) * len(costs)}개")

    res = {nm: {k: [] for k in ('rank', 'Cal', 'MDD', 'ht95', 'per_yr')} for nm, _, _ in arm_list}
    bases = {nm: {k: [] for k in ('Cal', 'MDD', 'ht95', 'per_yr')} for nm, _, _ in arm_list}
    for anchor in anchors:
        for cost in costs:
            ctx = cell_ctx(anchor, cost == 'x5')
            n_days = len(ctx[0])
            for nm, kw, ms in arm_list:
                kw = dict(kw)
                if ms is not None:
                    kw['miss'] = miss_vector(n_days, ms[0], ms[1])
                acc, base = score(ctx, spec, kw)
                for k in res[nm]:
                    res[nm][k].append(acc[k])
                for k in bases[nm]:
                    bases[nm][k].append(base[k])
            print(f"  앵커 {anchor} {cost} 끝", flush=True)

    out = {}
    for nm in res:
        for k in res[nm]:
            out[f"{nm}__{k}"] = np.mean(res[nm][k], axis=0)
        for k in bases[nm]:
            out[f"{nm}__base_{k}"] = float(np.mean(bases[nm][k]))
    out['idx'] = idx
    np.savez_compressed(os.path.join(OUT, "stress.npz"),
                        **{k: np.asarray(v) for k, v in out.items()})

    ref = out['기준__rank']
    order_ref = np.argsort(ref)
    top20 = order_ref[:20]
    best = int(order_ref[0])
    print("\n팔별 (칸 평균)")
    print(f"{'팔':<12}{'현행Cal':>8}{'후보Cal중앙':>12}{'현행MDD':>9}{'후보MDD중앙':>12}"
          f"{'순위중앙':>9}{'㉮칼마':>7}{'㉯낙폭':>7}{'㉰상관':>8}{'㉱1등':>7}")
    rows = []
    for nm, _, _ in arm_list:
        cal, mdd = out[f"{nm}__Cal"], out[f"{nm}__MDD"]
        bc, bm = out[f"{nm}__base_Cal"], out[f"{nm}__base_MDD"]
        c1 = int((cal >= bc).sum())
        c2 = int(((np.abs(mdd) - abs(bm)) <= 0.01).sum())
        rho = spearman(ref[top20], out[f"{nm}__rank"][top20]) if nm != '기준' else 1.0
        thr = np.percentile(out[f"{nm}__rank"], 10)
        in10 = out[f"{nm}__rank"][best] <= thr
        rows.append((nm, c1, c2, rho, bool(in10)))
        print(f"{nm:<12}{bc:>8.3f}{np.median(cal):>12.3f}{bm * 100:>8.2f}%"
              f"{np.median(mdd) * 100:>11.2f}%{np.median(out[f'{nm}__rank']):>9.4f}"
              f"{c1:>7}{c2:>7}{rho:>8.3f}{'예' if in10 else '아니오':>7}")

    nstress = len(idx)
    ok = np.ones(nstress, dtype=bool)
    for nm, _, _ in arm_list:
        if nm == '기준':
            continue
        ok &= out[f"{nm}__Cal"] >= out[f"{nm}__base_Cal"]
        ok &= (np.abs(out[f"{nm}__MDD"]) - abs(out[f"{nm}__base_MDD"])) <= 0.01
    print(f"\n㉮㉯ 를 전 팔에서 통과한 후보 {int(ok.sum()):,} / {nstress:,}")
    bad = [nm for nm, _, _, rho, in10 in rows
           if nm != '기준' and (not (rho >= 0.5) or not in10)]
    print("㉰㉱ 를 어긴 팔:", ", ".join(bad) if bad else "없음")
    with open(os.path.join(OUT, "stress_summary.json"), "w") as f:
        json.dump(dict(n=int(nstress), pass_ab=int(ok.sum()), bad_cd=bad,
                       rows=[dict(arm=r[0], cal_ok=r[1], mdd_ok=r[2],
                                  rho=None if np.isnan(r[3]) else r[3], top10=r[4])
                             for r in rows]), f, ensure_ascii=False, indent=1)
    np.save(os.path.join(OUT, "stress_pass_idx.npy"), idx[ok])
    return 0


if __name__ == '__main__':
    sys.exit(main())
