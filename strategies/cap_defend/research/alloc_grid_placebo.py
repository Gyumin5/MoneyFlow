#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""자산간 리밸런싱 격자의 위약(음성 대조).

왜: 후보가 94만이면 그 중 하나는 우연만으로도 현행을 이긴다. 그 "우연의 크기"를
재지 않으면 1등의 성적이 실력인지 자릿수인지 가릴 수 없다. 사전등록한 보정은
"실제 격자의 최대 개선이 위약 최대 개선의 95분위를 넘어야 한다" 이다.

두 가족 — 둘 다 전이 사건 신호 하나만 무의미하게 만든다:
  shift — 전이 사건 행렬을 같은 칸만큼 원형으로 민다. 사건의 개수·간격 구조는 그대로고
          시점만 어긋난다. 전이의 "시점" 이 값인지 가른다.
  rand  — 전이 사건을 발화일 수가 같은 무작위 날짜집합으로 갈아끼운다. 그 횟수의
          아무 날짜집합이나 같은 성적을 내는지 가른다.

게이트(카나리)는 양쪽 다 실제값을 쓴다. 게이트를 같이 밀면 T1·T3U 와 현행 기준선까지
바뀌어서, 후보가 좋아 보이는 게 아니라 기준선이 나빠진 것을 재게 된다 — 실제로 첫
시험에서 위약인데도 후보 71%가 "현행보다 낫다" 로 나왔다.

재는 통계는 격자 전체에 대한 최대값들이다 — 한 위약이 한 줄을 낸다.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import alloc_grid_run as RUN          # noqa: E402
import alloc_grid_space as G          # noqa: E402
import alloc_ledger_sim as L          # noqa: E402
import alloc_trigger_space as S       # noqa: E402

OUT = os.path.join(RUN.OUT, "placebo")


def lean_cal(pv, axis, bounds):
    """창별 칼마만. 순위는 칼마로만 매기므로 샤프·연복리는 여기서 안 센다."""
    nb = len(bounds)
    cal = np.empty((nb, pv.shape[1]), dtype=np.float64)
    for k, (i0, i1) in enumerate(bounds):
        seg = pv[i0:i1 + 1]
        yrs = (axis[i1] - axis[i0]).days / 365.25
        g = (seg[-1] / seg[0]) ** (1.0 / yrs) - 1.0
        dd = (seg / np.maximum.accumulate(seg, axis=0) - 1.0).min(axis=0)
        cal[k] = np.where(dd < 0, g / np.abs(np.where(dd < 0, dd, 1)), 0.0)
    return cal


def random_events(EV, rng):
    """발화일 수를 칸마다 맞춘 무작위 날짜집합."""
    n, m = EV.shape
    out = np.zeros_like(EV)
    cnt = EV.sum(axis=0)
    for j in range(m):
        c = int(cnt[j])
        if c:
            out[rng.choice(n, size=c, replace=False), j] = True
    return out


def one_placebo(kind, seed, anchor, stress, spec, ctx, chunk):
    axis, Rm, is_bd, tab, gate, EV0, month_id, bounds, cal_b, base, yrs = ctx
    rng = np.random.default_rng(seed)
    if kind == 'shift':
        # 축 길이의 10~90% 안에서 고른다 — 가장자리는 원본과 거의 같아진다
        k = int(rng.integers(int(len(axis) * 0.10), int(len(axis) * 0.90)))
        EV = np.roll(EV0, k, axis=0)
        note = dict(shift=k)
    else:
        EV = random_events(EV0, rng)
        note = dict(shift=0)

    n_all = len(spec['e_idx'])
    best = dict(rank=9.9, dCal=-9.9, dMDD=-9.9, dht95=-9.9)
    n140 = 0
    npass = 0
    for s0 in range(0, n_all, chunk):
        s1 = min(s0 + chunk, n_all)
        sub = {k2: v[s0:s1] for k2, v in spec.items()}
        pv, inf = L.simulate_ledger(Rm, is_bd, gate, EV, sub, tab, month_id=month_id)
        cal = lean_cal(pv, axis, bounds)
        rk = np.where(cal > cal_b[:, None], 1.0,
                      np.where(cal < cal_b[:, None], 2.0, 1.5)).mean(axis=0)
        fs = RUN.full_stats(pv, axis)
        ht95 = inf['ht_p95']; mx = inf['max_ht']
        per_yr = inf['dones'] / yrs
        ok = (rk <= 1.40) & (fs['Cal'] >= base['Cal'])
        ok &= (np.abs(fs['MDD']) - abs(base['MDD'])) <= 0.01
        ok &= (ht95 <= base['ht95']) & (mx <= base['maxht'])
        ok &= per_yr <= min(base['per_yr'] * 1.5, 12.0)
        ok &= inf['multi_months'] <= base['multi']
        n140 += int((rk <= 1.40).sum())
        npass += int(ok.sum())
        best['rank'] = min(best['rank'], float(rk.min()))
        best['dCal'] = max(best['dCal'], float((fs['Cal'] - base['Cal']).max()))
        best['dMDD'] = max(best['dMDD'], float((abs(base['MDD']) - np.abs(fs['MDD'])).max()))
        best['dht95'] = max(best['dht95'], float((base['ht95'] - ht95).max()))
    return dict(kind=kind, seed=seed, anchor=anchor, stress=bool(stress),
                n=n_all, base=base, best=best, n140=n140, npass=npass, **note)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', type=int, default=0)
    ap.add_argument('--cost', choices=['base', 'x5'], default='base')
    ap.add_argument('--kind', choices=['shift', 'rand'], default='shift')
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--count', type=int, default=1)
    ap.add_argument('--e', type=int, default=113)
    ap.add_argument('--chunk', type=int, default=5000)
    ap.add_argument('--limit', type=int, default=0, help='후보 수 제한 (속도 재기용)')
    a = ap.parse_args()

    stress = a.cost == 'x5'
    eq_st, eq_sp, eq_fu = RUN.sleeve_curves(a.anchor, stress)
    axis, Rm, is_bd = L.build_ledger(eq_st, eq_sp, eq_fu)
    tab = L.bd_tables(axis, is_bd)
    g0 = L.gate_series_ledger(axis)
    EV0, _, _ = L.event_matrix(S.build_E(g0), axis)
    month_id = np.asarray(pd.DatetimeIndex(axis).year * 12
                          + pd.DatetimeIndex(axis).month, dtype=np.int64)
    bounds = RUN.window_bounds(axis, is_bd)
    gate = np.column_stack([np.asarray(g0[s].values, dtype=bool) for s in S.SLEEVES])

    # 현행 기준선은 전이 신호를 안 쓰므로 위약마다 다시 안 센다 — 한 번만 센다
    pv_b, inf_b = L.simulate_ledger(Rm, is_bd, gate, EV0, RUN.control_spec("현행"), tab,
                                    month_id=month_id)
    cal_b = lean_cal(pv_b, axis, bounds)[:, 0]
    fs_b = RUN.full_stats(pv_b, axis)
    yrs = (axis[-1] - axis[0]).days / 365.25
    base = dict(Cal=float(fs_b['Cal'][0]), MDD=float(fs_b['MDD'][0]),
                ht95=float(inf_b['ht_p95'][0]), maxht=float(inf_b['max_ht'][0]),
                per_yr=float(inf_b['dones'][0]) / yrs,
                multi=int(inf_b['multi_months'][0]))
    print("기준선(현행) Cal %.4f MDD %.2f%% ht95 %.2fpp 송금 %.2f/년"
          % (base['Cal'], base['MDD'] * 100, base['ht95'] * 100, base['per_yr']), flush=True)
    ctx = (axis, Rm, is_bd, tab, gate, EV0, month_id, bounds, cal_b, base, yrs)

    spec, _ = G.build(a.e, 0)
    if a.limit:
        spec = {k: v[:a.limit] for k, v in spec.items()}
    print(f"후보 {len(spec['e_idx']):,}개  창 {len(bounds)}개  축 {len(axis)}일", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for i in range(a.count):
        seed = a.seed0 + i
        t0 = time.time()
        row = one_placebo(a.kind, seed, a.anchor, stress, spec, ctx, a.chunk)
        row['초'] = round(time.time() - t0, 1)
        row['팔당ms'] = round(row['초'] / max(row['n'], 1) * 1000, 3)
        tag = f"{a.kind}_a{a.anchor}_{a.cost}_s{seed}"
        with open(os.path.join(OUT, f"pl_{tag}.json"), "w") as f:
            json.dump(row, f, ensure_ascii=False, indent=1)
        b = row['best']
        print(f"{tag}  {row['초']}s  최소순위 {b['rank']:.4f}  "
              f"최대 칼마개선 {b['dCal']:+.3f}  최대 낙폭개선 {b['dMDD']*100:+.2f}pp  "
              f"최대 이탈개선 {b['dht95']*100:+.2f}pp  1.40이하 {row['n140']:,}  "
              f"전기준통과 {row['npass']:,}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
