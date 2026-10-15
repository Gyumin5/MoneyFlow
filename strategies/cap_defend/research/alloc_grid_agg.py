#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""본 채점 14칸을 모아 사전등록 기준으로 거른다.

기준은 토론이 결과를 보기 전에 못박은 것이고 여기서 옮기지 않는다:
  ⑴ 일대일 평균순위가 두 비용 팔 각각에서 1.40 이하
  ⑵ 앵커 7 중 6 이상에서 1.5 미만
  ⑶ 앵커마다 — 칼마가 현행 이상, 최대낙폭 악화 1pp 이하, 이탈 95분위와 최대이탈이 현행 이하
  ⑷ 앵커 7 평균 — 최대낙폭이 0.5pp 이상 좋아지거나 이탈 95분위가 2pp 이상 좋아지거나
  ⑸ 송금 횟수가 min(현행×1.5, 12)회/년 이하
  ⑹ 한 달에 2회 이상 송금한 달 수가 현행 이하
살아남은 것에 대해서만 칼마·샤프·연복리·최대낙폭 넷을 다 적는다.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import alloc_grid_space as G     # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261015-allocrebal"
ANCHORS = [0, 1, 2, 3, 4, 5, 6]
COSTS = ['base', 'x5']


def load_cells():
    cells, metas, missing = {}, {}, []
    for a in ANCHORS:
        for c in COSTS:
            p = os.path.join(OUT, f"grid_a{a}_{c}.npz")
            q = os.path.join(OUT, f"grid_a{a}_{c}.json")
            if not (os.path.exists(p) and os.path.exists(q)):
                missing.append(f"a{a}_{c}")
                continue
            cells[(a, c)] = np.load(p)
            metas[(a, c)] = json.load(open(q))
    return cells, metas, missing


def main() -> int:
    cells, metas, missing = load_cells()
    if missing:
        print("아직 없는 칸:", ", ".join(missing))
    if not cells:
        return 1
    ns = {v['rank'].shape[0] for v in cells.values()}
    if len(ns) != 1:
        print("칸마다 후보 수가 다르다:", ns)
        return 1
    n = ns.pop()
    print(f"칸 {len(cells)}/14, 후보 {n:,}")

    def stack(key, cost):
        return np.vstack([cells[(a, cost)][key] for a in ANCHORS if (a, cost) in cells])

    ok = np.ones(n, dtype=bool)
    rank_mean = {}
    for c in COSTS:
        if not any((a, c) in cells for a in ANCHORS):
            continue
        r = stack('rank', c)
        rank_mean[c] = r.mean(axis=0)
        ok &= rank_mean[c] <= 1.40                                    # ⑴

    per_anchor = np.stack([
        np.mean([cells[(a, c)]['rank'] for c in COSTS if (a, c) in cells], axis=0)
        for a in ANCHORS if any((a, c) in cells for c in COSTS)])
    n_anchor = per_anchor.shape[0]
    ok &= (per_anchor < 1.5).sum(axis=0) >= 6                         # ⑵

    dmdd, dht = [], []
    for a in ANCHORS:
        for c in COSTS:
            if (a, c) not in cells:
                continue
            z, b = cells[(a, c)], metas[(a, c)]['base']
            ok &= z['Cal'] >= b['Cal']                                # ⑶
            ok &= (np.abs(z['MDD']) - abs(b['MDD'])) <= 0.01
            ok &= z['ht95'] <= b['ht95']
            ok &= z['maxht'] <= b['maxht']
            ok &= z['per_yr'] <= min(b['per_yr'] * 1.5, 12.0)         # ⑸
            ok &= z['multi'] <= b['multi']                            # ⑹
            dmdd.append(abs(b['MDD']) - np.abs(z['MDD']))
            dht.append(b['ht95'] - z['ht95'])
    dmdd = np.mean(dmdd, axis=0); dht = np.mean(dht, axis=0)
    ok &= (dmdd >= 0.005) | (dht >= 0.02)                             # ⑷

    if n_anchor < 7:
        print(f"\n[경고] 앵커가 {n_anchor}곳만 모였다 — 아래 숫자는 중간값이고 판정이 아니다."
              f" 기준 ⑵ 는 7곳 중 6곳을 요구하므로 {n_anchor}곳으로는 구조적으로 통과가 0 이다.")
    print(f"\n전 기준 통과 {int(ok.sum()):,}개 / {n:,}")
    if not ok.any():
        rows = [("⑴ 순위 1.40", np.all([rank_mean[c] <= 1.40 for c in rank_mean], axis=0)),
                ("⑷ 평균 개선", (dmdd >= 0.005) | (dht >= 0.02))]
        if n_anchor >= 7:
            rows.insert(1, ("⑵ 앵커 6/7", (per_anchor < 1.5).sum(axis=0) >= 6))
        for nm, m in rows:
            print(f"  {nm} 단독 통과 {int(m.sum()):,}")
        if missing or n_anchor < 7:
            print("\n칸이 덜 모였다 — 통과 색인을 쓰지 않는다. 칸이 다 모이면 같은 명령으로 다시 본다.")
            return 0
        # 빈 색인을 남긴다 — 안 남기면 아래 단계가 "아직 안 돌렸다" 와 "돌렸는데 0" 을 못 가른다
        np.save(os.path.join(OUT, "pass_idx.npy"), np.empty(0, dtype=np.int64))
        print(f"\n통과 색인 저장(빈 것): {OUT}/pass_idx.npy")
        print("사전등록 기준을 다 넘은 후보가 없다 — 이 연구의 결론은 현행 유지다.")
        return 0

    spec, _ = G.build(int(metas[list(metas)[0]]['발화E']), 0)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(np.mean([rank_mean[c][idx] for c in rank_mean], axis=0))]
    print("\n상위 20 (평균순위 오름차순)")
    for j in order[:20]:
        s = {k: int(v[j]) for k, v in spec.items()}
        cal = np.mean([cells[(a, c)]['Cal'][j] for a in ANCHORS for c in COSTS
                       if (a, c) in cells])
        shp = np.mean([cells[(a, c)]['Sharpe'][j] for a in ANCHORS for c in COSTS
                       if (a, c) in cells])
        cg = np.mean([cells[(a, c)]['CAGR'][j] for a in ANCHORS for c in COSTS
                      if (a, c) in cells])
        md = np.mean([cells[(a, c)]['MDD'][j] for a in ANCHORS for c in COSTS
                      if (a, c) in cells])
        pr = np.mean([cells[(a, c)]['per_yr'][j] for a in ANCHORS for c in COSTS
                      if (a, c) in cells])
        print(f"  #{j}  순위 {np.mean([rank_mean[c][j] for c in rank_mean]):.4f}  "
              f"칼마 {cal:.3f} 샤프 {shp:.3f} 연복리 {cg*100:.1f}% 낙폭 {md*100:.2f}% "
              f"송금 {pr:.2f}/년  {s}")
    np.save(os.path.join(OUT, "pass_idx.npy"), idx)
    print(f"\n통과 색인 저장: {OUT}/pass_idx.npy")
    return 0


if __name__ == '__main__':
    sys.exit(main())
