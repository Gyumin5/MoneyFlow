"""합동 격자 집계 — 비중 × 리밸런싱 조건 (2026-10-31).

alloc_joint_grid.py 가 앵커 7곳 × 비용 2조건으로 뱉은 npz 14개를 모아 판정한다.
설계는 ai-debate run-20261031T005336Z 확정분이고, 토론이 요구한 "안 라"(셋을 다 내고
어긋나는 자리를 보고)를 그대로 따른다.

내는 것:
  주표   — 합동 조합 2,240개를 앵커 합산 순위합으로 세운 상위·하위.
  보조1 — 비중마다 자기 최적 조건을 고른 뒤 비중끼리 겨루기(직전 측정의 2단계).
  보조2 — 조건을 평균해 지우고 비중만 겨루기(직전 측정의 3단계).
           유한 활성 42칸 균등평균과 행동 중복을 지운 평균을 같이 낸다.
  진단   — 대표 칸의 이웃 평지, 두 비용 조건 일치, 앵커 7곳 중 상위 비율.

행동 중복: 반회전율이 아주 크면 그 트리거가 사실상 안 걸려 상대부족만 남는다. 그러면
서로 다른 칸이 같은 매매를 하므로 평균에서 같은 행동이 여러 번 세어진다. 발화일이
완전히 같은 칸을 한 묶음으로 보고 묶음마다 한 표씩만 준 평균을 따로 낸다.

실행: python3 alloc_joint_agg.py [--dir state/20261031-jointgrid]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alloc_joint_grid as J  # noqa: E402

ANCHORS = list(range(7))
STRESSES = [0, 1]
TOP_SELF = 0.10      # 대표 칸은 자기 비용·자기 축에서 상위 10% 안
TOP_NEIGH = 0.20     # 이웃은 상위 20% 안이면 "같이 좋다"


def load(d: str):
    """npz 14개를 (앵커, 비용) 키로 읽는다. 격자 축이 전부 같은지 확인한다."""
    got, missing = {}, []
    for s in STRESSES:
        for a in ANCHORS:
            p = os.path.join(d, f'a{a}_s{s}.npz')
            if not os.path.exists(p):
                missing.append(os.path.basename(p))
                continue
            got[(a, s)] = np.load(p, allow_pickle=True)
    if missing:
        raise SystemExit(f'없는 결과: {", ".join(missing)}')
    ref = got[(0, 0)]
    for k, z in got.items():
        for col in ('mix', 'ht', 't3u'):
            if not np.array_equal(z[col], ref[col]):
                raise SystemExit(f'{k} 의 격자 축 {col} 이 기준과 다르다')
    return got, ref


def stack(got, key: str, stress: int) -> np.ndarray:
    """(앵커, 조합) 행렬."""
    return np.vstack([got[(a, stress)][key] for a in ANCHORS])


def fmt_cond(ht: float, t3u: float) -> str:
    h = '꺼짐' if not np.isfinite(ht) else f'{ht * 100:.0f}'
    t = '꺼짐' if not np.isfinite(t3u) else f'{t3u * 100:.0f}'
    return f'{h}/{t}'


def avg_rank(rs_sum: np.ndarray, n_windows: int) -> np.ndarray:
    """순위합을 창 하나당 평균순위로. 낮을수록 우세."""
    return rs_sum / n_windows


def head(title: str):
    print()
    print(title)
    print('-' * len(title) * 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='state/20261031-jointgrid')
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()

    got, ref = load(a.dir)
    mix, ht, t3u = ref['mix'], ref['ht'], ref['t3u']
    K = len(mix)
    nwin = int(ref['n']) * len(ANCHORS)
    live = (ht == J.LIVE_HT) & (t3u == J.LIVE_T3U)
    mixes = sorted(set(mix.tolist()))

    print(f'합동 격자 집계 — 조합 {K}개 (비중 {len(mixes)} × 조건 {K // len(mixes)})')
    print(f'구간 {ref["d0"]} ~ {ref["d1"]}, 영업일 {ref["ndays"]}, '
          f'창 {int(ref["n"])}개 × 앵커 {len(ANCHORS)} = {nwin}')
    if K != len(J.build_combos(J.build_mixes())):
        raise SystemExit('격자 크기가 지금의 alloc_joint_grid 정의와 다르다')

    res = {}
    for s in STRESSES:
        res[s] = {
            'rank': avg_rank(stack(got, 'rs', s).sum(axis=0), nwin),
            'rank_by_anchor': stack(got, 'rs', s) / int(ref['n']),
            'cal': stack(got, 'cal', s).mean(axis=0),
            'mdd': stack(got, 'mdd', s).mean(axis=0),
            'cagr': stack(got, 'cagr', s).mean(axis=0),
            'sharpe': stack(got, 'sharpe', s).mean(axis=0),
            'fires': stack(got, 'fires_per_year', s).mean(axis=0),
        }

    # --- 주표 ---------------------------------------------------------------
    for s in STRESSES:
        lab = '기본 비용' if s == 0 else '비용 5배'
        r = res[s]
        head(f'[주표/{lab}] 합동 조합 순위 (평균순위 낮을수록 우세)')
        print(f'{"순":>3} {"비중":<14} {"조건":>8} {"평균순위":>9} {"칼마":>6} '
              f'{"낙폭%":>7} {"연복리%":>8} {"샤프":>6} {"송금/년":>7}')
        order = np.argsort(r['rank'], kind='stable')
        for i in list(order[:a.top]):
            print(f'{int(np.where(order == i)[0][0]) + 1:>3} {mix[i]:<14} '
                  f'{fmt_cond(ht[i], t3u[i]):>8} {r["rank"][i]:>9.1f} '
                  f'{r["cal"][i]:>6.2f} {r["mdd"][i]:>7.1f} {r["cagr"][i]:>8.1f} '
                  f'{r["sharpe"][i]:>6.2f} {r["fires"][i]:>7.2f}')
        # 현행 비중 × 라이브 조건이 어디에 있나
        cur = np.array([('현행' in m) for m in mix])
        k = np.where(cur & live)[0]
        if len(k):
            k = int(k[0])
            pos = int(np.where(order == k)[0][0]) + 1
            print(f'    현행 비중 + 라이브 조건(20/20) = {pos}위 / {K}, '
                  f'평균순위 {r["rank"][k]:.1f}, 칼마 {r["cal"][k]:.2f}, '
                  f'낙폭 {r["mdd"][k]:.1f}%, 송금 {r["fires"][k]:.2f}회')
        best_cur = np.argsort(np.where(cur, r['rank'], np.inf), kind='stable')[0]
        print(f'    현행 비중 안 최적 조건 = {fmt_cond(ht[best_cur], t3u[best_cur])}, '
              f'{int(np.where(order == best_cur)[0][0]) + 1}위, '
              f'칼마 {r["cal"][best_cur]:.2f}, 낙폭 {r["mdd"][best_cur]:.1f}%, '
              f'송금 {r["fires"][best_cur]:.2f}회')

    # --- 보조1: 비중마다 자기 최적으로 겨루기 --------------------------------
    for s in STRESSES:
        lab = '기본 비용' if s == 0 else '비용 5배'
        r = res[s]
        head(f'[보조1/{lab}] 비중마다 자기 최적 조건을 고른 뒤 겨루기')
        rows = []
        for m in mixes:
            sel = mix == m
            k = int(np.argsort(np.where(sel, r['rank'], np.inf), kind='stable')[0])
            rows.append((r['rank'][k], m, k))
        rows.sort()
        print(f'{"순":>3} {"비중":<14} {"최적조건":>8} {"평균순위":>9} {"칼마":>6} '
              f'{"낙폭%":>7} {"송금/년":>7}')
        for i, (v, m, k) in enumerate(rows[:a.top], 1):
            print(f'{i:>3} {m:<14} {fmt_cond(ht[k], t3u[k]):>8} {v:>9.1f} '
                  f'{r["cal"][k]:>6.2f} {r["mdd"][k]:>7.1f} {r["fires"][k]:>7.2f}')

    # --- 보조2: 조건을 평균해 지우고 비중만 ----------------------------------
    fin = np.isfinite(ht) & np.isfinite(t3u)      # 유한 활성 42칸
    for s in STRESSES:
        lab = '기본 비용' if s == 0 else '비용 5배'
        r = res[s]
        head(f'[보조2/{lab}] 조건을 평균해 지우고 비중만 겨루기')
        rows = []
        for m in mixes:
            sel = mix == m
            flat = r['rank'][sel & fin].mean()
            # 행동 중복 제거: 같은 발화 빈도는 같은 매매로 본다(발화수 자체가 지표)
            fr = np.round(r['fires'][sel & fin], 6)
            uniq = np.array([r['rank'][sel & fin][fr == v].mean() for v in sorted(set(fr))])
            rows.append((flat, uniq.mean(), m))
        rows.sort()
        print(f'{"순":>3} {"비중":<14} {"42칸평균":>9} {"중복제거":>9}')
        for i, (f1, f2, m) in enumerate(rows[:a.top], 1):
            print(f'{i:>3} {m:<14} {f1:>9.1f} {f2:>9.1f}')

    # --- 진단: 대표 칸 -------------------------------------------------------
    head('[진단] 대표 칸 — 기본 비용 상위 10% 안에서 이웃이 같이 좋은 칸')
    r0, r1 = res[0], res[1]
    thr_self = np.quantile(r0['rank'], TOP_SELF)
    thr_nb = np.quantile(r0['rank'], TOP_NEIGH)
    mix_list = [m for m in mixes]
    # 비중 이웃 = 주식 축 ±1칸(같은 비율), 비율 축 ±1칸(같은 주식)
    axis_st = J.STOCK_AXIS
    axis_rt = [c[0] for c in J.COIN_RATIOS]

    def parse(m: str):
        base = m.split(' ')[0]
        s_, rt = base.split('/', 1)
        return float(s_) / 100.0, rt

    def mix_neighbors(m: str):
        s_, rt = parse(m)
        out = []
        i = axis_st.index(round(s_, 10)) if round(s_, 10) in axis_st else None
        j = axis_rt.index(rt)
        for di in (-1, 1):
            if i is not None and 0 <= i + di < len(axis_st):
                out.append((axis_st[i + di], rt))
        for dj in (-1, 1):
            if 0 <= j + dj < len(axis_rt):
                out.append((s_, axis_rt[j + dj]))
        names = []
        for ss, rr in out:
            for mm in mix_list:
                a2, b2 = parse(mm)
                if abs(a2 - ss) < 1e-9 and b2 == rr:
                    names.append(mm)
        return names

    # 조건 이웃 = 각 축 ±1칸. 꺼짐은 별도 정책이라 유한 임계의 이웃으로 안 센다.
    def cond_neighbors(h: float, t: float):
        out = []
        if np.isfinite(h) and np.isfinite(t):
            hs = [x for x in J.GRID_HT if np.isfinite(x)]
            ts = [x for x in J.GRID_T3U if np.isfinite(x)]
            i, j = hs.index(h), ts.index(t)
            for di in (-1, 1):
                if 0 <= i + di < len(hs):
                    out.append((hs[i + di], t))
            for dj in (-1, 1):
                if 0 <= j + dj < len(ts):
                    out.append((h, ts[j + dj]))
        return out

    cand = []
    for k in range(K):
        if r0['rank'][k] > thr_self or not fin[k]:
            continue
        nb = []
        for nm in mix_neighbors(str(mix[k])):
            idx = np.where((mix == nm) & (ht == ht[k]) & (t3u == t3u[k]))[0]
            if len(idx):
                nb.append(int(idx[0]))
        for h2, t2 in cond_neighbors(ht[k], t3u[k]):
            idx = np.where((mix == mix[k]) & (ht == h2) & (t3u == t2))[0]
            if len(idx):
                nb.append(int(idx[0]))
        if not nb:
            continue
        ok = sum(1 for x in nb if r0['rank'][x] <= thr_nb)
        if ok * 2 <= len(nb):          # 이웃 과반이 상위 20% 안이어야 한다
            continue
        cand.append((np.mean([r0['rank'][x] for x in nb]), k, ok, len(nb)))
    cand.sort()

    if not cand:
        print('  없다 — 상위 10% 안에서 이웃 과반이 같이 좋은 칸이 하나도 없다.')
        print('  이 격자는 평지가 아니다. 1위를 운영값으로 옮기지 않는다.')
    else:
        print(f'{"비중":<14} {"조건":>8} {"이웃평균":>9} {"이웃":>7} {"본인":>8} '
              f'{"5배순위":>8} {"앵커상위":>8}')
        for nbavg, k, ok, tot in cand[:a.top]:
            ord1 = np.argsort(r1['rank'], kind='stable')
            pos1 = int(np.where(ord1 == k)[0][0]) + 1
            per = r0['rank_by_anchor']
            top_anchor = sum(1 for ai in range(len(ANCHORS))
                             if per[ai, k] <= np.quantile(per[ai], TOP_NEIGH))
            print(f'{mix[k]:<14} {fmt_cond(ht[k], t3u[k]):>8} {nbavg:>9.1f} '
                  f'{ok:>3}/{tot:<3} {r0["rank"][k]:>8.1f} {pos1:>8} '
                  f'{top_anchor:>6}/7')

    # --- 진단: 세 판정이 어긋나는 자리 ---------------------------------------
    head('[진단] 주표·보조1·보조2 가 어긋나는 자리 (기본 비용)')
    r = res[0]
    main_best = mix[int(np.argsort(r['rank'], kind='stable')[0])]
    rows1 = sorted((r['rank'][int(np.argsort(np.where(mix == m, r['rank'], np.inf),
                                             kind='stable')[0])], m) for m in mixes)
    rows2 = sorted((r['rank'][(mix == m) & fin].mean(), m) for m in mixes)
    print(f'  주표 1위 비중   : {main_best}')
    print(f'  보조1 1위 비중  : {rows1[0][1]}')
    print(f'  보조2 1위 비중  : {rows2[0][1]}')
    o1 = [m for _v, m in rows1]
    o2 = [m for _v, m in rows2]
    diffs = [(m, o1.index(m) + 1, o2.index(m) + 1) for m in mixes
             if abs(o1.index(m) - o2.index(m)) >= 5]
    if diffs:
        print('  두 보조표 순위가 5칸 이상 갈리는 비중:')
        for m, p1, p2 in sorted(diffs, key=lambda x: x[1]):
            print(f'    {m:<14} 보조1 {p1:>2}위 / 보조2 {p2:>2}위')
    else:
        print('  두 보조표가 5칸 이상 갈리는 비중은 없다.')

    # --- 주표2: 송금 한도마다의 최적 ------------------------------------------
    # 상대부족 축은 낮출수록 계속 좋아져서 봉우리가 없다 — 바닥을 2.5% 까지 내려도
    # 최적이 거기 붙고 송금이 연 25회쯤 된다. 그건 조건의 최적이 아니라 "되도록 자주
    # 되돌려라" 이고, 실제로 멈추는 자리는 격자가 아니라 사람이 손으로 보낼 수 있는
    # 횟수다. 그래서 한도마다 최적을 따로 낸다 — 운영자가 고르는 축이 그쪽이다.
    BUDGETS = [2, 3, 4, 6, 12, 1e9]
    for s in STRESSES:
        lab = '기본 비용' if s == 0 else '비용 5배'
        r = res[s]
        head(f'[주표2/{lab}] 송금 한도마다의 최적 조합')
        print(f'{"한도/년":>7} {"비중":<14} {"조건":>8} {"평균순위":>9} {"칼마":>6} '
              f'{"낙폭%":>7} {"연복리%":>8} {"샤프":>6} {"실제송금":>8}')
        for b in BUDGETS:
            ok = r['fires'] <= b + 1e-9
            if not ok.any():
                continue
            k = int(np.argsort(np.where(ok, r['rank'], np.inf), kind='stable')[0])
            lb = '제한없음' if b > 1e8 else f'{b:g}회'
            print(f'{lb:>7} {mix[k]:<14} {fmt_cond(ht[k], t3u[k]):>8} '
                  f'{r["rank"][k]:>9.1f} {r["cal"][k]:>6.2f} {r["mdd"][k]:>7.1f} '
                  f'{r["cagr"][k]:>8.1f} {r["sharpe"][k]:>6.2f} {r["fires"][k]:>8.2f}')

    # --- 진단: 상대부족 축에 봉우리가 있나 ------------------------------------
    head('[진단] 상대부족 축 — 최적 비중에서 축을 내리면 계속 좋아지나 (기본 비용)')
    r = res[0]
    best_mix = mix[int(np.argsort(r['rank'], kind='stable')[0])]
    print(f'  기준 비중 {best_mix}, 반회전율은 그 비중의 최적값으로 고정')
    sel = mix == best_mix
    kb = int(np.argsort(np.where(sel, r['rank'], np.inf), kind='stable')[0])
    print(f'{"상대부족":>8} {"평균순위":>9} {"칼마":>6} {"낙폭%":>7} {"송금/년":>8}')
    for t in J.GRID_T3U:
        k = np.where(sel & (ht == ht[kb]) & (t3u == t))[0]
        if not len(k):
            continue
        k = int(k[0])
        lb = '꺼짐' if not np.isfinite(t) else f'{t * 100:g}%'
        print(f'{lb:>8} {r["rank"][k]:>9.1f} {r["cal"][k]:>6.2f} '
              f'{r["mdd"][k]:>7.1f} {r["fires"][k]:>8.2f}')

    # --- 진단: 라이브 조건이 어디쯤인가 --------------------------------------
    head('[진단] 라이브 조건(20/20) 이 비중마다 몇 등인가 (기본 비용, 56칸 중)')
    print(f'{"비중":<14} {"라이브등수":>10} {"최적조건":>8} {"최적칼마":>9} {"라이브칼마":>10}')
    for m in mixes:
        sel = mix == m
        sub = np.where(sel)[0]
        order_sub = sub[np.argsort(r['rank'][sub], kind='stable')]
        kl = int(np.where(sel & live)[0][0])
        pos = int(np.where(order_sub == kl)[0][0]) + 1
        kb = int(order_sub[0])
        print(f'{m:<14} {pos:>10} {fmt_cond(ht[kb], t3u[kb]):>8} '
              f'{r["cal"][kb]:>9.2f} {r["cal"][kl]:>10.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
