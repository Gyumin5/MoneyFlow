"""격자를 실제로 돌려 채점한다 — 토론이 사전등록한 기준 그대로.

채점 규약(run-20261015T005408Z verification_plan 그대로, 결과를 보고 안 바꾼다):
  · 창 길이 504·756·1008 과 보폭 63·126·252 의 아홉 조합을 전부 쓴다. 창은 영업일
    자리로 센다 — 장부 축은 주말을 포함하므로 날짜 칸으로 세면 창 길이의 뜻이 바뀐다.
  · 후보와 현행을 일대일로 비교한다(창마다 둘 중 칼마가 큰 쪽이 1위). 후보가 몇 개든
    순위가 안 흔들리게 하려는 것이다. 동률은 1.5.
  · 사전 지정한 7개 위상(bt_coin_param_resweep.ANCHORS)의 평균을 본다.
  · 비용 기본 / 비용 5배 두 벌을 따로 채점한다.
  · 창마다 칼마·샤프·연복리·최대낙폭을 같이 적는다. 홀드아웃·워크포워드·연도별 순위는 안 쓴다.

통과 기준(사전등록):
  ① 두 비용 모두 일대일 평균순위 1.40 이하
  ② 7앵커 중 6개 이상에서 1.5 미만
  ③ 앵커마다 칼마 현행 이상 · 최대낙폭 악화 1pp 이내 · 이탈 95분위와 최대 현행 이하
  ④ 7앵커 평균에서 최대낙폭 0.5pp 이상 개선 또는 이탈 95분위 2pp 이상 개선
  ⑤ 연평균 완료 송금이 현행의 1.5배 이하이면서 12회 이하, 두 번 이상 송금한 달의 비율도 현행 이하

주식 비용 배수: bt_coin_param_resweep.run_stock 은 stress 인자를 받고도 주식 체결
비용에 전달하지 않는다(bt_stock_mom3.TX 가 모듈 상수 0.001 고정). 토론이 고치라고
지목한 자리라 여기서는 호출 앞뒤로 그 상수를 바꿔 끼운다. 공용 모듈은 안 건드린다 —
그 캐시를 쓰는 다른 측정들이 같이 흔들린다.

실행:
  python3 alloc_grid_run.py --quick            작은 조각으로 채점 코드만 확인
  python3 alloc_grid_run.py --anchor 0 --cost base   한 칸 (병렬 실행 단위)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alloc_grid_space as G  # noqa: E402
import alloc_ledger_sim as L  # noqa: E402
import alloc_trigger_space as S  # noqa: E402
import bt_coin_param_resweep as R  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261015-allocrebal"
EQDIR = os.path.join(OUT, "eq")
WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]
STOCK_TX = 0.001
STRESS_MULT = 5.0
CHUNK = 5000   # 한 번에 세우는 팔 수 — 봉우리 메모리가 여기 비례한다


# ───────────────────────────────────────────────────── 슬리브 곡선 (캐시)
def _eq_path(kind, anchor, stress):
    return os.path.join(EQDIR, f"{kind}_a{anchor}_{'x5' if stress else 'x1'}.csv")


def _init_resweep():
    """bt_coin_param_resweep 는 main() 에서 전역을 채운다 — 모듈로 쓰려면 직접 채운다."""
    if R._bars is not None:
        return
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)


def sleeve_curves(anchor: int, stress: bool):
    os.makedirs(EQDIR, exist_ok=True)
    _init_resweep()
    out = {}
    for kind in ("stock", "spot", "fut"):
        p = _eq_path(kind, anchor, stress)
        if os.path.exists(p):
            s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
            if len(s) > 100:
                out[kind] = s
                continue
        if kind == "stock":
            import bt_stock_mom3 as bsm3
            old = bsm3.TX
            bsm3.TX = STOCK_TX * (STRESS_MULT if stress else 1.0)
            try:
                s = R.run_stock(anchor, stress)
            finally:
                bsm3.TX = old
        elif kind == "spot":
            s = R.run_spot({}, R.PH_SPOT[anchor], stress)
        else:
            s = R.run_fut({}, R.PH_FUT[anchor], stress)
        s = s.dropna()
        # 임시 파일에 쓰고 바꿔 끼운다 — 바로 쓰면 중간에 끊긴 파일이 남고,
        # 다음 일꾼이 그걸 "길이 100 넘는 캐시" 로 읽어 잘린 곡선으로 돈다.
        tmp = f"{p}.tmp.{os.getpid()}"
        s.rename("eq").to_frame().to_csv(tmp)
        os.replace(tmp, p)
        out[kind] = s
    return out["stock"], out["spot"], out["fut"]


# ───────────────────────────────────────────────────── 창 채점
def window_bounds(axis, is_bd):
    """창은 영업일 자리로 센다 — 장부 축의 날짜 칸으로 세면 길이의 뜻이 바뀐다."""
    bpos = np.flatnonzero(is_bd)
    out = []
    for size in WIN_SIZES:
        for stride in STRIDES:
            for i in range(0, len(bpos) - size, stride):
                out.append((int(bpos[i]), int(bpos[i + size - 1])))
    return out


def window_stats(pv, axis, bounds):
    """(창, 팔) 의 칼마·샤프·연복리·최대낙폭."""
    nb = len(bounds)
    A = pv.shape[1]
    cal = np.empty((nb, A)); mdd = np.empty((nb, A))
    cagr = np.empty((nb, A)); shp = np.empty((nb, A))
    for k, (i0, i1) in enumerate(bounds):
        seg = pv[i0:i1 + 1]
        yrs = (axis[i1] - axis[i0]).days / 365.25
        g = (seg[-1] / seg[0]) ** (1.0 / yrs) - 1.0
        dd = (seg / np.maximum.accumulate(seg, axis=0) - 1.0).min(axis=0)
        r = seg[1:] / seg[:-1] - 1.0
        sd = r.std(axis=0)
        cagr[k] = g
        mdd[k] = dd
        shp[k] = np.where(sd > 0, r.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)
        cal[k] = np.where(dd < 0, g / np.abs(np.where(dd < 0, dd, 1)), 0.0)
    return cal, mdd, cagr, shp


def full_stats(pv, axis):
    yrs = (axis[-1] - axis[0]).days / 365.25
    g = (pv[-1] / pv[0]) ** (1.0 / yrs) - 1.0
    dd = (pv / np.maximum.accumulate(pv, axis=0) - 1.0).min(axis=0)
    r = pv[1:] / pv[:-1] - 1.0
    sd = r.std(axis=0)
    sh = np.where(sd > 0, r.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)
    cal = np.where(dd < 0, g / np.abs(np.where(dd < 0, dd, 1)), 0.0)
    return dict(CAGR=g, MDD=dd, Sharpe=sh, Cal=cal)


# ───────────────────────────────────────────────────── 한 칸 돌리기
def control_spec(name):
    kw = dict(G.CONTROLS)[name]
    d = dict(e_idx=-1, combo=0, w_key=0, w_all=1, r_key=0, r_all=1,
             r_from_done=0, band_i=0, tscope=0, block_mode=0)
    d.update({k: v for k, v in kw.items() if k in d})
    return {k: np.array([v], dtype=np.int64) for k, v in d.items()}


def run_cell(anchor: int, stress: bool, spec, quick=False):
    # 음성 대조는 여기 없다 — alloc_grid_placebo.py 가 맡는다. 게이트를 밀면 T1·T3U 와
    # 현행 기준선까지 같이 나빠져서, 후보가 좋아진 게 아니라 기준선이 떨어진 것을 재게 된다.
    eq_st, eq_sp, eq_fu = sleeve_curves(anchor, stress)
    axis, Rm, is_bd = L.build_ledger(eq_st, eq_sp, eq_fu)
    tab = L.bd_tables(axis, is_bd)
    g = L.gate_series_ledger(axis)
    gate = np.column_stack([np.asarray(g[s].values, dtype=bool) for s in S.SLEEVES])
    E = S.build_E(g)
    EV, names, _ = L.event_matrix(E, axis)
    live = np.flatnonzero(EV.any(axis=0))
    month_id = np.asarray(pd.DatetimeIndex(axis).year * 12
                          + pd.DatetimeIndex(axis).month, dtype=np.int64)
    bounds = window_bounds(axis, is_bd)
    yrs = (axis[-1] - axis[0]).days / 365.25

    # 현행 기준선
    pv_b, inf_b = L.simulate_ledger(Rm, is_bd, gate, EV, control_spec("현행"), tab,
                                    month_id=month_id)
    cal_b, mdd_b, _, _ = window_stats(pv_b, axis, bounds)
    fs_b = full_stats(pv_b, axis)
    base = dict(cal=cal_b[:, 0], Cal=float(fs_b['Cal'][0]), MDD=float(fs_b['MDD'][0]),
                Sharpe=float(fs_b['Sharpe'][0]), CAGR=float(fs_b['CAGR'][0]),
                ht95=float(inf_b['ht_p95'][0]), maxht=float(inf_b['max_ht'][0]),
                per_yr=float(inf_b['dones'][0]) / yrs,
                multi=int(inf_b['multi_months'][0]))

    n_all = len(spec['e_idx'])
    if quick:
        n_all = min(n_all, 3 * CHUNK)
    acc = {k: np.empty(n_all, dtype=np.float32) for k in
           ('rank', 'Cal', 'MDD', 'Sharpe', 'CAGR', 'ht95', 'maxht', 'per_yr', 'multi')}
    t0 = time.time()
    for s0 in range(0, n_all, CHUNK):
        s1 = min(s0 + CHUNK, n_all)
        sub = {k: v[s0:s1] for k, v in spec.items()}
        pv, inf = L.simulate_ledger(Rm, is_bd, gate, EV, sub, tab, month_id=month_id)
        cal, mdd, _, _ = window_stats(pv, axis, bounds)
        rk = np.where(cal > base['cal'][:, None], 1.0,
                      np.where(cal < base['cal'][:, None], 2.0, 1.5))
        acc['rank'][s0:s1] = rk.mean(axis=0)
        fs = full_stats(pv, axis)
        for k in ('Cal', 'MDD', 'Sharpe', 'CAGR'):
            acc[k][s0:s1] = fs[k]
        acc['ht95'][s0:s1] = inf['ht_p95']
        acc['maxht'][s0:s1] = inf['max_ht']
        acc['per_yr'][s0:s1] = inf['dones'] / yrs
        acc['multi'][s0:s1] = inf['multi_months']
    dt = time.time() - t0
    meta = dict(anchor=anchor, stress=bool(stress),
                n=n_all, 창=len(bounds), 영업일=int(is_bd.sum()), 연수=yrs,
                발화E=int(len(live)), 초=round(dt, 1),
                base={k: v for k, v in base.items() if k != 'cal'},
                팔당ms=round(dt / max(n_all, 1) * 1000, 3))
    acc['_base_cal'] = base['cal'].astype(np.float32)
    return acc, meta, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', type=int, default=0)
    ap.add_argument('--cost', choices=['base', 'x5'], default='base')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--e', type=int, default=0, help='고유 E 개수 (0=자동)')
    a = ap.parse_args()

    e_live = a.e
    if e_live <= 0:
        # 장부 축에서 실제로 발화하는 E 를 센다
        eq_st, eq_sp, eq_fu = sleeve_curves(a.anchor, a.cost == 'x5')
        axis, _, _ = L.build_ledger(eq_st, eq_sp, eq_fu)
        E = S.build_E(L.gate_series_ledger(axis))
        EV, names, _ = L.event_matrix(E, axis)
        e_live = int(EV.any(axis=0).sum())
        print(f"장부 축에서 발화하는 고유 E {e_live}종 / 전체 {len(names)}종")

    spec, counts = G.build(e_live, 0)
    print(f"후보 {len(spec['e_idx']):,}개 (+대조 {counts['대조']})")

    acc, meta, base = run_cell(a.anchor, a.cost == 'x5', spec, a.quick)
    os.makedirs(OUT, exist_ok=True)
    tag = f"a{a.anchor}_{a.cost}"
    np.savez_compressed(os.path.join(OUT, f"grid_{tag}.npz"), **acc)
    with open(os.path.join(OUT, f"grid_{tag}.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(json.dumps(meta, ensure_ascii=False, indent=1))
    r = acc['rank']
    print(f"\n일대일 평균순위  최소 {r.min():.4f}  중앙 {np.median(r):.4f}"
          f"  최대 {r.max():.4f}")
    print(f"  1.40 이하 {int((r <= 1.40).sum()):,}개"
          f"  1.50 미만 {int((r < 1.50).sum()):,}개")
    ok = (r <= 1.40) & (acc['Cal'] >= base['Cal'])
    ok &= (np.abs(acc['MDD']) - abs(base['MDD'])) <= 1.0
    ok &= acc['ht95'] <= base['ht95']
    ok &= acc['maxht'] <= base['maxht']
    ok &= acc['per_yr'] <= min(base['per_yr'] * 1.5, 12.0)
    ok &= acc['multi'] <= base['multi']
    print(f"  이 칸에서 전 기준 통과 {int(ok.sum()):,}개")
    return 0


if __name__ == '__main__':
    sys.exit(main())
