"""평가 배터리: 레짐별 분해 + 블록 부트스트랩 CI + T3O 실행지연·부분이행 stress.
full(라이브) 과 U5(완전 de-outlier) 두 유니버스에서 base vs T3O20 비교.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)
from bt_v25_t1_t3u_t3o import run_spot, run_fut, run_stock, load_canaries, simulate_alloc, metrics

W = {'stock': 0.60, 'spot': 0.25, 'fut': 0.15}
REGIMES = [
    ('불장 20-11~21', '2020-11-13', '2021-12-31'),
    ('베어 2022',     '2022-01-01', '2022-12-31'),
    ('회복 23-24',    '2023-01-01', '2024-12-31'),
    ('최근 25-26',    '2025-01-01', '2026-05-01'),
]


def _target(pvs, total, ft1, fire):
    if ft1 or len(fire) >= 2:
        return {k: total * W[k] for k in W}
    if len(fire) == 1:
        k = next(iter(fire)); others = [o for o in W if o != k]; wo = sum(W[o] for o in others)
        tp = total * W[k]; delta = tp - pvs[k]
        t = dict(pvs); t[k] = tp
        for o in others:
            t[o] = pvs[o] - delta * (W[o] / wo)
        return t
    return dict(pvs)


def simulate_realistic(eq_st, eq_sp, eq_fu, can, T1=0.20, T3U=0.20, T3O=None, t3o_canary='none',
                       exec_delay=0, fill_frac=1.0):
    common = sorted(eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index))
    r = {'stock': eq_st.loc[common].pct_change().fillna(0),
         'spot': eq_sp.loc[common].pct_change().fillna(0),
         'fut': eq_fu.loc[common].pct_change().fillna(0)}
    cser = {k: can[k].reindex(common).ffill().fillna(False).astype(bool) for k in can}
    pvs = dict(W); pending = []; eq = []
    for i, d in enumerate(common):
        for k in pvs:
            pvs[k] *= (1 + r[k].iloc[i])
        total = sum(pvs.values())
        if total <= 0:
            eq.append(0); continue
        cur = {k: pvs[k] / total for k in pvs}
        ht = sum(abs(cur[k] - W[k]) for k in W) / 2
        ft1 = ht >= T1
        fire = {}
        for k in W:
            can_on = bool(cser[k].iloc[i])
            ru = max(0.0, (W[k] - cur[k]) / W[k]); ro = max(0.0, (cur[k] - W[k]) / W[k])
            if ru >= T3U and can_on:
                fire[k] = 'u'
            elif T3O is not None and ro >= T3O:
                g = (not can_on) if t3o_canary == 'off' else (can_on if t3o_canary == 'on' else True)
                if g:
                    fire[k] = 'o'
        if ft1 or fire:
            pending.append((i + exec_delay, ft1, dict(fire)))
        still = []
        for (ei, pft1, pfd) in pending:
            if ei <= i:
                tgt = _target(pvs, total, pft1, pfd)
                for k in pvs:
                    pvs[k] = pvs[k] + fill_frac * (tgt[k] - pvs[k])
            else:
                still.append((ei, pft1, pfd))
        pending = still
        eq.append(sum(pvs.values()))
    return pd.Series(eq, index=common)


def reg_metrics(eq, s, e):
    sl = eq[(eq.index >= pd.Timestamp(s)) & (eq.index <= pd.Timestamp(e))].dropna()
    if len(sl) < 20:
        return None
    yrs = (sl.index[-1] - sl.index[0]).days / 365.25
    cagr = (sl.iloc[-1] / sl.iloc[0]) ** (1 / yrs) - 1
    peak = sl.cummax(); mdd = (sl / peak - 1).min()
    rets = sl.pct_change().dropna()
    sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return cagr * 100, mdd * 100, sh, cal


def block_boot_cal(eq, L=21, N=800, seed_offset=0):
    rets = eq.pct_change().dropna().values
    n = len(rets); nb = n // L + 1
    cals = []
    for it in range(N):
        # deterministic pseudo-random block starts (Math.random 불가 환경 회피)
        starts = [((it * 1103515245 + j * 12345 + seed_offset) % (n - L)) for j in range(nb)]
        seq = np.concatenate([rets[s:s + L] for s in starts])[:n]
        e = np.cumprod(1 + seq)
        yrs = n / 252.0
        cagr = e[-1] ** (1 / yrs) - 1
        peak = np.maximum.accumulate(e); mdd = (e / peak - 1).min()
        cals.append(cagr / abs(mdd) if mdd < 0 else 0)
    return np.percentile(cals, [5, 50, 95])


def main():
    t0 = time.time()
    eq_st = run_stock()
    eq_sp_f = run_spot(20, 127, 217, 7, [])
    eq_fu_f = run_fut(18, 127, 95, 5, [])
    EX5 = ['BNB', 'SOL', 'DOGE', 'XRP', 'ADA']
    eq_sp_5 = run_spot(20, 127, 217, 7, EX5)
    eq_fu_5 = run_fut(18, 127, 95, 5, EX5)
    common0 = sorted(eq_st.index.intersection(eq_sp_f.index).intersection(eq_fu_f.index))
    can = load_canaries(common0)
    BASE = dict(T1=0.20, T3U=0.20, T3O=None)
    T3O = dict(T1=0.20, T3U=0.20, T3O=0.20, t3o_canary='none')

    curves = {
        'full base':  simulate_alloc(eq_st, eq_sp_f, eq_fu_f, can, **BASE)[0],
        'full T3O':   simulate_alloc(eq_st, eq_sp_f, eq_fu_f, can, **T3O)[0],
        'U5 base':    simulate_alloc(eq_st, eq_sp_5, eq_fu_5, can, **BASE)[0],
        'U5 T3O':     simulate_alloc(eq_st, eq_sp_5, eq_fu_5, can, **T3O)[0],
    }

    print("\n=== 1. 레짐별 분해 (CAGR% / MDD% / Cal) ===")
    print(f"  {'레짐':<14} " + " ".join(f"{k:>16}" for k in curves))
    for rname, s, e in REGIMES:
        cells = []
        for k in curves:
            m = reg_metrics(curves[k], s, e)
            cells.append(f"{m[0]:>5.0f}/{m[1]:>+5.0f}/{m[3]:>4.2f}" if m else "       -       ")
        print(f"  {rname:<14} " + " ".join(f"{c:>16}" for c in cells))

    print("\n=== 2. 블록 부트스트랩 Cal CI (block=21d, N=800) [p5, median, p95] ===")
    for k in curves:
        p = block_boot_cal(curves[k])
        print(f"  {k:<12} Cal [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]")
    # T3O ΔCal paired significance (full)
    rb = curves['full base'].pct_change().dropna(); rt = curves['full T3O'].pct_change().dropna()
    idx = rb.index.intersection(rt.index); rb = rb.loc[idx].values; rt = rt.loc[idx].values
    n = len(rb); L = 21; nb = n // L + 1; diffs = []
    for it in range(800):
        starts = [((it * 1103515245 + j * 12345) % (n - L)) for j in range(nb)]
        eb = np.cumprod(1 + np.concatenate([rb[s:s+L] for s in starts])[:n])
        et = np.cumprod(1 + np.concatenate([rt[s:s+L] for s in starts])[:n])
        yrs = n / 252.0
        def cal(e):
            cg = e[-1]**(1/yrs)-1; pk=np.maximum.accumulate(e); md=(e/pk-1).min()
            return cg/abs(md) if md<0 else 0
        diffs.append(cal(et) - cal(eb))
    dp = np.percentile(diffs, [5, 50, 95]); frac = np.mean(np.array(diffs) > 0) * 100
    print(f"  full T3O−base ΔCal [{dp[0]:+.2f}, {dp[1]:+.2f}, {dp[2]:+.2f}], ΔCal>0 비율 {frac:.0f}%")

    print("\n=== 3. T3O 실행지연·부분이행 stress (ΔCal vs 해당 유니버스 base) ===")
    for utag, sp, fu in [('full', eq_sp_f, eq_fu_f), ('U5', eq_sp_5, eq_fu_5)]:
        base_cal = metrics(simulate_realistic(eq_st, sp, fu, can, T1=0.20, T3U=0.20, T3O=None))[3]
        print(f"  [{utag}] base Cal {base_cal:.2f}")
        for delay in [0, 1, 2, 3, 5]:
            for fill in [1.0, 0.5]:
                eq = simulate_realistic(eq_st, sp, fu, can, T1=0.20, T3U=0.20, T3O=0.20,
                                        t3o_canary='none', exec_delay=delay, fill_frac=fill)
                m = metrics(eq)
                print(f"    delay {delay}d fill {int(fill*100)}%: CAGR {m[0]:>5.1f}% MDD {m[1]:>+6.1f}% Cal {m[3]:.2f}  ΔCal {m[3]-base_cal:+.2f}")
        print()

    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
