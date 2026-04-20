#!/usr/bin/env python3
"""C 현물 train(~2023)/holdout(2024+) 분리 sweep — V21 coin baseline.

m3_engine_final.simulate (spot, 1x lev) 사용.
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from m3_engine_final import (load_v21, load_universe_hist, list_available_futures,
                             load_coin_daily, simulate, metrics)
from c_engine_v5 import run_c_v5, load_coin

OUT = os.path.join(HERE, "c_spot_holdout")
os.makedirs(OUT, exist_ok=True)

N_JOBS = 24
TRAIN_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")


def extract_events(avail, P):
    rows = []
    for c in avail:
        df = load_coin(c + "USDT")
        if df is None: continue
        _, evs = run_c_v5(df, **P)
        for e in evs:
            e["coin"] = c
            rows.append(e)
    return pd.DataFrame(rows)


def slice_v21(v21, start, end):
    sub = v21[(v21.index >= start) & (v21.index <= end)].copy()
    if len(sub) < 30: return None
    sub["equity"] = sub["equity"] / sub["equity"].iloc[0]
    sub["v21_ret"] = sub["equity"].pct_change().fillna(0)
    sub["prev_cash"] = sub["cash_ratio"].shift(1).fillna(sub["cash_ratio"].iloc[0])
    return sub


def eval_port(events, coin_daily, v21_slice, hist, **FP):
    if v21_slice is None:
        return {"Cal":0, "CAGR":0, "MDD":0, "Sharpe":0, "n_entries":0}
    ev_sub = events[(events["entry_ts"] >= v21_slice.index[0])
                    & (events["entry_ts"] <= v21_slice.index[-1])]
    _, stats = simulate(ev_sub, coin_daily, v21_slice, hist, **FP)
    return {**{k: stats[k] for k in ["Sharpe","CAGR","MDD","Cal"]},
            "n_entries": stats["n_entries"]}


def phase1_worker(args):
    P, avail_pkl, cd_pkl, v21_pkl, hist_pkl = args
    avail = pd.read_pickle(avail_pkl)
    coin_daily = pd.read_pickle(cd_pkl)
    v21_full = pd.read_pickle(v21_pkl)
    hist = pd.read_pickle(hist_pkl)

    events = extract_events(avail, P)
    if len(events) == 0:
        return {**P, "Cal_train":0, "Cal_holdout":0, "Cal_full":0, "n_events":0}

    v21_train = slice_v21(v21_full, v21_full.index[0], TRAIN_END)
    v21_hout = slice_v21(v21_full, HOLDOUT_START, v21_full.index[-1])
    v21_full_norm = slice_v21(v21_full, v21_full.index[0], v21_full.index[-1])

    FP = dict(n_pick=1, cap_per_slot=0.333, universe_size=15,
              tx_cost=0.003, swap_edge_threshold=1)
    mt = eval_port(events, coin_daily, v21_train, hist, **FP)
    mh = eval_port(events, coin_daily, v21_hout, hist, **FP)
    mf = eval_port(events, coin_daily, v21_full_norm, hist, **FP)

    return {**P,
            "Cal_train": round(mt["Cal"], 4),
            "Cal_holdout": round(mh["Cal"], 4),
            "Cal_full": round(mf["Cal"], 4),
            "CAGR_train": round(mt["CAGR"], 4),
            "CAGR_holdout": round(mh["CAGR"], 4),
            "MDD_train": round(mt["MDD"], 4),
            "MDD_holdout": round(mh["MDD"], 4),
            "n_events": len(events),
            "n_entries_train": mt["n_entries"],
            "n_entries_holdout": mh["n_entries"]}


def main():
    print("Loading (spot)...")
    v21 = load_v21()
    hist = load_universe_hist()
    avail = sorted(list_available_futures())
    coin_daily = load_coin_daily(avail)

    v21_train = slice_v21(v21, v21.index[0], TRAIN_END)
    v21_hout = slice_v21(v21, HOLDOUT_START, v21.index[-1])
    v21_full = slice_v21(v21, v21.index[0], v21.index[-1])
    print("=== V21 spot 단독 baseline ===")
    print(f"Full:    {metrics(v21_full['equity'])}")
    print(f"Train:   {metrics(v21_train['equity'])}")
    print(f"Holdout: {metrics(v21_hout['equity'])}")

    avail_pkl = "/tmp/csh_avail.pkl"; cd_pkl = "/tmp/csh_cd.pkl"
    v21_pkl = "/tmp/csh_v21.pkl"; hist_pkl = "/tmp/csh_hist.pkl"
    pd.to_pickle(avail, avail_pkl); pd.to_pickle(coin_daily, cd_pkl)
    pd.to_pickle(v21, v21_pkl); pd.to_pickle(hist, hist_pkl)

    configs = []
    for db in [12, 24, 48, 72]:
        for dt in [-0.10, -0.12, -0.15, -0.18, -0.20]:
            for tp in [0.04, 0.06, 0.08, 0.12]:
                for ts in [12, 24, 48]:
                    configs.append({"dip_bars":db,"dip_thr":dt,"tp":tp,"tstop":ts})

    print(f"\n===== Phase 1: {len(configs)} configs × {N_JOBS} parallel =====")
    t0 = time.time()
    args_list = [(P, avail_pkl, cd_pkl, v21_pkl, hist_pkl) for P in configs]
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(phase1_worker)(a) for a in args_list)
    print(f"완료 ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "phase1_holdout.csv"), index=False)

    print("\n=== Train Cal Top 15 ===")
    print(df.sort_values("Cal_train", ascending=False).head(15)[
        ["dip_bars","dip_thr","tp","tstop","Cal_train","Cal_holdout","Cal_full",
         "CAGR_holdout","MDD_holdout"]].to_string(index=False))

    best = df.sort_values("Cal_train", ascending=False).iloc[0]
    best_P = {"dip_bars":int(best["dip_bars"]),"dip_thr":float(best["dip_thr"]),
              "tp":float(best["tp"]),"tstop":int(best["tstop"])}
    print(f"\n=== Train best: {best_P} ===")

    events_best = extract_events(avail, best_P)
    print(f"\n=== Phase 3: cap_per_slot 민감도 ===")
    rows = []
    for cap in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.333]:
        FP = dict(n_pick=1, cap_per_slot=cap, universe_size=15,
                  tx_cost=0.003, swap_edge_threshold=1)
        mt = eval_port(events_best, coin_daily, v21_train, hist, **FP)
        mh = eval_port(events_best, coin_daily, v21_hout, hist, **FP)
        mf = eval_port(events_best, coin_daily, v21_full, hist, **FP)
        rows.append({"cap":cap, "Cal_train":round(mt["Cal"],4),
                     "Cal_holdout":round(mh["Cal"],4), "Cal_full":round(mf["Cal"],4),
                     "CAGR_holdout":round(mh["CAGR"],4), "MDD_holdout":round(mh["MDD"],4)})
        print(f"  cap={cap:.3f}: train Cal={mt['Cal']:.2f} / holdout Cal={mh['Cal']:.2f} "
              f"CAGR_h={mh['CAGR']:.2%} MDD_h={mh['MDD']:.2%}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase3_cap.csv"), index=False)

    v21_h_m = metrics(v21_hout["equity"])
    v21_f_m = metrics(v21_full["equity"])
    print("\n=== V21 spot 단독 vs V21+C ===")
    print(f"V21 단독 Full:    Cal={v21_f_m['Cal']:.2f} CAGR={v21_f_m['CAGR']:.2%} MDD={v21_f_m['MDD']:.2%}")
    print(f"V21 단독 Holdout: Cal={v21_h_m['Cal']:.2f} CAGR={v21_h_m['CAGR']:.2%} MDD={v21_h_m['MDD']:.2%}")
    best_row = pd.DataFrame(rows).sort_values("Cal_holdout", ascending=False).iloc[0]
    print(f"Best cap={best_row['cap']}: V21+C Holdout Cal={best_row['Cal_holdout']} "
          f"CAGR={best_row['CAGR_holdout']:.2%} MDD={best_row['MDD_holdout']:.2%}")

    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
