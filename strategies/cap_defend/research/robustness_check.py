#!/usr/bin/env python3
"""Phase-3 사후 강건성 검증: LOYO (Leave-One-Year-Out) + LOAO (Leave-One-Asset-Out).

phase3 top 앙상블 대상으로:
1. LOYO: 연도 하나씩 제외 → 순위 안정성 (Spearman)
2. LOAO: 코인 하나씩 유니버스에서 제외 → 순위 안정성 (Spearman)

입력: phase3_ensembles_floor{30,40}/all_combos.csv + spot_top.csv + fut_top.csv
출력: robustness_results_floor{30,40}/loyo.csv, loao.csv, summary.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import (
    FULL_END, atomic_write_csv, build_trace, equity_metrics, parse_tag,
    preload_futures, preload_spot, run_spot_ensemble, write_manifest,
)

ANCHOR = "2020-10-01"
LOYO_YEARS = [2021, 2022, 2023, 2024, 2025]


def _load_top_ensembles(p3_dir: str, top_n: int = 100) -> pd.DataFrame:
    """phase3/all_combos.csv 에서 status==ok 전체 중 rank_sum 상위 top_n.
    버킷별로 나누지 않고 통합 rank 사용 — robustness는 앙상블 절대 순위 안정성이 목적."""
    all_p = os.path.join(p3_dir, "all_combos.csv")
    if not os.path.exists(all_p):
        return pd.DataFrame()
    df = pd.read_csv(all_p)
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        return df
    df["rank_sum"] = (
        df["Cal"].rank(ascending=False, method="min")
        + df["Sharpe"].rank(ascending=False, method="min")
        + df["CAGR"].rank(ascending=False, method="min")
    )
    return df.sort_values("rank_sum").head(top_n).reset_index(drop=True)


def _run_ensemble(members: list[str], asset: str, lev: float,
                  anchor: str, end: str,
                  exclude_coin: str | None = None) -> dict:
    """단일 앙상블 재실행. exclude_coin 지정 시 해당 코인을 유니버스에서 제외."""
    k = len(members)
    weights = {m: 1.0 / k for m in members}

    if asset == "spot":
        return _run_spot_with_exclusion(members, weights, anchor, end, exclude_coin)
    return _run_fut_with_exclusion(members, weights, lev, anchor, end, exclude_coin)


def _run_spot_with_exclusion(members, weights, anchor, end, exclude_coin):
    preload_spot()
    import run_current_coin_v20_backtest as spot_bt
    from phase_common import _cfg_to_member

    member_cfgs = {}
    for m_tag in members:
        meta = parse_tag(m_tag)
        member_cfgs[m_tag] = {
            "interval": meta["interval"], "sma": meta["sma"],
            "ms": meta["ms"], "ml": meta["ml"],
            "vol_mode": meta["vol_mode"], "vol_thr": meta["vol_thr"],
            "snap": meta["snap"],
        }
    members_dict = {k: _cfg_to_member(v) for k, v in member_cfgs.items()}
    total = sum(weights.values())
    norm_w = {k: w / total for k, w in weights.items()}

    orig_m = spot_bt.MEMBERS
    orig_w = spot_bt.ENSEMBLE_WEIGHTS
    orig_load_univ = spot_bt.load_universe

    if exclude_coin:
        _orig_fn = orig_load_univ
        def _patched_universe(top_n=40):
            um = _orig_fn(top_n=top_n)
            return {d: [c for c in coins if c != exclude_coin]
                    for d, coins in um.items()}
        spot_bt.load_universe = _patched_universe

    spot_bt.MEMBERS = members_dict
    spot_bt.ENSEMBLE_WEIGHTS = norm_w
    try:
        res = spot_bt.run_backtest(start=anchor, end=end)
    finally:
        spot_bt.MEMBERS = orig_m
        spot_bt.ENSEMBLE_WEIGHTS = orig_w
        spot_bt.load_universe = orig_load_univ

    eq = res["equity"]
    m = equity_metrics(eq)
    return {"Cal": m["Cal"], "Sharpe": m["Sh"], "CAGR": m["CAGR"], "MDD": m["MDD"]}


def _run_fut_with_exclusion(members, weights, lev, anchor, end, exclude_coin):
    from futures_ensemble_engine import SingleAccountEngine, combine_targets
    import phase_common as pc

    _ = preload_futures()
    cache = pc._FUT_DATA
    saved = {iv: cache[iv] for iv in cache}
    try:
        if exclude_coin:
            for iv, (b, f) in saved.items():
                if exclude_coin in b:
                    cache[iv] = (
                        {k: v for k, v in b.items() if k != exclude_coin},
                        {k: v for k, v in f.items() if k != exclude_coin},
                    )
        bars_1h, funding_1h = cache["1h"]
        traces = {}
        for m_tag in members:
            meta = parse_tag(m_tag)
            cfg = {"interval": meta["interval"], "sma": meta["sma"],
                   "ms": meta["ms"], "ml": meta["ml"],
                   "vol_mode": meta["vol_mode"], "vol_thr": meta["vol_thr"],
                   "snap": meta["snap"]}
            tr = build_trace("fut", cfg, lev, anchor, end=end)["trace"]
            traces[m_tag] = tr
        all_dates = bars_1h[next(iter(bars_1h))].index
        dates = all_dates[(all_dates >= anchor) & (all_dates <= end)]
        combined = combine_targets(traces, weights, dates)
        engine = SingleAccountEngine(
            bars_1h, funding_1h,
            leverage=float(lev), leverage_mode="fixed",
            per_coin_leverage_mode="none",
            stop_kind="none", stop_pct=0.0, stop_lookback_bars=0,
            stop_gate="always",
        )
        res = engine.run(combined)
    finally:
        for iv, val in saved.items():
            cache[iv] = val
    return {
        "Cal": float(res.get("Cal", 0)), "Sharpe": float(res.get("Sharpe", 0)),
        "CAGR": float(res.get("CAGR", 0)), "MDD": float(res.get("MDD", 0)),
    }


STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "UST"}


def _get_all_universe_coins(top_n: int = 10, min_months: int = 1) -> list[str]:
    """historical_universe.json에서 top_n 안에 min_months 이상 등장한 코인.
    anchor (2020-10) 이후 구간만 카운트. stablecoin은 제외.
    default: anchor 이후 top-10에 한번이라도 등장한 코인 전부."""
    import json
    for p in (
        os.path.join(HERE, "..", "data", "historical_universe.json"),
        os.path.join(HERE, "..", "..", "data", "historical_universe.json"),
        os.path.join(HERE, "..", "..", "..", "data", "historical_universe.json"),
    ):
        ap = os.path.abspath(p)
        if os.path.exists(ap):
            with open(ap) as f:
                raw = json.load(f)
            counts: dict[str, int] = {}
            for ds, tickers in raw.items():
                if ds < ANCHOR:
                    continue
                for t in tickers[:top_n]:
                    c = t.replace("-USD", "")
                    if c in STABLECOINS:
                        continue
                    counts[c] = counts.get(c, 0) + 1
            return sorted([c for c, n in counts.items() if n >= min_months])
    return []


# ─── LOYO ───
def _year_range(exclude_year: int) -> list[tuple[str, str]]:
    """연도 제외 → 남은 구간 리스트. 제외 연도 전후를 분리."""
    segments = []
    yr_start = f"{exclude_year}-01-01"
    yr_end = f"{exclude_year}-12-31"
    if ANCHOR < yr_start:
        segments.append((ANCHOR, f"{exclude_year - 1}-12-31"))
    if yr_end < FULL_END:
        segments.append((f"{exclude_year + 1}-01-01", FULL_END))
    return segments


def run_loyo_single(args_tuple):
    """(ensemble_row_dict, exclude_year) → result dict."""
    row, year = args_tuple
    members = row["members"].split(";")
    asset = row["asset"]
    lev = float(row["lev"])

    segments = _year_range(year)
    if not segments:
        return None

    seg_eqs = []
    for seg_start, seg_end in segments:
        try:
            res = _run_ensemble(members, asset, lev, seg_start, seg_end)
            seg_eqs.append(res)
        except Exception:
            return None

    if len(seg_eqs) == 1:
        met = seg_eqs[0]
    else:
        met = {k: np.mean([s[k] for s in seg_eqs]) for k in ("Cal", "Sharpe", "CAGR", "MDD")}

    return {
        "ensemble_tag": row["ensemble_tag"],
        "exclude_year": year,
        "Cal": met["Cal"], "Sharpe": met["Sharpe"],
        "CAGR": met["CAGR"], "MDD": met["MDD"],
    }


# ─── LOAO ───
def run_loao_single(args_tuple):
    """(ensemble_row_dict, exclude_coin) → result dict."""
    row, coin = args_tuple
    members = row["members"].split(";")
    asset = row["asset"]
    lev = float(row["lev"])

    try:
        met = _run_ensemble(members, asset, lev, ANCHOR, FULL_END, exclude_coin=coin)
    except Exception:
        return None

    return {
        "ensemble_tag": row["ensemble_tag"],
        "exclude_coin": coin,
        "Cal": met["Cal"], "Sharpe": met["Sharpe"],
        "CAGR": met["CAGR"], "MDD": met["MDD"],
    }


def compute_per_ensemble_stability(baseline_df: pd.DataFrame, perturbed_df: pd.DataFrame,
                                   group_col: str, metric: str = "Cal") -> pd.DataFrame:
    """앙상블별 섭동(연도/코인 제외)에 대한 통계.

    아웃라이어(worst 1건) 제거한 평균·CV를 함께 보고 → "항상 이기는" 기준이
    아니라 "아웃라이어 제외 후에도 장기적으로 우수한" 앙상블을 고르기 위함.
    """
    import numpy as np
    base_map = baseline_df.set_index("ensemble_tag")[metric].to_dict()
    rows = []
    for tag, grp in perturbed_df.groupby("ensemble_tag"):
        vals = grp[metric].astype(float).values
        if len(vals) < 2:
            continue
        base = float(base_map.get(tag, np.nan))
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        cv = float(sigma / abs(mu)) if mu != 0 else 0.0
        # 최저 1건 제외 trimmed 평균
        trimmed = float(np.mean(sorted(vals)[1:])) if len(vals) >= 3 else mu
        # max-drop = baseline - worst (양수=baseline 대비 손실폭)
        worst = float(np.min(vals))
        max_drop = base - worst if not np.isnan(base) else 0.0
        # drop 목록에서 worst(제외 시 가장 큰 drop) 하나 뺀 평균 drop
        drops = sorted([base - v for v in vals], reverse=True)
        trimmed_drop = float(np.mean(drops[1:])) if len(drops) >= 3 else float(np.mean(drops))
        rows.append({
            "ensemble_tag": tag,
            f"{group_col}_baseline_{metric}": round(base, 4),
            f"{group_col}_mean_{metric}": round(mu, 4),
            f"{group_col}_trimmed_{metric}": round(trimmed, 4),
            f"{group_col}_worst_{metric}": round(worst, 4),
            f"{group_col}_max_drop": round(max_drop, 4),
            f"{group_col}_trimmed_drop": round(trimmed_drop, 4),
            f"{group_col}_cv": round(cv, 4),
            f"{group_col}_n": len(vals),
        })
    return pd.DataFrame(rows)


def compute_rank_stability(baseline_df: pd.DataFrame, perturbed_df: pd.DataFrame,
                           group_col: str, metric: str = "Cal") -> pd.DataFrame:
    """baseline 순위 vs 제외 후 순위 Spearman."""
    base_rank = baseline_df.set_index("ensemble_tag")[metric].rank(ascending=False)
    rows = []
    for label, grp in perturbed_df.groupby(group_col):
        pert_rank = grp.set_index("ensemble_tag")[metric].rank(ascending=False)
        common = base_rank.index.intersection(pert_rank.index)
        if len(common) < 3:
            continue
        rho, pval = spearmanr(base_rank.loc[common], pert_rank.loc[common])
        top5_base = set(base_rank.loc[common].nsmallest(5).index)
        top5_pert = set(pert_rank.loc[common].nsmallest(5).index)
        overlap = len(top5_base & top5_pert)
        rows.append({
            group_col: label,
            "spearman_rho": round(rho, 4),
            "p_value": round(pval, 6),
            "top5_overlap": overlap,
            "n_strategies": len(common),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3-dir", required=True, help="phase3 output dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    write_manifest(manifest_path, {"status": "running", "stage": "robustness_check"})

    top_ens = _load_top_ensembles(args.p3_dir, args.top_n)
    if top_ens.empty:
        write_manifest(manifest_path, {"status": "done", "note": "no ensembles"})
        print("No ensembles found.")
        return

    baseline = top_ens[["ensemble_tag", "Cal", "Sharpe", "CAGR", "MDD"]].copy()
    ens_rows = top_ens.to_dict("records")
    n_ens = len(ens_rows)

    # ── LOYO ──
    print(f"LOYO: {n_ens} ensembles × {len(LOYO_YEARS)} years = {n_ens * len(LOYO_YEARS)} runs")
    loyo_tasks = [(r, y) for r in ens_rows for y in LOYO_YEARS]
    loyo_results = []
    with Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_loyo_single, loyo_tasks), 1):
            if res:
                loyo_results.append(res)
            if i % 20 == 0 or i == len(loyo_tasks):
                print(f"  LOYO [{i}/{len(loyo_tasks)}] ok={len(loyo_results)}")

    loyo_df = pd.DataFrame(loyo_results) if loyo_results else pd.DataFrame()
    if not loyo_df.empty:
        atomic_write_csv(loyo_df, os.path.join(args.out_dir, "loyo_raw.csv"))
        loyo_stability = compute_rank_stability(baseline, loyo_df, "exclude_year")
        atomic_write_csv(loyo_stability, os.path.join(args.out_dir, "loyo_stability.csv"))
        print(f"LOYO stability:\n{loyo_stability.to_string(index=False)}")
        loyo_per_ens = compute_per_ensemble_stability(baseline, loyo_df, "loyo")
        atomic_write_csv(loyo_per_ens, os.path.join(args.out_dir, "loyo_per_ensemble.csv"))

    # ── LOAO ──
    all_coins = _get_all_universe_coins()
    print(f"LOAO: {n_ens} ensembles × {len(all_coins)} coins = {n_ens * len(all_coins)} runs")
    print(f"  Coins: {', '.join(all_coins)}")
    loao_tasks = [(r, c) for r in ens_rows for c in all_coins]
    loao_results = []
    with Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_loao_single, loao_tasks), 1):
            if res:
                loao_results.append(res)
            if i % 50 == 0 or i == len(loao_tasks):
                print(f"  LOAO [{i}/{len(loao_tasks)}] ok={len(loao_results)}")

    loao_df = pd.DataFrame(loao_results) if loao_results else pd.DataFrame()
    if not loao_df.empty:
        atomic_write_csv(loao_df, os.path.join(args.out_dir, "loao_raw.csv"))
        loao_stability = compute_rank_stability(baseline, loao_df, "exclude_coin")
        atomic_write_csv(loao_stability, os.path.join(args.out_dir, "loao_stability.csv"))
        print(f"LOAO stability:\n{loao_stability.to_string(index=False)}")
        loao_per_ens = compute_per_ensemble_stability(baseline, loao_df, "loao")
        atomic_write_csv(loao_per_ens, os.path.join(args.out_dir, "loao_per_ensemble.csv"))

    # ── HHI (수익 집중도) — baseline equity에서 추출 ──
    # (equity 재실행 없이 baseline 메트릭 기반 간이 산출은 불가 → LOAO 결과로 대체)
    # LOAO에서 특정 코인 제외 시 Cal 하락폭이 큰 앙상블 = 해당 코인 집중도 높음

    # ── Summary ──
    summary_rows = []
    if not loyo_df.empty:
        summary_rows.append({
            "test": "LOYO",
            "n_perturbations": len(LOYO_YEARS),
            "mean_spearman": round(loyo_stability["spearman_rho"].mean(), 4) if not loyo_stability.empty else 0,
            "min_spearman": round(loyo_stability["spearman_rho"].min(), 4) if not loyo_stability.empty else 0,
            "mean_top5_overlap": round(loyo_stability["top5_overlap"].mean(), 2) if not loyo_stability.empty else 0,
        })
    if not loao_df.empty:
        summary_rows.append({
            "test": "LOAO",
            "n_perturbations": len(all_coins),
            "mean_spearman": round(loao_stability["spearman_rho"].mean(), 4) if not loao_stability.empty else 0,
            "min_spearman": round(loao_stability["spearman_rho"].min(), 4) if not loao_stability.empty else 0,
            "mean_top5_overlap": round(loao_stability["top5_overlap"].mean(), 2) if not loao_stability.empty else 0,
        })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        atomic_write_csv(summary_df, os.path.join(args.out_dir, "summary.csv"))
        print(f"\nSummary:\n{summary_df.to_string(index=False)}")

    write_manifest(manifest_path, {
        "status": "done", "stage": "robustness_check",
        "n_ensembles": n_ens,
        "loyo_runs": len(loyo_results),
        "loao_runs": len(loao_results),
        "loao_coins": len(all_coins),
    })
    print("Done.")


if __name__ == "__main__":
    main()
