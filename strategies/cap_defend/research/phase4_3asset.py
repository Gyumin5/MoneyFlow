#!/usr/bin/env python3
"""Phase-4 three-asset portfolio search (single-stage grid, 2026-04-15).

- 입력: phase3_ensembles/{spot_top.csv, fut_top.csv}
- 주식: V17 (고정, run_3asset_grid.load_stock_v17 재사용)
- Weight grid: stock=60% 고정, spot ∈ {20,25,30,35,40}%, fut = 40-spot (5pp).
- Band grid: {3, 5, 8, 10, 15}%.
- 자산별 리밸 비용 (편도): spot 40bps / fut 4bps / stock 10bps. band hit 시
  자산별 drift × 비용을 합산 차감.
- 출력:
    phase4_3asset/raw.csv
    phase4_3asset/tables.json (필터×정렬 테이블, Cal/Sh/CAGR/MDD 포함)
    phase4_3asset/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import (
    FULL_END, atomic_write_csv, build_trace, equity_metrics, parse_tag,
    preload_futures, run_single_target, run_spot_ensemble, write_manifest,
)

OUT_DIR = os.path.join(HERE, "phase4_3asset")
START = "2020-10-01"

STOCK_FIXED = 0.60       # 주식 60% 고정
# spot ∈ {20,25,30,35,40}, fut = 40 - spot (5pp 단위), 합=100.
COARSE_STOCK = [STOCK_FIXED]
COARSE_SPOT = [0.20, 0.25, 0.30, 0.35, 0.40]
COARSE_FUT = [round(0.40 - s, 4) for s in COARSE_SPOT]
COARSE_BANDS = [0.03, 0.05, 0.08, 0.10, 0.15]
# sleeve-relative band: band = weight × ratio. 자산별 독립 band.
SLEEVE_RATIOS = [0.20, 0.30, 0.40, 0.50]

# 자산별 리밸 비용 (bps, 편도). band hit 시 자산별 drift 규모에 비례.
REBAL_COST_BPS_BY_ASSET = {"st": 10.0, "sp": 40.0, "fut": 4.0}


def _valid_weights(st: float, sp: float, fu: float) -> bool:
    return (abs(st - STOCK_FIXED) < 1e-9
            and abs((st + sp + fu) - 1.0) < 1e-9
            and min(st, sp, fu) >= 0.0)


def valid_weight_grid(stocks: list[float], spots: list[float],
                      futs: list[float]) -> list[tuple[float, float, float]]:
    out = []
    for st, sp, fu in product(stocks, spots, futs):
        if _valid_weights(st, sp, fu):
            out.append((round(st, 4), round(sp, 4), round(fu, 4)))
    # dedup
    return sorted(set(out))


def build_ensemble_full_equity(ens_row: pd.Series) -> pd.Series:
    """ensemble row (spot_top / fut_top) → START ~ FULL_END 단일 equity.

    Phase-3 재현: member parse → combine_targets → SingleAccountEngine.run (fut)
    혹은 member equity 가중 평균 (spot).
    """
    asset = ens_row["asset"]
    lev = float(ens_row["lev"])
    members = str(ens_row["members"]).split(";")
    k = len(members)
    w_each = min(1.0 / k, 0.5)
    total = w_each * k
    weights = {m: w_each / total for m in members}

    if asset == "fut":
        from futures_ensemble_engine import SingleAccountEngine, combine_targets
        data = preload_futures()
        bars_1h, funding_1h = data["1h"]
        all_dates_1h = bars_1h["BTC"].index
        traces = {}
        for m_tag in members:
            meta = parse_tag(m_tag)
            cfg = {"interval": meta["interval"], "sma": meta["sma"],
                   "ms": meta["ms"], "ml": meta["ml"],
                   "vol_mode": meta["vol_mode"], "vol_thr": meta["vol_thr"],
                   "snap": meta["snap"]}
            tr = build_trace("fut", cfg, lev, START, end=FULL_END)["trace"]
            traces[m_tag] = tr
        dates = all_dates_1h[(all_dates_1h >= START) & (all_dates_1h <= FULL_END)]
        combined = combine_targets(traces, weights, dates)
        engine = SingleAccountEngine(
            bars_1h, funding_1h,
            leverage=lev, leverage_mode="fixed", per_coin_leverage_mode="none",
            stop_kind="none", stop_pct=0.0, stop_lookback_bars=0, stop_gate="always",
        )
        res = engine.run(combined)
        eq = res.get("_equity")
        if eq is not None and not isinstance(eq, pd.Series):
            eq = pd.Series(eq)
        return eq
    # spot — Phase-3와 동일 경로: coin_live_engine MEMBERS 주입 단일계정 앙상블
    member_cfgs = {}
    for m_tag in members:
        meta = parse_tag(m_tag)
        member_cfgs[m_tag] = {
            "interval": meta["interval"], "sma": meta["sma"],
            "ms": meta["ms"], "ml": meta["ml"],
            "vol_mode": meta["vol_mode"], "vol_thr": meta["vol_thr"],
            "snap": meta["snap"],
        }
    r = run_spot_ensemble(member_cfgs, weights, START,
                          end=FULL_END, want_equity=True)
    eq = r.get("_equity")
    if eq is None:
        raise RuntimeError(f"no equity for spot ensemble {members}")
    if not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    return eq


def mix_eq(series_dict: dict, weights: dict, band: float | dict,
           cost_bps_by_asset: dict | None = None,
           init: float = 1.0) -> pd.Series:
    """band: float → 모든 자산 동일 절대 band,
    dict → 자산별 독립 band (예: {"st":0.18,"sp":0.10,"fut":0.015}).
    dict 모드에서는 '어떤' 자산이든 자기 band 초과하면 전체 리밸."""
    def _strip_tz(s: pd.Series) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s = s.copy()
            s.index = s.index.tz_localize(None)
        return s
    df = pd.concat([_strip_tz(s).rename(k) for k, s in series_dict.items()],
                   axis=1).dropna()
    rets = {k: df[k].pct_change().fillna(0).values for k in series_dict}
    cur = dict(weights)
    eq = init
    out = np.empty(len(df))
    keys = list(series_dict.keys())
    cost_map = cost_bps_by_asset or REBAL_COST_BPS_BY_ASSET
    band_dict = band if isinstance(band, dict) else {k: band for k in keys}
    for i in range(len(df)):
        total = 0.0
        vals = {}
        for k in keys:
            v = eq * cur[k] * (1 + rets[k][i])
            vals[k] = v
            total += v
        eq = total
        if eq > 0:
            for k in keys:
                cur[k] = vals[k] / eq
        breached = any(abs(cur[k] - weights[k]) >= band_dict.get(k, 1.0)
                       for k in keys)
        if breached:
            cost_frac = 0.0
            for k in keys:
                d = abs(cur[k] - weights[k])
                cost_frac += d * (cost_map.get(k, 0.0) / 10000.0)
            cur = dict(weights)
            eq = eq * (1.0 - cost_frac)
        out[i] = eq
    return pd.Series(out, index=df.index)


def run_mix(stock_eq: pd.Series, spot_eq: pd.Series, fut_eq: pd.Series,
            stock_id: str, spot_id: str, fut_id: str, fut_lev: float,
            weights: tuple[float, float, float], band: float | dict,
            band_mode: str = "abs") -> dict:
    st_w, sp_w, fu_w = weights
    mix = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_eq},
                 {"st": st_w, "sp": sp_w, "fut": fu_w}, band)
    m = equity_metrics(mix)
    if isinstance(band, dict):
        band_label = f"st{band['st']:.3f}_sp{band['sp']:.3f}_fu{band['fut']:.3f}"
    else:
        band_label = f"{band:.4f}"
    return {
        "stock": stock_id, "spot": spot_id, "fut": fut_id, "fut_lev": fut_lev,
        "st_w": st_w, "sp_w": sp_w, "fu_w": fu_w,
        "band": band_label, "band_mode": band_mode,
        "Cal": round(m["Cal"], 4),
        "Sh": round(m["Sh"], 4),
        "CAGR": round(m["CAGR"], 4),
        "MDD": round(m["MDD"], 4),
    }


def build_tables(df: pd.DataFrame) -> dict:
    """12 테이블: 3 필터 × 4 정렬."""
    tables = {}
    filters = {
        "all": df,
        "mdd25": df[df["MDD"] >= -0.25],
        "mdd35": df[df["MDD"] >= -0.35],
    }
    if "fut_lev" in df.columns:
        for lev_val in sorted(df["fut_lev"].dropna().unique()):
            lev_tag = f"L{int(lev_val)}"
            sub = df[df["fut_lev"] == lev_val]
            filters[lev_tag] = sub
            filters[f"{lev_tag}_mdd25"] = sub[sub["MDD"] >= -0.25]
            filters[f"{lev_tag}_mdd35"] = sub[sub["MDD"] >= -0.35]
    cols = [c for c in ["stock", "spot", "fut", "fut_lev", "st_w", "sp_w", "fu_w", "band",
                        "Cal", "Sh", "CAGR", "MDD", "CalxCAGR", "rank_sum"] if c in df.columns]
    for suffix, sub in filters.items():
        if sub.empty:
            continue
        tables[f"Cal_{suffix}"] = sub.sort_values("Cal", ascending=False).head(15)[cols].to_dict("records")
        tables[f"Sh_{suffix}"] = sub.sort_values("Sh", ascending=False).head(15)[cols].to_dict("records")
        tables[f"CAGR_{suffix}"] = sub.sort_values("CAGR", ascending=False).head(15)[cols].to_dict("records")
        tables[f"CalxCAGR_{suffix}"] = sub.sort_values("CalxCAGR", ascending=False).head(15)[cols].to_dict("records")
        if "rank_sum" in sub:
            tables[f"rank_{suffix}"] = sub.sort_values("rank_sum").head(15)[cols].to_dict("records")
    return tables


def _load_stock_v17() -> pd.Series:
    # run_3asset_grid.load_stock_v17 재사용
    import run_3asset_grid as r3
    return r3.load_stock_v17()


def _resolve_tops(spot_top_csv: str, fut_top_csv: str
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    sp = pd.read_csv(spot_top_csv) if os.path.exists(spot_top_csv) else pd.DataFrame()
    fu = pd.read_csv(fut_top_csv) if os.path.exists(fut_top_csv) else pd.DataFrame()
    return sp, fu


def add_rank_sum(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CalxCAGR"] = out["Cal"] * out["CAGR"]
    out["rank_Cal"] = out["Cal"].rank(ascending=False, method="min")
    out["rank_Sh"] = out["Sh"].rank(ascending=False, method="min")
    out["rank_CAGR"] = out["CAGR"].rank(ascending=False, method="min")
    out["rank_CalxCAGR"] = out["CalxCAGR"].rank(ascending=False, method="min")
    out["rank_sum"] = out["rank_Cal"] + out["rank_Sh"] + out["rank_CAGR"] + out["rank_CalxCAGR"]
    return out


def refine_weights(coarse_rows: pd.DataFrame, step: float = 0.05,
                   pp: int = 5) -> list[tuple[float, float, float]]:
    """coarse top-10 기준 ±pp% 주변 refine."""
    seen = set()
    out = []
    for _, r in coarse_rows.head(10).iterrows():
        for dst in (-pp, 0, pp):
            for dsp in (-pp, 0, pp):
                for dfu in (-pp, 0, pp):
                    st = round(r["st_w"] + dst / 100, 4)
                    sp = round(r["sp_w"] + dsp / 100, 4)
                    fu = round(r["fu_w"] + dfu / 100, 4)
                    if min(st, sp, fu) < 0 or max(st, sp, fu) > 1.0:
                        continue
                    if abs((st + sp + fu) - 1.0) > 1e-9:
                        continue
                    if not _valid_weights(st, sp, fu):
                        continue
                    key = (st, sp, fu)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-top", required=True)
    ap.add_argument("--fut-top", required=True)
    ap.add_argument("--stock-tag", default="V17")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--top-n", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    write_manifest(manifest_path, {"status": "running", "stage": "phase4_3asset"})

    spot_df, fut_df = _resolve_tops(args.spot_top, args.fut_top)
    if spot_df.empty or fut_df.empty:
        write_manifest(manifest_path, {"status": "done", "stage": "phase4_3asset",
                                       "note": "missing spot_top or fut_top"})
        print("spot_top or fut_top empty. Exit.")
        return

    print("Loading V17 stock equity...")
    stock_eq = _load_stock_v17()

    print(f"Building {len(spot_df)} spot + {len(fut_df)} fut ensembles full equity...")
    spot_eqs = {}
    for _, r in spot_df.iterrows():
        print(f"  spot ensemble {r['ensemble_tag']}")
        spot_eqs[r["ensemble_tag"]] = build_ensemble_full_equity(r)
    fut_eqs = {}
    fut_lev_map = {}
    for _, r in fut_df.iterrows():
        print(f"  fut ensemble {r['ensemble_tag']} (L{int(r['lev'])})")
        fut_eqs[r["ensemble_tag"]] = build_ensemble_full_equity(r)
        fut_lev_map[r["ensemble_tag"]] = float(r["lev"])

    # ── grid: weight × (absolute band + sleeve-relative band) ──
    weights = valid_weight_grid(COARSE_STOCK, COARSE_SPOT, COARSE_FUT)

    # sleeve-relative bands: band_k = weight_k × ratio
    sleeve_band_specs: list[tuple[dict, float]] = []
    for ratio in SLEEVE_RATIOS:
        for w in weights:
            st_w, sp_w, fu_w = w
            band_d = {"st": round(st_w * ratio, 4),
                      "sp": round(sp_w * ratio, 4),
                      "fut": round(fu_w * ratio, 4)}
            sleeve_band_specs.append((band_d, ratio))

    n_abs = len(weights) * len(COARSE_BANDS)
    n_sleeve = len(sleeve_band_specs)
    print(f"Grid: {len(weights)} weight tuples × "
          f"({len(COARSE_BANDS)} abs bands + {len(SLEEVE_RATIOS)} sleeve ratios) "
          f"× {len(spot_eqs)} spot × {len(fut_eqs)} fut")
    print(f"  abs configs={n_abs}, sleeve configs={n_sleeve}")
    rows = []
    for sp_id, sp_eq in spot_eqs.items():
        for fu_id, fu_eq in fut_eqs.items():
            # absolute bands
            for w in weights:
                for band in COARSE_BANDS:
                    try:
                        rows.append(run_mix(stock_eq, sp_eq, fu_eq,
                                            args.stock_tag, sp_id, fu_id,
                                            fut_lev_map[fu_id], w, band,
                                            band_mode="abs"))
                    except Exception as e:
                        print(f"FAIL {sp_id}/{fu_id} {w} band={band}: {e}")
            # sleeve-relative bands
            for w in weights:
                st_w, sp_w, fu_w = w
                for ratio in SLEEVE_RATIOS:
                    band_d = {"st": round(st_w * ratio, 4),
                              "sp": round(sp_w * ratio, 4),
                              "fut": round(fu_w * ratio, 4)}
                    try:
                        rows.append(run_mix(stock_eq, sp_eq, fu_eq,
                                            args.stock_tag, sp_id, fu_id,
                                            fut_lev_map[fu_id], w, band_d,
                                            band_mode="sleeve"))
                    except Exception as e:
                        print(f"FAIL {sp_id}/{fu_id} {w} sleeve={ratio}: {e}")
    if not rows:
        write_manifest(manifest_path, {
            "status": "done", "stage": "phase4_3asset",
            "note": "no successful mix runs"})
        print("No successful mix runs. Exit.")
        return
    all_df = add_rank_sum(pd.DataFrame(rows))
    atomic_write_csv(all_df, os.path.join(args.out_dir, "raw.csv"))
    with open(os.path.join(args.out_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(build_tables(all_df), f, ensure_ascii=False, indent=2, default=str)

    write_manifest(manifest_path, {
        "status": "done", "stage": "phase4_3asset",
        "n_rows": int(len(all_df)),
        "n_spot": len(spot_eqs), "n_fut": len(fut_eqs),
    })
    print(f"Done. rows={len(all_df)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
