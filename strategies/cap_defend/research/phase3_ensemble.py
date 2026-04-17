#!/usr/bin/env python3
"""Phase-3 single-account ensemble (iter_refine 후속, 단일 앵커).

핵심 설계 (2026-04-15 확정):
- 입력: Phase-2 plateau survivors (plateau_ok=True)
- Pool 선정 전 CAGR 바닥 필터: CAGR >= CAGR_FLOOR_PER_LEV × max(1, lev)
- Pool: (bucket, interval) 그룹 내 mCal/mSh/mCAGR/rank_sum_m 각 top-N 합집합
- 조합: bucket 내 k=1,2,3. D+4h 혼합시 snap 동일 제약 (4h_snap == D_snap 그대로 일치).
  - k=1은 단순 참조 (이미 phase2 survivor 메트릭 그대로).
  - k>=2만 실제 앙상블 백테스트.
- Corr 필터 전면 제거 (다양성은 pool top-N 합집합의 지표 다양화로 확보).
- 가중치: EW (동일).
- 엔진: futures_ensemble_engine.SingleAccountEngine + combine_targets (fut),
        run_spot_ensemble (spot).
- 출력:
    phase3_ensembles/all_combos.csv
    phase3_ensembles/spot_top.csv, fut_top.csv
    phase3_ensembles/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import (
    FULL_END, atomic_write_csv, build_trace, parse_tag, preload_futures,
    run_spot_ensemble, write_manifest,
)

OUT_DIR = os.path.join(HERE, "phase3_ensembles")
ANCHOR = "2020-10-01"
CAGR_FLOOR_PER_LEV = 0.30  # CAGR >= 30% × lev


def classify_bucket(row: pd.Series) -> str:
    if row["asset"] == "spot":
        return "spot"
    return f"fut_L{int(row['lev'])}"


def ensemble_tag(bucket: str, members: list[str]) -> str:
    h = hashlib.sha1("|".join(sorted(members)).encode()).hexdigest()[:8]
    return f"ENS_{bucket}_k{len(members)}_{h}"


IV_HOURS = {"D": 24, "4h": 4, "2h": 2, "1h": 1}

# diversity proxy: 같은 interval + 모든 수치축(sma/ms/ml/snap) relative diff < NEAR_DUP_TOL → near-duplicate
NEAR_DUP_TOL = 0.15


def _snap_hours(iv: str, snap: int) -> int:
    return int(snap) * IV_HOURS.get(str(iv), 0)


def _rel_diff(a: float, b: float) -> float:
    m = max(abs(a), abs(b))
    return abs(a - b) / m if m > 0 else 0.0


def _is_near_duplicate(meta1: dict, meta2: dict, tol: float = NEAR_DUP_TOL) -> bool:
    """같은 interval + 4축 상대차 모두 tol 미만이면 near-duplicate."""
    if meta1.get("interval") != meta2.get("interval"):
        return False
    for ax in ("sma", "ms", "ml", "snap"):
        if _rel_diff(float(meta1.get(ax, 0)), float(meta2.get(ax, 0))) >= tol:
            return False
    return True


def enumerate_combos(df: pd.DataFrame, sizes=(2, 3)) -> list[dict]:
    """bucket 내 k=1,2,3 조합.
    - 혼합 interval 조합은 wall-clock snap 일치만 허용 (D snap=30 == 4h snap=180).
    - k>=2 에서 near-duplicate 쌍 포함 combo 는 제외 (diversity gate).
    """
    rows: list[dict] = []
    for bucket, g in df.groupby("bucket"):
        tag_to_iv = dict(zip(g["tag"], g["interval"]))
        tag_to_snap_h = {t: _snap_hours(tag_to_iv[t], s)
                         for t, s in zip(g["tag"], g["snap"].astype(int))}
        tag_to_meta: dict[str, dict] = {}
        for t in g["tag"]:
            try:
                tag_to_meta[t] = parse_tag(str(t))
            except Exception:
                tag_to_meta[t] = {}
        tags = list(g["tag"])
        for k in sizes:
            if len(tags) < k:
                continue
            for combo in itertools.combinations(tags, k):
                ivs = {tag_to_iv[t] for t in combo}
                if len(ivs) > 1:
                    snap_hs = {tag_to_snap_h[t] for t in combo}
                    if len(snap_hs) > 1:
                        continue  # 혼합인데 wall-clock snap 불일치 → skip
                if k >= 2:
                    dup = False
                    for a, b in itertools.combinations(combo, 2):
                        if _is_near_duplicate(tag_to_meta.get(a, {}),
                                              tag_to_meta.get(b, {})):
                            dup = True; break
                    if dup:
                        continue
                rows.append({"bucket": str(bucket), "k": k, "members": list(combo)})
    return rows


_EMPTY_YC = {"trimmed_year_cal": 0.0, "trimmed_year_ret": 0.0,
             "year_cal_cv": 0.0, "year_ret_cv": 0.0, "n_years": 0}


def _yearly_consistency(eq: pd.Series) -> dict:
    """equity 시계열 → 연도별 Cal/return 분산 지표.

    아웃라이어(최저 1년) 제거 후 평균(trimmed_year_*)과 연도간 변동계수(CV)를 반환.
    "항상 이기는" 기준(negative_years, min_year_cal) 대신 "아웃라이어 제외 후
    장기 성과와 연도간 일관성"을 측정한다."""
    import numpy as np
    from phase_common import equity_metrics
    if eq is None or len(eq) < 30:
        return dict(_EMPTY_YC)
    eq = pd.Series(eq).dropna()
    if not isinstance(eq.index, pd.DatetimeIndex):
        try:
            eq.index = pd.to_datetime(eq.index)
        except Exception:
            return dict(_EMPTY_YC)
    year_cals, year_rets = [], []
    for y in sorted(set(eq.index.year)):
        sub = eq[eq.index.year == y]
        if len(sub) < 10 or sub.iloc[0] <= 0:
            continue
        year_rets.append(float(sub.iloc[-1] / sub.iloc[0] - 1))
        m = equity_metrics(sub)
        year_cals.append(float(m.get("Cal", 0.0)))
    if not year_cals:
        return dict(_EMPTY_YC)

    # trimmed mean: 최저 1년 제외 (연도 수 ≤2면 단순 평균)
    def _trim_mean(xs):
        if len(xs) <= 2:
            return float(np.mean(xs))
        return float(np.mean(sorted(xs)[1:]))

    def _cv(xs):
        if len(xs) < 2:
            return 0.0
        mu = float(np.mean(xs))
        if mu == 0:
            return 0.0
        return float(np.std(xs) / abs(mu))

    return {"trimmed_year_cal": round(_trim_mean(year_cals), 4),
            "trimmed_year_ret": round(_trim_mean(year_rets), 4),
            "year_cal_cv": round(_cv(year_cals), 4),
            "year_ret_cv": round(_cv(year_rets), 4),
            "n_years": len(year_cals)}


_SUMMARY_DF: pd.DataFrame | None = None


def _run_combo_worker(combo: dict) -> dict:
    assert _SUMMARY_DF is not None
    return run_combo(combo, _SUMMARY_DF)


def run_combo(combo: dict, summary_df: pd.DataFrame) -> dict:
    members = combo["members"]
    bucket = combo["bucket"]
    ens_tag = ensemble_tag(bucket, members)
    rows = summary_df[summary_df["tag"].isin(members)]
    assets = rows["asset"].unique()
    if len(assets) != 1:
        return {"ensemble_tag": ens_tag, "status": "skip_mixed_asset"}
    asset = str(assets[0])
    lev = float(rows["lev"].iloc[0])
    k = len(members)

    # k=1: phase2 survivor 메트릭 그대로 복사
    if k == 1:
        r = rows.iloc[0]
        return {
            "ensemble_tag": ens_tag, "bucket": bucket, "asset": asset,
            "lev": lev, "k": 1, "members": ";".join(members),
            "Cal": float(r.get("mCal", 0)), "Sharpe": float(r.get("mSh", 0)),
            "CAGR": float(r.get("mCAGR", 0)), "MDD": float(r.get("wMDD", 0)),
            "status": "ok",
        }

    # k>=2: 실제 앙상블 백테스트
    weights = {m: 1.0 / k for m in members}
    eq_series = None
    try:
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
                tr = build_trace("fut", cfg, lev, ANCHOR, end=FULL_END)["trace"]
                traces[m_tag] = tr
            dates = all_dates_1h[(all_dates_1h >= ANCHOR) & (all_dates_1h <= FULL_END)]
            combined = combine_targets(traces, weights, dates)
            engine = SingleAccountEngine(
                bars_1h, funding_1h,
                leverage=lev, leverage_mode="fixed", per_coin_leverage_mode="none",
                stop_kind="none", stop_pct=0.0, stop_lookback_bars=0,
                stop_gate="always",
            )
            res = engine.run(combined)
            met = {"Cal": float(res.get("Cal", 0)),
                   "Sharpe": float(res.get("Sharpe", 0)),
                   "CAGR": float(res.get("CAGR", 0)),
                   "MDD": float(res.get("MDD", 0))}
            eq_series = res.get("_equity")
        else:
            member_cfgs = {}
            for m_tag in members:
                meta = parse_tag(m_tag)
                member_cfgs[m_tag] = {
                    "interval": meta["interval"], "sma": meta["sma"],
                    "ms": meta["ms"], "ml": meta["ml"],
                    "vol_mode": meta["vol_mode"], "vol_thr": meta["vol_thr"],
                    "snap": meta["snap"],
                }
            r = run_spot_ensemble(member_cfgs, weights, ANCHOR,
                                  end=FULL_END, want_equity=True)
            met = {"Cal": float(r.get("Cal", 0)), "Sharpe": float(r.get("Sh", 0)),
                   "CAGR": float(r.get("CAGR", 0)), "MDD": float(r.get("MDD", 0))}
            eq_series = r.get("_equity")
    except Exception as e:
        return {"ensemble_tag": ens_tag, "bucket": bucket, "asset": asset,
                "lev": lev, "k": k, "members": ";".join(members),
                "status": f"error:{str(e)[:120]}"}

    yc = _yearly_consistency(eq_series) if eq_series is not None else dict(_EMPTY_YC)

    return {
        "ensemble_tag": ens_tag, "bucket": bucket, "asset": asset, "lev": lev,
        "k": k, "members": ";".join(members),
        **met, **yc, "status": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-summary", required=True)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--pool-per-metric", type=int, default=5)
    ap.add_argument("--cagr-floor-per-lev", type=float, default=CAGR_FLOOR_PER_LEV)
    ap.add_argument("--processes", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    write_manifest(manifest_path, {"status": "running", "stage": "phase3_ensemble"})

    df = pd.read_csv(args.phase2_summary)
    if "plateau_ok" in df.columns:
        df = df[df["plateau_ok"] == True].copy()
    if df.empty:
        write_manifest(manifest_path, {"status": "done", "stage": "phase3_ensemble",
                                       "note": "no survivors"})
        print("No survivors. Exit.")
        return

    df["bucket"] = df.apply(classify_bucket, axis=1)
    if "interval" not in df.columns:
        df["interval"] = df["tag"].apply(lambda t: parse_tag(str(t))["interval"])

    # CAGR floor — pool 전 필터
    floor = args.cagr_floor_per_lev * df["lev"].astype(float).clip(lower=1.0)
    n_before = len(df)
    df = df[df["mCAGR"].astype(float) >= floor].copy()
    print(f"CAGR floor: {n_before} → {len(df)} (>= {args.cagr_floor_per_lev*100:.0f}%×lev)")
    if df.empty:
        write_manifest(manifest_path, {"status": "done", "stage": "phase3_ensemble",
                                       "note": "all filtered by CAGR floor"})
        print("All filtered by CAGR floor.")
        return

    # rank_sum_m
    df["rank_mCal"] = df.groupby(["bucket", "interval"])["mCal"].rank(ascending=False, method="min")
    df["rank_mSh"] = df.groupby(["bucket", "interval"])["mSh"].rank(ascending=False, method="min")
    df["rank_mCAGR"] = df.groupby(["bucket", "interval"])["mCAGR"].rank(ascending=False, method="min")
    df["rank_sum_m"] = df["rank_mCal"] + df["rank_mSh"] + df["rank_mCAGR"]

    # Pool: (bucket, interval)별 각 지표 top-N 합집합
    selected: set[str] = set()
    pool_stats: dict = {}
    for (bucket, interval), sub in df.groupby(["bucket", "interval"]):
        picks: set[str] = set()
        for col, asc in (("mCal", False), ("mSh", False),
                         ("mCAGR", False), ("rank_sum_m", True)):
            picks.update(sub.sort_values(col, ascending=asc).head(
                args.pool_per_metric)["tag"].tolist())
        pool_stats[(bucket, interval)] = len(picks)
        selected.update(picks)
    df = df[df["tag"].isin(selected)].copy()
    print(f"Pool after top-{args.pool_per_metric} union (Cal/Sh/CAGR/rank_sum) per "
          f"(bucket,interval): {len(df)}")
    for (bkt, iv), n in sorted(pool_stats.items()):
        print(f"  {bkt}_{iv}: {n}")

    combos = enumerate_combos(df, sizes=(1, 2, 3))
    print(f"Combos: {len(combos)} (k=1,2,3, snap-aligned mixed only, "
          f"near-duplicate pairs excluded for k>=2)")

    global _SUMMARY_DF
    _SUMMARY_DF = df

    rows: list[dict] = []
    # k=1은 IO 없음 → 직렬 처리. k>=2만 Pool.
    k1 = [c for c in combos if c["k"] == 1]
    k_multi = [c for c in combos if c["k"] >= 2]
    for c in k1:
        rows.append(run_combo(c, df))
    if args.processes > 1 and len(k_multi) > 1:
        with Pool(processes=args.processes) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_combo_worker, k_multi,
                                                         chunksize=1), 1):
                rows.append(res)
                if i % 50 == 0 or i == len(k_multi):
                    ok_n = sum(1 for r in rows if r.get("status") == "ok")
                    print(f"  [{i}/{len(k_multi)}] ok={ok_n}")
    else:
        for c in k_multi:
            rows.append(run_combo(c, df))

    res_df = pd.DataFrame(rows)
    atomic_write_csv(res_df, os.path.join(args.out_dir, "all_combos.csv"))

    ok = res_df[res_df["status"] == "ok"].copy()
    if not ok.empty:
        ok["rank_sum"] = (ok["Cal"].rank(ascending=False, method="min")
                          + ok["Sharpe"].rank(ascending=False, method="min")
                          + ok["CAGR"].rank(ascending=False, method="min"))

        def _union_top(sub: pd.DataFrame, n: int) -> pd.DataFrame:
            if sub.empty:
                return sub
            parts = [sub.sort_values(m, ascending=False).head(n)
                     for m in ("Cal", "Sharpe", "CAGR")]
            parts.append(sub.sort_values("rank_sum", ascending=True).head(n))
            return (pd.concat(parts)
                    .drop_duplicates(subset=["ensemble_tag"])
                    .sort_values("rank_sum"))

        spot_top = _union_top(ok[ok["asset"] == "spot"], args.top_n)
        fut_parts = []
        for lev_val, sub in ok[ok["asset"] == "fut"].groupby("lev"):
            fut_parts.append(_union_top(sub, args.top_n))
        fut_top = (pd.concat(fut_parts).sort_values(["lev", "rank_sum"])
                   if fut_parts else pd.DataFrame())
    else:
        spot_top = fut_top = pd.DataFrame()
    atomic_write_csv(spot_top, os.path.join(args.out_dir, "spot_top.csv"))
    atomic_write_csv(fut_top, os.path.join(args.out_dir, "fut_top.csv"))

    write_manifest(manifest_path, {
        "status": "done", "stage": "phase3_ensemble",
        "n_combos": int(len(res_df)),
        "n_ok": int((res_df["status"] == "ok").sum()),
        "n_spot_top": int(len(spot_top)),
        "n_fut_top": int(len(fut_top)),
        "cagr_floor_per_lev": args.cagr_floor_per_lev,
        "pool_per_metric": args.pool_per_metric,
    })
    print(f"combos={len(res_df)} ok={(res_df['status']=='ok').sum()} "
          f"spot_top={len(spot_top)} fut_top={len(fut_top)}")


if __name__ == "__main__":
    main()
