#!/usr/bin/env python3
"""iter_refine 결과의 snap을 3일 배수(72h wall-clock)로 정렬.

동작:
1. raw_combined.csv 읽기
2. 이미 정렬된 config (D snap*24 in ALLOWED_H, 4h snap*4 in ALLOWED_H) → keep
3. 정렬 안 된 config 중 top-N (Cal 상위) → snap을 가장 가까운 허용값으로 round → 재실행
4. 중복 제거 후 raw_aligned.csv 저장

ALLOWED_H = multiples of 72h (= 3 days) up to 2592h (= 108 days).
→ D snap ∈ {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, ..., 108}
→ 4h snap ∈ {18, 36, 54, 72, 90, 108, ..., 648}
→ 모든 D wall-clock == 4h wall-clock 가능
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# wall-clock hours: 3일(72h) 배수, 3일~108일 범위
ALLOWED_H = sorted(range(72, 2592 + 1, 72))
IV_HOURS = {"D": 24, "4h": 4, "1D": 24}


def snap_hours(iv: str, snap: int) -> int:
    return int(snap) * IV_HOURS.get(str(iv), 0)


def is_aligned(iv: str, snap: int) -> bool:
    return snap_hours(iv, snap) in set(ALLOWED_H)


def round_snap(iv: str, snap: int) -> int:
    """가장 가까운 ALLOWED_H로 snap 반올림."""
    h = snap_hours(iv, snap)
    if h == 0:
        return snap
    target = min(ALLOWED_H, key=lambda x: abs(x - h))
    return target // IV_HOURS.get(str(iv), 1)


def _run_one(task: dict) -> dict:
    """단일 config 재실행. 키: asset, iv, lev, sma, ms, ml, vmode, vthr, snap, anchor, end."""
    from unified_backtest import run_single
    try:
        res = run_single(
            asset=task["asset"], iv=task["iv"], lev=task["lev"],
            sma=task["sma"], ms=task["ms"], ml=task["ml"],
            vmode=task["vmode"], vthr=task["vthr"], snap=task["snap"],
            anchor=task["anchor"], end=task["end"],
        )
        return {**task, **res, "error": ""}
    except Exception as e:
        return {**task, "Sh": 0, "Cal": 0, "CAGR": 0, "MDD": 0,
                "rebal": 0, "liq": 0, "error": str(e)[:100]}


def _build_tag(row: dict) -> str:
    iv = "1D" if row["iv"] in ("D", "1D") else row["iv"]
    vtag = f"{row['vmode'][0]}{float(row['vthr']):.2f}"
    return (f"{row['asset']}_{iv}_S{int(row['sma'])}_M{int(row['ms'])}_{int(row['ml'])}"
            f"_{vtag}_SN{int(row['snap'])}_L{int(row['lev'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw_combined.csv")
    ap.add_argument("--out", required=True, help="raw_aligned.csv")
    ap.add_argument("--top-round", type=int, default=500,
                    help="정렬 안 된 top-N을 라운딩해서 재실행")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--anchor", default="2020-10-01")
    ap.add_argument("--end", default="2026-04-13")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""].copy()
    print(f"[snap_align] total rows: {len(df)}")

    # iv 정규화 (1D → D)
    df["iv_norm"] = df["iv"].replace({"1D": "D"})

    df["aligned"] = df.apply(
        lambda r: is_aligned(r["iv_norm"], int(r["snap"])), axis=1)
    aligned = df[df["aligned"]].copy()
    unaligned = df[~df["aligned"]].copy()
    print(f"[snap_align] already aligned: {len(aligned)}, "
          f"unaligned: {len(unaligned)}")

    # 정렬 안 된 config 중 Cal 상위 top-N을 라운딩해서 재실행
    if not unaligned.empty:
        # (asset, iv, lev, sma, ms, ml, vmode, vthr) 조합별 Cal 최대만 뽑아
        # 같은 구조에서 snap만 다른 중복 라운딩 방지
        key_cols = ["asset", "iv_norm", "lev", "sma", "ms", "ml", "vmode", "vthr"]
        unaligned_best = (
            unaligned.sort_values("Cal", ascending=False)
            .drop_duplicates(subset=key_cols, keep="first")
            .head(args.top_round)
        )

        # 각 row의 snap을 round
        round_tasks = []
        for _, r in unaligned_best.iterrows():
            iv = r["iv_norm"]
            rounded_snap = round_snap(iv, int(r["snap"]))
            if rounded_snap == int(r["snap"]):
                continue
            task = {
                "asset": str(r["asset"]), "iv": iv,
                "lev": float(r["lev"]), "sma": int(r["sma"]),
                "ms": int(r["ms"]), "ml": int(r["ml"]),
                "vmode": str(r["vmode"]), "vthr": float(r["vthr"]),
                "snap": int(rounded_snap),
                "anchor": args.anchor, "end": args.end,
            }
            task["tag"] = _build_tag(task)
            round_tasks.append(task)

        # 이미 aligned에 존재하는 tag는 skip
        existing_tags = set(aligned["tag"].astype(str))
        round_tasks = [t for t in round_tasks if t["tag"] not in existing_tags]
        print(f"[snap_align] round tasks after dedup: {len(round_tasks)}")

        results = []
        if round_tasks:
            with Pool(processes=args.workers) as pool:
                for i, res in enumerate(pool.imap_unordered(_run_one, round_tasks), 1):
                    results.append(res)
                    if i % 50 == 0 or i == len(round_tasks):
                        ok = sum(1 for r in results if not r.get("error"))
                        print(f"  [round {i}/{len(round_tasks)}] ok={ok}")

        rounded_df = pd.DataFrame(results)
        if not rounded_df.empty:
            rounded_df = rounded_df[rounded_df["error"].fillna("") == ""].copy()
            rounded_df["aligned"] = True
            rounded_df["iv_norm"] = rounded_df["iv"]
            # 컬럼 순서 맞춤
            common_cols = [c for c in df.columns if c in rounded_df.columns]
            for c in df.columns:
                if c not in rounded_df.columns:
                    rounded_df[c] = None
            rounded_df = rounded_df[df.columns.tolist()]
            print(f"[snap_align] rounded re-ran rows: {len(rounded_df)}")
        else:
            rounded_df = pd.DataFrame(columns=df.columns)
    else:
        rounded_df = pd.DataFrame(columns=df.columns)

    # 병합 + 중복 제거 (tag 기준)
    merged = pd.concat([aligned, rounded_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["tag"], keep="first")
    print(f"[snap_align] output total: {len(merged)}")

    # aligned 컬럼 제거하고 저장 (하위 단계 호환)
    for c in ("aligned", "iv_norm"):
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    merged.to_csv(tmp, index=False)
    os.replace(tmp, args.out)
    print(f"[snap_align] wrote {args.out}")


if __name__ == "__main__":
    main()
