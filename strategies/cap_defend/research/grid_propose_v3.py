#!/usr/bin/env python3
"""Edge-only drop + intermediate fill 규칙으로 grid v3 제안.

규칙:
- 각 (asset, iv) 버킷, 각 축에 대해 lev 통합 평균 top-rate 계산
- "keep" 값들의 min~max 범위 안쪽은 모두 보존 (중간 골짜기는 노이즈로 간주)
- DROP은 surviving range 바깥(양 끝)에서만
- ADD: 인접 값의 비율이 >= 1.5x이면 중간값 삽입 (densify)
- ADD: 양쪽 끝이 강한 keep이면 바깥 1단 확장 (boundary push)
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
import pandas as pd

HERE = "/home/gmoh/mon/251229/strategies/cap_defend/research"
RAW1 = f"{HERE}/phase1_sweep/raw.csv"
RAW2 = f"{HERE}/phase1_v2_sweep/raw.csv"

TAG_RE = re.compile(
    r"^(?P<asset>spot|fut)_(?P<iv>1D|4h|2h)"
    r"_S(?P<sma>\d+)_M(?P<ms>\d+)_(?P<ml>\d+)"
    r"_(?P<vmode>[db])(?P<vthr>[\d.]+)"
    r"_SN(?P<snap>\d+)_L(?P<lev>\d+)$"
)

# 엄격 임계: 정직한 keep만 인정
KEEP_RATE = 0.20   # top quintile에 20% 이상 등장
TOP_Q = 0.20

NUMERIC_AXES = ["sma", "ms", "ml", "snap"]
ALL_AXES = NUMERIC_AXES + ["vmode_thr"]
DENSIFY_RATIO = 1.5  # 인접 비율이 1.5x 이상이면 중간값 추가


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        return None
    d = m.groupdict()
    return {
        "asset": d["asset"],
        "iv": d["iv"].replace("1D", "D"),
        "lev": int(d["lev"]),
        "sma": int(d["sma"]),
        "ms": int(d["ms"]),
        "ml": int(d["ml"]),
        "vmode_thr": f"{d['vmode']}{float(d['vthr']):.2f}",
        "snap": int(d["snap"]),
    }


def load_all() -> pd.DataFrame:
    parts = []
    for p in [RAW1, RAW2]:
        df = pd.read_csv(p)
        df = df[df["error"].isna() | (df["error"] == "")]
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    parsed = raw["tag"].apply(parse_tag)
    raw = raw[parsed.notna()].copy()
    pdf = pd.DataFrame(parsed.dropna().tolist(), index=raw.index)
    for c in pdf.columns:
        if c in raw.columns:
            raw = raw.drop(columns=[c])
    raw = pd.concat([raw, pdf], axis=1)
    keys = ["asset", "iv", "lev"] + ALL_AXES
    agg = raw.groupby(keys, as_index=False).agg(
        Cal=("Cal", "mean"), Sh=("Sh", "mean"),
        CAGR=("CAGR", "mean"), n=("Cal", "size"),
    )
    return agg


def edge_drop_axis(df: pd.DataFrame, asset: str, iv: str,
                   axis: str, sort_key) -> dict:
    """단일 (asset, iv, axis) — lev 통합 후 edge-only drop 규칙 적용."""
    sub = df[(df["asset"] == asset) & (df["iv"] == iv)]
    if sub.empty:
        return {"keep": [], "drop": [], "all": []}
    # lev별 top-rate 평균
    lev_rates = []
    for lev, g in sub.groupby("lev"):
        if len(g) < 5:
            continue
        thr = g["Cal"].quantile(1 - TOP_Q)
        for v, ssub in g.groupby(axis):
            n = len(ssub)
            top = (ssub["Cal"] >= thr).sum()
            lev_rates.append({
                "lev": lev, "value": v, "rate": top / n if n else 0,
                "mean": ssub["Cal"].mean(), "n": n,
            })
    rdf = pd.DataFrame(lev_rates)
    if rdf.empty:
        return {"keep": [], "drop": [], "all": []}
    # lev 평균 (보수적: 모든 lev에서 keep이어야 keep, 하나라도 keep이면 keep?)
    # 사용자 원칙: 양 끝만 자른다. 그러니 union 의미로 → max rate 사용.
    g2 = rdf.groupby("value", as_index=False).agg(
        rate=("rate", "max"), mean=("mean", "mean"),
    )
    g2["sort_key"] = g2["value"].apply(sort_key)
    g2 = g2.sort_values("sort_key").reset_index(drop=True)
    # 정직한 keep 마킹
    g2["keep"] = g2["rate"] >= KEEP_RATE
    keeps_idx = g2.index[g2["keep"]].tolist()
    if not keeps_idx:
        # 아무것도 keep이 안 되면 → 전부 보존 (정보 부족)
        return {
            "keep": g2["value"].tolist(), "drop": [],
            "detail": g2.to_dict("records"),
        }
    lo, hi = keeps_idx[0], keeps_idx[-1]
    survive_mask = (g2.index >= lo) & (g2.index <= hi)
    keep_vals = g2.loc[survive_mask, "value"].tolist()
    drop_vals = g2.loc[~survive_mask, "value"].tolist()
    return {
        "keep": keep_vals, "drop": drop_vals,
        "detail": g2.to_dict("records"),
    }


def densify_numeric(values: list[int]) -> list[int]:
    """인접 비율 >=1.5x이면 기하평균 근사로 중간값 추가."""
    if len(values) < 2:
        return list(values)
    out = []
    for i, v in enumerate(values):
        out.append(v)
        if i + 1 < len(values):
            nxt = values[i + 1]
            if nxt / v >= DENSIFY_RATIO:
                # 기하평균 → 가장 가까운 정수, 단 v보다 크고 nxt보다 작은 값
                mid = int(round((v * nxt) ** 0.5))
                if v < mid < nxt:
                    out.append(mid)
    return sorted(set(out))


def main():
    df = load_all()
    print(f"[load] aggregated rows: {len(df)}")

    out_lines = []
    new_grid = {}  # iv → axis → list

    pairs = [("spot", "D"), ("spot", "4h"),
             ("fut", "D"), ("fut", "4h"), ("fut", "2h")]

    for asset, iv in pairs:
        out_lines.append(f"\n{'='*60}\n[{asset} {iv}]\n{'='*60}")
        bucket_grid = {}
        for axis in ALL_AXES:
            sk = (lambda v: v) if axis in NUMERIC_AXES else (lambda v: str(v))
            res = edge_drop_axis(df, asset, iv, axis, sk)
            keep = res["keep"]
            drop = res["drop"]
            # densify (수치형만)
            if axis in NUMERIC_AXES and keep:
                densified = densify_numeric(sorted(int(v) for v in keep))
                added = sorted(set(densified) - set(int(v) for v in keep))
            else:
                densified = keep
                added = []
            bucket_grid[axis] = densified
            out_lines.append(
                f"  {axis:>10s}: keep={keep} +add={added} drop={drop}"
            )
        new_grid[f"{asset}_{iv}"] = bucket_grid

    # iv 단위로 통합 (asset 다른 lev/cost지만 같은 데이터 → union)
    iv_grid = {}
    for iv in ["D", "4h", "2h"]:
        merged = {}
        for axis in ALL_AXES:
            vals = set()
            for k, g in new_grid.items():
                if k.endswith(f"_{iv}") and axis in g:
                    vals.update(g[axis])
            if not vals:
                continue
            if axis in NUMERIC_AXES:
                merged[axis] = sorted(vals)
            else:
                merged[axis] = sorted(vals)
        if merged:
            iv_grid[iv] = merged

    out_lines.append(f"\n\n{'='*60}\nIV 통합 grid (asset/lev union)\n{'='*60}")
    for iv, g in iv_grid.items():
        out_lines.append(f"\n[{iv}]")
        for axis, vals in g.items():
            out_lines.append(f"  {axis}: {vals}")

    # 사이즈 추정
    out_lines.append(f"\n\n{'='*60}\nGrid 크기 추정 (Mom_l>=2*Mom_s 미반영 raw)\n{'='*60}")
    for iv, g in iv_grid.items():
        sz = 1
        for axis in ALL_AXES:
            sz *= max(1, len(g.get(axis, [])))
        out_lines.append(f"  {iv}: ~{sz}")

    with open(f"{HERE}/grid_v3_proposal.txt", "w") as f:
        f.write("\n".join(out_lines))
    with open(f"{HERE}/grid_v3.json", "w") as f:
        json.dump(iv_grid, f, indent=2, default=str)
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
