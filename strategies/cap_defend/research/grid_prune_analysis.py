#!/usr/bin/env python3
"""Phase-1 결과 기반 그리드 가지치기/확장 분석.

각 축 값에 대해:
- top quintile inclusion rate (Cal 기준)
- 살아남은 값의 간격 → 중간값 추천
- 경계값이 우수 → 바깥쪽 확장 추천
"""
from __future__ import annotations
import re
from collections import defaultdict
import pandas as pd

HERE = "/home/gmoh/mon/251229/strategies/cap_defend/research"
RAW1 = f"{HERE}/phase1_sweep/raw.csv"
RAW2 = f"{HERE}/phase1_v2_sweep/raw.csv"

# tag 형식: spot_1D_S20_M10_60_d0.03_SN30_L1
TAG_RE = re.compile(
    r"^(?P<asset>spot|fut)_(?P<iv>1D|4h|2h)"
    r"_S(?P<sma>\d+)_M(?P<ms>\d+)_(?P<ml>\d+)"
    r"_(?P<vmode>[db])(?P<vthr>[\d.]+)"
    r"_SN(?P<snap>\d+)_L(?P<lev>\d+)$"
)

AXES = ["sma", "ms", "ml", "vmode_thr", "snap"]


def parse_tag(tag: str) -> dict | None:
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
    # raw에 이미 asset/lev 컬럼이 있으므로 충돌 방지
    for c in pdf.columns:
        if c in raw.columns:
            raw = raw.drop(columns=[c])
    raw = pd.concat([raw, pdf], axis=1)
    # anchor 평균 (현재는 1 anchor지만 향후 대비)
    keys = ["asset", "iv", "lev"] + AXES
    agg = raw.groupby(keys, as_index=False).agg(
        Cal=("Cal", "mean"),
        Sh=("Sh", "mean"),
        CAGR=("CAGR", "mean"),
        MDD=("MDD", "mean"),
        n=("Cal", "size"),
    )
    return agg


def axis_inclusion(df: pd.DataFrame, bucket_keys, axis: str,
                   metric: str = "Cal", topq: float = 0.20) -> pd.DataFrame:
    """축 값별 top quintile 포함률 (버킷 내)."""
    rows = []
    for bk, g in df.groupby(bucket_keys):
        if len(g) < 5:
            continue
        thr = g[metric].quantile(1 - topq)
        for v, sub in g.groupby(axis):
            n_total = len(sub)
            n_top = (sub[metric] >= thr).sum()
            rows.append({
                "bucket": "_".join(map(str, bk)) if isinstance(bk, tuple) else str(bk),
                "axis": axis,
                "value": v,
                "n_total": n_total,
                "n_top": int(n_top),
                "top_rate": n_top / n_total if n_total else 0,
                "mean_metric": sub[metric].mean(),
                "max_metric": sub[metric].max(),
            })
    return pd.DataFrame(rows)


def main():
    df = load_all()
    print(f"[load] aggregated rows: {len(df)}")
    print(f"[load] buckets: {df.groupby(['asset','iv','lev']).size().to_dict()}")

    out_path = f"{HERE}/grid_prune_report.txt"
    out = []
    bucket_keys = ["asset", "iv", "lev"]

    for axis in AXES:
        out.append(f"\n{'='*70}\n축: {axis}\n{'='*70}")
        rep = axis_inclusion(df, bucket_keys, axis)
        if rep.empty:
            out.append("(빈 결과)")
            continue
        for bk, g in rep.groupby("bucket"):
            g_sorted = g.sort_values(
                "value",
                key=lambda x: pd.to_numeric(x, errors="coerce")
                .fillna(pd.Series(range(len(x)), index=x.index)),
            )
            out.append(f"\n[{bk}]")
            for _, r in g_sorted.iterrows():
                mark = ""
                if r["top_rate"] >= 0.30:
                    mark = "✓✓ KEEP"
                elif r["top_rate"] >= 0.15:
                    mark = "✓ keep"
                elif r["top_rate"] <= 0.05:
                    mark = "✗ DROP"
                out.append(
                    f"  {str(r['value']):>8s}  top%={r['top_rate']*100:5.1f}  "
                    f"mean={r['mean_metric']:+.3f}  max={r['max_metric']:+.3f}  "
                    f"n={r['n_total']}  {mark}"
                )

    # 권장 grid 후보 추출
    out.append(f"\n\n{'='*70}\n축별 권장 (drop/keep, 자동 추천)\n{'='*70}")
    suggestions = defaultdict(dict)  # bucket -> axis -> {keep:[], drop:[]}
    for axis in AXES:
        rep = axis_inclusion(df, bucket_keys, axis)
        for bk, g in rep.groupby("bucket"):
            keep = sorted(g[g["top_rate"] >= 0.15]["value"].tolist(),
                          key=lambda v: (isinstance(v, str), v))
            drop = sorted(g[g["top_rate"] <= 0.05]["value"].tolist(),
                          key=lambda v: (isinstance(v, str), v))
            mid = sorted(g[(g["top_rate"] > 0.05) & (g["top_rate"] < 0.15)]["value"].tolist(),
                         key=lambda v: (isinstance(v, str), v))
            suggestions[bk][axis] = {"keep": keep, "mid": mid, "drop": drop}

    for bk, ax_map in suggestions.items():
        out.append(f"\n[{bk}]")
        for axis, d in ax_map.items():
            out.append(f"  {axis}: keep={d['keep']}  mid={d['mid']}  drop={d['drop']}")

    text = "\n".join(out)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[done] report → {out_path}")
    print(text[-2000:])


if __name__ == "__main__":
    main()
