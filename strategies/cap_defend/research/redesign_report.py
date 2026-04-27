"""Step 8 — 자산별 markdown 리포트 생성.

입력:
  redesign_rank_{asset}.csv
  redesign_ensemble_bt_{asset}.csv
  redesign_stress_{asset}.csv

출력:
  redesign_report_{asset}.md

구성:
  - Primary 3 (rank_sum top 3, stress all pass, ensemble 포함)
  - Backup 5~9 (rank_sum 4~12)
  - Rejected but interesting (rank 상위지만 stress 에서 열화 큰 후보)
"""
from __future__ import annotations
import argparse
import os

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))


def load_or_empty(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def stress_summary(stress_df, tag):
    sub = stress_df[stress_df["tag"] == tag]
    if sub.empty:
        return "(stress 미실행)"
    baseline = sub[sub["scenario"] == "baseline"]
    if baseline.empty:
        return "(baseline 없음)"
    base_cal = float(baseline["Cal"].iloc[0])
    lines = []
    for _, r in sub.iterrows():
        if r["scenario"] == "baseline":
            continue
        decay = (float(r["Cal"]) - base_cal) / base_cal * 100 if base_cal else None
        lines.append(f"  - {r['scenario']}: Cal {r['Cal']:.3f} ({decay:+.1f}%)" if decay is not None
                     else f"  - {r['scenario']}: Cal {r['Cal']:.3f}")
    return "\n".join(lines)


def render_asset(asset):
    rank = load_or_empty(os.path.join(HERE, f"redesign_rank_{asset}.csv"))
    ensemble = load_or_empty(os.path.join(HERE, f"redesign_ensemble_bt_{asset}.csv"))
    stress = load_or_empty(os.path.join(HERE, f"redesign_stress_{asset}.csv"))

    out = [f"# {asset.upper()} 최종 후보 리포트", ""]
    out.append(f"총 rank 후보: {len(rank)} / ensemble: {len(ensemble)} / stress: {len(stress)}")
    out.append("")

    # stock 은 v17_iter 가 CAGR/MDD 를 % 형태 (12.49 = 12.49%) 로 저장.
    # coin/fut 은 fraction (0.12 = 12%). 일관성 위해 stock RANK 만 /100 정규화.
    # ensemble 결과는 redesign_ensemble_bt 가 fraction 으로 직접 산출 → /100 안 함.
    if asset == "stock":
        for col in ("CAGR", "MDD"):
            if col in rank.columns:
                rank = rank.copy()
                rank[col] = rank[col].astype(float) / 100.0

    if rank.empty:
        out.append("(rank csv 없음)")
        with open(os.path.join(HERE, f"redesign_report_{asset}.md"), "w") as f:
            f.write("\n".join(out))
        return

    # Primary 선정: rank top 에서 stress 전 시나리오 pass 인 후보만
    # stress.csv 가 없으면 rank top 3 그대로 (fallback)
    stress_pass_tags = set()
    if not stress.empty:
        # 각 tag 의 모든 scenario status == ok 면 pass
        for tag, g in stress.groupby("tag"):
            if "status" in g.columns and (g["status"] == "ok").all():
                stress_pass_tags.add(str(tag))
        primary_pool = rank[rank["tag"].astype(str).isin(stress_pass_tags)]
        if len(primary_pool) == 0:
            primary = rank.head(3)
            primary_note = "(stress pass 후보 없음, rank top3 fallback)"
        else:
            primary = primary_pool.head(3)
            primary_note = f"(stress 전 시나리오 pass, {len(stress_pass_tags)}개 중)"
    else:
        primary = rank.head(3)
        primary_note = "(stress 미실행)"
    backup = rank.iloc[3:12]

    out.append(f"## Primary 3 {primary_note}")
    for _, r in primary.iterrows():
        out.append(f"### {r['tag']}")
        cal = r.get('Cal', float('nan'))
        cagr = r.get('CAGR', float('nan'))
        composite = (cal * cagr) if (pd.notna(cal) and pd.notna(cagr)) else float('nan')
        out.append(f"- rank_sum: {r.get('rank_sum', 'n/a')} | Cal: {cal:.3f} | CAGR: {cagr:.3%} | Cal×CAGR: {composite:.3f} | MDD: {r.get('MDD', 0):.2%} | Sh: {r.get('Sh', 0):.2f}")
        for c in ("phase_med", "yearly_med", "worst_year", "snap_nudge_CV"):
            if c in r and pd.notna(r[c]):
                out.append(f"- {c}: {r[c]:.3f}" if isinstance(r[c], (int, float)) else f"- {c}: {r[c]}")
        out.append(f"- stress:\n{stress_summary(stress, r['tag'])}")
        out.append("")

    out.append("## Backup 5~9")
    for _, r in backup.iterrows():
        cal = r.get('Cal', 0); cagr = r.get('CAGR', 0)
        out.append(f"- {r['tag']} | rank_sum {r.get('rank_sum', 'n/a')} | Cal {cal:.2f} | CAGR {cagr:.2%} | Cal×CAGR {cal*cagr:.2f}")
    out.append("")

    # ── Cal / CAGR / Cal×CAGR / rank_sum 별 top 10 통합 ─────────
    out.append("## 다중 metric 정렬 — top 10 (단독 멤버 k=1)")
    rk = rank.copy()
    if "Cal" in rk.columns and "CAGR" in rk.columns:
        rk["Cal_x_CAGR"] = rk["Cal"].astype(float) * rk["CAGR"].astype(float)
    for metric, label, asc in [
        ("rank_sum", "rank_sum 낮은순", True),
        ("Cal", "Cal 높은순", False),
        ("CAGR", "CAGR 높은순", False),
        ("Cal_x_CAGR", "Cal×CAGR 높은순", False),
    ]:
        if metric not in rk.columns:
            continue
        sorted_rk = rk.sort_values(metric, ascending=asc).head(10)
        out.append(f"\n### {label}")
        for _, r in sorted_rk.iterrows():
            cal = r.get('Cal', 0); cagr = r.get('CAGR', 0)
            out.append(f"- {r['tag']} | rank_sum {r.get('rank_sum', '?')} | Cal {cal:.2f} | CAGR {cagr:.2%} | Cal×CAGR {cal*cagr:.2f} | MDD {r.get('MDD', 0):.1%}")
    out.append("")

    if not ensemble.empty:
        out.append("## Ensemble 후보 (k=2/3) — 다중 metric 정렬")
        # 앙상블 special gate 모두 제거 (2026-04-25). status==ok 만 필터.
        ok = ensemble.copy()
        if "status" in ok.columns:
            ok = ok[ok["status"] == "ok"]
        if ok.empty:
            out.append("(BT ok 후보 없음)")
        else:
            if "Cal" in ok.columns and "CAGR" in ok.columns:
                ok = ok.copy()
                ok["Cal_x_CAGR"] = ok["Cal"].astype(float) * ok["CAGR"].astype(float)
            for metric, label, asc in [
                ("Cal", "Cal 높은순", False),
                ("CAGR", "CAGR 높은순", False),
                ("Cal_x_CAGR", "Cal×CAGR 높은순", False),
            ]:
                if metric not in ok.columns:
                    continue
                sub = ok.sort_values(metric, ascending=asc).head(10)
                out.append(f"\n### Ensemble {label} — top 10")
                for _, r in sub.iterrows():
                    cal = r.get('Cal', 0); cagr = r.get('CAGR', 0)
                    out.append(
                        f"- k={r['k']}: {r['members']} → "
                        f"Cal {cal:.3f} | CAGR {cagr:.2%} | Cal×CAGR {cal*cagr:.3f} | "
                        f"MDD {r.get('MDD', 0):.2%} | Sh {r.get('Sh', 0):.2f} | "
                        f"improve {r.get('improve_count', '?')}/3 corr {r.get('corr_max', 0):.2f} bad {r.get('bad_day_overlap', 0):.2f}"
                    )

    path = os.path.join(HERE, f"redesign_report_{asset}.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"[{asset}] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="all")
    args = ap.parse_args()
    assets = ("fut", "spot", "stock") if args.asset == "all" else (args.asset,)
    for a in assets:
        render_asset(a)


if __name__ == "__main__":
    main()
