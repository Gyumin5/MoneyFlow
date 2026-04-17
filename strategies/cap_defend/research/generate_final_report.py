#!/usr/bin/env python3
"""Phase B + D 결과로 최종 markdown 리포트 생성."""
from __future__ import annotations
import argparse
import json


def fmt_pct(x):
    try:
        return f"{float(x)*100:+.1f}%"
    except Exception:
        return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-d-top", required=True)
    p.add_argument("--phase-b-csv", required=True)
    p.add_argument("--winners", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.phase_d_top) as f:
        d = json.load(f)
    with open(args.winners) as f:
        winners = json.load(f)["winners"]

    spot = d.get("spot_baseline", {})
    accepted = d.get("accepted_top", [])

    lines = []
    lines.append("# 선물 전략 탐색 최종 리포트")
    lines.append("")
    lines.append("## 현물 V20 baseline")
    lines.append(f"- Sharpe {spot.get('Sharpe',0):.2f} / CAGR {fmt_pct(spot.get('CAGR',0))} / "
                 f"MDD {fmt_pct(spot.get('MDD',0))} / Cal {spot.get('Cal',0):.2f}")
    lines.append("")

    lines.append("## Lev별 선물 winner (Phase B)")
    lines.append("| Lev | Interval | Guard | Cal | CAGR | MDD | Sharpe | Liq | Stops | Label |")
    lines.append("|-----|----------|-------|-----|------|-----|--------|-----|-------|-------|")
    for w in sorted(winners, key=lambda x: x["lev"]):
        lines.append(f"| {int(w['lev'])}x | {w['interval']} | {w['guard_name']} | "
                     f"{w['Cal']:.2f} | {fmt_pct(w['CAGR'])} | {fmt_pct(w['MDD'])} | "
                     f"{w['Sharpe']:.2f} | {w['Liq']} | {w['Stops']} | {w['label'][:40]} |")
    lines.append("")

    lines.append(f"## 현물+선물 mix accepted ({d.get('n_accepted',0)}/{d.get('total',0)})")
    lines.append("합격선: |MDD_mix| ≤ |MDD_spot|×1.10 AND Cal_mix ≥ Cal_spot×1.05")
    lines.append("")
    if not accepted:
        lines.append("⚠ accepted 조합 없음 — 단독 선물이 현물 안전성을 깨뜨림")
    else:
        lines.append("| Lev | Spot% | Band | Cal | CAGR | MDD | MDD_ratio | Cal_ratio | Fut Label |")
        lines.append("|-----|-------|------|-----|------|-----|-----------|-----------|-----------|")
        for r in accepted[:15]:
            lines.append(f"| {int(r['lev'])}x | {int(r['spot_ratio']*100)}% | "
                         f"{r['band']} | {r['mix_Cal']:.2f} | {fmt_pct(r['mix_CAGR'])} | "
                         f"{fmt_pct(r['mix_MDD'])} | {r['MDD_ratio']} | {r['Cal_ratio']} | "
                         f"{r['fut_label'][:30]} |")
    lines.append("")

    if accepted:
        best = accepted[0]
        lines.append("## 추천")
        lines.append(f"- 현물 {int(best['spot_ratio']*100)}% / 선물 {int(best['fut_ratio']*100)}% "
                     f"({int(best['lev'])}x, {best['fut_label'][:30]})")
        lines.append(f"- 밴드 리밸런싱: {best['band']}")
        lines.append(f"- 예상 Cal {best['mix_Cal']:.2f} (현물 대비 {best['Cal_ratio']:.2f}x), "
                     f"MDD {fmt_pct(best['mix_MDD'])} (현물 대비 {best['MDD_ratio']:.2f}x)")
    lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
