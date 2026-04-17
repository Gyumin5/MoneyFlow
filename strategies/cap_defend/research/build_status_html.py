#!/usr/bin/env python3
"""Phase 체인 통합 상태 HTML 렌더 (v3: iter_refine→plateau→bridge→phase3→robustness→phase4).

- 각 phase 디렉터리의 manifest.json + 대표 summary CSV를 읽어 진행률/top 결과 표시.
- 결과 테이블은 Cal/Sh/CAGR/MDD 4개 메트릭을 모두 포함.
- iter_refine 은 run_local.log tail + stage별 raw.csv row 수 표시.
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
OUT = os.path.join(HERE, "phase_status.html")


def _count_rows(p: str) -> int:
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return 0
    try:
        with open(p, "rb") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def _tail(path: str, n: int = 15) -> str:
    if not os.path.exists(path):
        return "(no log)"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return html.escape("".join(lines[-n:]))
    except Exception:
        return "(read error)"


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_df(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        return df if not df.empty else None
    except Exception:
        return None


def progress_bar(meta: dict) -> str:
    total = int(meta.get("total_tasks", 0) or 0)
    done = int(meta.get("done_tasks", 0) or 0)
    if total <= 0:
        return ""
    pct = min(100, int(done * 100 / total)) if total else 0
    return (f"<div class='pb'><div class='pb-fill' style='width:{pct}%'></div>"
            f"<span class='pb-text'>{done}/{total} ({pct}%)</span></div>")


def top_table(df: pd.DataFrame | None, cols: list[str], n: int = 12) -> str:
    if df is None or df.empty:
        return "<p class='meta'>no rows</p>"
    sort_col = None
    for candidate in ("rank_sum", "Cal", "mCal", "spearman_rho"):
        if candidate in df.columns:
            sort_col = candidate
            break
    if sort_col:
        asc = sort_col in ("rank_sum",)
        df = df.sort_values(sort_col, ascending=asc)
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        use_cols = list(df.columns)[:10]
    return df[use_cols].head(n).to_html(index=False, escape=True, classes="tbl",
                                        float_format=lambda x: f"{x:.4f}")


def render_iter_refine() -> str:
    base = os.path.join(HERE, "iter_refine")
    log = os.path.join(base, "run_local.log")
    stages = []
    for s in range(1, 7):
        p = os.path.join(base, f"stage_{s}", "raw.csv")
        if os.path.exists(p):
            stages.append(f"stage{s}={_count_rows(p)}")
    combined = os.path.join(base, "raw_combined.csv")
    done_flag = "DONE" if os.path.exists(combined) else "RUNNING"
    color = "#16a34a" if done_flag == "DONE" else "#f59e0b"
    return (
        f"<section id='iter_refine'><h2>iter_refine (5-stage peak refinement)</h2>"
        f"<p class='meta'>status=<b style='color:{color}'>{done_flag}</b> "
        f"stages: {' '.join(stages) or '(not started)'}</p>"
        f"<pre class='log'>{_tail(log, 12)}</pre></section>"
    )


def render_plateau() -> str:
    base = os.path.join(HERE, "plateau_check")
    mf = os.path.join(base, "manifest.json")
    surv = os.path.join(base, "survivors.csv")
    meta = load_json(mf)
    df = load_df(surv)
    status = meta.get("status", "missing")
    extras = [f"{k}={v}" for k, v in meta.items()
              if k in ("n_centers", "n_survivors", "pass_ratio")]
    return (
        f"<section id='plateau'><h2>plateau_check (parameter robustness)</h2>"
        f"<p class='meta'>status=<b>{html.escape(status)}</b> "
        f"{' '.join(extras)} updated={html.escape(meta.get('updated_at','-'))}</p>"
        f"{top_table(df, ['tag','asset','lev','mCal','mSh','mCAGR','wMDD','plateau_ok','plateau_min_ratio'])}"
        f"</section>"
    )


def render_bridge() -> str:
    base = os.path.join(HERE, "phase1_sweep")
    summary = os.path.join(base, "summary.csv")
    n = _count_rows(summary)
    present = "DONE" if n > 0 else "WAITING"
    color = "#16a34a" if n > 0 else "#f59e0b"
    return (
        f"<section id='bridge'><h2>bridge (plateau → phase1_sweep/summary.csv)</h2>"
        f"<p class='meta'>status=<b style='color:{color}'>{present}</b> "
        f"summary rows={n}</p></section>"
    )


def render_phase3(floor: str) -> str:
    base = os.path.join(HERE, f"phase3_ensembles_floor{floor}")
    mf = os.path.join(base, "manifest.json")
    all_p = os.path.join(base, "all_combos.csv")
    spot_p = os.path.join(base, "spot_top.csv")
    fut_p = os.path.join(base, "fut_top.csv")
    meta = load_json(mf)
    status = meta.get("status", "missing")
    extras = []
    for k in ("n_combos", "n_ok", "n_spot_top", "n_fut_top",
              "cagr_floor_per_lev", "pool_per_metric"):
        if k in meta:
            extras.append(f"{k}={meta[k]}")
    cols = ["ensemble_tag", "bucket", "k", "members",
            "Cal", "Sharpe", "CAGR", "MDD", "min_year_cal",
            "worst_year_ret", "negative_years", "status"]
    df_all = load_df(all_p)
    df_spot = load_df(spot_p)
    df_fut = load_df(fut_p)
    return (
        f"<section id='phase3_{floor}'><h2>Phase-3 ensemble (CAGR floor {floor[:1]}.{floor[1:]})</h2>"
        f"<p class='meta'>status=<b>{html.escape(status)}</b> "
        f"{' '.join(extras)} updated={html.escape(meta.get('updated_at','-'))}</p>"
        f"<h3>spot_top</h3>{top_table(df_spot, cols)}"
        f"<h3>fut_top</h3>{top_table(df_fut, cols)}"
        f"<h3>all combos (sample)</h3>{top_table(df_all, cols, n=8)}"
        f"</section>"
    )


def render_robustness(floor: str) -> str:
    base = os.path.join(HERE, f"robustness_floor{floor}")
    mf = os.path.join(base, "manifest.json")
    summary_p = os.path.join(base, "summary.csv")
    loyo_p = os.path.join(base, "loyo_stability.csv")
    loao_p = os.path.join(base, "loao_stability.csv")
    meta = load_json(mf)
    status = meta.get("status", "missing")
    extras = []
    for k in ("n_ensembles", "loyo_runs", "loao_runs", "loao_coins"):
        if k in meta:
            extras.append(f"{k}={meta[k]}")
    sum_df = load_df(summary_p)
    loyo_df = load_df(loyo_p)
    loao_df = load_df(loao_p)
    return (
        f"<section id='robustness_{floor}'><h2>robustness_check "
        f"(LOYO + LOAO, floor {floor[:1]}.{floor[1:]})</h2>"
        f"<p class='meta'>status=<b>{html.escape(status)}</b> "
        f"{' '.join(extras)} updated={html.escape(meta.get('updated_at','-'))}</p>"
        f"<h3>Summary</h3>{top_table(sum_df, ['test','n_perturbations','mean_spearman','min_spearman','mean_top5_overlap'])}"
        f"<h3>LOYO (Leave-One-Year-Out)</h3>"
        f"{top_table(loyo_df, ['exclude_year','spearman_rho','p_value','top5_overlap','n_strategies'])}"
        f"<h3>LOAO (Leave-One-Asset-Out)</h3>"
        f"{top_table(loao_df, ['exclude_coin','spearman_rho','p_value','top5_overlap','n_strategies'], n=20)}"
        f"</section>"
    )


def render_phase4(floor: str) -> str:
    base = os.path.join(HERE, f"phase4_3asset_floor{floor}")
    mf = os.path.join(base, "manifest.json")
    raw = os.path.join(base, "raw.csv")
    meta = load_json(mf)
    status = meta.get("status", "missing")
    extras = [f"{k}={v}" for k, v in meta.items()
              if k in ("n_rows", "n_combos")]
    df = load_df(raw)
    cols = ["stock", "spot", "fut", "st_w", "sp_w", "fu_w", "band_mode", "band_label",
            "Cal", "Sh", "CAGR", "MDD", "rank_sum"]
    return (
        f"<section id='phase4_{floor}'><h2>Phase-4 3-asset mix (floor {floor[:1]}.{floor[1:]})</h2>"
        f"<p class='meta'>status=<b>{html.escape(status)}</b> "
        f"{' '.join(extras)} updated={html.escape(meta.get('updated_at','-'))}</p>"
        f"{top_table(df, cols, n=15)}</section>"
    )


def render_plan() -> str:
    return """<section id='plan'><h2>실험 계획 (2026-04-16 업데이트)</h2>
<ul>
<li><b>목표:</b> 주식 V17 + 업비트 현물 + 바이낸스 선물 앙상블 재설계. 자산배분 60/35/5 (8pp band).</li>
<li><b>탐색 범위:</b> D + 4h 봉 (2h 제거), 현물 L1 + 선물 L2/L3/L4.</li>
<li><b>앵커:</b> 2020-10-01 단일. FULL_END 2026-04-13. OOS 미분리 (IS only).</li>
</ul>
<h3>체인 순서</h3>
<ol>
<li><b>iter_refine</b> — 5-stage peak refinement. AXIS_CAP=7, prominence≥0.15, PEAK_CAP=3.</li>
<li><b>plateau_check</b> — 버킷별 top-100 × sma/ms/ml/snap ±5/±10% perturbation. min/center Cal ≥ 0.85 통과.</li>
<li><b>bridge_iter_to_phase1</b> — survivor → phase1_sweep/summary.csv 포맷 변환.</li>
<li><b>phase3_ensemble</b> — (bucket×interval) 지표4 top-5 합집합 pool, k=1/2/3 조합. CAGR floor 0.30/0.40 두 세트. diversity gate + yearly consistency 출력.</li>
<li><b>robustness_check</b> — <u>NEW</u>. phase3 top-20 앙상블 대상 LOYO(연도 제외) + LOAO(코인 제외). Spearman 순위 안정성 + top-5 overlap.</li>
<li><b>phase4_3asset</b> — 주식 60% 고정, 현물 {20..40}%, 선물 40-현물. 절대 band {3,5,8,10,15}pp + sleeve-relative {0.2,0.3,0.4,0.5}. 자산별 비용 반영.</li>
</ol>
<h3>핵심 수정 (2026-04-16)</h3>
<ul>
<li>snap 매칭 버그 수정: raw bar → wall-clock hour (D 30봉 = 4h 180봉 일치).</li>
<li>diversity gate: 같은 interval + 4축 상대차 &lt;15% 쌍 제외.</li>
<li>yearly consistency: equity 기반 연도별 Cal, min_year_cal/worst_year_ret/negative_years 출력.</li>
<li>sleeve-relative band: phase4에서 자산 weight × ratio로 독립 band 추가.</li>
<li>robustness_check 신규 작성: LOYO 5년 + LOAO 16코인 (anchor이후 top-10 등장, stable 제외).</li>
</ul>
</section>"""


def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        render_plan(),
        render_iter_refine(),
        render_plateau(),
        render_bridge(),
        render_phase3("30"),
        render_phase3("40"),
        render_robustness("30"),
        render_robustness("40"),
        render_phase4("30"),
        render_phase4("40"),
    ]
    body = "\n".join(sections)
    nav_items = [
        ("plan", "계획"),
        ("iter_refine", "iter_refine"),
        ("plateau", "plateau"),
        ("bridge", "bridge"),
        ("phase3_30", "phase3 floor30"),
        ("phase3_40", "phase3 floor40"),
        ("robustness_30", "robustness 30"),
        ("robustness_40", "robustness 40"),
        ("phase4_30", "phase4 floor30"),
        ("phase4_40", "phase4 floor40"),
    ]
    nav = " ".join(f"<a href='#{k}'>{v}</a>" for k, v in nav_items)
    out = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cap Defend Phase Status</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:20px;line-height:1.5;max-width:1400px}}
h1{{margin:0 0 10px}} h2{{margin:0 0 8px;font-size:1.15rem}} h3{{margin:8px 0 4px;font-size:0.98rem}}
section{{margin:18px 0;padding:12px;border:1px solid #e5e7eb;border-radius:6px;background:#fafafa}}
.meta{{color:#555;font-size:0.85rem;margin:2px 0}}
table.tbl{{border-collapse:collapse;width:100%;font-size:0.82rem}}
table.tbl th,table.tbl td{{border:1px solid #ddd;padding:3px 6px}} th{{background:#f3f4f6}}
.pb{{position:relative;background:#eee;height:16px;border-radius:3px;overflow:hidden;margin:4px 0}}
.pb-fill{{background:#4ade80;height:100%}}
.pb-text{{position:absolute;left:8px;top:0;font-size:0.78rem;color:#111;line-height:16px}}
nav a{{margin-right:8px;font-size:0.85rem;color:#2563eb;text-decoration:none}}
nav a:hover{{text-decoration:underline}}
pre.log{{background:#f1f5f9;padding:8px;font-size:0.78rem;max-height:200px;overflow:auto;border-radius:4px}}
ul,ol{{margin:4px 0 4px 20px}}
</style></head><body>
<h1>Cap Defend Phase Status</h1>
<p class="meta">updated {now}</p>
<nav>{nav}</nav>
{body}
</body></html>"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
