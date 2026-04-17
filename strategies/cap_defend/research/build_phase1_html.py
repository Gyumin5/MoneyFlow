#!/usr/bin/env python3
"""phase1_status.html 생성기.

계획 + 실행 상태 + 결과(Calmar/Sharpe/CAGR/MDD TOP)를 통합.
disk 상태(run.log / raw.csv / summary.csv)를 읽어 HTML로 렌더.
"""
from __future__ import annotations
import os
import re
import sys
import html
import time
from datetime import datetime

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
SWEEP = os.path.join(HERE, "phase1_sweep")
OUT = os.path.join(HERE, "phase1_status.html")

RUN_LOG = os.path.join(SWEEP, "run.log")
RAW_CSV = os.path.join(SWEEP, "raw.csv")
SUMMARY_CSV = os.path.join(SWEEP, "summary.csv")


def read_progress():
    if not os.path.exists(RUN_LOG):
        return {"status": "not_started", "lines": []}
    with open(RUN_LOG) as f:
        lines = [ln.rstrip() for ln in f.readlines()]
    prog = [ln for ln in lines if re.match(r"\s*\[\d+/\d+\]", ln)]
    last = prog[-1] if prog else ""
    m = re.match(r"\s*\[(\d+)/(\d+)\]\s+pending=(\d+)\s+elapsed=(\d+)s\s+eta=(\d+)s\s+last=(.+)", last)
    if not m:
        m2 = re.match(r"\s*\[(\d+)/(\d+)\]\s+elapsed=(\d+)s\s+eta=(\d+)s\s+last=(.+)", last)
        if m2:
            done, total, el, eta, tag = m2.groups()
            return {"done": int(done), "total": int(total), "pending": int(total)-int(done),
                    "elapsed": int(el), "eta": int(eta), "last_tag": tag, "lines": lines}
        return {"status": "parse_fail", "lines": lines}
    done, total, pending, el, eta, tag = m.groups()
    return {"done": int(done), "total": int(total), "pending": int(pending),
            "elapsed": int(el), "eta": int(eta), "last_tag": tag, "lines": lines}


def fmt_hms(seconds: int) -> str:
    s = int(seconds); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h}h{m:02d}m"


def load_summary() -> pd.DataFrame | None:
    if not os.path.exists(SUMMARY_CSV) or os.path.getsize(SUMMARY_CSV) == 0:
        return None
    try:
        df = pd.read_csv(SUMMARY_CSV)
        return df if not df.empty else None
    except Exception:
        return None


def load_raw_count() -> dict:
    if not os.path.exists(RAW_CSV) or os.path.getsize(RAW_CSV) == 0:
        return {"rows": 0, "errors": 0}
    try:
        df = pd.read_csv(RAW_CSV, on_bad_lines="skip")
        errs = int(df["error"].notna().sum()) if "error" in df.columns else 0
        return {"rows": int(len(df)), "errors": errs}
    except Exception:
        return {"rows": 0, "errors": 0}


def top_table(sdf: pd.DataFrame, by: str, n: int = 10) -> str:
    """by: one of mCal, mSh, mCAGR, mMDD (for mMDD, ascending=True)"""
    if sdf is None or sdf.empty:
        return "<p class='meta'>결과 없음</p>"
    asc = (by == "mMDD")
    d = sdf.sort_values(by, ascending=asc).head(n)
    rows = []
    rows.append("<tr><th>#</th><th>tag</th><th>asset</th><th>lev</th>"
                "<th>Calmar</th><th>Sharpe</th><th>CAGR</th><th>MDD</th>"
                "<th>wMDD</th><th>win_rate</th><th>sCal</th></tr>")
    for i, r in enumerate(d.itertuples(index=False), 1):
        rows.append(
            f"<tr><td>{i}</td><td><code>{html.escape(str(r.tag))}</code></td>"
            f"<td>{r.asset}</td><td>{r.lev:.1f}</td>"
            f"<td>{r.mCal:.2f}</td><td>{r.mSh:.2f}</td>"
            f"<td>{r.mCAGR*100:.1f}%</td><td>{r.mMDD*100:.1f}%</td>"
            f"<td>{r.wMDD*100:.1f}%</td><td>{r.win_rate:.2f}</td>"
            f"<td>{getattr(r, 'sCal', 0):.2f}</td></tr>"
        )
    return "<div class='scroll'><table>" + "".join(rows) + "</table></div>"


def render_results(sdf: pd.DataFrame | None) -> str:
    if sdf is None:
        return "<p class='meta'>summary.csv 없음 — 첫 SUMMARY_EVERY(2000건) 체크포인트 이후 집계</p>"
    out = []
    out.append(f"<p class='meta'>집계 config 수: {len(sdf)}</p>")
    # spot 전용
    spot = sdf[sdf["asset"] == "spot"]
    fut = sdf[sdf["asset"] == "fut"]
    for title, d in [("Spot Top10 (Calmar)", spot), ("Futures Top10 (Calmar)", fut),
                     ("전체 Top10 (Sharpe)", sdf), ("전체 Top10 (CAGR)", sdf),
                     ("전체 Top10 (MDD 낮음)", sdf)]:
        if d is None or d.empty:
            continue
        by = "mCal"
        if "Sharpe" in title: by = "mSh"
        elif "CAGR" in title: by = "mCAGR"
        elif "MDD" in title: by = "mMDD"
        out.append(f"<h3>{title}</h3>" + top_table(d, by, 10))
    return "".join(out)


def render_html(prog: dict, raw: dict, sdf: pd.DataFrame | None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M KST")
    pct = 0.0
    eta_str = "-"
    if "total" in prog and prog.get("total"):
        pct = 100.0 * prog["done"] / prog["total"]
        eta_str = fmt_hms(prog.get("eta", 0))
    status_box = f"""
<div class='kvp'>
<dt>상태</dt><dd>{'진행중' if 'done' in prog else prog.get('status','?')}</dd>
<dt>진행</dt><dd>{prog.get('done','?')} / {prog.get('total','?')} ({pct:.2f}%)</dd>
<dt>대기</dt><dd>{prog.get('pending','?')}</dd>
<dt>경과</dt><dd>{fmt_hms(prog.get('elapsed',0))}</dd>
<dt>ETA</dt><dd>{eta_str}</dd>
<dt>최근 tag</dt><dd><code>{html.escape(prog.get('last_tag','-'))}</code></dd>
<dt>raw rows</dt><dd>{raw['rows']} (error {raw['errors']})</dd>
<dt>갱신</dt><dd>{now}</dd>
</div>
"""
    last_log = "<pre>" + html.escape("\n".join(prog.get("lines", [])[-30:])) + "</pre>"
    results_html = render_results(sdf)

    return f"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Cap Defend — Phase-1 상태</title>
<style>
:root{{--fg:#222;--muted:#666;--bd:#ddd;--bg:#fff;--accent:#2a5;--warn:#c60}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;padding:14px;color:var(--fg);line-height:1.55;background:var(--bg);font-size:15px}}
h1{{font-size:1.35rem;border-bottom:2px solid #333;padding-bottom:6px;margin:6px 0 10px}}
h2{{font-size:1.1rem;margin-top:22px;border-bottom:1px solid var(--bd);padding-bottom:4px}}
h3{{font-size:1rem;margin:14px 0 6px}}
p,li{{font-size:0.95rem}}
ul{{padding-left:20px}}
.meta{{color:var(--muted);font-size:0.8rem;margin-bottom:10px}}
.box{{background:#f6faf6;border-left:4px solid var(--accent);padding:10px 12px;margin:10px 0;border-radius:4px}}
.warn{{background:#fff8f0;border-left-color:var(--warn)}}
.scroll{{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:6px -14px;padding:0 14px}}
table{{border-collapse:collapse;width:100%;min-width:520px;font-size:0.82rem}}
th,td{{border:1px solid var(--bd);padding:5px 7px;text-align:left;vertical-align:top}}
th{{background:#eef1f7;font-weight:600;white-space:nowrap}}
code{{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:0.88em}}
.kvp{{display:grid;grid-template-columns:120px 1fr;gap:4px 10px;font-size:0.88rem;margin:6px 0}}
pre{{background:#0c0c0c;color:#d4d4d4;padding:10px;font-size:0.78rem;border-radius:4px;overflow-x:auto;white-space:pre}}
</style></head><body>

<h1>Cap Defend — Phase-1 상태 (계획 + 실행 + 결과)</h1>
<p class="meta">자동 갱신 (build_phase1_html.py) · {now}</p>

<div class="box">
목표: 현물/선물 전략 브루트포스 재탐색. Phase-1 1앵커 풀기간 + Phase-2 axis-neighbor plateau 필터 → Phase-3 단일계정 앙상블 → Phase-4 3자산 비중/밴드 grid.
</div>

<h2>A. 실행 상태 (라이브)</h2>
{status_box}

<h2>B. 결과 (4지표 Top)</h2>
<p class="meta">지표: Calmar · Sharpe · CAGR · MDD (+ wMDD, win_rate, sCal)</p>
{results_html}

<h2>C. 계획 / 설계</h2>
<ul>
<li>평가 기간 2020-10-01 ~ 2026-04-13</li>
<li>Phase-1 anchors: 시작일 3개 × 3주 간격 (2020-10-01 / 10-22 / 11-12)</li>
<li>제약 <code>Mom_l ≥ 2×Mom_s</code></li>
<li>거래비용 현물 0.4% · 선물 0.04% · 유지증거금 0.4%</li>
<li>n_snapshots=3, snap_interval_bars는 3의 배수</li>
<li>스탑/갭/excl 탈출 없음</li>
</ul>

<h3>그리드</h3>
<div class="scroll"><table>
<tr><th>봉</th><th>SMA</th><th>Mom_s</th><th>Mom_l</th><th>vol</th><th>snap</th></tr>
<tr><td>1D</td><td>20,30,40,50,60,90,150</td><td>10,20,30,40,60,90</td><td>60,90,120,240</td><td>d0.03,d0.05</td><td>12,21,30,45,60</td></tr>
<tr><td>4h</td><td>120,180,240,360,480,720</td><td>20,30,40,60,90,120</td><td>120,240,480,720</td><td>d0.03,b0.50</td><td>21,30,60,84,90,168</td></tr>
<tr><td>2h (선물)</td><td>60,120,180,240,360,480</td><td>10,20,30,40,60,90</td><td>60,120,240,480</td><td>b0.50,d0.03</td><td>12,21,30,60,84,120,168</td></tr>
</table></div>

<h2>D. 코드 안정성 (AI 크로스체크 v3)</h2>
<ul>
<li>RAW_COLUMNS 고정 + append CSV 헤더 중복 방지</li>
<li>매 200건 raw.csv append (flush+fsync) · 매 2000건 summary.csv atomic (tmp+os.replace)</li>
<li>Resume: (tag, anchor) skip. error 행은 재시도</li>
<li>on_bad_lines='skip'으로 partial write 내성</li>
<li>log 'a' append + RUN START 구분선</li>
<li>메모리: 결과 in-memory 누적 제거, checkpoint마다 disk append 후 pending 비우기, 2000건마다 disk 재로드 → del + gc.collect</li>
<li>백테스트 무결성: sig=prev close / 체결=current · tx/slippage/funding/liq 반영 · anchor 간 독립 cold boot</li>
</ul>

<h2>E. 로드맵</h2>
<ol>
<li>Phase-1 실행 (현재)</li>
<li>Phase-2: Top 후보 50-100 선정 + plateau 체크 + 10-anchor OOS 확장</li>
<li>Phase-3: 단일 계정 앙상블 (상관/weight cap)</li>
<li>Phase-4: 3자산 × 12 테이블 (MDD 필터 3단계)</li>
</ol>

<h2>F. 최근 로그 (run.log tail)</h2>
{last_log}

</body></html>
"""


def main():
    prog = read_progress()
    raw = load_raw_count()
    sdf = load_summary()
    html_out = render_html(prog, raw, sdf)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"[OK] {OUT}  ({len(html_out)} bytes)")


if __name__ == "__main__":
    main()
