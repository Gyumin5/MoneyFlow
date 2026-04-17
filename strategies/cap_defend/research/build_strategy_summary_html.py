#!/usr/bin/env python3
"""전략 비교 요약 HTML (모바일 최적화)."""
from __future__ import annotations
import os
import numpy as np
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(HERE, "strategy_summary.html")

START = "2020-10-01"
END = "2026-04-13"


def metrics(eq):
    eq = eq.dropna()
    if len(eq) < 2:
        return {"Sh": 0, "CAGR": 0, "MDD": 0, "Cal": 0}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if eq.iloc[0] <= 0 or yrs <= 0:
        return {"Sh": 0, "CAGR": 0, "MDD": 0, "Cal": 0}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd else 0
    return {"Sh": float(sh), "CAGR": float(cagr), "MDD": float(mdd), "Cal": float(cal)}


print("Fetching SPY/QQQ/BTC...")
df = yf.download(["SPY", "QQQ", "BTC-USD"], start=START, end=END, auto_adjust=True,
                 progress=False)["Close"]
BH = {t: metrics(df[t].dropna()) for t in ["SPY", "QQQ", "BTC-USD"]}


# Strategy data: (name, Cal, CAGR, MDD, Sh)
BENCH = [
    ("SPY", BH["SPY"]["Cal"], BH["SPY"]["CAGR"], BH["SPY"]["MDD"], BH["SPY"]["Sh"]),
    ("QQQ", BH["QQQ"]["Cal"], BH["QQQ"]["CAGR"], BH["QQQ"]["MDD"], BH["QQQ"]["Sh"]),
    ("BTC", BH["BTC-USD"]["Cal"], BH["BTC-USD"]["CAGR"], BH["BTC-USD"]["MDD"], BH["BTC-USD"]["Sh"]),
]

OLD_ALLOC = [
    ("23-01 9자산+Quant*", 0.52, 0.196, -0.377, 0.89),
    ("24-10 8자산", 0.51, 0.154, -0.300, 0.92),
    ("22-01 10자산", 0.45, 0.148, -0.325, 0.84),
    ("25-12 Cash+3주식", 0.44, 0.078, -0.177, 0.92),
    ("21-01 8자산", 0.41, 0.133, -0.327, 0.81),
    ("20-12 12자산", 0.39, 0.138, -0.358, 0.77),
]

SINGLES = [
    ("주식 V17", 1.12, 0.141, -0.125, 1.30),
    ("현물 V20 (D+4h)", 1.83, 0.475, -0.260, 1.69),
    ("현물 spot_D", 1.58, 0.385, -0.244, 1.33),
    ("현물 spot_4h", 1.99, 0.561, -0.282, 1.68),
    ("선물 L2_4h", 2.86, 1.024, -0.358, 1.75),
    ("선물 L2_1D", 2.26, 0.764, -0.338, 1.45),
    ("선물 L2 ENS", 2.63, 0.882, -0.336, 1.73),
    ("선물 L3_4h", 3.01, 1.506, -0.500, 1.70),
    ("선물 L3 ENS", 3.02, 1.425, -0.472, 1.73),
    ("선물 L4_4h", 3.01, 1.902, -0.632, 1.64),
    ("선물 L4 ENS", 2.54, 1.549, -0.610, 1.54),
]

COMBOS = [
    # (desc, ratio, band, Cal, CAGR, MDD, Sh, tag)
    ("spot_4h+L4_4h", "60/10/30", "10%", 3.82, 0.901, -0.236, 2.41, "rank#1"),
    ("spot_V20+L3_4h", "50/10/40", "10%", 3.95, 0.904, -0.229, 2.40, "균형"),
    ("spot_4h+L3_ENS", "60/20/20", "5%", 3.62, 0.538, -0.149, 2.50, "선물≤현물 #1"),
    ("spot_V20+L3_ENS", "60/20/20", "10%", 3.47, 0.537, -0.155, 2.49, "안정형"),
    ("spot_4h+L4_4h", "30/20/50", "no_rebal", 3.41, 2.152, -0.631, 2.01, "고CAGR"),
    ("spot_V20+L3_4h", "70/10/20", "5%", 3.02, 0.509, -0.169, 2.57, "고Sh"),
]

ABLATION_SPOT = [
    ("ON (기본 D15/4h10)", 1.80, 0.474, -0.264, 0.72),
    ("OFF", 1.69, 0.470, -0.278, 0.70),
    ("STRICT (D10/4h7)", 1.64, 0.346, -0.211, 0.64),
    ("LOOSE (D25/4h20)", 1.69, 0.470, -0.278, 0.70),
]

SPOT_D_SWEEP = [
    ("현재 -15%/30d", 1.80, 0.474, -0.264, 0.72),
    ("최적 -12%/21d", 1.89, 0.477, -0.253, 0.74),
    ("-12%/30d", 1.88, 0.477, -0.253, 0.74),
    ("-12%/10d", 1.87, 0.474, -0.253, 0.74),
    ("-18%/7d", 1.81, 0.478, -0.264, 0.72),
]

SPOT_4H_SWEEP = [
    ("현재 -10%/10d", 1.80, 0.474, -0.264, 0.72),
    ("-10%/30d", 1.80, 0.463, -0.257, 0.72),
    ("-10%/45d (스파이크)", 1.92, 0.468, -0.244, 0.73),
    ("-7%/60d", 1.80, 0.368, -0.204, 0.67),
    ("-15%/60d", 1.78, 0.476, -0.268, 0.71),
]

FUT_GUARD = [
    ("none (기본)", 2.98, 1.425, -0.479, 1.73),
    ("prev_close_eq10", 0.88, 0.452, -0.512, 1.02),
    ("prev_close_eq20", 1.61, 0.816, -0.506, 1.29),
    ("trail_N5_eq20", 1.35, 0.737, -0.544, 1.26),
    ("trail_N10_eq30", 1.69, 0.924, -0.546, 1.40),
]

# 선물 gap exclusion sweep (L3 single, 63 combos × 2 members)
FUT_GAP_4H = [
    ("baseline (no guard)", 3.82, 1.900, -0.521, 1.85),
    ("best -15%/30d", 4.02, 2.100, -0.522, 1.90),
    ("-15%/45d", 4.02, 2.095, -0.522, 1.90),
    ("-15%/60d", 4.02, 2.095, -0.522, 1.90),
    ("-12%/7d", 3.87, 2.021, -0.522, 1.88),
    ("-10%/30d (기존)", 3.82, 1.900, -0.521, 1.85),
]

FUT_GAP_1D = [
    ("baseline (no guard)", 2.80, 1.434, -0.513, 1.58),
    ("best -10%/14d", 2.91, 1.493, -0.513, 1.62),
    ("-12%/60d", 2.80, 1.436, -0.513, 1.59),
    ("-15%/30d (현재)", 2.80, 1.434, -0.513, 1.58),
]


def card(name, cal, cagr, mdd, sh, tag=""):
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    return f'''<div class="card">
<div class="card-title">{name}{tag_html}</div>
<div class="metrics">
<div class="m"><span class="lbl">Cal</span><span class="val">{cal:.2f}</span></div>
<div class="m"><span class="lbl">CAGR</span><span class="val {"pos" if cagr>0 else "neg"}">{cagr:+.0%}</span></div>
<div class="m"><span class="lbl">MDD</span><span class="val neg">{mdd:+.0%}</span></div>
<div class="m"><span class="lbl">Sh</span><span class="val">{sh:.2f}</span></div>
</div>
</div>'''


def cards_simple(data):
    return "\n".join(card(*row) for row in data)


def cards_combo(data):
    out = []
    for desc, ratio, band, cal, cagr, mdd, sh, tag in data:
        name = f"{desc} {ratio} b{band}"
        out.append(card(name, cal, cagr, mdd, sh, tag))
    return "\n".join(out)


def cards_ablation(data):
    return "\n".join(card(*row) for row in data)


HTML = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>전략 비교 요약</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Noto Sans KR', sans-serif;
  margin: 0; padding: 12px;
  background: #f5f5f7; color: #1d1d1f;
  font-size: 15px; line-height: 1.45;
}}
h1 {{
  font-size: 22px; margin: 8px 0 4px;
  text-align: center;
}}
.subtitle {{
  text-align: center; color: #666;
  font-size: 12px; margin-bottom: 20px;
}}
h2 {{
  font-size: 17px; margin: 28px 0 10px;
  color: #0070c9; border-bottom: 2px solid #0070c9;
  padding-bottom: 6px;
}}
h3 {{
  font-size: 15px; margin: 18px 0 8px;
  color: #444;
}}
p {{ margin: 6px 0 12px; color: #444; font-size: 14px; }}
.cards {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}}
.card {{
  background: white;
  border-radius: 10px;
  padding: 10px 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.card-title {{
  font-weight: 600; font-size: 14px;
  margin-bottom: 6px;
  display: flex; align-items: center; gap: 6px;
}}
.tag {{
  background: #007aff; color: white;
  padding: 1px 6px; border-radius: 4px;
  font-size: 10px; font-weight: 500;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 4px;
}}
.m {{
  display: flex; flex-direction: column;
  align-items: center;
  background: #fafafa;
  border-radius: 6px;
  padding: 4px 2px;
}}
.lbl {{ font-size: 10px; color: #888; }}
.val {{ font-size: 14px; font-weight: 600; }}
.pos {{ color: #00aa44; }}
.neg {{ color: #d00020; }}
.note {{
  background: #fffbe6; border-left: 3px solid #ffa500;
  padding: 8px 10px; margin: 10px 0;
  font-size: 13px; border-radius: 4px;
  color: #555;
}}
.note b {{ color: #222; }}
ul {{ padding-left: 20px; margin: 8px 0; }}
li {{ margin: 4px 0; font-size: 13px; }}
table {{
  width: 100%; border-collapse: collapse;
  font-size: 13px; margin: 8px 0;
}}
th, td {{
  padding: 5px 6px; border: 1px solid #e0e0e0;
  text-align: center;
}}
th {{ background: #f0f0f0; font-weight: 600; }}
td:first-child {{ text-align: left; font-weight: 500; }}
.summary {{
  background: #e8f4ff; border-radius: 10px;
  padding: 12px; margin: 14px 0;
}}
.summary h3 {{ margin-top: 0; color: #0070c9; }}
@media (min-width: 600px) {{
  .cards {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
</head>
<body>
<h1>전략 비교 요약</h1>
<div class="subtitle">백테스트 2020-10 ~ 2026-04 (5.5년)<br>주식 11-anchor 평균, 코인/선물 tx 0.04%</div>

<h2>벤치마크 (바이앤홀드)</h2>
<div class="cards">{cards_simple(BENCH)}</div>

<h2>옛날 정적 배분 (엑셀)</h2>
<p>주식포트폴리오.xlsx의 5년치 배분. 월간 5% 밴드 리밸 가정.</p>
<div class="cards">{cards_simple(OLD_ALLOC)}</div>
<div class="note">모든 배분이 Cal 0.4~0.5. BTC 바이앤홀드(Cal {BH["BTC-USD"]["Cal"]:.2f})보다는 낮고 SPY(Cal {BH["SPY"]["Cal"]:.2f}) 수준. Quant는 BTC×2 proxy.</div>

<h2>현재 단일 전략</h2>
<p>각 자산별 단독 성과.</p>
<div class="cards">{cards_simple(SINGLES)}</div>

<h2>3자산 조합 베스트</h2>
<p>주식 30-80% × 현물 10-50% × 선물 5-50%, band {{None,3,5,8,10,15%}}, 3078개 조합 탐색.</p>
<div class="cards">{cards_combo(COMBOS)}</div>

<div class="summary">
<h3>주력 후보</h3>
<ul>
<li><b>안정 (선물≤현물):</b> spot_V20 + L3_ENS 60/20/20 b10% → Cal 3.47, MDD-15%</li>
<li><b>균형:</b> spot_V20 + L3_4h 50/10/40 b10% → Cal 3.95, MDD-23%</li>
<li><b>공격:</b> spot_4h + L4_4h 60/10/30 b10% → Cal 3.82, MDD-24%</li>
</ul>
</div>

<h2>핵심 ablation 결론</h2>

<h3>현물 exclusion 규칙 효과</h3>
<p>V20 앙상블에서 gap_threshold 변경 시.</p>
<div class="cards">{cards_ablation(ABLATION_SPOT)}</div>
<div class="note">기본값이 <b>최적</b>. OFF 대비 Cal +0.11, MDD -1.4pp 개선. STRICT는 CAGR 13pp 손실로 과도.</div>

<h3>현물 2D sweep: D_SMA50 (gap × days)</h3>
<p>gap -8%~-25%, days 3~60d 총 56조합.</p>
<div class="cards">{cards_ablation(SPOT_D_SWEEP)}</div>
<div class="note">gap=-12%에서 days 전 구간 Cal 1.84~1.89 <b>강한 plateau</b>. 현재 -15%는 plateau 밖. 개선폭 +0.09 Cal 존재하나 코드 4곳 동기화 부담으로 보류.</div>

<h3>현물 2D sweep: 4h_SMA240 (gap × days)</h3>
<p>gap -5%~-20%, days 3~60d 총 56조합.</p>
<div class="cards">{cards_ablation(SPOT_4H_SWEEP)}</div>
<div class="note">gap=-10%에서 days 3~30 전부 Cal 1.80 근처 plateau. 45d 단독 스파이크(1.92)는 <b>과적합 신호</b>로 무시. 현재값 plateau 중심에 있어 유지.</div>

<h3>선물 스탑 효과 (L3 ENS)</h3>
<p>Phase B guard sweep 결과.</p>
<div class="cards">{cards_ablation(FUT_GUARD)}</div>
<div class="note">스탑 추가는 <b>모든 지표 악화</b>. MDD 방어도 실패(whipsaw). 백테스트 기준 guard=none 최적. 운영 스탑은 블랙스완 보험료 성격.</div>

<h3>선물 gap exclusion 최적 파라미터</h3>
<p>현물식 gap exclusion을 선물에 이식. bl_drop × bl_days 63개 조합 스윕 (L3 단독, 청산 허용).</p>
<h3 style="font-size:13px;color:#666;">4h_S240 (top 후보)</h3>
<div class="cards">{cards_simple(FUT_GAP_4H)}</div>
<h3 style="font-size:13px;color:#666;">1D_S40 (top 후보)</h3>
<div class="cards">{cards_simple(FUT_GAP_1D)}</div>
<div class="note">4h: <b>-15%/30d</b>가 최적 (Cal +0.21, CAGR +20pp). 1D: <b>-10%/14d</b>가 최적 (Cal +0.11). 단 4h는 청산 발생 유지 — gap은 bar close 후 작동하므로 intra-bar 청산은 못 막음. MDD 방어 효과는 미미, 재진입 타이밍 개선이 주효과.</div>

<h3>현물 전략을 선물에 얹으면?</h3>
<p>V20 현물 일별 수익률에 L배 적용 시 (funding/drag 반영).</p>
<table>
<tr><th>레버리지</th><th>Cal</th><th>CAGR</th><th>MDD</th></tr>
<tr><td>1.0x</td><td>1.80</td><td>+47%</td><td>-26%</td></tr>
<tr><td>1.5x</td><td>0.43</td><td>+22%</td><td>-52%</td></tr>
<tr><td>2.0x</td><td>0.00</td><td>+0%</td><td>-86%</td></tr>
<tr><td>3.0x</td><td>-0.36</td><td>-36%</td><td><b>-99%</b></td></tr>
</table>
<div class="note">현물 V20 파라미터(빠른 SMA/Mom)는 레버리지 variance drag에 취약. 선물엔 별도 최적화 필요 → 그래서 d005(느린 SMA240/Mom720).</div>

<h3>봉주기 그리드 비교</h3>
<p>같은 파라미터를 D/4h/2h/1h에 적용.</p>
<table>
<tr><th>봉</th><th>현물 Cal</th><th>현물 Sh</th><th>선물 Cal</th><th>선물 Sh</th></tr>
<tr><td>D</td><td>4.07</td><td>1.86</td><td>5.41</td><td>1.90</td></tr>
<tr><td>4h</td><td>3.35</td><td>1.85</td><td>5.12</td><td>1.86</td></tr>
<tr><td>2h</td><td>2.01</td><td>1.59</td><td>3.60</td><td>1.77</td></tr>
<tr><td>1h</td><td>1.54</td><td>1.33</td><td>2.19</td><td>1.46</td></tr>
</table>
<div class="note">D와 4h가 압도적. 2h/1h는 노이즈 과다. 앙상블은 D+4h만.</div>

<h3>앙상블 효과 (단독 vs ENS)</h3>
<ul>
<li>현물 spot_4h → V20: MDD -28% → -26% (2pp, 작음)</li>
<li>선물 L3_4h → ENS: MDD -52% → -48% (4pp, 유의)</li>
<li>선물은 레버리지 꼬리 리스크가 커서 분산 효과 큼</li>
<li>현물은 Top5 cross-sectional 분산이 이미 있어 추가 효과 작음</li>
</ul>

<h2>운영 권고 (3자 AI 합의)</h2>
<div class="summary">
<h3>현물 V20</h3>
<ul>
<li>앙상블 구조 유지 (D_SMA50 + 4h_SMA240 50:50)</li>
<li>gap exclusion 규칙 유지: D=-15%/30d, 4h=-10%/10d</li>
<li>ablation 기반 ON > OFF 검증 (Cal +0.11, MDD 2pp 방어)</li>
<li>D의 -12% 완화는 다음 업데이트 기회에 패키지로</li>
</ul>
<h3>선물 d005</h3>
<ul>
<li>4전략 EW 앙상블 유지, 3x 레버리지</li>
<li>guard=none 유지 (스탑 추가 시 모든 지표 악화)</li>
<li>gap exclusion 미적용 (1D는 overfit, 4h만 적용 시 비대칭)</li>
<li>청산 방어는 다자산 분산으로</li>
</ul>
<h3>자산배분</h3>
<ul>
<li>현 60/35/5 → 그리드 최적 60/20/20 band 8-10%로 단계적 조정 검토</li>
<li>밴드 리밸: 8pp drift 초과 시만</li>
<li>옛날 정적 배분(Cal 0.4)은 전면 열위</li>
</ul>
</div>

<div class="subtitle" style="margin-top: 30px;">생성 2026-04-13 · 상세 CSV는 research/ 참조</div>
</body>
</html>
"""

with open(OUT_HTML, "w") as f:
    f.write(HTML)
print(f"Wrote {OUT_HTML}")
