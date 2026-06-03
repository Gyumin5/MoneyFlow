"""V25 라이브 spec BT — 3자산 + 합성 (60/25/15) 일별 equity → 월별 수익률 HTML.

완벽 BT (단순화 없음):
- 주식 V25: 3-mom (30/72/230) + multi-snap (n=3 stag=23 int=69) + cap+Cash + thr=0.05
- 현물 V24=V25: D_SMA42 + mom2vol (20,127) + sn=217 n=7 + drift=0.10
- 선물 V25: backtest_futures_v25.run + 동적 K2 (per-coin L from build_K2_signal)
- 합성 60/25/15: 일별 PV 합성 (주식 60% + 현물 25% + 선물 15%), 매월 1일 reset

기간: 2020-10-01 ~ 2026-04-13
출력: /tmp/v25_monthly.html
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"


def run_stock_v25():
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
    sd = pd.Timestamp(START); ed = pd.Timestamp(END)
    eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor=0,
                       drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
                       snap_int=69, n_snaps=3)
    return eq


def run_spot_v25():
    from unified_backtest import run as bt_run, load_data
    bars, funding = load_data('D')
    m = bt_run(
        bars, funding, interval='D',
        asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
    )
    return m['_equity'] if m and '_equity' in m else None


def run_fut_v25_dynamic():
    """V25 완벽 BT — backtest_futures_v25 + 동적 K2 per-coin L."""
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars,
                         btc_cap_sma_period=42,
                         btc_cap_thr_mid=1.015, btc_cap_thr_max=1.05,
                         k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    m = fbt_run(
        bars, funding, interval='D',
        leverage=k2,  # per-coin dynamic L
        universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015,
        drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END,
    )
    return m.get('_equity') if m else None


def to_monthly_pct(eq):
    eq = eq.dropna()
    monthly = eq.resample('M').last()
    pct = monthly.pct_change() * 100
    return monthly, pct


def build_alloc_eq(eq_st, eq_sp, eq_fu, w_st=0.60, w_sp=0.25, w_fu=0.15):
    """B 안: 매월 1일 가상 리셋 (자산배분 가정).
    매월 시작 시 60/25/15 비중. 월 안에서 단순 결합 수익률 = 가중합 일별 수익률.
    """
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    if len(common) < 30:
        return None
    s_st = eq_st.loc[common]; s_sp = eq_sp.loc[common]; s_fu = eq_fu.loc[common]
    r_st = s_st.pct_change().fillna(0)
    r_sp = s_sp.pct_change().fillna(0)
    r_fu = s_fu.pct_change().fillna(0)
    r_alloc = w_st * r_st + w_sp * r_sp + w_fu * r_fu
    eq = (1 + r_alloc).cumprod()
    return eq


def metrics(eq):
    eq = eq.dropna()
    if len(eq) < 30: return ('-', '-', '-', '-')
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1]/eq.iloc[0]) ** (1/yrs) - 1
    peak = eq.cummax(); mdd = (eq/peak - 1).min()
    rets = eq.pct_change().dropna()
    sh = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else 0
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return (f"{cagr*100:.1f}%", f"{mdd*100:.1f}%", f"{sh:.2f}", f"{cal:.2f}")


def cell_pct(p):
    if p is None or pd.isna(p): return '<td class="num">-</td>'
    cls = 'up' if p > 0 else ('down' if p < 0 else 'flat')
    return f'<td class="num {cls}">{p:+.2f}%</td>'


def main():
    t0 = time.time()
    print("[stock V25 BT]")
    eq_st = run_stock_v25()
    print(f"  stock eq: {len(eq_st)} dates, last={eq_st.iloc[-1]:.3f}")

    print("[spot V25 BT (=V24 spec)]")
    eq_sp = run_spot_v25()
    print(f"  spot eq: {len(eq_sp)} dates, last={eq_sp.iloc[-1]:.3f}")

    print("[fut V25 BT — 동적 K2 per-coin L]")
    eq_fu = run_fut_v25_dynamic()
    if eq_fu is None:
        print("  fut BT FAILED"); sys.exit(1)
    print(f"  fut eq: {len(eq_fu)} dates, last={eq_fu.iloc[-1]:.3f}")

    print("[합성 60/25/15]")
    eq_al = build_alloc_eq(eq_st, eq_sp, eq_fu, 0.60, 0.25, 0.15)
    if eq_al is None:
        print("  합성 BT FAILED"); sys.exit(1)
    print(f"  alloc eq: {len(eq_al)} dates, last={eq_al.iloc[-1]:.3f}")

    m_st, p_st = to_monthly_pct(eq_st)
    m_sp, p_sp = to_monthly_pct(eq_sp)
    m_fu, p_fu = to_monthly_pct(eq_fu)
    m_al, p_al = to_monthly_pct(eq_al)

    months = sorted(set(m_st.index) | set(m_sp.index) | set(m_fu.index) | set(m_al.index))

    cum_st = 1.0; cum_sp = 1.0; cum_fu = 1.0; cum_al = 1.0
    rows_html = []
    for m in months:
        ym = m.strftime('%Y-%m')
        p1 = p_st.get(m); p2 = p_sp.get(m); p3 = p_fu.get(m); p4 = p_al.get(m)
        if not pd.isna(p1): cum_st *= (1 + p1/100)
        if not pd.isna(p2): cum_sp *= (1 + p2/100)
        if not pd.isna(p3): cum_fu *= (1 + p3/100)
        if not pd.isna(p4): cum_al *= (1 + p4/100)
        rows_html.append(
            f"<tr><td>{ym}</td>"
            f"{cell_pct(p1)}{cell_pct(p2)}{cell_pct(p3)}{cell_pct(p4)}"
            f"<td class=\"num\">{(cum_st-1)*100:+.1f}%</td>"
            f"<td class=\"num\">{(cum_sp-1)*100:+.1f}%</td>"
            f"<td class=\"num\">{(cum_fu-1)*100:+.1f}%</td>"
            f"<td class=\"num strong\">{(cum_al-1)*100:+.1f}%</td>"
            f"</tr>"
        )

    m_st_m = metrics(eq_st); m_sp_m = metrics(eq_sp); m_fu_m = metrics(eq_fu); m_al_m = metrics(eq_al)

    # chart data
    import json
    chart_labels = [m.strftime('%Y-%m') for m in months]
    def cum_series(p_series):
        out = []; c = 1.0
        for m in months:
            v = p_series.get(m)
            if not pd.isna(v): c *= (1 + v/100)
            out.append((c - 1) * 100)
        return out
    chart_data = json.dumps({
        'labels': chart_labels,
        'stock': cum_series(p_st),
        'spot': cum_series(p_sp),
        'fut': cum_series(p_fu),
        'alloc': cum_series(p_al),
    })

    html_doc = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V25 BT 월별 수익률 (완벽 + 60/25/15)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 12px; background: #fafafa; color: #222; max-width: 800px; margin: 0 auto; font-size: 16px; }}
h1 {{ font-size: 1.25em; margin: 8px 0; }}
.sub {{ color: #666; font-size: 0.88em; margin-bottom: 14px; }}
.summary {{ background: #fff; padding: 12px; border-radius: 8px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.summary table {{ width: 100%; font-size: 0.85em; }}
.summary th, .summary td {{ padding: 6px 4px; text-align: right; }}
.summary th:first-child, .summary td:first-child {{ text-align: left; font-weight: 600; }}
.note {{ background: #e3f2fd; border-left: 4px solid #1976d2; padding: 10px 12px; border-radius: 6px; font-size: 0.85em; margin: 8px 0 16px; }}
.chart-wrap {{ background: #fff; padding: 12px; border-radius: 8px; margin: 14px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
canvas {{ max-height: 320px; }}
.legend {{ display: flex; gap: 10px; font-size: 0.85em; margin-top: 6px; flex-wrap: wrap; }}
.legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; }}
table.mret {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 0.78em; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
table.mret th, table.mret td {{ padding: 6px 3px; border-bottom: 1px solid #eee; }}
table.mret th {{ background: #f5f5f5; font-weight: 700; font-size: 0.75em; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.up {{ color: #c62828; font-weight: 600; }}
.down {{ color: #1565c0; font-weight: 600; }}
.flat {{ color: #888; }}
.strong {{ font-weight: 700; }}
</style></head><body>
<h1>V25 BT 월별 수익률</h1>
<div class="sub">기간 {START} ~ {END} · 일별 equity → 월말 resample → MoM %</div>

<div class="note">BT 전략 (완벽 spec):
주식 V25 — 3-mom (30/72/230), multi-snap n=3 stag=23 int=69, cap+Cash, thr=0.05<br>
현물 V25 — D_SMA42, mom2vol (20,127), sn=217 n=7, drift=0.10<br>
선물 V25 — D_SMA42, mom2vol (18,127), sn=95 n=5, drift=0.03, 동적 K2 per-coin L (Lmin=2 Lmid=3 Lmax=4, BTC_cap × K2 min)<br>
합성 — 자산배분 60% 주식 / 25% 현물 / 15% 선물 (일별 가중합 수익률)
</div>

<div class="summary">
<h3 style="margin:0 0 8px;font-size:1.0em;">전체 성과 ({START} ~ {END})</h3>
<table>
<tr><th></th><th>주식</th><th>현물</th><th>선물</th><th>60/25/15</th></tr>
<tr><td>CAGR</td><td class="num">{m_st_m[0]}</td><td class="num">{m_sp_m[0]}</td><td class="num">{m_fu_m[0]}</td><td class="num strong">{m_al_m[0]}</td></tr>
<tr><td>MDD</td><td class="num">{m_st_m[1]}</td><td class="num">{m_sp_m[1]}</td><td class="num">{m_fu_m[1]}</td><td class="num strong">{m_al_m[1]}</td></tr>
<tr><td>Sharpe</td><td class="num">{m_st_m[2]}</td><td class="num">{m_sp_m[2]}</td><td class="num">{m_fu_m[2]}</td><td class="num strong">{m_al_m[2]}</td></tr>
<tr><td>Calmar</td><td class="num">{m_st_m[3]}</td><td class="num">{m_sp_m[3]}</td><td class="num">{m_fu_m[3]}</td><td class="num strong">{m_al_m[3]}</td></tr>
</table>
</div>

<div class="chart-wrap">
<h3 style="margin:0 0 8px;font-size:1.0em;">누적 수익률 추이 (%)</h3>
<canvas id="cum"></canvas>
<div class="legend">
  <span><span class="dot" style="background:#7627bb"></span>60/25/15 합성</span>
  <span><span class="dot" style="background:#1976d2"></span>주식</span>
  <span><span class="dot" style="background:#388e3c"></span>현물</span>
  <span><span class="dot" style="background:#f57c00"></span>선물</span>
</div>
</div>

<table class="mret">
<thead><tr>
<th>월</th>
<th>주식<br>월%</th><th>현물<br>월%</th><th>선물<br>월%</th><th>합성<br>월%</th>
<th>주식<br>누적%</th><th>현물<br>누적%</th><th>선물<br>누적%</th><th>합성<br>누적%</th>
</tr></thead>
<tbody>
{chr(10).join(rows_html)}
</tbody></table>

<script>
const D = {chart_data};
new Chart(document.getElementById('cum'), {{
  type: 'line', data: {{ labels: D.labels, datasets: [
    {{ label: '60/25/15 합성', data: D.alloc, borderColor: '#7627bb', borderWidth: 2.5, tension: 0.2, pointRadius: 0 }},
    {{ label: '주식', data: D.stock, borderColor: '#1976d2', borderWidth: 1.5, tension: 0.2, pointRadius: 0 }},
    {{ label: '현물', data: D.spot, borderColor: '#388e3c', borderWidth: 1.5, tension: 0.2, pointRadius: 0 }},
    {{ label: '선물', data: D.fut, borderColor: '#f57c00', borderWidth: 1.5, tension: 0.2, pointRadius: 0 }},
  ]}},
  options: {{ responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ mode: 'index', intersect: false }} }},
    interaction: {{ mode: 'index', intersect: false }},
    scales: {{ y: {{ ticks: {{ callback: v => v + '%' }} }} }}
  }}
}});
</script>
</body></html>
"""
    with open('/tmp/v25_monthly.html', 'w') as f:
        f.write(html_doc)
    print(f"\nwrote /tmp/v25_monthly.html ({len(months)} months, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
