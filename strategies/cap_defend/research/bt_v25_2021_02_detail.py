"""V25 spot/fut BT — 2021-02 일별 상세 (PV/수익률/보유 weight/종목별 일별 수익률).

출력: /tmp/v25_detail_spot_202102.html, /tmp/v25_detail_fut_202102.html
"""
from __future__ import annotations
import os, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

FOCUS_YM = "2021-02"
START = "2020-10-01"
END = "2021-03-15"


def _run_spot_trace():
    from unified_backtest import run as bt_run, load_data
    bars, funding = load_data('D')
    trace = []
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
        _trace=trace,
    )
    return m['_equity'], trace, bars


def _run_fut_trace():
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars,
                         btc_cap_sma_period=42,
                         btc_cap_thr_mid=1.015, btc_cap_thr_max=1.05,
                         k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    trace = []
    m = fbt_run(
        bars, funding, interval='D',
        leverage=k2,
        universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015,
        drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END,
        _trace=trace,
    )
    return m.get('_equity'), trace, bars, k2


def daily_target_at(trace_list):
    """trace -> {date -> {coin: w}}."""
    return {pd.Timestamp(t['date']).normalize(): t['target'] for t in trace_list}


def render_html(title, eq, targets, bars, focus_ym, asset, k2=None):
    eq = eq.dropna()
    mask = eq.index.strftime('%Y-%m') == focus_ym
    eq_m = eq.loc[mask]
    if eq_m.empty:
        return f"<html><body>{title}: no data</body></html>"
    # daily returns (전월 마지막일 → 첫째날 시작)
    first_in_focus = eq_m.index[0]
    prev_idx = eq.index.get_indexer([first_in_focus])[0] - 1
    if prev_idx >= 0:
        eq_m_ext = pd.concat([eq.iloc[[prev_idx]], eq_m])
    else:
        eq_m_ext = eq_m
    daily_ret = eq_m_ext.pct_change().dropna() * 100  # %

    # holdings per day: use trace target nearest <= day
    sorted_dates = sorted(targets.keys())
    def target_on(d):
        d_norm = pd.Timestamp(d).normalize()
        # walk back
        for sd in reversed(sorted_dates):
            if sd <= d_norm:
                return targets[sd]
        return {}

    # all coins ever held in focus month
    coin_set = set()
    for d in eq_m.index:
        for c, w in target_on(d).items():
            if c not in ('CASH', 'Cash') and w > 0.001:
                coin_set.add(c)
    coins = sorted(coin_set)

    # per-coin daily price returns within focus_ym
    coin_returns = {}  # coin -> Series of % daily ret
    for c in coins:
        df = bars.get(c)
        if df is None:
            coin_returns[c] = pd.Series(dtype=float); continue
        close = pd.Series(df['Close'].values, index=df.index)
        if prev_idx >= 0:
            start_d = eq.index[prev_idx]
        else:
            start_d = first_in_focus
        seg = close.loc[start_d:eq_m.index[-1]]
        r = seg.pct_change().dropna() * 100
        # restrict to focus month
        coin_returns[c] = r[r.index.strftime('%Y-%m') == focus_ym]

    # Build table
    def cell(v):
        if v is None or pd.isna(v): return '<td class="num">-</td>'
        cls = 'up' if v > 0 else ('down' if v < 0 else 'flat')
        return f'<td class="num {cls}">{v:+.2f}</td>'

    def pct_w(v):
        if v is None or pd.isna(v) or v <= 0.001: return '<td class="num">-</td>'
        return f'<td class="num">{v*100:.1f}%</td>'

    head_coins = ''.join(f'<th>{c}<br>weight</th><th>{c}<br>일%</th>' for c in coins)
    if asset == 'fut':
        head_coins = ''.join(f'<th>{c}<br>weight</th><th>{c}<br>L</th><th>{c}<br>일%</th>' for c in coins)
    rows = []
    for d in eq_m.index:
        pv = eq_m.loc[d]
        dret = daily_ret.get(d)
        tgt = target_on(d)
        c_cells = []
        for c in coins:
            w = tgt.get(c, 0)
            r = coin_returns[c].get(d) if d in coin_returns[c].index else None
            if asset == 'fut':
                lev = None
                if k2 is not None and c in k2:
                    s = k2[c]
                    try: lev = float(s.asof(d))
                    except: lev = None
                lev_cell = f'<td class="num">{lev:.1f}</td>' if lev else '<td class="num">-</td>'
                c_cells.append(pct_w(w) + lev_cell + cell(r))
            else:
                c_cells.append(pct_w(w) + cell(r))
        rows.append(f"<tr><td>{d.strftime('%Y-%m-%d')}</td>"
                    f"<td class=\"num strong\">{pv:.4f}</td>"
                    f"{cell(dret)}"
                    + ''.join(c_cells) + "</tr>")

    asset_label = '현물' if asset == 'spot' else '선물'
    return f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V25 {asset_label} {focus_ym} 상세</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 12px; background: #fafafa; color: #222; font-size: 14px; max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 1.2em; margin: 8px 0; }}
.sub {{ color: #666; font-size: 0.88em; margin-bottom: 12px; }}
.scroll {{ overflow-x: auto; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
th, td {{ padding: 6px 4px; border-bottom: 1px solid #eee; }}
th {{ background: #f5f5f5; font-weight: 700; font-size: 0.78em; white-space: nowrap; position: sticky; top: 0; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.up {{ color: #c62828; font-weight: 600; }}
.down {{ color: #1565c0; font-weight: 600; }}
.flat {{ color: #888; }}
.strong {{ font-weight: 700; }}
</style></head><body>
<h1>V25 {asset_label} {focus_ym} 일별 상세</h1>
<div class="sub">PV 시작가→종가, 일별 % 수익률, 종목별 weight + 일별 % 수익률{' + 동적 L' if asset == 'fut' else ''}</div>
<div class="scroll"><table>
<thead><tr><th>날짜</th><th>PV</th><th>일%</th>{head_coins}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table></div>
</body></html>
"""


def main():
    t0 = time.time()
    print("[spot trace]")
    eq_sp, tr_sp, bars = _run_spot_trace()
    print(f"  spot eq: {len(eq_sp)} dates")
    tgt_sp = daily_target_at(tr_sp)

    print("[fut trace]")
    eq_fu, tr_fu, bars_fu, k2 = _run_fut_trace()
    print(f"  fut eq: {len(eq_fu)} dates")
    tgt_fu = daily_target_at(tr_fu)

    html_sp = render_html("V25 spot 2021-02", eq_sp, tgt_sp, bars, FOCUS_YM, 'spot')
    html_fu = render_html("V25 fut 2021-02", eq_fu, tgt_fu, bars_fu, FOCUS_YM, 'fut', k2)

    with open('/tmp/v25_detail_spot_202102.html', 'w') as f:
        f.write(html_sp)
    with open('/tmp/v25_detail_fut_202102.html', 'w') as f:
        f.write(html_fu)
    print(f"\nwrote spot+fut detail HTML, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
