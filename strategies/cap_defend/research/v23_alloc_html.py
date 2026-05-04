"""V23 자산배분 그리드 BT + HTML 보고서.

V23 sleeve 고정
- spot: D_SMA42 sn=217 n=7 drift=0.10
- fut:  D_SMA42 sn=57 n=3 drift=0.05 L3
- stock: SNAP=69 STAGGER=23 N=3 (sd=69 n=3)

자산배분 그리드: 2.5% step, 주식 ≥ 현물 ≥ 선물 (3개 weight 합=100)
베이스라인: 60/40/0

산출: v23_alloc_report.html (정렬 가능 테이블, baseline 강조, plateau 시각화)
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START = '2020-10-01'
END = pd.Timestamp.now().strftime('%Y-%m-%d')
BASELINE = (60.0, 40.0, 0.0)

print(f'== V23 alloc grid BT ({START} ~ {END}) ==')


def daily_norm(eq):
    s = eq.resample('1D').last().dropna()
    return s / s.iloc[0]


def metrics(eq):
    eq = eq.dropna()
    if len(eq) < 2:
        return dict(cal=0, cagr=0, mdd=0, sh=0, ymin=0)
    rt = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    sh = rt.mean() / rt.std() * np.sqrt(252) if rt.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    yearly = eq.resample('A').last().pct_change().dropna()
    ymin = yearly.min() if len(yearly) > 0 else 0
    return dict(cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin)


# ─── sleeve BT ───
print('\n[1/3] coin spot BT (D_SMA42 sn=217 n=7 d=0.10)...')
t0 = time.time()
bars_D, funding = ub.load_data('D')
spot_eq = ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
                 sma_days=42, mom_short_days=20, mom_long_days=127,
                 vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=7,
                 universe_size=3, cap=1/3, tx_cost=0.004,
                 health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
                 snap_interval_bars=217, start_date=START, end_date=END)['_equity']
spot_eq = daily_norm(spot_eq)
print(f'  done ({time.time()-t0:.0f}s, bars={len(spot_eq)})')

print('\n[2/3] coin fut BT (D_SMA42 sn=57 n=3 d=0.05 L3)...')
t0 = time.time()
fut_eq = ub.run(bars_D, funding, interval='D', asset_type='fut', leverage=3.0,
                sma_days=42, mom_short_days=18, mom_long_days=127,
                vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
                universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
                health_mode='mom2vol', vol_mode='daily', drift_threshold=0.05,
                snap_interval_bars=57, start_date=START, end_date=END)['_equity']
fut_eq = daily_norm(fut_eq)
print(f'  done ({time.time()-t0:.0f}s, bars={len(fut_eq)})')

print('\n[3/3] stock BT (sd=69 n=3)...')
t0 = time.time()
OFF = ('SPY', 'QQQ', 'VEA', 'EEM', 'EWJ', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')
CAN = ('EEM',)
ts._g_prices = ts.load_prices(list(set(OFF + DEF + CAN)), start='2014-01-01')
ts._g_ind = ts.precompute(ts._g_prices)
sp = ts.SP(offensive=OFF, defensive=DEF, canary_assets=CAN,
           canary_sma=300, canary_hyst=0.020, canary_type='sma',
           health='none', defense='top2', defense_sma=100, def_mom_period=126,
           select='zscore3', n_mom=3, n_sh=3, sharpe_lookback=126,
           weight='ew', crash='none', tx_cost=0.001, start=START, end=END, capital=10000.0)
stock_eq = tss.run_snapshot(sp, snap_days=69, n_snap=3)['Value']
stock_eq = daily_norm(stock_eq)
print(f'  done ({time.time()-t0:.0f}s, bars={len(stock_eq)})')

# ─── 공통 인덱스로 정렬 ───
common = stock_eq.index.intersection(spot_eq.index).intersection(fut_eq.index)
stock_eq = stock_eq.loc[common]
spot_eq = spot_eq.loc[common]
fut_eq = fut_eq.loc[common]
print(f'\n공통 인덱스: {len(common)} bars ({common[0].date()} ~ {common[-1].date()})')

# ─── alloc 그리드 ───
STEP = 2.5
weights = np.arange(0, 100 + STEP / 2, STEP)
allocs = []
for f in weights:
    for s in weights:
        st = 100 - f - s
        if abs(st - round(st / STEP) * STEP) > 1e-6:
            continue
        if st < 0:
            continue
        # constraint: stock >= spot >= fut
        if not (st >= s >= f):
            continue
        allocs.append((st, s, f))
print(f'\n그리드 alloc 개수: {len(allocs)} (step {STEP}%, 주식≥현물≥선물)')

# ─── 포트폴리오 BT ───
rows = []
for (w_st, w_sp, w_f) in allocs:
    # 일별 수익률 가중 합 → cumulative
    rt_st = stock_eq.pct_change().fillna(0)
    rt_sp = spot_eq.pct_change().fillna(0)
    rt_f = fut_eq.pct_change().fillna(0)
    port_rt = (w_st / 100 * rt_st + w_sp / 100 * rt_sp + w_f / 100 * rt_f)
    port_eq = (1 + port_rt).cumprod()
    m = metrics(port_eq)
    rows.append(dict(stock=w_st, spot=w_sp, fut=w_f,
                     **m,
                     is_baseline=(w_st, w_sp, w_f) == BASELINE))

df = pd.DataFrame(rows)
df = df.sort_values('cal', ascending=False).reset_index(drop=True)

# ─── HTML 출력 ───
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', 'v23_alloc_report.html')
HTML_PATH = os.path.normpath(HTML_PATH)


def fmt_pct(v, decimals=1):
    return f'{v * 100:+.{decimals}f}%'


# 베이스라인 row
bl = df[df['is_baseline']].iloc[0] if df['is_baseline'].any() else None

# 색상 (Cal 기준 heatmap)
cal_min = df['cal'].min()
cal_max = df['cal'].max()


def cal_color(v):
    if cal_max == cal_min:
        return '#fff'
    t = (v - cal_min) / (cal_max - cal_min)
    # 빨강 → 노랑 → 초록
    if t < 0.5:
        r, g, b = 255, int(255 * 2 * t), 0
    else:
        r, g, b = int(255 * 2 * (1 - t)), 255, 0
    return f'rgb({r},{g},{b})'


def build_table(df, title):
    rows_html = []
    for _, r in df.iterrows():
        bg = cal_color(r['cal'])
        bl_mark = ' style="border:2px solid #000; font-weight:bold;"' if r['is_baseline'] else ''
        cls = 'baseline' if r['is_baseline'] else ''
        rows_html.append(
            f'<tr class="{cls}"{bl_mark}>'
            f'<td>{r["stock"]:.1f}</td>'
            f'<td>{r["spot"]:.1f}</td>'
            f'<td>{r["fut"]:.1f}</td>'
            f'<td style="background:{bg}">{r["cal"]:.2f}</td>'
            f'<td>{fmt_pct(r["cagr"], 1)}</td>'
            f'<td>{fmt_pct(r["mdd"], 1)}</td>'
            f'<td>{r["sh"]:.2f}</td>'
            f'<td>{fmt_pct(r["ymin"], 1)}</td>'
            f'</tr>'
        )
    return (
        f'<h2>{title}</h2>'
        '<table>'
        '<thead><tr>'
        '<th>주식%</th><th>현물%</th><th>선물%</th>'
        '<th>Cal</th><th>CAGR</th><th>MDD</th><th>Sharpe</th><th>ymin</th>'
        '</tr></thead>'
        '<tbody>' + '\n'.join(rows_html) + '</tbody>'
        '</table>'
    )


# 베이스라인 + top10 + bottom5
top10 = df.head(10)
bot5 = df.tail(5)
baseline_html = ''
if bl is not None:
    baseline_html = (
        f'<h2>📍 베이스라인 60/40/0</h2>'
        f'<table><thead><tr><th>지표</th><th>값</th></tr></thead><tbody>'
        f'<tr><td>Cal</td><td><b>{bl["cal"]:.2f}</b></td></tr>'
        f'<tr><td>CAGR</td><td>{fmt_pct(bl["cagr"], 1)}</td></tr>'
        f'<tr><td>MDD</td><td>{fmt_pct(bl["mdd"], 1)}</td></tr>'
        f'<tr><td>Sharpe</td><td>{bl["sh"]:.2f}</td></tr>'
        f'<tr><td>ymin</td><td>{fmt_pct(bl["ymin"], 1)}</td></tr>'
        f'<tr><td>전체 순위</td><td>{df.index[df["is_baseline"]][0] + 1} / {len(df)}</td></tr>'
        f'</tbody></table>'
    )

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>V23 자산배분 BT 보고서</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 24px; max-width: 1200px; }}
h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 8px; }}
h2 {{ color: #555; margin-top: 32px; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
th, td {{ padding: 6px 12px; border: 1px solid #ccc; text-align: right; }}
th {{ background: #f5f5f5; font-weight: 600; }}
.baseline {{ background: #fff3cd !important; }}
.note {{ color: #666; font-size: 0.9em; line-height: 1.6; }}
</style>
</head>
<body>
<h1>V23 자산배분 백테스트 보고서</h1>
<div class="note">
기간: {common[0].date()} ~ {common[-1].date()} ({len(common)} bars)<br>
Sleeve: spot D_SMA42 sn=217 n=7 d=0.10 / fut D_SMA42 sn=57 n=3 d=0.05 L3 / stock sd=69 n=3<br>
그리드: 2.5% step, 주식 ≥ 현물 ≥ 선물 ({len(allocs)} alloc)<br>
베이스라인: {BASELINE[0]:.0f}/{BASELINE[1]:.0f}/{BASELINE[2]:.0f} (주식/현물/선물)
</div>

{baseline_html}

{build_table(top10, '🏆 Top 10 Cal')}

{build_table(bot5, '⬇ Bottom 5 Cal')}

{build_table(df, f'전체 alloc {len(df)} 개 (Cal 내림차순)')}

</body>
</html>
"""

with open(HTML_PATH, 'w') as fh:
    fh.write(html)
print(f'\n✅ HTML 저장: {HTML_PATH}')
print(f'\n베이스라인 60/40/0:')
if bl is not None:
    print(f'  Cal={bl["cal"]:.2f} CAGR={bl["cagr"]:+.1%} MDD={bl["mdd"]:+.1%} Sh={bl["sh"]:.2f} ymin={bl["ymin"]:+.1%}')
    print(f'  전체 순위 {df.index[df["is_baseline"]][0] + 1} / {len(df)}')
print(f'\nTop 5:')
for _, r in df.head(5).iterrows():
    print(f"  {r['stock']:.1f}/{r['spot']:.1f}/{r['fut']:.1f}  Cal={r['cal']:.2f}  CAGR={r['cagr']:+.1%}  MDD={r['mdd']:+.1%}  ymin={r['ymin']:+.1%}")
