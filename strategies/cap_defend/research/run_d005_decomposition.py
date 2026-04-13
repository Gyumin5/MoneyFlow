#!/usr/bin/env python3
"""d005 4멤버를 봉주기 D/4h/2h/1h × 방식 A/B로 분해.

방식 A: 변수 숫자 그대로 (예: 4h_d005=SMA240 → D/2h/1h도 SMA240)
방식 B: 시간 환산 (4h SMA240 ≈ 40일 → D=SMA40, 2h=SMA480, 1h=SMA960)

멤버:
  M1 = 4h_d005 (SMA240/Ms20/Ml720/daily5%/Sn60)
  M2 = 2h_S240 (SMA240/Ms20/Ml720/bar60%/Sn120) ← 2h 원본
  M3 = 2h_S120 (SMA120/Ms20/Ml720/bar60%/Sn120) ← 2h 원본
  M4 = 4h_M20  (SMA240/Ms20/Ml120/bar60%/Sn21)

각 멤버의 원본 봉은 A=B=원본.
총 28개 단일전략 평가 + 멤버×방식별 4봉 앙상블 8개.
"""
import csv
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets

START = '2020-10-01'
END = '2026-03-28'
OUT_CSV = os.path.join(_here, 'd005_decomposition.csv')

BASE = dict(canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
            bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3)

# 멤버 정의: (이름, 원본봉, sma_4h, ms_4h, ml_4h, sn_4h, vol_mode, vol_th)
# 시간 단위 환산 기준: 1봉 = (D=24h, 4h=4h, 2h=2h, 1h=1h)
MEMBERS = {
    'M1_d005':   dict(orig='4h', sma=240, ms=20, ml=720, sn=60,  vol_mode='daily', vol_th=0.05),
    'M2_2hS240': dict(orig='2h', sma=240, ms=20, ml=720, sn=120, vol_mode='bar',   vol_th=0.60),
    'M3_2hS120': dict(orig='2h', sma=120, ms=20, ml=720, sn=120, vol_mode='bar',   vol_th=0.60),
    'M4_4hM20':  dict(orig='4h', sma=240, ms=20, ml=120, sn=21,  vol_mode='bar',   vol_th=0.60),
}

# 봉별 시간계수: 원본 4h 또는 2h 기준 봉 단위로 환산
# bars per day: D=1, 4h=6, 2h=12, 1h=24
BPD = {'D': 1, '4h': 6, '2h': 12, '1h': 24}


def make_cfg_A(member, target_iv):
    """A: 변수 숫자 그대로, 봉만 변경."""
    m = MEMBERS[member]
    return dict(
        interval=target_iv,
        sma_bars=m['sma'], mom_short_bars=m['ms'], mom_long_bars=m['ml'],
        snap_interval_bars=m['sn'],
        vol_mode=m['vol_mode'], vol_threshold=m['vol_th'],
        **BASE,
    )


def make_cfg_B(member, target_iv):
    """B: 시간 환산. 원본 봉의 시간 길이를 target 봉 단위로 변환."""
    m = MEMBERS[member]
    orig = m['orig']
    factor = BPD[target_iv] / BPD[orig]
    sma = max(2, int(round(m['sma'] * factor)))
    ms = max(2, int(round(m['ms'] * factor)))
    ml = max(2, int(round(m['ml'] * factor)))
    sn = max(1, int(round(m['sn'] * factor)))
    return dict(
        interval=target_iv,
        sma_bars=sma, mom_short_bars=ms, mom_long_bars=ml,
        snap_interval_bars=sn,
        vol_mode=m['vol_mode'], vol_threshold=m['vol_th'],
        **BASE,
    )


def gen_trace(data, cfg):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **cfg)
    return trace


def run_combined(bars_1h, funding_1h, traces_dict, weights, guard):
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    combined = combine_targets(traces_dict, weights, all_dates)
    g = dict(guard)
    eng = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=g.get('leverage', 4.0),
        stop_kind=g.get('stop_kind', 'none'),
        stop_pct=g.get('stop_pct', 0.0),
        stop_gate=g.get('stop_gate', 'always'),
        stop_gate_cash_threshold=g.get('stop_gate_cash_threshold', 0.0),
        leverage_mode=g.get('leverage_mode', 'fixed'),
        per_coin_leverage_mode=g.get('per_coin_leverage_mode', 'none'),
        leverage_floor=3.0, leverage_mid=4.0, leverage_ceiling=5.0,
        leverage_cash_threshold=0.34,
        leverage_partial_cash_threshold=0.0,
        leverage_count_floor_max=2, leverage_count_mid_max=4,
        leverage_canary_floor_gap=0.015,
        leverage_canary_mid_gap=0.04,
        leverage_canary_high_gap=0.08,
        leverage_canary_sma_bars=1200,
        leverage_mom_lookback_bars=24*30,
        leverage_vol_lookback_bars=24*90,
    )
    return eng.run(combined)


GUARDS = {
    'G0': dict(leverage=3.0, leverage_mode='fixed', per_coin_leverage_mode='none', stop_kind='none'),
    'G4': dict(leverage=4.0, leverage_mode='fixed', per_coin_leverage_mode='cap_mom_blend_543_cash',
               stop_kind='prev_close_pct', stop_pct=0.15,
               stop_gate='cash_guard', stop_gate_cash_threshold=0.34),
}


def main():
    t0 = time.time()
    print('Loading data: D, 4h, 2h, 1h ...', flush=True)
    data = {iv: load_data(iv) for iv in ['D', '4h', '2h', '1h']}
    bars_1h, funding_1h = data['1h']

    # Build all single-strategy variants. Dedup by (mode,iv,sma,ms,ml,sn,vol_mode,vol_th).
    variants = {}  # name → cfg
    for mname in MEMBERS:
        orig = MEMBERS[mname]['orig']
        for iv in ['D', '4h', '2h', '1h']:
            for mode, fn in [('A', make_cfg_A), ('B', make_cfg_B)]:
                # 원본 봉이면 mode 표기 생략(A=B=원본)
                cfg = fn(mname, iv)
                key = (cfg['interval'], cfg['sma_bars'], cfg['mom_short_bars'],
                       cfg['mom_long_bars'], cfg['snap_interval_bars'],
                       cfg['vol_mode'], cfg['vol_threshold'])
                if iv == orig:
                    name = f'{mname}__{iv}_orig'
                else:
                    name = f'{mname}__{iv}_{mode}'
                # Skip duplicates (e.g. A/B identical at orig bar)
                if any(v == key for v in variants.values() if isinstance(v, tuple)):
                    pass
                variants[name] = (key, cfg)

    # Build unique trace set
    print(f'Total variants: {len(variants)}', flush=True)
    unique_keys = {}
    for name, (key, cfg) in variants.items():
        if key not in unique_keys:
            unique_keys[key] = (name, cfg)
    print(f'Unique configs: {len(unique_keys)}', flush=True)

    # Generate traces for unique configs
    traces = {}
    name_to_key = {name: key for name, (key, cfg) in variants.items()}
    for i, (key, (name, cfg)) in enumerate(unique_keys.items(), 1):
        ts = time.time()
        traces[key] = gen_trace(data, cfg)
        print(f'  [{i}/{len(unique_keys)}] {name} ({cfg["interval"]} sma{cfg["sma_bars"]} '
              f'ms{cfg["mom_short_bars"]} ml{cfg["mom_long_bars"]} sn{cfg["snap_interval_bars"]}) '
              f'{time.time()-ts:.0f}s', flush=True)

    # Evaluate each variant as single-strategy fixed_3x and capmom G4
    rows = []
    print('\n=== Single strategy evaluation ===', flush=True)
    for i, (name, (key, cfg)) in enumerate(variants.items(), 1):
        for gd in ['G0', 'G4']:
            ts = time.time()
            m = run_combined(bars_1h, funding_1h, {name: traces[key]}, {name: 1.0}, GUARDS[gd])
            dt = time.time() - ts
            rows.append(dict(
                kind='single', case=f'{name}|{gd}', member=name.split('__')[0],
                bar=cfg['interval'], mode=name.split('__')[1].split('_')[-1],
                guard=gd,
                sma=cfg['sma_bars'], ms=cfg['mom_short_bars'], ml=cfg['mom_long_bars'],
                sn=cfg['snap_interval_bars'], vol_mode=cfg['vol_mode'], vol_th=cfg['vol_threshold'],
                Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                Liq=m.get('Liq', 0), Stops=m.get('Stops', 0), elapsed=round(dt, 1),
            ))
        if i % 10 == 0 or i == len(variants):
            r = rows[-1]
            print(f'  [{i}/{len(variants)}] {name:<28} G4 Sh{r["Sharpe"]:.2f} Cal{r["Cal"]:.2f}',
                  flush=True)

    # Member×Mode 4봉 앙상블 (D+4h+2h+1h, EW)
    print('\n=== Member×Mode 4-bar ensemble ===', flush=True)
    for mname in MEMBERS:
        for mode in ['A', 'B']:
            members = []
            sub = {}
            for iv in ['D', '4h', '2h', '1h']:
                orig = MEMBERS[mname]['orig']
                if iv == orig:
                    name = f'{mname}__{iv}_orig'
                else:
                    name = f'{mname}__{iv}_{mode}'
                members.append(name)
                sub[name] = traces[name_to_key[name]]
            weights = {n: 1.0/len(members) for n in members}
            for gd in ['G0', 'G4']:
                ts = time.time()
                m = run_combined(bars_1h, funding_1h, sub, weights, GUARDS[gd])
                dt = time.time() - ts
                rows.append(dict(
                    kind='ens4bar', case=f'{mname}_{mode}_4bar|{gd}',
                    member=mname, bar='4bar', mode=mode, guard=gd,
                    sma=0, ms=0, ml=0, sn=0, vol_mode='mix', vol_th=0,
                    Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                    MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                    Liq=m.get('Liq', 0), Stops=m.get('Stops', 0), elapsed=round(dt, 1),
                ))
                print(f'  {mname}_{mode}_4bar|{gd}  Sh{m.get("Sharpe",0):.2f} '
                      f'Cal{m.get("Cal",0):.2f} ({dt:.0f}s)', flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {OUT_CSV}  ({time.time()-t0:.0f}s total)', flush=True)

    # Top by Cal (G4)
    g4 = [r for r in rows if r['guard'] == 'G4']
    g4.sort(key=lambda r: -r['Cal'])
    print('\nTop 25 by Cal (G4):')
    print(f'{"case":<45} {"kind":<7} {"bar":<5} {"mode":<5} {"Sh":>5} {"Cal":>5} {"CAGR":>7} {"MDD":>7}')
    for r in g4[:25]:
        print(f'{r["case"][:45]:<45} {r["kind"]:<7} {r["bar"]:<5} {str(r["mode"]):<5} '
              f'{r["Sharpe"]:>5.2f} {r["Cal"]:>5.2f} {r["CAGR"]:>+7.1%} {r["MDD"]:>+7.1%}')


if __name__ == '__main__':
    main()
