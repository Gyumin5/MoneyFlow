"""통제 패리티 하니스 — 코인 선물 (fut).

라이브 선정함수(auto_trade_binance.compute_strategy_target)와
BT 선정함수(unified_backtest.run/_compute_weights asset_type='fut')가 동일 유니버스·
동일 OHLCV·동일 BTC 카나리 입력에서 일별 picks/weights 100% 일치하는지 증명.

선물 라이브 컨벤션 차이 처리:
- compute_strategy_target 은 bars[-1] 을 진행중(미완성) 봉으로 보고 signal=index[-2].
  → 라이브 driver 는 bslice 를 d+1 까지 줘서 index[-2]=d(signal), index[-1]=d+1(in-progress).
- 종가 자름: 라이브=[:-1], BT=ffill[:ci+1]. 정상(무결측) 일봉에서 동일.

레버리지/마진은 execution 영역 → target weights(선정)에는 무관. read-only.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'trade'))

import unified_backtest as ub
import auto_trade_binance as atb

START = '2020-10-01'
END = '2026-05-09'   # 마지막 날은 in-progress 봉 필요 → 하루 여유

_REF_ORDER = ['BTC','ETH','BNB','SOL','XRP','ADA','AVAX','DOGE','DOT','MATIC',
              'TRX','LINK','LTC','UNI','ATOM','BCH','XLM','VET','FIL','THETA',
              'EOS','AAVE','ALGO','GRT','NEAR','FTM','SAND','MANA','AXS','GALA',
              'CHZ','APE','FLOW','ICP','APT','ARB','OP','SUI']


def build_universe(bars):
    return [c for c in _REF_ORDER if c in bars]


def norm(w):
    o = {}
    for k, v in w.items():
        kk = 'CASH' if k in ('CASH', 'Cash') else k
        if v and abs(v) > 1e-9:
            o[kk] = round(float(v), 6)
    return o


def run_bt(bars, funding, universe):
    ub.get_mcap = lambda d, _u=tuple(universe): list(_u)
    trace = []
    ub.run(bars, funding, interval='D', asset_type='fut', leverage=3.0,
           sma_days=42, mom_short_days=18, mom_long_days=127,
           vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
           n_snapshots=1, snap_interval_bars=1,         # snap 무력화
           universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
           health_mode='mom2vol', vol_mode='daily',
           drift_threshold=0.03,
           dd_lookback=0, bl_drop=0.0, crash_threshold=-10.0,  # 가드 OFF
           start_date=START, end_date=END, _trace=trace)
    btc = bars['BTC']
    all_dates = list(btc.index[(btc.index >= START) & (btc.index <= END)])
    pos = {d: i for i, d in enumerate(all_dates)}
    m = {}
    for e in trace:
        i = pos.get(e['date'])
        if i is None or i == 0:
            continue
        m[all_dates[i - 1]] = norm(e['target'])
    return m


def run_live(bars, universe):
    # 모듈 글로벌 주입 (고정 유니버스)
    atb.UNIVERSE = [c + 'USDT' for c in universe]
    atb.UNIVERSE_SIZE = 3
    atb.CAP = 1 / 3
    sp = dict(atb.STRATEGIES['D_SMA42'])
    sp['n_snapshots'] = 1
    sp['snap_interval_bars'] = 1

    import pandas as pd
    s, e = pd.Timestamp(START), pd.Timestamp(END)
    btc = bars['BTC']
    full_dates = list(btc.index)
    sig_dates = [d for d in full_dates if (d >= s and d <= e)]
    fpos = {d: i for i, d in enumerate(full_dates)}

    state = {}
    out = {}
    for d in sig_dates:
        fi = fpos[d]
        if fi + 1 >= len(full_dates):
            break
        d_next = full_dates[fi + 1]  # in-progress 봉
        bslice = {c: df[df.index <= d_next] for c, df in bars.items()}
        data = {'D': bslice}
        combined = atb.compute_strategy_target('D_SMA42', sp, data, state)
        out[d] = norm(combined)
    return out


def main():
    print('== 통제 패리티: 코인 선물 (fut) ==')
    print(f'기간 {START} ~ {END}')
    bars, funding = ub.load_data('D')
    universe = build_universe(bars)
    print(f'고정 유니버스 {len(universe)}종')

    bt = run_bt(bars, funding, universe)
    live = run_live(bars, universe)

    common = sorted(set(bt) & set(live))
    diffs, picks_diff = [], []
    for d in common:
        b, l = bt[d], live[d]
        if b != l:
            diffs.append(d)
            if sorted(k for k in b if k != 'CASH') != sorted(k for k in l if k != 'CASH'):
                picks_diff.append((d, b, l))
    print(f'\n총 비교일: {len(common)} (bt={len(bt)} live={len(live)})')
    print(f'weights 불일치일: {len(diffs)}')
    print(f'picks(종목집합) 불일치일: {len(picks_diff)}')
    if picks_diff:
        print('\n-- picks 불일치 샘플 (최대 20) --')
        for d, b, l in picks_diff[:20]:
            print(f'  {str(d)[:10]}  BT={sorted(b)}  LIVE={sorted(l)}')
    elif diffs:
        print('\n-- weights-only 불일치 (최대 10) --')
        for d in diffs[:10]:
            print(f'  {str(d)[:10]}  BT={bt[d]}  LIVE={live[d]}')
    if common:
        print(f'\n==> weights 정합률 {100.0*(len(common)-len(diffs))/len(common):.4f}%  '
              f'picks 정합률 {100.0*(len(common)-len(picks_diff))/len(common):.4f}%')
    if not diffs:
        print('==> 100% 정합: 선물 선정 로직 동일 증명 완료.')


if __name__ == '__main__':
    main()
