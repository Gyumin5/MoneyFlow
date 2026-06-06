"""통제 패리티 하니스 — 코인 현물 (spot).

목적: 라이브 선정함수(coin_live_engine.compute_member_target)와
BT 선정함수(unified_backtest.run/_compute_weights)가 동일 유니버스·동일 OHLCV·
동일 BTC 카나리 입력에서 일별 picks/weights 100% 일치하는지 증명.

설계 (ai-debate 20260606 합의 A):
- 유니버스 소스 차이(라이브 CoinGecko vs BT historical_universe.json)를 제거하기 위해
  양쪽에 동일한 고정 후보 순서(UNIVERSE)를 주입. (BT get_mcap monkeypatch.)
- 가드 OFF (DD/BL/crash) — 라이브 compute_member_target 는 가드가 없으므로 BT 도 끔.
- snap 머신 무력화: n_snapshots=1, snap_interval_bars=1 → combined = 매 봉 fresh 선정.
  drift refill 도 단일 snap 매봉 재계산이라 no-op.
- 정렬: signal date 기준. BT trace[i] 는 prev_date=all_dates[i-1] 로 계산 → live[all_dates[i-1]] 와 비교.

실거래 매매 코드는 read-only (변경 없음).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'trade'))
from datetime import timedelta, timezone

import unified_backtest as ub
import coin_live_engine as cle

START = '2020-10-01'
END = '2026-05-10'

# ── 고정 후보 유니버스 (상수 순서) ──
# 유니버스 소스 변수를 제거. mcap 기준 순서를 한 번 고정해 양쪽에 동일 주입.
_REF_ORDER = ['BTC','ETH','BNB','SOL','XRP','ADA','AVAX','DOGE','DOT','MATIC',
              'TRX','LINK','LTC','UNI','ATOM','BCH','XLM','VET','FIL','THETA',
              'EOS','AAVE','ALGO','GRT','NEAR','FTM','SAND','MANA','AXS','GALA',
              'CHZ','APE','FLOW','ICP','APT','ARB','OP','SUI']


def build_universe(bars):
    return [c for c in _REF_ORDER if c in bars]


def run_bt(bars, funding, universe):
    # get_mcap monkeypatch → 고정 순서
    ub.get_mcap = lambda d, _u=tuple(universe): list(_u)
    trace = []
    ub.run(bars, funding, interval='D', asset_type='spot', leverage=1.0,
           sma_days=42, mom_short_days=20, mom_long_days=127,
           vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
           n_snapshots=1, snap_interval_bars=1,        # snap 무력화
           universe_size=3, cap=1/3, tx_cost=0.004,
           health_mode='mom2vol', vol_mode='daily',
           drift_threshold=0.10,
           dd_lookback=0, bl_drop=0.0, crash_threshold=-10.0,  # 가드 OFF
           start_date=START, end_date=END, _trace=trace)
    # signal date(prev_date) → target.
    # 주의: BT 루프는 bar_i=sma_period+1 부터 시작 → trace[0].date != all_dates[0].
    # 각 trace 항목의 date 직전 all_dates 항목이 그 target 의 signal date(prev_date).
    m = {}
    btc = bars['BTC']
    all_dates = list(btc.index[(btc.index >= START) & (btc.index <= END)])
    pos = {d: i for i, d in enumerate(all_dates)}
    for e in trace:
        i = pos.get(e['date'])
        if i is None or i == 0:
            continue
        sig = all_dates[i - 1]
        m[sig] = norm(e['target'])
    return m


def run_live(bars, universe):
    cfg = dict(cle.MEMBER_D_SMA42)
    cfg['n_snapshots'] = 1
    cfg['snap_interval_bars'] = 1
    state = cle.MemberState()
    btc = bars['BTC']
    sig_dates = list(btc.index[(btc.index >= START) & (btc.index <= END)])
    out = {}
    for d in sig_dates:
        bslice = {c: df[df.index <= d] for c, df in bars.items()}
        now_utc = d.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(days=1)
        res = cle.compute_member_target('D_SMA42', cfg, bslice, universe, state, now_utc)
        state = res.new_state
        out[d] = norm(res.target)
    return out


def norm(w):
    """CASH/Cash 통일 + 반올림 비교용 정규화."""
    o = {}
    for k, v in w.items():
        kk = 'CASH' if k in ('CASH', 'Cash') else k
        if v and abs(v) > 1e-9:
            o[kk] = round(float(v), 6)
    return o


def main():
    print('== 통제 패리티: 코인 현물 (spot) ==')
    print(f'기간 {START} ~ {END}')
    bars, funding = ub.load_data('D')
    universe = build_universe(bars)
    print(f'고정 유니버스 {len(universe)}종: {universe}')

    bt = run_bt(bars, funding, universe)
    live = run_live(bars, universe)

    common = sorted(set(bt) & set(live))
    print(f'비교 가능일 {len(common)} (bt={len(bt)} live={len(live)})')

    diffs = []
    picks_diff = []
    for d in common:
        b, l = bt[d], live[d]
        if b != l:
            diffs.append(d)
            bp = sorted([k for k in b if k != 'CASH'])
            lp = sorted([k for k in l if k != 'CASH'])
            if bp != lp:
                picks_diff.append((d, bp, lp))
    print(f'\n총 비교일: {len(common)}')
    print(f'weights 불일치일: {len(diffs)}')
    print(f'picks(종목집합) 불일치일: {len(picks_diff)}')
    if picks_diff:
        print('\n-- picks 불일치 샘플 (최대 20) --')
        for d, bp, lp in picks_diff[:20]:
            print(f'  {str(d)[:10]}  BT={bp}  LIVE={lp}')
    elif diffs:
        print('\n-- weights-only 불일치 샘플 (종목 동일, 비중만; 최대 10) --')
        for d in diffs[:10]:
            print(f'  {str(d)[:10]}  BT={bt[d]}  LIVE={live[d]}')
    match_pct = 100.0 * (len(common) - len(diffs)) / len(common) if common else 0
    print(f'\n==> weights 정합률 {match_pct:.4f}%  picks 정합률 '
          f'{100.0*(len(common)-len(picks_diff))/len(common):.4f}%')
    if not diffs:
        print('==> 100% 정합: 선정 로직 동일 증명 완료.')


if __name__ == '__main__':
    main()
