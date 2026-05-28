"""V25 cycle 4 P1 golden test — live `_calc_btc_cap_lev/_calc_percoin_k2_lev` 와
BT `build_K2_signal` 이 동일 OHLCV 에서 동일 leverage / 동일 기준 timestamp 산출하는지 검증.

실행: cd /home/gmoh/mon/251229 && python3 tests/test_k2_signal_golden.py
"""
import sys, os
sys.path.insert(0, '/home/gmoh/mon/251229')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')

import numpy as np
import pandas as pd

from trade.auto_trade_binance import _calc_btc_cap_lev, _calc_percoin_k2_lev
from strategies.cap_defend.backtest_futures_v25 import build_K2_signal


def _make_series(close_list, start='2024-01-01'):
    idx = pd.date_range(start, periods=len(close_list), freq='D')
    return pd.DataFrame({'Close': close_list}, index=idx)


def _make_bars(seed=42, n=80):
    rng = np.random.default_rng(seed)
    bars = {}
    for coin, base in [('BTC', 50000), ('ETH', 3000), ('SOL', 100)]:
        steps = rng.normal(0, 0.02, n).cumsum()
        close = base * np.exp(steps)
        bars[coin] = _make_series(list(close))
    return bars


def main():
    bars = _make_bars()
    # 라이브: cron 호출 시점은 진행중봉 + 어제까지 = bars 그대로. _calc_*  는 prev_close=close[:-1]
    # BT helper 와 비교하려면 BT 의 마지막 완성봉 시점 signal 을 보면 됨.
    # BT: shift(1) 적용된 signal[T] 이 사용되는 시점이 T. 라이브 의 prev_close[-1]=close[T-1].
    # 즉 라이브 cron 시점 = BT 의 signal[T] (T = bars 의 마지막 인덱스).

    # build_K2_signal 은 final L = min(btc_cap, per_coin_K2) 만 반환 (raw btc_cap 미노출).
    # 라이브의 final L = min(_calc_btc_cap_lev, _calc_percoin_k2_lev) 와 비교.
    for seed in (42, 1, 7, 99, 2026, 314, 271):
        bars = _make_bars(seed=seed)
        btc_cap_live = _calc_btc_cap_lev(bars)
        sig = build_K2_signal(bars)
        for coin in ('BTC', 'ETH', 'SOL'):
            k2_live = _calc_percoin_k2_lev(coin, bars)
            final_live = min(btc_cap_live, k2_live)
            bt_last = int(sig[coin].iloc[-1])
            assert final_live == bt_last, (
                f"seed={seed} {coin} final L mismatch: live=min({btc_cap_live},{k2_live})={final_live} bt={bt_last}"
            )
        print(f"  seed={seed} OK (btc_cap={btc_cap_live}, finals={[int(sig[c].iloc[-1]) for c in ('BTC','ETH','SOL')]})")

    # cycle 5 P1: 기준 timestamp 일치
    bars = _make_bars(seed=42)
    sig = build_K2_signal(bars)
    last_ts = bars['BTC'].index[-1]
    sig_idx_last = sig['BTC'].index[-1]
    assert last_ts == sig_idx_last, f"timestamp mismatch: bars[-1]={last_ts} vs sig[-1]={sig_idx_last}"

    # cycle 5 P1: close[:-1] 경계 — bars 의 마지막 봉(진행중) 이 변해도 live 결과 불변
    bars_mod = _make_bars(seed=42)
    bars_mod['BTC'] = bars_mod['BTC'].copy()
    # 마지막 봉 close 만 ±50% 흔들기 — close[:-1] 사용 시 결과 동일
    orig_btc_lev = _calc_btc_cap_lev(bars_mod)
    bars_mod['BTC'].iloc[-1, bars_mod['BTC'].columns.get_loc('Close')] *= 1.5
    after_btc_lev = _calc_btc_cap_lev(bars_mod)
    assert orig_btc_lev == after_btc_lev, f"close[:-1] 위반: 진행중 봉 변화로 leverage 가 변함 ({orig_btc_lev}->{after_btc_lev})"

    # cycle 5 P1: shift(1) lag — BT 의 signal[T] 은 T-1 까지의 정보만 사용
    # (bars 끝에 한 봉 추가 시 sig[-2] 가 이전 sig[-1] 과 같아야 함)
    bars2 = _make_bars(seed=42)
    sig_a = build_K2_signal(bars2)
    bars2_ext = {c: bars2[c].copy() for c in bars2}
    next_ts = bars2['BTC'].index[-1] + pd.Timedelta(days=1)
    for c in bars2_ext:
        last_close = bars2_ext[c]['Close'].iloc[-1]
        bars2_ext[c].loc[next_ts] = {'Close': last_close * 2.0}  # 큰 변화로 가짜 시그널 유도
    sig_b = build_K2_signal(bars2_ext)
    # sig_b 의 끝-2 = sig_a 의 끝-1 (같은 timestamp 에 대해)
    for coin in ('BTC', 'ETH', 'SOL'):
        common_ts = bars2['BTC'].index[-1]
        assert int(sig_a[coin].loc[common_ts]) == int(sig_b[coin].loc[common_ts]), \
            f"shift(1) 위반: {coin} 미래 봉이 과거 시그널에 영향 ({sig_a[coin].loc[common_ts]} vs {sig_b[coin].loc[common_ts]})"

    # cycle 5 P1: min() clip 명시 검증 — BTC_cap=2, per-coin K2=4 면 final=2
    # 인위적 fixture: BTC 횡보(cap=2) + ETH 강한 상승(K2=4) → min=2
    n = 80
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    btc_flat = [50000.0] * n  # ratio=1.0 → cap=L_min(2)
    eth_strong = [3000.0 * (1.03 ** i) for i in range(n)]  # 매일 3% → 7-day ratio ≈ 1.03^3 ≈ 1.093 > 1.075
    bars_clip = {
        'BTC': pd.DataFrame({'Close': btc_flat}, index=idx),
        'ETH': pd.DataFrame({'Close': eth_strong}, index=idx),
    }
    btc_cap = _calc_btc_cap_lev(bars_clip)
    eth_k2 = _calc_percoin_k2_lev('ETH', bars_clip)
    sig_clip = build_K2_signal(bars_clip)
    assert btc_cap == 2, f"flat BTC cap 예상 2, got {btc_cap}"
    assert eth_k2 == 4, f"strong ETH K2 예상 4, got {eth_k2}"
    assert int(sig_clip['ETH'].iloc[-1]) == 2, f"min() clip 위반: BTC_cap=2 이면 ETH final 도 2 (got {sig_clip['ETH'].iloc[-1]})"
    print(f"  clip OK: BTC_cap={btc_cap} ETH_K2={eth_k2} → ETH_final={int(sig_clip['ETH'].iloc[-1])}")

    print("\n✅ K2 golden test PASSED (BTC cap + per-coin K2 + final L all match across seeds)")


if __name__ == '__main__':
    main()
