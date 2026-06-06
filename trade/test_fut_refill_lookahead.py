"""선물 refill v2 look-ahead 수정 검증.

apply_refill_v2_fut 가 진행 중(미완성) 일봉을 모멘텀 판정에 쓰지 않는지(=BT t-1 정합) 확인.
핵심 파리티: 진행중 봉이 있든 없든 refill 결과(combined)가 동일해야 함.
수정 전(raw .values)이면 진행중 봉 스파이크가 fail 판정을 뒤집어 결과가 달라짐.
"""
import sys, os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_trade_binance as a


def _mk_df(index, closes):
    return pd.DataFrame({'Open': closes, 'High': closes, 'Low': closes,
                         'Close': closes, 'Volume': [1.0] * len(closes)}, index=index)


def main():
    fails = []
    sp = a.STRATEGIES['D_SMA42']
    mom_l = sp['mom_long_bars']
    n = mom_l + 5  # 충분한 길이

    # UTC 일봉 인덱스: 마지막 봉 = 오늘 00:00 UTC(진행중), 직전 = 어제(완성)
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    cur_open = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    idx = pd.to_datetime([cur_open - timedelta(days=(n - 1 - i)) for i in range(n)])

    # ALT_FAIL: 완성봉 기준 하락(모멘텀 음수=fail), 진행중 봉은 큰 스파이크(있으면 mom 양수로 뒤집힘)
    fail_closes = list(np.linspace(100.0, 70.0, n - 1)) + [200.0]
    # ALT_HEAL: 상승(모멘텀 양수=healthy)
    heal_closes = list(np.linspace(70.0, 140.0, n))
    # BTC: 무관(healthy pool 제외 대상)
    btc_closes = list(np.linspace(100.0, 110.0, n))

    data_full = {'D': {
        'BTC': _mk_df(idx, btc_closes),
        'ALT_FAIL': _mk_df(idx, fail_closes),
        'ALT_HEAL': _mk_df(idx, heal_closes),
    }}
    # 진행중 봉 제거한 버전(마지막 행 drop) — 완성봉 기준 truth
    data_trunc = {'D': {k: df.iloc[:-1].copy() for k, df in data_full['D'].items()}}

    def base_state():
        return {'strategies': {'D_SMA42': {
            'snapshots': [{'ALT_FAIL': 1.0 / 3.0, 'CASH': 2.0 / 3.0}]
        }}}

    r_full = a.apply_refill_v2_fut(base_state(), data_full)
    r_trunc = a.apply_refill_v2_fut(base_state(), data_trunc)

    def norm(d):
        return {k: round(v, 6) for k, v in d.items() if v > 1e-9}

    # 1) 파리티: 진행중 봉 유무와 무관하게 동일 결과
    if norm(r_full) != norm(r_trunc):
        fails.append(f"파리티 실패: full={norm(r_full)} != trunc={norm(r_trunc)}")

    # 2) 완성봉 기준 ALT_FAIL 은 fail → 결과에서 빠지고 ALT_HEAL 로 교체
    if 'ALT_FAIL' in norm(r_full):
        fails.append(f"ALT_FAIL 이 결과에 남음(진행중 스파이크로 fail 판정 누락 의심): {norm(r_full)}")
    if 'ALT_HEAL' not in norm(r_full):
        fails.append(f"ALT_HEAL 로 교체 안 됨: {norm(r_full)}")

    # 3) _finalize 가 진행중 봉을 실제로 drop 하는지 직접 확인
    fin = a._finalize_daily_bar_for_signal(data_full['D']['ALT_FAIL'])
    if float(fin['Close'].values[-1]) == 200.0:
        fails.append("_finalize 가 진행중 스파이크 봉을 drop 안 함")

    print("# 선물 refill look-ahead 검증")
    print(f"  result(full,진행중봉포함)  = {norm(r_full)}")
    print(f"  result(trunc,완성봉만)     = {norm(r_trunc)}")
    print(f"  _finalize 마지막 close     = {float(fin['Close'].values[-1]):.1f} (스파이크200 drop 기대)")
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
