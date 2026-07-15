#!/usr/bin/env python3
"""V24 코인 현물 — "카나리 ON인데 후보 healthy=0" 상태 계측 (read-only).

목적: 빈 스냅 트랜치가 현금에 고정되는 근본원인의 발생빈도/지속/원인/시점을
      정량화한다. 전략/실매매/코드 변경 없음. 순수 측정.

방법:
  - unified_backtest.py 의 검증된 판정 함수(calc_sma/calc_mom/calc_vol_daily/get_mcap)를
    그대로 재사용해 BT-of-record 와 동일 임계로 매 거래일을 재평가한다.
  - 카나리: run() 메인루프의 히스테리시스 로직을 그대로 복제 (t-1 종가 기준).
  - 헬스: _compute_weights 의 mom2vol 판정을 그대로 복제 (sig_date=t-1).
  - 파라미터는 trade/v24_shadow_today.py 의 spot V24 호출과 100% 동일.

주의(설계상 한계, 명시):
  - blacklist / crash-breaker 상태는 모델링하지 않는다. 이 둘은 보유코인 제거 +
    일시적 후보제외라 healthy_count 를 더 낮출 수만 있다. 즉 본 스크립트의
    "healthy==0" 일수는 실제 엔진 대비 하한(과소추정)이다. 구조적 후보가용성 측정.
  - 유니버스 소스는 BT-of-record 와 동일한 point-in-time historical_universe.json.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))  # strategies/cap_defend
import unified_backtest as ub  # noqa: E402

# ─── V24 spot 파라미터 (v24_shadow_today.py 와 동일) ───
INTERVAL = 'D'
BPD = 1
SMA_PERIOD = 42          # sma_days=42
MOM_SHORT = 20           # mom_short_days=20
MOM_LONG = 127           # mom_long_days=127
VOL_LB = 90              # vol_days=90 (bpd=1 -> 90 bars, vol_mode='daily')
VOL_THRESH = 0.05
CANARY_HYST = 0.015
UNIVERSE_SIZE = 3
START = '2020-10-01'
END = '2026-05-14'       # BTC 데이터 마지막 봉
MIN_BARS = max(MOM_SHORT, MOM_LONG, VOL_LB, SMA_PERIOD)  # =127

FWD_HORIZONS = (20, 60)


def health_eval(c):
    """mom2vol 헬스 판정 + 실패원인 분해. c=Close 배열(sig_date 까지, inclusive).
    반환 (ok: bool, ms_fail, ml_fail, vol_fail, ms, ml, vol)."""
    ms = ub.calc_mom(c, MOM_SHORT)                       # >0 요구
    ml = ub.calc_mom(c, MOM_LONG)                        # >0 요구
    vol = ub.calc_vol_daily(c, BPD, lookback_bars=VOL_LB)  # <=0.05 요구
    ms_fail = not (ms > 0)
    ml_fail = not (ml > 0)
    vol_fail = not (vol <= VOL_THRESH)
    ok = (ms > 0) and (ml > 0) and (vol <= VOL_THRESH)
    return ok, ms_fail, ml_fail, vol_fail, ms, ml, vol


def main():
    bars, _funding = ub.load_data(INTERVAL)
    btc_df = bars.get('BTC')
    assert btc_df is not None, 'BTC 데이터 없음'
    btc_close = btc_df['Close'].values
    btc_index = list(btc_df.index)
    btc_idx_map = {d: i for i, d in enumerate(btc_df.index)}

    all_dates = btc_df.index[(btc_df.index >= START) & (btc_df.index <= END)]

    # 코인별 Close 배열 + 인덱서 캐시
    close_cache = {c: df['Close'].values for c, df in bars.items()}

    prev_canary = False

    # 일별 레코드
    recs = []  # dict per processed bar

    for bar_i in range(SMA_PERIOD + 1, len(all_dates)):
        date = all_dates[bar_i]
        prev_date = all_dates[bar_i - 1]
        btc_i = btc_idx_map.get(date, -1)
        btc_i_prev = btc_idx_map.get(prev_date, -1)
        if btc_i < SMA_PERIOD or btc_i_prev < SMA_PERIOD:
            continue

        # ── 카나리 (t-1 기준, run() 복제) ──
        btc_c_prev = btc_close[:btc_i_prev + 1]
        sma_val = ub.calc_sma(btc_c_prev, SMA_PERIOD)
        cur_btc_prev = btc_c_prev[-1] if len(btc_c_prev) > 0 else 0
        if prev_canary:
            canary_on = not (cur_btc_prev < sma_val * (1 - CANARY_HYST))
        else:
            canary_on = cur_btc_prev > sma_val * (1 + CANARY_HYST)
        prev_canary = canary_on

        # ── 후보 헬스 (sig_date=prev_date, _compute_weights 복제) ──
        sig_date = prev_date
        mcap_order = ub.get_mcap(sig_date)
        healthy = []
        n_cand = 0            # 데이터 충분한 후보 수
        ms_fail_n = ml_fail_n = vol_fail_n = 0
        for coin in mcap_order:
            df = bars.get(coin)
            if df is None:
                continue
            ci = df.index.get_indexer([sig_date], method='ffill')[0]
            if ci < 0 or ci < MIN_BARS:
                continue
            n_cand += 1
            c = close_cache[coin][:ci + 1]
            ok, msf, mlf, volf, ms, ml, vol = health_eval(c)
            if ok:
                healthy.append(coin)
            else:
                if msf:
                    ms_fail_n += 1
                if mlf:
                    ml_fail_n += 1
                if volf:
                    vol_fail_n += 1

        recs.append({
            'date': date,
            'btc_i': btc_i,
            'canary_on': canary_on,
            'n_cand': n_cand,
            'healthy_count': len(healthy),
            'ms_fail_n': ms_fail_n,
            'ml_fail_n': ml_fail_n,
            'vol_fail_n': vol_fail_n,
            'mcap_top': [c for c in mcap_order if c in bars][:UNIVERSE_SIZE],
        })

    # ─────────────────────── 집계 ───────────────────────
    total = len(recs)
    canary_on_days = sum(1 for r in recs if r['canary_on'])
    empty_days = [r for r in recs if r['canary_on'] and r['healthy_count'] == 0]
    n_empty = len(empty_days)

    print('=' * 68)
    print('V24 코인 현물 — 카나리 ON & healthy==0 계측')
    print(f'기간 {START} ~ {END} (interval=D)')
    print(f'파라미터: SMA42 hyst1.5% | mom2vol ms20 ml127 vol90d<=0.05 | uni_size3')
    print('=' * 68)
    print(f'총 거래일           : {total}')
    print(f'카나리 ON 일수       : {canary_on_days}  ({100*canary_on_days/total:.1f}% of total)')
    print(f'카나리 OFF 일수      : {total - canary_on_days}')
    print(f'ON & healthy==0 일수 : {n_empty}')
    if total:
        print(f'   빈도(전체 대비)   : {100*n_empty/total:.2f}%')
    if canary_on_days:
        print(f'   빈도(ON 대비)     : {100*n_empty/canary_on_days:.2f}%')

    # healthy_count 분포 (canary ON 일 한정)
    on_recs = [r for r in recs if r['canary_on']]
    from collections import Counter
    hc_dist = Counter(r['healthy_count'] for r in on_recs)
    print('\n[카나리 ON 일의 healthy_count 분포]')
    for k in sorted(hc_dist):
        print(f'   healthy={k:>2} : {hc_dist[k]:>4} 일  ({100*hc_dist[k]/canary_on_days:.1f}%)')

    # ── 에피소드 (연속 ON&empty 구간) ──
    episodes = []
    cur = None
    empty_set = set(id(r) for r in empty_days)
    for r in recs:
        is_empty = (r['canary_on'] and r['healthy_count'] == 0)
        if is_empty:
            if cur is None:
                cur = [r]
            else:
                cur.append(r)
        else:
            if cur is not None:
                episodes.append(cur)
                cur = None
    if cur is not None:
        episodes.append(cur)

    durations = [len(e) for e in episodes]
    print(f'\n[에피소드 (연속 ON&healthy==0 구간)]')
    print(f'   에피소드 개수     : {len(episodes)}')
    if durations:
        arr = np.array(durations)
        print(f'   지속일수 평균     : {arr.mean():.1f}')
        print(f'   지속일수 p50/p90/p99/max : '
              f'{np.percentile(arr,50):.0f} / {np.percentile(arr,90):.0f} / '
              f'{np.percentile(arr,99):.0f} / {arr.max()}')
        print(f'   총 empty-days     : {arr.sum()}')

    # ── 헬스 OFF 주 원인 (empty 일들의 후보 실패 집계) ──
    tot_ms = sum(r['ms_fail_n'] for r in empty_days)
    tot_ml = sum(r['ml_fail_n'] for r in empty_days)
    tot_vol = sum(r['vol_fail_n'] for r in empty_days)
    tot_cand = sum(r['n_cand'] for r in empty_days)
    print(f'\n[헬스 OFF 원인 분해 — ON&healthy==0 일의 후보 실패 카운트]')
    print(f'   (한 후보가 복수 조건 동시 위반 가능. 분모=후보평가 총 {tot_cand}건)')
    if tot_cand:
        print(f'   ms<=0  (단기모멘텀 음수) : {tot_ms:>6}  ({100*tot_ms/tot_cand:.1f}% of 후보평가)')
        print(f'   ml<=0  (장기모멘텀 음수) : {tot_ml:>6}  ({100*tot_ml/tot_cand:.1f}%)')
        print(f'   vol>5% (고변동)          : {tot_vol:>6}  ({100*tot_vol/tot_cand:.1f}%)')

    # ── 연도별 카운트 ──
    yr = Counter(r['date'].year for r in empty_days)
    yr_on = Counter(r['date'].year for r in on_recs)
    yr_tot = Counter(r['date'].year for r in recs)
    print(f'\n[연도별 분포]')
    print(f'   {"year":>6} {"거래일":>6} {"ON":>6} {"empty":>6} {"empty/ON%":>9}')
    for y in sorted(yr_tot):
        e = yr.get(y, 0)
        o = yr_on.get(y, 0)
        pct = (100*e/o) if o else 0
        print(f'   {y:>6} {yr_tot[y]:>6} {o:>6} {e:>6} {pct:>8.1f}%')

    # ── forward return (empty-day 기준, 놓친수익 vs 회피손실 힌트) ──
    print(f'\n[Forward return — ON&healthy==0 일 이후 20/60봉]')
    print(f'   양수=현금고정이 놓친 수익 / 음수=현금고정이 회피한 손실')

    def fwd_ret(close_arr, i, h):
        if i + h < len(close_arr) and close_arr[i] > 0:
            return close_arr[i + h] / close_arr[i] - 1
        return None

    for h in FWD_HORIZONS:
        btc_rets = []
        top_rets = []
        for r in empty_days:
            br = fwd_ret(btc_close, r['btc_i'], h)
            if br is not None:
                btc_rets.append(br)
            # top-3 mcap 가용후보 EW forward (health 무시, 순수 시장 방향)
            per = []
            for coin in r['mcap_top']:
                arr = close_cache.get(coin)
                if arr is None:
                    continue
                df = bars[coin]
                j = df.index.get_indexer([r['date']], method='ffill')[0]
                if j < 0:
                    continue
                fr = fwd_ret(arr, j, h)
                if fr is not None:
                    per.append(fr)
            if per:
                top_rets.append(float(np.mean(per)))
        def _stat(x):
            if not x:
                return 'n/a'
            a = np.array(x)
            return (f'n={len(a):>4} mean={a.mean():+.3f} median={np.median(a):+.3f} '
                    f'p10={np.percentile(a,10):+.3f} p90={np.percentile(a,90):+.3f} '
                    f'win={100*(a>0).mean():.0f}%')
        print(f'   +{h:>3}봉  BTC        : {_stat(btc_rets)}')
        print(f'   +{h:>3}봉  top3mcapEW : {_stat(top_rets)}')

    print('\n' + '=' * 68)
    print('완료.')


if __name__ == '__main__':
    main()
