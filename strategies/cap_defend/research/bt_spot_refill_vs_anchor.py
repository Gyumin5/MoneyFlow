"""코인 현물(V24 spot) anchor-only vs drift-refill(라이브 정합) 성과 비교.

라이브(coin_live_engine)는 drift 발화 시 _apply_refill_v2_to_state 로 mom2 음수 코인을
fresh healthy 로 교체(refill). 채택 spot BT(unified_backtest)는 기본 anchor-only(교체 없음).
unified_backtest 에 추가한 DRIFT_HEALTH_MODE='refill' 모드로 두 동작을 동일 데이터·비용에서 비교.

라이브 V24 spot 파라미터: SMA42, mom 20/127, vol 90d/0.05, n_snap=7, snap_int=217,
cap=1/3, drift=0.10, health=mom2vol. 비용 스트레스 1x/3x/5x (base tx=0.004).
phase_offset 7종 평균으로 robustness.
"""
import os, sys, time
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import numpy as np
import unified_backtest as ub

BASE_TX = 0.004
START = '2020-10-01'
END = '2026-05-13'
PHASES = list(range(0, 217, 31))  # 7 위상 (stagger 31)


def run_cfg(mode, tx, phase):
    if mode == 'refill':
        os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    else:
        os.environ.pop('DRIFT_HEALTH_MODE', None)
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
                  sma_days=42, mom_short_days=20, mom_long_days=127,
                  vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=7,
                  universe_size=3, cap=1/3, tx_cost=tx,
                  health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
                  snap_interval_bars=217, phase_offset_bars=phase,
                  start_date=START, end_date=END)


def main():
    global bars_D, funding
    t0 = time.time()
    print('코인 D 데이터 로드...')
    bars_D, funding = ub.load_data('D')
    print(f'  로드 완료 ({time.time()-t0:.1f}s). 기간 {START}~{END}, phases={len(PHASES)}\n')
    print(f"  {'cost':>5} {'mode':<8} {'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'Rebal':>6} {'Trades':>7}")
    for mult in (1, 3, 5):
        tx = BASE_TX * mult
        for mode in ('anchor', 'refill'):
            cagrs, mdds, cals, rebs, trds = [], [], [], [], []
            for ph in PHASES:
                r = run_cfg(mode, tx, ph)
                if not r or 'CAGR' not in r:
                    continue
                cagrs.append(r['CAGR']); mdds.append(r['MDD']); cals.append(r['Cal'])
                rebs.append(r.get('Rebal', 0)); trds.append(r.get('Trades', 0))
            if not cagrs:
                print(f"  {mult}x    {mode:<8} (no data)"); continue
            print(f"  {mult}x    {mode:<8} {np.mean(cagrs)*100:>6.1f}% {np.mean(mdds)*100:>6.1f}% "
                  f"{np.mean(cals):>7.2f} {np.mean(rebs):>6.0f} {np.mean(trds):>7.0f}")
        print()
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
