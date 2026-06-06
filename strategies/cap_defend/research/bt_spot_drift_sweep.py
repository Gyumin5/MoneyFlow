"""현물(V24 spot) 드리프트 임계 스윕 + refill 방식 교차 변형.

질문: (1) 드리프트 임계를 낮추면 비용 차감 후에도 나은가? (2) refill 방식(슬롯교체 vs 전체재선정)이
성과를 바꾸는가?

modes:
  off      = anchor-only (drift 발화해도 종목 교체 없음)
  refill   = 슬롯교체 (라이브 현행: mom2 음수 슬롯만 fresh healthy 로)
  reselect = 전체 재선정 (주식식: 전 snapshot 을 fresh full selection 으로)

drift: 0.02~0.12. 비용 1x/3x (base tx=0.004). 위상 5종 평균.
"""
import os, sys, time
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import numpy as np
import unified_backtest as ub

BASE_TX = 0.004
START = '2020-10-01'
END = '2026-05-13'
PHASES = [0, 43, 87, 130, 173]
DRIFTS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
MODES = ['off', 'refill', 'reselect']


def run_cfg(mode, tx, phase, drift):
    if mode == 'off':
        os.environ.pop('DRIFT_HEALTH_MODE', None)
    else:
        os.environ['DRIFT_HEALTH_MODE'] = mode
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
                  sma_days=42, mom_short_days=20, mom_long_days=127,
                  vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=7,
                  universe_size=3, cap=1/3, tx_cost=tx,
                  health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
                  snap_interval_bars=217, phase_offset_bars=phase,
                  start_date=START, end_date=END)


def main():
    global bars_D, funding
    t0 = time.time()
    print('코인 D 데이터 로드...')
    bars_D, funding = ub.load_data('D')
    print(f'  완료 ({time.time()-t0:.1f}s). {START}~{END}, 위상 {len(PHASES)}종 평균\n')
    for mult in (1, 3):
        tx = BASE_TX * mult
        print(f"=== 비용 {mult}x (tx={tx:.4f}) ===")
        print(f"  {'mode':<9} {'drift':>5} {'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'Rebal':>6} {'Trades':>7}")
        for mode in MODES:
            for d in DRIFTS:
                cg, md, cl, rb, td = [], [], [], [], []
                for ph in PHASES:
                    r = run_cfg(mode, tx, ph, d)
                    if not r or 'CAGR' not in r:
                        continue
                    cg.append(r['CAGR']); md.append(r['MDD']); cl.append(r['Cal'])
                    rb.append(r.get('Rebal', 0)); td.append(r.get('Trades', 0))
                if not cg:
                    continue
                star = '  *현행*' if (mode == 'refill' and abs(d - 0.10) < 1e-9) else ''
                print(f"  {mode:<9} {d:>5.2f} {np.mean(cg)*100:>6.1f}% {np.mean(md)*100:>6.1f}% "
                      f"{np.mean(cl):>7.2f} {np.mean(rb):>6.0f} {np.mean(td):>7.0f}{star}")
            print()
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
