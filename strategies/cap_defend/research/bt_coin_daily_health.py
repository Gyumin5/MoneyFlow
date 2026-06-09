"""코인 현물·선물: daily-health-exit ON vs OFF 비교 BT.

질문(사용자): 보유 코인 헬스(mom2vol) 를 매일 체크해서 탈락 즉시 빼기(daily_health_exit)
vs 현행(앵커/드리프트 발화일에만 교체) — 어느 쪽이 나은가?

- daily_health_exit=False : 현행 (combined target = snap merge, 헬스 재평가는 앵커/드리프트 때만).
- daily_health_exit=True  : 매일 보유 코인 헬스 재평가, 탈락 코인 → CASH 즉시 (unified_backtest 기존 플래그).

설정: live-equivalent 가드 OFF (dd/bl/crash), DRIFT_HEALTH_MODE='refill'(BT-of-record).
robustness: phase_offset(앵커 위상) 다중 평균 × 비용 1x/3x/5x.
지표: CAGR/MDD/Calmar/Rebal/Trades.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ['DRIFT_HEALTH_MODE'] = 'refill'
import numpy as np
import unified_backtest as ub

START = '2020-10-01'
END = '2026-05-10'

SPOT = dict(asset_type='spot', leverage=1.0, sma_days=42, mom_short_days=20, mom_long_days=127,
            vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=7,
            snap_interval_bars=217, universe_size=3, cap=1/3,
            health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
            dd_lookback=0, bl_drop=0.0, crash_threshold=-10.0)
FUT = dict(asset_type='fut', leverage=3.0, sma_days=42, mom_short_days=18, mom_long_days=127,
           vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=5,
           snap_interval_bars=95, universe_size=3, cap=1/3, maint_rate=0.004,
           health_mode='mom2vol', vol_mode='daily', drift_threshold=0.03,
           dd_lookback=0, bl_drop=0.0, crash_threshold=-10.0)


def run_cfg(bars, funding, base, tx, phase, dhe):
    m = ub.run(bars, funding, interval='D', tx_cost=tx,
               phase_offset_bars=phase, daily_health_exit=dhe,
               start_date=START, end_date=END, **base)
    if not m or 'CAGR' not in m:
        return None
    return (m['CAGR'], m['MDD'], m['Cal'], m.get('Rebal', 0), m.get('Trades', 0))


def main():
    t0 = time.time()
    bars, funding = ub.load_data('D')
    print(f"# 코인 daily_health_exit ON/OFF 비교. 기간 {START}~{END}")
    print(f"# 가드 OFF, DRIFT_HEALTH_MODE=refill. phase 평균 × 비용 1/3/5x\n")

    for name, base, phases, base_tx in (
        ('SPOT(V24)', SPOT, [0, 31, 62, 93, 124], 0.004),
        ('FUT(V25)',  FUT,  [0, 19, 38, 57, 76], 0.0004),
    ):
        print(f"=== {name} ===")
        print(f"  {'cost':>4} {'mode':<10} {'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'Rebal':>6} {'Trades':>7}")
        for mult in (1, 3, 5):
            tx = base_tx * mult
            for dhe in (False, True):
                rows = []
                for ph in phases:
                    r = run_cfg(bars, funding, base, tx, ph, dhe)
                    if r:
                        rows.append(r)
                if not rows:
                    print(f"  {mult}x  {'(no data)'}"); continue
                cagr = np.mean([x[0] for x in rows]); mdd = np.mean([x[1] for x in rows])
                cal = np.mean([x[2] for x in rows]); rb = np.mean([x[3] for x in rows])
                tr = np.mean([x[4] for x in rows])
                lbl = 'daily-exit' if dhe else 'current'
                print(f"  {mult}x  {lbl:<10} {cagr*100:>6.1f}% {mdd*100:>6.1f}% {cal:>7.2f} {rb:>6.0f} {tr:>7.0f}")
            print()
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
