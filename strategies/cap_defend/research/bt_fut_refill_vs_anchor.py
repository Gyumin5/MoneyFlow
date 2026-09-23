"""선물(V25) 문턱을 조이면 종목 교체가 실제로 빨라지는가.

물음의 배경. 드리프트 문턱을 조이면 발화가 는다는 것은 이미 쟀다(bt_fut_drift_plateau.py).
그런데 발화가 는다고 종목 교체까지 느는 것은 아니다. 발화일의 교체는 refill 경로에서만
일어나고, 그 경로는 보유 코인의 두 모멘텀이 모두 음수일 때만 코인을 갈아낀다. 그 조건이
발화일에 한 번도 성립하지 않으면 교체는 앵커(95봉)에서만 일어나고, 문턱을 조여도 교체
속도는 그대로다.

현물은 같은 물음을 2026-06-06 에 재서 교체 0일이었다(bt_spot_refill_vs_anchor.py,
drift=0.10 에서 refill 과 anchor-only 가 완전히 같은 곡선). 선물은 안 쟀다.

재는 법. 선물 엔진의 DRIFT_HEALTH_MODE 를 off(앵커에서만 교체)와 refill(라이브 정합)로
두고 같은 데이터·같은 비용으로 돌려 자산곡선을 비교한다. 두 곡선이 한 점도 안 다르면
그 문턱에서는 교체가 한 번도 안 일어난 것이다. 다르면 일어난 것이고, 그 차이가 성과에
얼마나 닿는지도 같이 보인다.

주의. 선물 엔진의 DRIFT_HEALTH_MODE 기본값은 refill 이다(현물은 off). 그래서 off 쪽을
잴 때 환경변수를 반드시 명시해야 한다 — 지우기만 하면 refill 로 돈다.
캐시는 쓰지도 쓰지도 않는다. state/20260922-resweep/eq 는 교정된 버퍼 곡선만 담아야 한다.
"""
import os
import sys
import time

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')

BASE_DIR = '/home/gmoh/mon/251229/strategies/cap_defend'
START, END = '2020-10-01', '2026-09-18'
TX_FUT = 0.0006
# 선물 앵커 위상 — resweep 하니스와 같은 값에서 셋만 뽑는다(양끝+가운데).
PHASES = [0, 8, 16]
THRESHOLDS = [0.005, 0.01, 0.02, 0.03]
FUT_BASE = dict(sma_days=42, mom_short_days=18, mom_long_days=127,
                vol_threshold=0.05, n_snapshots=5, snap_interval_bars=95,
                canary_hyst=0.015, universe_size=3, cap=1 / 3)

_bars = None
_funding = None
_k2 = None


def run_fut(mode: str, thr: float, phase: int):
    from backtest_futures_v25 import run as fbt_run
    os.environ['DRIFT_HEALTH_MODE'] = mode
    kw = dict(FUT_BASE)
    kw['drift_threshold'] = thr
    return fbt_run(_bars, _funding, interval='D', leverage=_k2,
                   tx_cost=TX_FUT, maint_rate=0.004,
                   vol_days=90, vol_mode='daily', post_flip_delay=5,
                   phase_offset_bars=phase, health_mode='mom2vol',
                   start_date=START, end_date=END, **kw)


def main() -> None:
    global _bars, _funding, _k2
    t0 = time.time()
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print('코인 일봉 로드 중...')
    _bars, _funding = load_data('D')
    _k2 = build_K2_signal(_bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                          btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                          l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f'  심볼 {len(_bars)}종 ({time.time()-t0:.0f}초)\n')

    print('선물 교체(refill) 대 앵커 전용(off) — 같은 데이터·같은 비용')
    print(f'  기간 {START}~{END}, 비용 {TX_FUT}, 위상 {PHASES}\n')
    hdr = (f"  {'문턱':>6} {'위상':>4} {'모드':<7} {'CAGR':>7} {'MDD':>8} "
           f"{'Calmar':>7} {'발화':>5} {'체결':>6}  판정")
    print(hdr)
    for thr in THRESHOLDS:
        for ph in PHASES:
            res = {}
            for mode in ('off', 'refill'):
                res[mode] = run_fut(mode, thr, ph)
            eq_off = res['off'].get('_equity')
            eq_ref = res['refill'].get('_equity')
            if eq_off is None or eq_ref is None:
                verdict = '자산곡선 없음'
            else:
                common = eq_off.index.intersection(eq_ref.index)
                if len(common) != len(eq_off.index) or len(common) != len(eq_ref.index):
                    verdict = '기간 불일치'
                else:
                    gap = (eq_ref.reindex(common) - eq_off.reindex(common)).abs().max()
                    rel = gap / max(abs(eq_off.iloc[-1]), 1e-9)
                    verdict = ('교체 0일(두 곡선 동일)' if rel < 1e-12
                               else f'교체 있었음(최대 괴리 {rel:.2e})')
            for mode in ('off', 'refill'):
                m = res[mode]
                print(f"  {thr:>6.4f} {ph:>4d} {mode:<7} {m['CAGR']*100:>6.1f}% "
                      f"{m['MDD']*100:>7.2f}% {m['Cal']:>7.2f} "
                      f"{m.get('Rebal', 0):>5d} {m.get('Trades', 0):>6d}"
                      f"  {verdict if mode == 'refill' else ''}")
        print()
    print(f'총 소요: {time.time()-t0:.0f}초')


if __name__ == '__main__':
    main()
