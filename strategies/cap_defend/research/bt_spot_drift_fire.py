"""현물 드리프트 문턱별로 실제 발화가 몇 번이나 되나 (운영자 물음 2026-10-15).

물음은 "0.2 면 너무 커서 그냥 끄는 거나 다름없지 않나" 였다. 그걸 세는 방법은 하나다 —
앵커(217봉)·카나리 플립은 문턱과 무관하게 발화하므로, 문턱을 끈 판(0)을 기준선으로 두고
거기서 늘어난 리밸런싱 수를 드리프트가 더한 몫으로 읽는다.

run() 은 CAGR·MDD 를 분수로 돌려준다(0.6 = +60%). 그래서 출력에서 100 을 곱한다.
"""
import sys
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import bt_coin_param_resweep as R
from unified_backtest import load_data, run as bt_run

R._bars, R._funding = load_data("D")


def one(thr):
    kw = dict(R.SPOT_BASE)
    kw["drift_threshold"] = thr
    return bt_run(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
                  tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
                  vol_mode="daily", phase_offset_bars=R.PH_SPOT[0],
                  post_flip_delay=5, selection="greedy", stop_kind="none", stop_pct=0.0,
                  dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7,
                  crash_threshold=-99.0, health_mode="mom2vol", **kw)


base = one(0.0)
print(f"{'문턱':>8} {'리밸 총':>7} {'드리프트 몫':>10} {'연 평균':>7} "
      f"{'거래':>6} {'칼마':>7} {'CAGR':>8} {'MDD':>8}")
yrs = 5.97
b = base["Rebal"]
print(f"{'끔(0)':>8} {b:7d} {0:10d} {0:7.1f} {base['Trades']:6d} "
      f"{base['Cal']:7.3f} {base['CAGR'] * 100:7.1f}% {base['MDD'] * 100:7.1f}%")
for thr in (0.06, 0.10, 0.15, 0.20, 0.30):
    m = one(thr)
    d = m["Rebal"] - b
    print(f"{thr:8.2f} {m['Rebal']:7d} {d:10d} {d / yrs:7.1f} {m['Trades']:6d} "
          f"{m['Cal']:7.3f} {m['CAGR'] * 100:7.1f}% {m['MDD'] * 100:7.1f}%")
