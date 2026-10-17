import os, sys
import pandas as pd
CAP = "/home/gmoh/mon/251229/strategies/cap_defend"
sys.path.insert(0, CAP); sys.path.insert(0, os.path.join(CAP, "research"))
import alloc_grid_run as RUN
import bt_coin_param_resweep as R
from unified_backtest import run as bt_run
RUN._init_resweep(); R.END = "2026-10-17"
print("현물 BT 거래비용 민감도 (창 2026-06-01~2026-10-17)")
for tx, tag in ((0.004, "0.40% 편도 (저장소 표준)"), (0.0005, "0.05% 편도 (업비트 실제)")):
    cums = []
    for a in range(7):
        os.environ["DRIFT_HEALTH_MODE"] = "refill"
        m = bt_run(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
                   tx_cost=tx, start_date=R.START, end_date=R.END,
                   vol_mode="daily", phase_offset_bars=R.PH_SPOT[a],
                   post_flip_delay=5, selection="greedy",
                   stop_kind="none", stop_pct=0.0,
                   dd_lookback=60, dd_threshold=-99.0,
                   bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
                   health_mode="mom2vol", **dict(R.SPOT_BASE))
        s = m.get("_equity").dropna()
        w = s[s.index >= pd.Timestamp("2026-06-01")]
        cums.append(w.iloc[-1] / w.iloc[0] - 1)
    print(f"  {tag}: 평균 {sum(cums)/7*100:+6.2f}%  "
          f"[{min(cums)*100:+6.2f} ~ {max(cums)*100:+6.2f}]")
