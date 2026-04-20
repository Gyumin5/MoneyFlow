#!/usr/bin/env python3
"""Range Mean Reversion grid validator."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_validator import run_grid_validator


def main():
    configs = []
    for bn in [14, 20, 30]:
        for rt in [20, 25, 30]:
            for bw in [0.04, 0.06, 0.10]:
                for tp in [0.03, 0.04, 0.06]:
                    for sl in [0.02, 0.03, 0.05]:
                        for ts in [24, 48, 72]:
                            configs.append({"bb_n":bn, "rsi_thr":rt, "bb_width_max":bw,
                                            "tp":tp, "stop_loss":sl, "tstop":ts})
    print(f"Range MR configs: {len(configs)}")
    run_grid_validator("range_mr", configs, checkpoint_every=2)


if __name__ == "__main__":
    main()
