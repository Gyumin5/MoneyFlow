#!/usr/bin/env python3
"""Pullback grid validator — parallel_validator 사용."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_validator import run_grid_validator


def main():
    configs = []
    for ef in [30, 50, 80]:
        for es in [150, 200, 300]:
            for pb in [0.01, 0.015, 0.025]:
                for tp in [0.04, 0.06, 0.10]:
                    for ts in [48, 72, 120]:
                        configs.append({"ema_fast":ef, "ema_slow":es,
                                        "pullback_min":pb, "tp":tp, "tstop":ts})
    print(f"Pullback configs: {len(configs)}")
    run_grid_validator("pullback", configs, checkpoint_every=2)


if __name__ == "__main__":
    main()
