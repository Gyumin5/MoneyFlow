#!/usr/bin/env python3
"""Breakdown Short grid validator."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_validator import run_grid_validator


def main():
    configs = []
    for dw in [24, 48, 96]:
        for sr in [120, 168, 240]:
            for tp in [0.04, 0.06, 0.10]:
                for ts in [48, 72, 120]:
                    for sab in [0.02, 0.03, 0.05]:
                        configs.append({"donch_window":dw, "sma_regime":sr,
                                        "tp":tp, "tstop":ts, "stop_above_ref":sab})
    print(f"Breakdown Short configs: {len(configs)}")
    run_grid_validator("breakdown_short", configs, checkpoint_every=2)


if __name__ == "__main__":
    main()
