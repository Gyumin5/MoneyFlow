#!/usr/bin/env python3
"""VBO grid validator."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_validator import run_grid_validator


def main():
    configs = []
    for dw in [24, 48, 96]:
        for aw in [10, 14, 20]:
            for tm in [1.5, 2.0, 3.0]:
                for tp in [0.05, 0.08, 0.15]:
                    for vf in [0.025, 0.035, 0.05]:
                        configs.append({"donch_window":dw, "atr_window":aw,
                                        "trail_atr_mult":tm, "tp":tp,
                                        "vol_filter_max":vf})
    print(f"VBO configs: {len(configs)}")
    run_grid_validator("vbo", configs, checkpoint_every=2)


if __name__ == "__main__":
    main()
