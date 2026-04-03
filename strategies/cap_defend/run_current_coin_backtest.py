#!/usr/bin/env python3
"""현재 공식 현물 코인 전략 단독 백테스트."""

import os
import sys
import subprocess


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(root, "backtest_official.py"),
        "--coin-only",
        "--version",
        "v18",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
