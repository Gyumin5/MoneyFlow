"""코인 백테스트 유니버스 구멍을 메운 효과를 기준선으로 잰다 (2026-09-22).

계기: 세 후보(C7·C6·C3)를 현재 전략에 붙여 재려고 데이터를 갱신하다가, 재던 자
자체가 어긋나 있었다는 것을 발견했다. 셋이 겹쳐 있었다.
  (1) download_futures_data.py 의 심볼 목록이 40종 고정이라 시점별 상위 40 중
      스물한 종의 가격이 아예 없었다.
  (2) 두 백테스트 엔진 안에도 40종 고정 이름표가 또 있어서, 데이터를 채워도
      엔진이 후보로 올리지 않았다.
  (3) 엔진이 시점별 시가총액 순위를 읽을 때 backup_20260125/ 사본을 먼저 봤는데
      그 사본은 2025-12-01 에서 멈춰 있다. 그 뒤 구간은 전부 2025년 12월 순위로
      돌았다.
셋 다 고쳤다. 이 스크립트는 고치기 전과 후의 차이를 수치로 남긴다 — 후보 비교를
새 기준선 위에서 하겠다고 말하려면 그 기준선이 얼마나 움직였는지를 먼저 보여야 한다.

축 분해 (네 팔):
  옛 심볼 + 옛 순위파일   = 고치기 전 상태
  옛 심볼 + 새 순위파일   = (3) 만 고친 것
  새 심볼 + 옛 순위파일   = (1)(2) 만 고친 것
  새 심볼 + 새 순위파일   = 지금 상태
전략 파라미터는 한 값도 안 바꾼다. 현물 V24 · 선물 V25 라이브 설정 그대로다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_universe_baseline.py
출력은 state/20260922-universe-baseline/ 로 리다이렉트한다.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAP))
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

START = "2020-10-01"
END = os.environ.get("BASE_END", "2026-09-18")

# 고치기 전에 두 엔진에 박혀 있던 목록 그대로.
OLD_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'TRXUSDT', 'LINKUSDT',
    'DOTUSDT', 'MATICUSDT', 'UNIUSDT', 'NEARUSDT', 'LTCUSDT',
    'BCHUSDT', 'APTUSDT', 'ICPUSDT', 'FILUSDT', 'ATOMUSDT',
    'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'SHIBUSDT', 'PEPEUSDT',
    'XLMUSDT', 'VETUSDT', 'ALGOUSDT', 'FTMUSDT', 'GRTUSDT',
    'AAVEUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'THETAUSDT',
    'EOSUSDT', 'FLOWUSDT', 'CHZUSDT', 'APEUSDT', 'GALAUSDT',
]
OLD_MCAP_FILE = os.path.join(ROOT, "backup_20260125", "historical_universe.json")
NEW_MCAP_FILE = os.path.join(ROOT, "data", "historical_universe.json")


def build_mcap(path: str, ticker_map: dict) -> dict:
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for ds, tickers in raw.items():
        out[ds] = [t.replace("-USD", "") for t in tickers
                   if t.replace("-USD", "") in ticker_map]
    return out


def apply_state(mod, symbols: list[str], mcap_file: str) -> None:
    """엔진 모듈의 심볼표와 시총 캐시를 갈아끼운다. 전략 파라미터는 안 건드린다."""
    tm = {s[:-4]: s for s in symbols}
    mod.TICKER_MAP = tm
    mod._MCAP = build_mcap(mcap_file, tm)


def main() -> None:
    import bt_fut_rebal_band as base
    import unified_backtest as ub
    import backtest_futures_v25 as fv25

    base.START, base.END = START, END

    new_symbols = sorted(f[:-7] for f in os.listdir(ub.DATA_DIR)
                         if f.endswith("_1h.csv"))
    print(f"기간 {START} ~ {END}")
    print(f"심볼 — 옛 {len(OLD_SYMBOLS)}종 / 새 {len(new_symbols)}종")
    for name, path in (("옛 순위파일", OLD_MCAP_FILE), ("새 순위파일", NEW_MCAP_FILE)):
        with open(path) as f:
            keys = sorted(json.load(f))
        print(f"{name}: {len(keys)}개월 {keys[0]} ~ {keys[-1]}")

    print("\n주식 슬리브 계산 중 (이 축과 무관한 고정 팔)...", flush=True)
    m_st = base.run_stock_live()
    eq_st = m_st["_equity"] if isinstance(m_st, dict) else m_st

    arms = [
        ("옛 심볼 + 옛 순위 (고치기 전)", OLD_SYMBOLS, OLD_MCAP_FILE),
        ("옛 심볼 + 새 순위", OLD_SYMBOLS, NEW_MCAP_FILE),
        ("새 심볼 + 옛 순위", new_symbols, OLD_MCAP_FILE),
        ("새 심볼 + 새 순위 (지금)", new_symbols, NEW_MCAP_FILE),
    ]

    rows: dict[str, dict] = {}
    comps: dict[str, pd.Series] = {}
    for label, syms, mfile in arms:
        print(f"\n[{label}] 계산 중...", flush=True)
        apply_state(ub, syms, mfile)
        apply_state(fv25, syms, mfile)
        eq_sp = base.run_spot_live()
        eq_fu = base.run_fut(0.05)["_equity"]
        comp = base.build_alloc(eq_st, eq_sp, eq_fu)
        comps[label] = comp
        rows[label] = dict(현물=base.metrics(eq_sp), 선물=base.metrics(eq_fu),
                           합성=base.metrics(comp))

    for sleeve in ("현물", "선물", "합성"):
        print(f"\n{sleeve} — 전략 파라미터는 전부 동일, 후보군만 다르다")
        print(f"{'팔':<30}{'CAGR':>9}{'MDD':>9}{'Sharpe':>8}{'Calmar':>8}")
        for label, _, _ in arms:
            mt = rows[label][sleeve]
            print(f"{label:<30}{mt['CAGR']:>8.1f}%{mt['MDD']:>8.1f}%"
                  f"{mt['Sharpe']:>8.2f}{mt['Cal']:>8.2f}")
        b = rows[arms[0][0]][sleeve]
        n = rows[arms[-1][0]][sleeve]
        print(f"  고치기 전 대비 지금: CAGR {n['CAGR']-b['CAGR']:+.1f}%p · "
              f"MDD {n['MDD']-b['MDD']:+.1f}%p · Calmar {n['Cal']-b['Cal']:+.2f}")

    sums, wins, n = base.window_rs(comps)
    print(f"\n윈도우 순위합 — 합성, 창 {base.WIN_SIZES} × 보폭 {base.STRIDES}, n={n}")
    print(f"{'팔':<30}{'평균순위':>10}{'1위횟수':>9}")
    for k, v in sorted(sums.items(), key=lambda x: x[1] / max(n, 1)):
        print(f"{k:<30}{v/max(n,1):>10.3f}{wins[k]:>9}")

    print("\n이 표는 채택 판정이 아니다. 후보 비교를 어느 기준선 위에서 하는지를 밝히는 것이다.")


if __name__ == "__main__":
    main()
