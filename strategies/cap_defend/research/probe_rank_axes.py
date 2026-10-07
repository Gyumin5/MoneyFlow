"""순위 축을 바꾸면 실제로 뽑히는 종목이 달라지는 날이 얼마나 되나 — 사전 측정 (2026-10-07).

측정이지 백테스트가 아니다. 자산곡선을 안 낸다. 엔진도 안 건드린다.
지금 선정 블록(헬스 통과분을 시총순으로 훑어 앞 3종)을 그대로 재현한 뒤, 같은 헬스 통과
집합을 다른 축으로 줄 세워 앞 3종이 달라지는 날의 비율만 센다.

왜 먼저 재나. 축을 바꿔도 뽑히는 셋이 거의 그대로면 어떤 팔을 설계하든 자산곡선이 안 갈린다.
2026-10-03 에 선물 자기시험이 같은 이유로 180일 중 130일 둔감했던 전례가 있다 — 교란을
넣기 전에 그 교란이 무엇을 바꾸는지부터 센다.

이 숫자는 채택 근거가 아니다. 어떤 축을 1차에 넣을지 고르는 데만 쓴다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 probe_rank_axes.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)

import backtest_futures_v25 as bf  # noqa: E402

START = "2020-10-01"
END = "2026-09-18"

# 헬스 기준 — 두 슬리브 공통값(선물 기준). 현물은 단기 20봉이라 따로 돌린다.
MOM_LONG = 127
VOL_BARS = 90
VOL_TH = 0.05
SMA = 42
UNIVERSE_SIZE = 3


# 헬스 계산은 엔진 함수를 그대로 쓴다. 다시 구현하면 정의가 어긋나도 조용히 지나간다 —
# 첫 실행에서 변동성을 연환산해 모든 코인이 탈락했던 것이 그 예다.
_mom = bf.calc_mom


def _vol(c: np.ndarray, n: int) -> float:
    return bf.calc_vol_daily(c, 1, lookback_bars=n)


def _dollar_vol(df, ci: int, n: int) -> float:
    lo = max(0, ci + 1 - n)
    v = df["Volume"].values[lo:ci + 1]
    c = df["Close"].values[lo:ci + 1]
    if len(v) == 0:
        return 0.0
    return float(np.mean(v * c))


def _sma_gap(c: np.ndarray, n: int) -> float:
    if len(c) < n:
        return -9.0
    return c[-1] / float(np.mean(c[-n:])) - 1.0


def _dd_from_high(c: np.ndarray, n: int) -> float:
    """최고가 대비 낙폭. 0 에 가까울수록 신고가 근처."""
    if len(c) < n:
        return -9.0
    w = c[-n:]
    return float(c[-1] / np.max(w) - 1.0)


def healthy_at(bars, mcap_order, sig_date, mom_short: int):
    """지금 엔진과 같은 순서로 헬스 통과 목록을 만든다 (시총 내림차순 유지)."""
    out = []
    min_bars = max(mom_short, MOM_LONG, VOL_BARS, SMA)
    for coin in mcap_order:
        df = bars.get(coin)
        if df is None:
            continue
        ci = df.index.get_indexer([sig_date], method="ffill")[0]
        if ci < 0 or ci < min_bars:
            continue
        c = df["Close"].values[:ci + 1]
        ms = _mom(c, mom_short)
        ml = _mom(c, MOM_LONG)
        vl = _vol(c, VOL_BARS)
        if ms > 0 and ml > 0 and vl <= VOL_TH:
            out.append((coin, df, ci, c, ms, ml, vl))
    return out


def axes_for(item, mom_short: int) -> dict[str, float]:
    """정렬 키. 값이 클수록 앞에 온다."""
    coin, df, ci, c, ms, ml, vl = item
    return {
        "모멘텀 단기": ms,
        "모멘텀 127봉": ml,
        "모멘텀 30봉": _mom(c, 30),
        "모멘텀 60봉": _mom(c, 60),
        "변동성 역수": -vl,
        "샤프 단기": ms / vl if vl > 1e-9 else -9.0,
        "샤프 127봉": ml / vl if vl > 1e-9 else -9.0,
        "거래대금 30일": _dollar_vol(df, ci, 30),
        "거래대금 90일": _dollar_vol(df, ci, 90),
        "이격도 42일선": _sma_gap(c, SMA),
        "신고가 근접 127봉": _dd_from_high(c, MOM_LONG),
    }


def main() -> None:
    t0 = time.time()
    from unified_backtest import load_data
    bars, _funding = load_data("D")
    print(f"일봉 로드 {len(bars)}종 {time.time() - t0:.0f}s", flush=True)

    btc = bars.get("BTC")
    dates = [d for d in btc.index if START <= str(d)[:10] <= END]
    print(f"신호일 후보 {len(dates)}일 ({str(dates[0])[:10]} ~ {str(dates[-1])[:10]})", flush=True)

    for label, mom_short in (("선물(단기 18봉)", 18), ("현물(단기 20봉)", 20)):
        stats: dict[str, list[int]] = {}
        n_eval = 0
        n_cash = 0
        n_ord = 0
        pool_sizes = []
        for i, d in enumerate(dates):
            mo = bf.get_mcap(d)
            hs = healthy_at(bars, mo, d, mom_short)
            if not hs:
                n_cash += 1
                continue
            n_eval += 1
            pool_sizes.append(len(hs))
            base = [x[0] for x in hs[:UNIVERSE_SIZE]]
            keys = {x[0]: axes_for(x, mom_short) for x in hs}
            for ax in next(iter(keys.values())).keys():
                alt = sorted(hs, key=lambda x: -keys[x[0]][ax])
                alt3 = [x[0] for x in alt[:UNIVERSE_SIZE]]
                same_set = set(alt3) == set(base)
                same_order = alt3 == base
                s = stats.setdefault(ax, [0, 0, 0])
                s[0] += 1
                if not same_set:
                    s[1] += 1
                if not same_order:
                    s[2] += 1
            n_ord += 1 if len(hs) >= 2 else 0
            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{len(dates)}일 ({time.time() - t0:.0f}s)", flush=True)

        print(f"\n=== {label} ===")
        if not pool_sizes:
            print("평가일 0일 — 모든 날 헬스 통과 0종이다. 헬스 정의가 엔진과 어긋났는지 봐라.")
            continue
        arr = np.array(pool_sizes)
        n_all = len(dates)
        # 통과 풀이 뽑는 수 이하이면 전부 뽑히므로 어떤 축을 써도 같은 셋이 나온다.
        # 그 날을 분모에 넣으면 축이 둔감한 것처럼 보인다 — 따로 센다.
        n_live = int((arr > UNIVERSE_SIZE).sum())
        print(f"신호일 {n_all}일 · 헬스 통과 0종이라 전액 현금인 날 {n_cash}일 "
              f"({n_cash / n_all * 100:.1f}%)")
        print(f"통과 {UNIVERSE_SIZE}종 이하라 순위가 무의미한 날 "
              f"{n_eval - n_live}일 ({(n_eval - n_live) / n_all * 100:.1f}%)")
        print(f"순위가 뽑는 셋을 가를 수 있는 날 {n_live}일 ({n_live / n_all * 100:.1f}%) "
              f"— 그 날 통과 풀 중앙 {int(np.median(arr[arr > UNIVERSE_SIZE]))} "
              f"평균 {arr[arr > UNIVERSE_SIZE].mean():.1f} 최대 {int(arr.max())}")
        # 순서는 통과 2종부터 갈린다 — greedy 흡수가 인접쌍의 단기 모멘텀을 비교하므로
        # 뽑히는 셋이 같아도 순서가 다르면 흡수 결과가 달라진다. 분모를 따로 쓴다.
        print(f"순서가 갈릴 수 있는 날(통과 2종 이상) {n_ord}일 ({n_ord / n_all * 100:.1f}%)")
        print(f"\n{'축':<18} {'셋이 다름 / 가를 수 있는 날':>24} {'순서가 다름 / 2종 이상인 날':>26}")
        for ax, (tot, diff_set, diff_ord) in sorted(stats.items(),
                                                    key=lambda kv: -kv[1][1]):
            print(f"{ax:<18} {diff_set:>7}/{n_live} ({diff_set / n_live * 100:5.1f}%)"
                  f"   {diff_ord:>7}/{n_ord} ({diff_ord / n_ord * 100:5.1f}%)")

    print(f"\n걸린 시간 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
