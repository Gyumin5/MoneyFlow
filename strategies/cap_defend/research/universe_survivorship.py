"""생존편향이 얼마나 되나 — 시점별 유니버스가 걸러지는 양을 센다.

왜 보나. 긴 구간 판정(history #116)이 2017년까지 이어 붙인 자료를 쓴다. 그 구간의
성과가 "그때 상위 시총 코인으로 돌린 결과" 가 아니라 "지금까지 살아남은 코인으로
돌린 결과" 면 부풀려진 값이다.

어디서 걸러지나(`unified_backtest._load_mcap`):
    result[ds] = [t for t in raw[ds] if t in TICKER_MAP]
원본 `data/historical_universe.json` 은 그 달의 상위 시총 목록이고, TICKER_MAP 은
지금 우리가 가격 자료를 가진 종목이다. 지금 자료가 없는 종목은 그 달 상위였어도
유니버스에서 사라진다 — 그게 생존편향이다.

다만 빠지는 것이 전부 편향은 아니다. 스테이블코인(테더·유에스디씨·다이 등)과 래핑
토큰(랩트비트코인)은 추세 전략의 대상이 아니라 빠지는 게 맞다. 그래서 둘을 갈라
센다 — 정당한 제외와 생존편향을 더해 하나로 보고하면 편향이 과장된다.

읽는 법. 편향 비율이 옛 구간일수록 크면, 옛 구간 수치를 그만큼 덜 믿어야 한다.
판정은 이 숫자 하나로 하지 않는다 — 관문 3(자료 출처 분할)이 2020-10 경계로
구간을 갈라 같은 방향인지 보는 것이 실제 판정이다.

실행: python3 universe_survivorship.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(CAP, '..', '..'))
sys.path.insert(0, CAP)

import unified_backtest as ub  # noqa: E402

UNIV = os.path.join(ROOT, 'data', 'historical_universe.json')

# 추세 전략의 대상이 아니라 빠지는 게 맞은 것들. 값이 1달러에 묶여 있거나(스테이블)
# 다른 자산을 그대로 담은 것(래핑)이라 모멘텀 자체가 성립하지 않는다.
NOT_TRADABLE = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDE', 'PYUSD', 'USDS', 'FDUSD',
    'USDP', 'GUSD', 'SUSD', 'LUSD', 'FRAX', 'USDD', 'EURS', 'EURC',
    'WBTC', 'WETH', 'STETH', 'WSTETH', 'WEETH', 'CBBTC', 'RETH', 'WBETH',
    'BSC-USD', 'SOLVBTC', 'LBTC', 'BTCB',
}


def main():
    if not os.path.exists(UNIV):
        print(f"유니버스 파일이 없다: {UNIV}")
        return 1
    raw = json.load(open(UNIV, encoding='utf-8'))
    have = set(ub.TICKER_MAP)

    print(f"원본 {os.path.relpath(UNIV, ROOT)} · 월 {len(raw)}개\n")
    print("  월          원본  대상외  실제대상  자료있음  편향수  편향비율")
    rows = []
    for ds in sorted(raw):
        orig = [t.replace('-USD', '') for t in raw[ds]]
        nottr = [t for t in orig if t in NOT_TRADABLE]
        target = [t for t in orig if t not in NOT_TRADABLE]
        kept = [t for t in target if t in have]
        bias = [t for t in target if t not in have]
        pct = 100.0 * len(bias) / len(target) if target else 0.0
        rows.append((ds, len(orig), len(nottr), len(target), len(kept),
                     len(bias), pct, bias))

    show = ['2017-01-01', '2017-07-01', '2018-01-01', '2018-07-01',
            '2019-01-01', '2019-07-01', '2020-01-01', '2020-10-01',
            '2021-07-01', '2023-01-01', '2024-07-01', '2026-01-01']
    for ds, o, nt, tg, kp, bs, pct, _ in rows:
        if ds in show:
            print(f"  {ds}  {o:6d} {nt:7d} {tg:9d} {kp:9d} {bs:7d} {pct:8.1f}%")

    # 구간 평균. 관문 3 의 경계(2020-10-01)와 같은 자리에서 가른다.
    def avg(sel):
        g = [r for r in rows if sel(r[0])]
        return (sum(r[6] for r in g) / len(g), len(g)) if g else (0.0, 0)

    old_pct, old_n = avg(lambda d: d < '2020-10-01')
    new_pct, new_n = avg(lambda d: d >= '2020-10-01')
    print(f"\n  2020-10 이전 {old_n}개월 평균 편향 {old_pct:.1f}%")
    print(f"  2020-10 이후 {new_n}개월 평균 편향 {new_pct:.1f}%")
    print(f"  차이 {old_pct - new_pct:+.1f}%포인트")

    # 옛 구간에서 빠진 종목 중 자주 나오는 것.
    from collections import Counter
    c = Counter()
    for row in rows:
        ds, bias = row[0], row[7]
        if ds < '2020-10-01':
            c.update(bias)
    print("\n  2020-10 이전에 가장 자주 빠진 종목 (달 수)")
    print("    " + ", ".join(f"{t} {n}" for t, n in c.most_common(12)))

    print("\n  읽는 법. 이 비율은 옛 구간 수치를 얼마나 덜 믿어야 하는지를 말한다.")
    print("  빠진 종목은 대개 그 뒤 순위가 밀렸거나 사라진 것이라, 넣고 돌렸다면")
    print("  성과가 지금보다 나빴을 가능성이 크다. 판정은 이 숫자가 아니라 관문 3")
    print("  (2020-10 경계로 옛자료·걸친창·새자료를 갈라 같은 방향인지)이 한다.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
