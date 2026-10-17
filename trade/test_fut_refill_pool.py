#!/usr/bin/env python3
"""선물 BT 의 드리프트 리필 후보 풀이 라이브와 같은 모양인지 본다 (2026-10-18).

라이브 `trade/auto_trade_binance.py` 의 apply_refill_v2_fut 는 그날 받아 온 유니버스
전체에서 단기·장기 모멘텀이 둘 다 양수인 코인을 모으고, 단기 모멘텀 내림차순(동률은
심볼 오름차순)으로 세운 뒤 앞에서부터 빈 자리를 채운다. 변동성 상한도, 상위 몇 개만
남기는 선발도 안 건다.

백테스트는 이 풀을 `_compute_weights` 결과(변동성까지 걸러 고른 상위 3개)로 만들고
심볼순으로 세우고 있었다. 후보가 셋 이하로 줄고 순서도 달라서 리필이 라이브와 다른
코인을 집는다. 전환 후 재평가는 이 리필 경로와 자리를 다투므로 재평가 유무에 공통으로
묻히는 오차가 아니다(토론 run-20261017T210537Z).

엔진 소스에서 실제 `_refill_pool` 함수를 떼어내 돌린다 — 식을 베낀 사본이 아니다.

실행: python3 trade/test_fut_refill_pool.py  (bash run_tests.sh 에 자동 포함)
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE = os.path.join(ROOT, 'strategies', 'cap_defend', 'backtest_futures_v25.py')

MOM_S, MOM_L = 5, 10
NBARS = 40


def _extract_refill_pool():
    """엔진 소스의 중첩 함수 `_refill_pool` 을 그대로 떼어내 실행 가능한 객체로 만든다."""
    src = open(ENGINE, encoding='utf-8').read()
    tree = ast.parse(src)
    seg = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_refill_pool':
            seg = ast.get_source_segment(src, node)
            break
    if seg is None:
        raise AssertionError('엔진에서 _refill_pool 을 못 찾았다 — 리필 풀 구현이 바뀌었다')
    # 들여쓰기를 벗겨 모듈 최상위 함수로 만든다.
    lines = seg.split('\n')
    pad = len(lines[0]) - len(lines[0].lstrip())
    body = '\n'.join(ln[pad:] if len(ln) >= pad else ln for ln in lines)
    return body


def _series(tail_x: float, tail_y: float, wild: bool = False) -> pd.DataFrame:
    """ms = y/x - 1, ml = y/100 - 1 이 되는 종가 배열을 만든다."""
    c = [100.0] * NBARS
    if wild:
        # 변동성만 크게 만든다. 새 풀은 변동성을 안 보므로 결과가 바뀌면 안 된다.
        for i in range(NBARS - 11, NBARS - 6):
            c[i] = 100.0 * (1.5 if i % 2 else 0.6)
    c[-11] = 100.0
    c[-6] = tail_x
    c[-1] = tail_y
    idx = pd.date_range('2024-01-01', periods=NBARS, freq='D')
    return pd.DataFrame({'Close': np.array(c)}, index=idx)


def main() -> int:
    ns = {}
    exec(_extract_refill_pool(), ns)

    bars = {
        'BTC': _series(100.0, 150.0),     # 항상 제외
        'AAA': _series(100.0, 101.0),     # ms +1%
        'BBB': _series(100.0, 130.0, wild=True),  # ms +30%, 변동성 큼
        'CCC': _series(100.0, 110.0),     # ms +10%
        'DDD': _series(100.0, 105.0),     # ms +5%
        'EEE': _series(100.0, 95.0),      # ms -5% → 탈락
        'ZZZ': _series(100.0, 140.0),     # 그날 유니버스 밖 → 제외돼야 함
    }
    universe = ['BTC', 'AAA', 'BBB', 'CCC', 'DDD', 'EEE']

    ns['bars'] = bars
    ns['get_mcap'] = lambda d: universe
    ns['calc_mom'] = lambda a, p: (-999 if len(a) < p + 1 else a[-1] / a[-p - 1] - 1)
    ns['mom30'] = MOM_S
    ns['mom90'] = MOM_L

    sig_date = bars['AAA'].index[-1]
    pool = ns['_refill_pool'](sig_date)

    want = ['BBB', 'CCC', 'DDD', 'AAA']
    ok = True

    print('후보 풀:', pool)
    print('기대값  :', want)
    if pool != want:
        print('실패: 풀의 구성 또는 순서가 라이브와 다르다')
        ok = False

    if len(pool) <= 3:
        print(f'실패: 후보가 {len(pool)}개뿐이다 — 상위 선발(3개)에 묶여 있다')
        ok = False

    if 'BBB' not in pool:
        print('실패: 변동성 큰 코인이 빠졌다 — 라이브 리필은 변동성 상한을 안 건다')
        ok = False

    if pool == sorted(pool):
        print('실패: 심볼순으로 서 있다 — 라이브는 단기 모멘텀 내림차순이다')
        ok = False

    if 'ZZZ' in pool:
        print('실패: 그날 유니버스 밖 코인이 들어왔다 — 생존편향이다')
        ok = False

    if 'BTC' in pool:
        print('실패: BTC 가 들어왔다 — 라이브는 BTC 를 제외한다')
        ok = False

    if 'EEE' in pool:
        print('실패: 단기 모멘텀 음수 코인이 들어왔다')
        ok = False

    print('통과' if ok else '실패')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
