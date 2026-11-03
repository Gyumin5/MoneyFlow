"""일일 보고가 들고 있는 상수가 실행층 정본과 같은지 본다.

왜 시험이 필요한가. recommend_personal.py 는 조회 전용이라 매매를 안 하지만,
운영자는 그 화면만 보고 송금을 정한다. 2026-11-03 에 대시보드 자바스크립트가
옛 배분(60/25/15)과 옛 문턱(T1 0.20)을 들고 있는 것이 드러났다 — 바로 위 주석은
새 값으로 갱신돼 있었는데 코드가 안 따라왔다. 그 상태로 보유가 정확히 목표대로
60/20/20 이어도 화면은 "리밸런싱 필요" 를 띄운다.

사본을 손으로 맞추는 것으로는 안 막힌다. 그래서 값을 읽어 대조한다.

엔진을 import 하지 않고 파일 글자에서 읽는 이유는 둘이다. auto_trade_binance 는
불러오는 순간 서버 전용 자격증명 모듈을 찾아서 집에서는 깨지고, coin_live_engine 은
import 부작용이 없더라도 같은 방식으로 읽어야 한쪽만 다른 경로가 되지 않는다.
"""
from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE: dict[str, dict[str, float]] = {}


def module_consts(rel: str) -> dict[str, float]:
    """그 파일 모듈 수준의 숫자 상수. 구문 트리로 읽어 import 를 안 탄다.

    `A = 0.1` 과 `A, B = 0.1, 0.2` 를 모두 받는다. 뒤에 다시 대입되면
    마지막 값이 남는다 — 실행 결과와 같은 쪽이다.
    """
    if rel in _CACHE:
        return _CACHE[rel]
    src = open(os.path.join(ROOT, rel), encoding='utf-8').read()
    out: dict[str, float] = {}

    def num(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                v = num(node.value)
                if v is not None:
                    out[tgt.id] = v
            elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for t, v in zip(tgt.elts, node.value.elts):
                    if isinstance(t, ast.Name):
                        n = num(v)
                        if n is not None:
                            out[t.id] = n
    _CACHE[rel] = out
    return out


def read_const(rel: str, name: str) -> float:
    vals = module_consts(rel)
    if name not in vals:
        raise AssertionError(f'{rel} 에 {name} 정의가 없다')
    return vals[name]


def main() -> int:
    # (보고 쪽 상수, 정본 파일, 정본 이름)
    pairs = [
        ('DRIFT_THRESHOLD_SPOT', 'trade/coin_live_engine.py', 'DRIFT_THRESHOLD_SPOT'),
        ('DRIFT_THRESHOLD_FUT', 'trade/auto_trade_binance.py', 'DRIFT_THRESHOLD_FUT'),
    ]
    rpt = 'strategies/cap_defend/recommend_personal.py'
    bad = []
    for rep_name, canon_rel, canon_name in pairs:
        got = read_const(rpt, rep_name)
        want = read_const(canon_rel, canon_name)
        if got != want:
            bad.append(f'{rep_name}: 보고 {got} vs 정본 {canon_rel} {want}')
        else:
            print(f'  OK {rep_name} = {got} (정본 {canon_rel})')

    # 배분·트리거는 recommend.py 가 정본이고 recommend_personal.py 가 같은 값을 든다.
    # 둘을 따로 고치는 날이 오면 여기서 걸린다.
    for name in ('STOCK_RATIO', 'COIN_RATIO', 'FUTURES_RATIO',
                 'REBAL_HT_THRESHOLD', 'REBAL_T3U_REL'):
        try:
            a = read_const(rpt, name)
        except AssertionError as exc:
            bad.append(str(exc))
            continue
        try:
            b = read_const('strategies/cap_defend/recommend.py', name)
        except AssertionError:
            continue  # recommend.py 가 안 들고 있는 이름은 건너뛴다
        if a != b:
            bad.append(f'{name}: recommend_personal {a} vs recommend {b}')
        else:
            print(f'  OK {name} = {a} (두 리포트 일치)')

    # 배분 셋의 합이 1 인지. 한 칸만 옮겨 적는 실수를 잡는다.
    s = sum(read_const(rpt, n) for n in ('STOCK_RATIO', 'COIN_RATIO', 'FUTURES_RATIO'))
    if abs(s - 1.0) > 1e-9:
        bad.append(f'배분 합이 1 이 아니다: {s}')
    else:
        print(f'  OK 배분 합 = {s}')

    # 화면 자바스크립트가 상수를 주입받는지. 숫자를 박아 두면 다시 갈린다.
    src = open(os.path.join(ROOT, rpt), encoding='utf-8').read()
    for js in ('T_STOCK', 'TARGET_STOCK_RATIO', 'REBAL_HT', 'REBAL_T3U'):
        m = re.search(r'const\s+' + js + r'\s*=\s*([^;\n]+)', src)
        if not m:
            continue
        rhs = m.group(1)
        if re.fullmatch(r'[\s0-9.,=A-Z_]*', rhs) and re.search(r'\d', rhs):
            bad.append(f'화면 상수 {js} 가 숫자를 직접 들고 있다: {rhs.strip()}')
        else:
            print(f'  OK 화면 {js} 는 주입받는다')

    if bad:
        for b in bad:
            print('  FAIL ' + b)
        return 1
    print('PASS 보고 상수와 정본이 같다')
    return 0


if __name__ == '__main__':
    sys.exit(main())
