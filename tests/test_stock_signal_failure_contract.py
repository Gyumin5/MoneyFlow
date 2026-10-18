#!/usr/bin/env python3
"""주식 신호의 실패 계약 (2026-10-18, 토론 run-20261018T060335Z 1라운드 합의 A안).

고친 결함. 일간 리포트의 주식 전략 함수는 계산에 실패하면 (전량 현금, "Error", meta) 를
돌려줬고, 신호 저장 자리는 모드 이름 문자열로 국면을 되짚었다("수비로 시작하지 않으면 공격").
그래서 전략 모듈을 못 불러온 회차가 risk_on=true · offense_picks=[] ·
offense_weights={'Cash':1.0} 로 저장됐다 — 계산 실패가 "오늘은 전량 현금이 목표다" 라는
정상 신호와 같은 모양이 된다. 실매매의 비중 검증(_picks_weights_ok)은 그 모양을 모멘텀
전멸로 보고 통과시키므로, 실행기 자체 계산까지 실패한 날 그 신호가 대체 경로로 넘어가면
주식 슬리브(총자산 60%) 목표가 전량 매도가 된다.

채택한 계약.
  · 성공 여부는 meta 의 값으로 알린다. 모드 이름·비중 합계·예외 유무로 추론하지 않는다.
  · 실패 회차에는 오늘 값을 한 칸도 쓰지 않는다. 직전 목표 넷만 보존하고 실패를 적는다.
    직전 값이 없으면 빈 목표를 둔다 — 전량 현금 목표를 지어내지 않는다.
  · 실패 블록에는 실행용 기준일(basis_*)을 하나도 붙이지 않는다. 읽는 쪽이 저장된
    basis_lag_bd 를 오늘 날짜로 다시 계산하지 않기 때문에, 기준일을 붙이면 갱신 시각만
    새것인 낡은 목표가 통과한다. 기준일이 없으면 기존 검사가 그대로 차단한다 —
    그래서 실매매 파일(executor_stock.py)은 안 고친다.
  · 자료가 충분한 정상 모멘텀 전멸의 전량 현금은 성공으로 남는다.

확인하는 것.
  S1~S8  전략 함수가 각 실패 종류를 meta 에 실제로 표시하는가 (성공은 성공으로 남는가).
  B1~B8  조립 함수의 성공·실패 블록 내용 — 보존, 기준일 부재, 성공 시각, 연속 실패, 복구.
  E1~E3  조립한 블록을 실매매의 진짜 검증 함수에 먹여 거절·통과를 확인. 되돌림 대조 포함.
  W1~W2  저장 경로가 이 조립 함수를 실제로 쓰는가 (기준일을 밖에서 다시 붙이지 않는가).

두 파일 모두 맨 위에서 비밀값·거래소 모듈을 import 하므로 통째로 불러오지 않는다.
함수와 상수를 소스에서 떼어 와 돌린다 — 판정식을 시험 안에 베껴 적지 않는다.
"""
import ast
import json
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PERSONAL = os.path.join(ROOT, 'strategies', 'cap_defend', 'recommend_personal.py')
EXEC_STOCK = os.path.join(ROOT, 'trade', 'executor_stock.py')

FAILED = []


def check(name, cond, detail=''):
    mark = 'OK ' if cond else 'FAIL'
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


def src_of(path):
    return open(path, encoding='utf-8').read()


def grab(src, name, path):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise SystemExit(f"{os.path.basename(path)} 에서 {name} 을 못 찾았다")


def const(src, name, path):
    """상수를 소스에서 읽어 온다 — 값을 시험에 베껴 적지 않는다."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise SystemExit(f"{os.path.basename(path)} 에서 상수 {name} 을 못 찾았다")


PSRC = src_of(PERSONAL)
ESRC = src_of(EXEC_STOCK)

BASIS_KEYS = const(PSRC, 'STOCK_BASIS_KEYS', PERSONAL)


# ── 전략 함수 / 조립 함수 들어 올리기 ────────────────────────────────────────
def load_personal_funcs(state_file):
    ns = {
        'pd': pd, 'json': json, 'os': os,
        'CASH_ASSET': const(PSRC, 'CASH_ASSET', PERSONAL),
        'DEFENSIVE_STOCK_UNIVERSE': const(PSRC, 'DEFENSIVE_STOCK_UNIVERSE', PERSONAL),
        'STOCK_CANARY_MA_PERIOD': const(PSRC, 'STOCK_CANARY_MA_PERIOD', PERSONAL),
        'STOCK_CANARY_HYST': const(PSRC, 'STOCK_CANARY_HYST', PERSONAL),
        'SIGNAL_STATE_FILE': state_file,
        'STOCK_BASIS_KEYS': BASIS_KEYS,
    }
    for fn in ('calc_ret', 'run_stock_strategy_v15', 'build_stock_signal_block'):
        exec(grab(PSRC, fn, PERSONAL), ns)
    return ns


def load_fresh_checker():
    logs = []
    ns = {
        'datetime': __import__('datetime').datetime,
        'log': logs.append,
        'DATE_ALIGN_TOLERANCE_BD': const(ESRC, 'DATE_ALIGN_TOLERANCE_BD', EXEC_STOCK),
        'V25_DEPLOY_DATE': const(ESRC, 'V25_DEPLOY_DATE', EXEC_STOCK),
        'V24_FALLBACK_BLOCK_DAYS': const(ESRC, 'V24_FALLBACK_BLOCK_DAYS', EXEC_STOCK),
    }
    exec(grab(ESRC, '_signal_state_fresh', EXEC_STOCK), ns)
    return ns['_signal_state_fresh'], logs


# ── 가짜 전략 모듈 ──────────────────────────────────────────────────────────
class FakeV25:
    """stock_strategy_v25 자리. 호출 결과를 시험이 정한다."""
    def __init__(self, offense=None, raise_offense=False):
        self._offense = offense
        self._raise = raise_offense

    def stock_basis(self, all_prices):
        return {'basis': '2026-10-16', 'oldest': '2026-10-16', 'mixed': False,
                'lag_bd': 0, 'expected': '2026-10-16', 'warn': None,
                'per_ticker': {'SPY': '2026-10-16'}}

    def compute_offense(self, all_prices):
        if self._raise:
            raise RuntimeError('야후 응답이 비었다')
        return self._offense


def with_fake(mod):
    sys.modules['stock_strategy_v25'] = mod
    return mod


def series(vals):
    idx = pd.date_range('2025-01-01', periods=len(vals), freq='B')
    return pd.Series(vals, index=idx)


def prices_canary(on=True, defense='positive'):
    """EEM 200일 이동평균 대비 위/아래. 수비 종목의 6개월 수익률도 같이 만든다."""
    base = [100.0] * 260
    eem = series(base[:-1] + [120.0]) if on else series(base[:-1] + [80.0])
    out = {'EEM': eem}
    for t in const(PSRC, 'DEFENSIVE_STOCK_UNIVERSE', PERSONAL):
        if defense == 'positive':
            out[t] = series([100.0] * 200 + [float(100 + i) for i in range(60)])
        elif defense == 'negative':
            out[t] = series([200.0] * 200 + [float(200 - i) for i in range(60)])
    return out


def run_strategy(ns, all_prices, fake):
    with_fake(fake)
    log = []
    try:
        w, stat, meta = ns['run_stock_strategy_v15'](log, all_prices,
                                                     pd.Timestamp('2026-10-19'))
    finally:
        sys.modules.pop('stock_strategy_v25', None)
    return w, stat, meta


def main():
    print("== 주식 신호 실패 계약 ==")
    tmp = tempfile.TemporaryDirectory()
    state_file = os.path.join(tmp.name, 'signal_state.json')
    ns = load_personal_funcs(state_file)
    build = ns['build_stock_signal_block']

    OK_OFFENSE = (['SPY', 'QQQ', 'GLD'],
                  {'SPY': 0.327, 'QQQ': 0.327, 'GLD': 0.327, 'Cash': 0.02},
                  {'df': None, 'cash_slot': 0.02})

    # ── S: 전략 함수가 실패를 meta 에 표시하는가 ────────────────────────────
    _, stat, m = run_strategy(ns, prices_canary(True), FakeV25(offense=OK_OFFENSE))
    check('S1 공격 정상 → 성공 + risk_on True',
          m['compute_ok'] is True and m['risk_on'] is True and m['canary_ok'] is True,
          str(m))

    _, stat, m = run_strategy(ns, prices_canary(True), None)  # sys.modules[...] = None
    check('S2 전략 모듈 적재 실패 → 실패 표시',
          m['compute_ok'] is False and m['fail_reason'] == 'module_load_failed'
          and stat == 'Error', f"{stat} {m.get('fail_reason')}")

    _, _, m = run_strategy(ns, prices_canary(True), FakeV25(raise_offense=True))
    check('S3 공격 계산 예외 → 실패 표시',
          m['compute_ok'] is False and m['fail_reason'] == 'offense_exception',
          str(m.get('fail_reason')))

    _, _, m = run_strategy(ns, prices_canary(True),
                           FakeV25(offense=([], {'Cash': 1.0}, {'reason': 'no_data'})))
    check('S4 공격 입력 부족(no_data) → 실패 표시 (정상 현금 아님)',
          m['compute_ok'] is False and m['fail_reason'] == 'offense_no_data',
          str(m.get('fail_reason')))

    w, stat, m = run_strategy(ns, prices_canary(False), FakeV25(offense=OK_OFFENSE))
    check('S5 카나리 꺼짐 + 수비 정상 → 성공 + risk_on False',
          m['compute_ok'] is True and m['risk_on'] is False and 'Cash' not in w,
          f"{stat} {m}")

    _, _, m = run_strategy(ns, {'EEM': series([100.0] * 10)}, FakeV25(offense=OK_OFFENSE))
    check('S6 카나리 자료 부족 → 실패 표시 + risk_on None',
          m['compute_ok'] is False and m['fail_reason'] == 'canary_no_data'
          and m['risk_on'] is None and m['canary_ok'] is False, str(m))

    p = prices_canary(False)
    for t in list(p):
        if t != 'EEM':
            p.pop(t)
    _, stat, m = run_strategy(ns, p, FakeV25(offense=OK_OFFENSE))
    check('S7 수비 자료 없음 → 실패 표시',
          m['compute_ok'] is False and m['fail_reason'] == 'defense_no_data',
          f"{stat} {m.get('fail_reason')}")

    w, stat, m = run_strategy(ns, prices_canary(False, defense='negative'),
                              FakeV25(offense=OK_OFFENSE))
    check('S8 수비 전부 음수(정상 전량 현금) → 성공으로 남는다',
          m['compute_ok'] is True and m['risk_on'] is False and w == {'Cash': 1.0},
          f"{stat} {m}")

    # ── B: 조립 함수 ────────────────────────────────────────────────────────
    BASIS = {'basis': '2026-10-16', 'oldest': '2026-10-16', 'mixed': False,
             'lag_bd': 0, 'per_ticker': {'SPY': '2026-10-16'}}
    PREV = {
        'offense_picks': ['GLD', 'PDBC', 'VNQ'],
        'offense_weights': {'GLD': 0.327, 'PDBC': 0.327, 'VNQ': 0.327, 'Cash': 0.02},
        'defense_picks': ['BIL', 'BNDX', 'GLD'],
        'defense_weights': {'BIL': 0.327, 'BNDX': 0.327, 'GLD': 0.327, 'Cash': 0.02},
        'risk_on': True, 'computed_side': 'offense', 'stale_side': 'defense',
        'basis_date': '2025-04-23', 'basis_oldest': '2025-04-23',
        'basis_mixed': False, 'basis_lag_bd': 0,
        'basis_per_ticker': {'SPY': '2025-04-23'},
        'last_success_at': '2026-10-16 09:15',
    }
    NOW = '2026-10-19 09:15'
    TODAY_PICKS = ['QQQ', 'SPY']
    TODAY_W = {'QQQ': 0.49, 'SPY': 0.49, 'Cash': 0.02}

    ok_on = {'compute_ok': True, 'risk_on': True, 'basis': BASIS}
    b = build(PREV, ok_on, TODAY_PICKS, TODAY_W, NOW)
    check('B1 성공(공격) — 오늘 공격 + 직전 수비 보존 + 기준일 기록',
          b['offense_picks'] == TODAY_PICKS and b['defense_picks'] == PREV['defense_picks']
          and b['computed_side'] == 'offense' and b['stale_side'] == 'defense'
          and b['compute_failed'] is False and b['last_success_at'] == NOW
          and b['basis_date'] == '2026-10-16' and b['risk_on'] is True, str(b))

    b = build(PREV, {'compute_ok': True, 'risk_on': False, 'basis': BASIS},
              TODAY_PICKS, TODAY_W, NOW)
    check('B2 성공(수비) — 오늘 수비 + 직전 공격 보존',
          b['defense_picks'] == TODAY_PICKS and b['offense_picks'] == PREV['offense_picks']
          and b['computed_side'] == 'defense' and b['risk_on'] is False, str(b))

    fail_meta = {'compute_ok': False, 'fail_reason': 'module_load_failed',
                 'risk_on': True, 'basis': BASIS}
    f = build(PREV, fail_meta, TODAY_PICKS, TODAY_W, NOW)
    check('B3 실패 — 목표 넷 전부 직전 값 그대로',
          all(f[k] == PREV[k] for k in ('offense_picks', 'offense_weights',
                                        'defense_picks', 'defense_weights')), str(f))
    check('B3 실패 — 실패 표시와 보존 표시',
          f['compute_failed'] is True and f['fail_reason'] == 'module_load_failed'
          and f['failed_at'] == NOW and f['computed_side'] is None
          and f['stale_side'] == 'both', str(f))
    check('B3 실패 — 성공 시각은 직전 값 그대로 (지금 시각을 넣지 않는다)',
          f['last_success_at'] == PREV['last_success_at'], str(f.get('last_success_at')))
    check('B3 실패 — 실행용 기준일 칸이 하나도 없다 (직전 것도 오늘 것도)',
          not any(k in f for k in BASIS_KEYS),
          f"남은 칸 {[k for k in BASIS_KEYS if k in f]}")
    check('B3 실패 — 카나리는 멀쩡했으므로 오늘 값을 쓴다',
          f['risk_on'] is True and f['risk_on_source'] == 'today', str(f))

    f0 = build({}, {'compute_ok': False, 'fail_reason': 'canary_no_data',
                    'risk_on': None, 'basis': BASIS}, TODAY_PICKS, TODAY_W, NOW)
    check('B4 실패 + 직전 값 없음 — 빈 목표 (전량 현금을 지어내지 않는다)',
          f0['offense_picks'] == [] and f0['offense_weights'] == {}
          and f0['defense_picks'] == [] and f0['defense_weights'] == {}, str(f0))
    check('B4 실패 + 카나리도 없음 + 직전 없음 → risk_on None',
          f0['risk_on'] is None and f0['risk_on_source'] == 'none'
          and f0['last_success_at'] is None, str(f0))

    f1 = build(PREV, {'compute_ok': False, 'fail_reason': 'canary_no_data',
                      'risk_on': None, 'basis': BASIS}, TODAY_PICKS, TODAY_W, NOW)
    check('B5 카나리를 못 쟀으면 직전 카나리를 재사용하고 그렇다고 적는다',
          f1['risk_on'] is True and f1['risk_on_source'] == 'prev', str(f1))

    f2 = build(f, fail_meta, TODAY_PICKS, TODAY_W, '2026-10-20 09:15')
    check('B6 연속 실패 — 목표와 성공 시각이 더 흔들리지 않는다',
          f2['offense_picks'] == PREV['offense_picks']
          and f2['last_success_at'] == PREV['last_success_at']
          and not any(k in f2 for k in BASIS_KEYS), str(f2))

    r = build(f2, ok_on, TODAY_PICKS, TODAY_W, '2026-10-21 09:15')
    check('B7 복구 — 실패 표시가 풀리고 새 성공 시각이 적힌다',
          r['compute_failed'] is False and r['last_success_at'] == '2026-10-21 09:15'
          and 'fail_reason' not in r and 'failed_at' not in r
          and r['basis_date'] == '2026-10-16', str(r))

    cash = build(PREV, {'compute_ok': True, 'risk_on': False, 'basis': BASIS},
                 [], {'Cash': 1.0}, NOW)
    check('B8 정상 전량 현금은 그대로 발행된다 (기준일 포함)',
          cash['defense_weights'] == {'Cash': 1.0} and cash['compute_failed'] is False
          and cash['basis_date'] == '2026-10-16', str(cash))

    # ── E: 실매매의 진짜 검증 함수에 먹여 본다 ──────────────────────────────
    fresh, _logs = load_fresh_checker()

    def wrap(stock_block):
        return {'stock': stock_block,
                'coin': {'picks': [], 'weights': {}, 'risk_on': True, 'guard_refs': {}},
                'meta': {'signal_date': '2026-10-19', 'updated_at': NOW_REAL}}

    from datetime import datetime as _dt
    NOW_REAL = _dt.now().strftime('%Y-%m-%d %H:%M')

    check('E1 실패 블록은 대체 신호로 거절된다 (갱신 시각이 방금이어도)',
          fresh(wrap(f)) is False)
    check('E2 정상 전량 현금 신호는 통과한다',
          fresh(wrap(cash)) is True)

    # 되돌림 대조: 실패 블록에 직전 기준일을 다시 붙이면 그대로 통과한다.
    leaked = dict(f)
    for k in BASIS_KEYS:
        leaked[k] = PREV[k]
    check('E3 되돌림 대조 — 기준일을 다시 붙이면 낡은 목표가 통과한다',
          fresh(wrap(leaked)) is True,
          '이 통과가 바로 기준일 칸을 빼는 이유다')

    # ── W: 저장 경로가 이 조립 함수를 실제로 쓰는가 ─────────────────────────
    check('W1 저장 블록이 조립 함수 결과를 그대로 쓴다',
          '_stock_block = build_stock_signal_block(' in PSRC
          and "'stock': _stock_block," in PSRC)
    outside = [ln for ln in PSRC.splitlines()
               if "_stock_block['basis" in ln or '_stock_block["basis' in ln]
    check('W2 조립 함수 밖에서 기준일을 다시 붙이지 않는다', not outside, str(outside))

    tmp.cleanup()
    print()
    if FAILED:
        print(f"실패 {len(FAILED)}건: {FAILED}")
        return 1
    print("전부 통과")
    return 0


if __name__ == '__main__':
    sys.exit(main())
