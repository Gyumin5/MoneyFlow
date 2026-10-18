#!/usr/bin/env python3
"""리포트 표시 경로 둘 (2026-10-18).

두 리포트 파일은 매매를 하지 않는다. 사람이 보는 화면만 만든다. 그래서 여기서 지키는 것은
"틀린 값을 사람에게 보여주지 않는다" 하나다.

고친 것 둘.

  ⑴ 공용 리포트(recommend.py)가 신호 파일의 최상위에서 카나리 값을 찾았다. 실제 스키마는
     주식 칸 안(stock.risk_on)이라 그 자리는 언제나 비고, 그러면 히스테리시스 죽은 구간에서
     이전 상태 대신 단순 비교로 떨어진다. 전 구간 2,514 세션 중 29일이 그 자리였고 그날들은
     실제 매매 슬리브와 반대 국면이 표시됐다.

  ⑵ 개인 리포트(recommend_personal.py)가 직전 신호의 공격 목표를 보유 종목인 것처럼 읽어
     "종목 변경" 알림과 대시보드 입력칸을 채웠다. 목표와 보유는 다른 값이다 — 앵커가 아직
     안 와서 안 샀거나 부분 체결로 남은 자리가 그대로 어긋난다. 값을 새로 읽어오는 대신
     이름과 문구를 사실대로 바로잡았다.

확인하는 것.
  T1~T6 카나리 판정을 실제 소스 조각으로 돌린다. 죽은 구간에서 주식 칸의 이전 상태를 쓰고,
        밴드를 넘으면 이전 상태와 무관하게 뒤집히고, 상태가 없을 때만 단순 비교로 떨어진다.
        최상위 risk_on 만 있는 파일은 상태 없음으로 친다 — 옛 읽기 자리로 되돌리면 T2 가 깨진다.
  T7    스키마 정본에서 risk_on 은 stock 칸 안에 있다 (읽는 자리의 근거).
  T8~T11 개인 리포트가 직전 목표를 목표라고 부르고 표시한다 (보유로 적지 않는다).
  T12   사람이 저장한 보유(localStorage)는 건드리지 않는다 — 그건 실제 보유다.

네트워크·외부 의존 없음. 두 리포트 파일은 맨 위에서 비밀값·거래소 모듈을 import 하므로
통째로 불러오지 않고 소스 조각만 떼어 돌린다.
"""
import json
import os
import sys
import tempfile
import textwrap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
GENERAL = os.path.join(ROOT, 'strategies', 'cap_defend', 'recommend.py')
PERSONAL = os.path.join(ROOT, 'strategies', 'cap_defend', 'recommend_personal.py')

FAILED = []


def check(name, cond, detail=''):
    mark = 'OK ' if cond else 'FAIL'
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


def slice_block(path, start_marker, end_marker):
    """소스에서 한 블록만 떼어 온다 (모듈 전체 import 회피)."""
    lines = open(path, encoding='utf-8').read().splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == start_marker)
        j = next(k for k, ln in enumerate(lines) if k > i and ln.strip() == end_marker)
    except StopIteration:
        raise SystemExit(f"{os.path.basename(path)} 에서 블록을 못 찾았다: {start_marker!r}")
    return textwrap.dedent('\n'.join(lines[i:j + 1]))


def esc(s):
    """소스에 \\uXXXX 로 적힌 한글을 그 표기 그대로 만든다."""
    return ''.join(ch if ord(ch) < 128 else f"\\u{ord(ch):04x}" for ch in s)


def run_canary(block, prev_state, dist, cur, sma):
    """떼어 온 카나리 판정을 돌린다. prev_state 는 신호 파일에 담을 내용 (None 이면 파일 없음)."""
    with tempfile.TemporaryDirectory() as td:
        if prev_state is not None:
            with open(os.path.join(td, 'signal_state.json'), 'w') as f:
                json.dump(prev_state, f)
        ns = {
            'os': os,
            '__file__': os.path.join(td, 'recommend.py'),
            'dist': dist,
            'eem_cur': cur,
            'eem_sma': sma,
            'STOCK_CANARY_HYST': 0.005,
        }
        exec(block, ns)
        return ns['risk_on']


def main():
    print("== 리포트 표시 경로 둘 ==")

    # ── 공용 리포트: 카나리 판정 (실제 소스 조각) ────────────────────────────
    block = slice_block(GENERAL,
                        'prev_stock_risk_on = None',
                        'risk_on = eem_cur > eem_sma  # no state: fallback')

    # 죽은 구간 = |dist| <= 0.005. 단순 비교와 이전 상태가 갈리게 값을 고른다.
    ON_FILE = {"stock": {"risk_on": True}}
    OFF_FILE = {"stock": {"risk_on": False}}

    check('T1 죽은 구간 + 이전 꺼짐 → 꺼짐 유지 (단순 비교는 켜짐)',
          run_canary(block, OFF_FILE, 0.001, 101.0, 100.9) is False)
    check('T2 죽은 구간 + 이전 켜짐 → 켜짐 유지 (단순 비교는 꺼짐)',
          run_canary(block, ON_FILE, -0.001, 100.9, 101.0) is True)
    check('T3 죽은 구간 + 신호 파일 없음 → 단순 비교',
          run_canary(block, None, -0.001, 100.9, 101.0) is False)
    check('T4 죽은 구간 + 최상위 risk_on 만 있는 파일 → 상태 없음 취급',
          run_canary(block, {"risk_on": True}, -0.001, 100.9, 101.0) is False,
          '옛 읽기 자리(최상위)로 되돌리면 여기서 True 가 나온다')
    check('T5 밴드 위로 넘김 → 이전 꺼짐이어도 켜짐',
          run_canary(block, OFF_FILE, 0.0072, 101.0, 100.3) is True)
    check('T6 밴드 아래로 넘김 → 이전 켜짐이어도 꺼짐',
          run_canary(block, ON_FILE, -0.0108, 62.85, 63.54) is False)

    sys.path.insert(0, os.path.join(ROOT, 'trade'))
    import schema
    check('T7 스키마 정본에서 risk_on 은 stock 칸 안에 있다',
          'risk_on' in schema.SIGNAL_REQUIRED_KEYS['stock']
          and 'risk_on' not in schema.SIGNAL_REQUIRED_KEYS
          and schema.SIGNAL_STATE_EXAMPLE.get('risk_on') is None)

    # ── 개인 리포트: 목표를 보유라고 부르지 않는다 ───────────────────────────
    psrc = open(PERSONAL, encoding='utf-8').read()

    check('T8 비교 기준을 직전 공격 목표로 읽고 그 이름으로 부른다',
          "prev_offense_picks = _prev.get('stock', {}).get('offense_picks', [])" in psrc
          and 'set(picks) - set(prev_offense_picks)' in psrc)
    check('T9 변경 알림 문구가 직전 추천 대비라고 밝힌다',
          '직전 추천 대비' in psrc and '직전 추천 종목이 기록돼 있지 않다' in psrc)
    check('T10 대시보드 사전 채움 값을 서버 공격 목표로 부른다',
          "server_offense_picks = _state.get('stock', {}).get('offense_picks', [])" in psrc
          and 'const SERVER_OFFENSE_PICKS' in psrc)
    check('T11 사전 채움 문구가 실제 보유가 아님을 밝힌다',
          esc('실제 보유 아님') in psrc,
          '표시 문구에서 보유 아님 표기가 사라졌다')
    check('T12 사람이 저장한 보유 칸(localStorage)은 그대로 둔다',
          "localStorage.getItem('cap_defend_stock_holdings')" in psrc
          and "localStorage.setItem('cap_defend_stock_holdings'" in psrc)

    # 보유가 아닌 값을 보유라고 부르던 옛 이름이 남아 있지 않다.
    for dead in ('SAVED_STOCK_HOLDINGS', 'saved_stock_holdings', 'prev_holdings'):
        check(f'T13 옛 이름 {dead} 가 남아 있지 않다', dead not in psrc)

    print()
    if FAILED:
        print(f"실패 {len(FAILED)}건: {FAILED}")
        return 1
    print("전부 통과")
    return 0


if __name__ == '__main__':
    sys.exit(main())
