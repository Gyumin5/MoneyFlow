#!/usr/bin/env python3
"""주식 입력이 불량한 회차 뒤에 재시도가 스스로 풀리나.

고정하는 것은 하나다. 가격 캐시를 다시 받게 하는 조건은 '파일을 받은 날이 오늘이
아니다' 이고, 입력 불량으로 멈춘 회차는 그 표시를 하루 뒤로 미뤄 둬야 한다.
안 미루면 5분 뒤 재시도가 같은 낡은 파일을 그대로 읽어 똑같이 멈춘다 —
2026-10-23 09:15 회차가 그랬고 사람이 손으로 풀어야 했다.

파일을 지우지 않는 것도 같이 고정한다. 다시 받는 것마저 실패하면 그 값을 낡은
자료로라도 써야 하는 경로(Final Safety Net)가 있다.

실행: python3 tests/test_recommend_cache_expire.py  (bash run_tests.sh 에 자동 포함)
"""
import os
import sys
import tempfile
from datetime import datetime

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(_root, 'strategies', 'cap_defend'))

import recommend as rc            # noqa: E402
import recommend_personal as rp   # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ''))
    if not cond:
        FAILED.append(name)


def stock_universe(mod):
    return sorted(set(mod.OFFENSIVE_STOCK_UNIVERSE
                      + mod.DEFENSIVE_STOCK_UNIVERSE
                      + mod.CANARY_ASSETS))


def make_cache(d, tickers, body='Date,Adj_Close\n2026-10-20,100.0\n'):
    """오늘 받은 것으로 찍힌 캐시 파일들을 만든다."""
    for t in tickers:
        with open(os.path.join(d, f'{t}.csv'), 'w') as f:
            f.write(body)
    return {t: os.path.join(d, f'{t}.csv') for t in tickers}


def received_today(fp):
    """다운로드 루프가 재사용 여부를 가를 때 쓰는 바로 그 조건."""
    return datetime.fromtimestamp(os.path.getmtime(fp)).date() == datetime.now().date()


for mod, label in ((rc, 'recommend'), (rp, 'recommend_personal')):
    print(f"\n===== {label}")
    need = stock_universe(mod)
    print(f"[설정] 주식 입력 {len(need)}종 {need}")

    print("\n[1] 멈춘 회차는 주식 캐시를 '오늘 받은 것' 에서 뺀다")
    tmp = tempfile.mkdtemp()
    paths = make_cache(tmp, need)
    old_dir = mod.DATA_DIR
    try:
        mod.DATA_DIR = tmp
        check("먼저 전부 오늘 받은 것으로 보인다",
              all(received_today(p) for p in paths.values()))
        done = mod.expire_stock_cache('주식 입력 기준일이 종목마다 다르다')
        check("되돌린 종목 수가 입력 종목 수와 같다", sorted(done) == need, str(done))
        check("이제 하나도 오늘 받은 것으로 안 보인다",
              not any(received_today(p) for p in paths.values()),
              str([t for t, p in paths.items() if received_today(p)]))

        print("\n[2] 파일을 지우지 않는다 — 다시 받기도 실패하면 낡은 값이라도 써야 한다")
        check("파일이 그대로 있다", all(os.path.exists(p) for p in paths.values()))
        check("내용이 그대로다",
              all(open(p).read() == 'Date,Adj_Close\n2026-10-20,100.0\n'
                  for p in paths.values()))

        print("\n[3] 주식이 아닌 캐시는 안 건드린다")
        other = os.path.join(tmp, 'BTC-USD.csv')
        with open(other, 'w') as f:
            f.write('Date,Adj_Close\n2026-10-22,1.0\n')
        mod.expire_stock_cache('재확인')
        check("코인 캐시는 오늘 받은 것 그대로", received_today(other))

        print("\n[4] 파일이 없는 종목이 있어도 멈추지 않는다")
        os.remove(paths[need[0]])
        try:
            done2 = mod.expire_stock_cache('일부 없음')
            ok = need[0] not in done2 and len(done2) == len(need) - 1
        except Exception as e:  # noqa: BLE001
            ok, done2 = False, repr(e)
        check("없는 것만 빼고 나머지를 처리한다", ok, str(done2))
    finally:
        mod.DATA_DIR = old_dir

    print("\n[5] 멈추는 자리에서 실제로 부른다 — 함수만 있고 안 부르면 아무 일도 안 난다")
    src = open(str(mod.__file__)).read()
    i_exit = src.find('🛑 주식 입력 불량으로 리포트를 중단한다')
    seg = src[i_exit:i_exit + 600] if i_exit >= 0 else ''
    check("중단 메시지가 있다", i_exit >= 0)
    check("그 뒤 sys.exit 전에 expire_stock_cache 를 부른다",
          'expire_stock_cache(' in seg
          and seg.find('expire_stock_cache(') < seg.find('sys.exit(2)'),
          seg[:200])

    print("\n[6] 입력이 온전하면 멈추지도 되돌리지도 않는다 (양성 대조)")
    import pandas as pd
    idx = pd.date_range('2025-01-01', periods=mod.STOCK_CANARY_MA_PERIOD + 60, freq='D')
    good = {t: pd.Series(1.0, index=idx) for t in need}
    good['BTC-USD'] = pd.Series(1.0, index=idx)
    check("require_stock_prices 가 통과시킨다",
          mod.require_stock_prices(good) is None, str(mod.require_stock_prices(good)))

    print("\n[7] 기준일이 하루라도 어긋나면 막는다 (오늘 사고 재현)")
    bad = dict(good)
    bad[need[0]] = pd.Series(1.0, index=idx[:-1])
    why = mod.require_stock_prices(bad)
    check("사유를 돌려준다", why is not None and '기준일' in why, str(why))

print()
if FAILED:
    print(f"❌ {len(FAILED)}건 실패: {FAILED}")
    sys.exit(1)
print("✅ 전부 통과")
