#!/usr/bin/env python3
"""가격 캐시 쓰기·읽기 가드 검증 (2026-10-12).

사고: 서버 크론 09:15 에 recommend 일반과 개인이 각자의 flock 으로 동시에 떠서 같은
data/<티커>.csv 를 대상 경로에 직접 썼다. 두 쓰기가 겹쳐 QQQ.csv 꼬리에
"026-10-09,751.27001953125" 가 한 글자 밀린 채 붙었다. 그러자 load_price 가 날짜를
문자열로 파싱하고 최신성 계산에서 TypeError 를 냈는데, 함수 끝의 bare except 가 그걸
삼키고 빈 시리즈를 돌려줬다. QQQ 가 로그 한 줄 없이 유니버스에서 빠져 그날 주식
공격 목표가 PDBC·QQQ·SPY 셋이 아니라 PDBC·SPY 둘로 나왔다.

설계는 토론 run-20261012T010744Z 의 조건부 채택안을 따른다. 깨진 줄만 버리고 진행하는
안은 기각됐다 — 날짜가 멀쩡한 채 값만 잘린 줄을 못 거르기 때문이다. 대신 깨진 파일은
통째로 안 쓰고, 오늘 받은 파일이어도 내용 검사를 통과해야 재사용하며(아니면 재수집),
주식 입력이 하나라도 빠지면 그 실행을 멈춘다.

확인하는 것 열하나.
  T1 깨진 꼬리 줄이 있으면 그 파일을 통째로 안 쓰고 사유가 남는다.
  T2 깨짐의 종류별로 사유가 갈린다 (열 없음·중복 날짜·미래 날짜·값 깨짐·빈 파일).
  T3 멀쩡한 파일은 그대로 읽히고 최신성은 정렬 후 최대 날짜로 본다.
  T4 진짜로 낡은 캐시는 제외되고 사유가 남는다.
  T5 파일이 없으면 사유를 남긴다 (조용한 실패 금지).
  T6 원자 저장은 임시파일을 안 남기고 같은 프로세스 안에서도 이름이 겹치지 않는다.
  T7 네 프로세스가 같은 파일에 동시에 저장해도 결과가 항상 완전한 파일이다.
  T8 주식 입력 검사는 입력 집합으로 판정한다 — 뽑힌 종목 수로 판정하지 않는다.
  T9 두 리포트 스크립트의 공용 함수가 같은 본문이다 (한쪽만 고치는 drift 방지).
  T10 대상 경로 직접 쓰기·사유 없는 except·고정 임시이름이 남아 있지 않다.
  T11 오늘 받은 캐시도 내용 검사를 거친 뒤에만 재사용한다 (깨진 파일이 굳지 않게).

네트워크를 타지 않는다. 파일은 전부 임시 폴더에 만든다.
"""
import ast
import contextlib
import io
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
GENERAL = os.path.join(ROOT, 'strategies', 'cap_defend', 'recommend.py')
PERSONAL = os.path.join(ROOT, 'strategies', 'cap_defend', 'recommend_personal.py')

SHARED = ['_atomic_tmp', '_save_csv_atomic', '_save_json_atomic',
          '_cache_problem', 'load_price', 'require_stock_prices']

OFFENSE = ['SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ']
DEFENSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY = ['EEM']

FAILED = []


def check(name, cond, detail=''):
    mark = 'OK ' if cond else 'FAIL'
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


def grab(path, names):
    """모듈 전체를 import 하지 않고 함수 본문만 떼어 온다.

    recommend*.py 는 맨 위에서 config(비밀값)·pyupbit 를 import 한다. 시험이 그걸
    끌고 오면 환경에 따라 실패하므로 함수 본문만 AST 로 떼어 쓴다.
    """
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    want = {n: None for n in names}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            want[node.name] = ast.get_source_segment(src, node)
    missing = [n for n, v in want.items() if v is None]
    if missing:
        raise SystemExit(f"{path} 에서 {missing} 를 못 찾았다")
    return want, src


def load_funcs(path, data_dir):
    want, src = grab(path, SHARED)
    ns = {'pd': pd, 'np': np, 'os': os, 'time': time, 'datetime': datetime,
          'DATA_DIR': data_dir, 'OFFENSIVE_STOCK_UNIVERSE': OFFENSE,
          'DEFENSIVE_STOCK_UNIVERSE': DEFENSE, 'CANARY_ASSETS': CANARY,
          'STOCK_CANARY_MA_PERIOD': 200}
    for name in SHARED:
        exec(compile(want[name], path, 'exec'), ns)
    return ns, src


def write_csv(path, rows, header='Date,Adj_Close'):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for d, v in rows:
            f.write(f'{d},{v}\n')


def clean_rows(n=40, end=None):
    end = end or datetime.now().date() - timedelta(days=1)
    return [((end - timedelta(days=n - 1 - i)).isoformat(), 100.0 + i) for i in range(n)]


def call(fn, *a, **kw):
    """표준출력을 가로채 결과와 같이 돌려준다."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


# --- T7 용 자식 프로세스 (모듈 최상위에 둬야 spawn 에서 집힌다) ---
def _writer(args):
    path, nrows, src_path = args
    ns, _ = load_funcs(src_path, os.path.dirname(path))
    df = pd.DataFrame(clean_rows(nrows), columns=['Date', 'Adj_Close'])
    for _ in range(25):
        ns['_save_csv_atomic'](df, path)
    return nrows


def main():
    tmpdir = tempfile.mkdtemp(prefix='pricecache-')
    try:
        ns, gsrc = load_funcs(GENERAL, tmpdir)
        load_price = ns['load_price']
        problem = ns['_cache_problem']
        save = ns['_save_csv_atomic']
        require = ns['require_stock_prices']

        rows = clean_rows()
        fp = os.path.join(tmpdir, 'QQQ.csv')

        print("[1] 깨진 꼬리 줄 — 그 파일은 통째로 안 쓴다")
        last_d, last_v = rows[-1]
        write_csv(fp, rows + [(last_d[1:], last_v)])  # "2026-10-09" → "026-10-09"
        s, out = call(load_price, 'QQQ')
        check("T1 빈 시리즈다 (깨진 줄만 버리고 진행하지 않는다)", len(s) == 0, f"길이 {len(s)}")
        check("T1 사유가 화면에 남는다", '못 쓴다' in out and 'QQQ' in out, out.strip())
        check("T1 사유가 깨진 줄을 가리킨다", '날짜가 깨진 줄 1개' in (problem(fp) or ''),
              str(problem(fp)))

        print("\n[2] 깨짐의 종류별 사유")
        cases = []
        write_csv(fp, rows, header='Date,Close')
        cases.append(('열 없음', problem(fp), 'Adj_Close 열이 없다'))
        write_csv(fp, rows + [rows[-1]])
        cases.append(('중복 날짜', problem(fp), '날짜가 겹치는 줄'))
        future = (datetime.now().date() + timedelta(days=3)).isoformat()
        write_csv(fp, rows + [(future, 101.0)])
        cases.append(('미래 날짜', problem(fp), '미래 날짜'))
        write_csv(fp, rows[:-1] + [(rows[-1][0], -1.0)])
        cases.append(('음수 값', problem(fp), '가격이 깨진 줄'))
        write_csv(fp, rows[:-1] + [(rows[-1][0], '')])
        cases.append(('빈 값', problem(fp), '가격이 깨진 줄'))
        write_csv(fp, [])
        cases.append(('줄 없음', problem(fp), '줄이 없다'))
        for label, got, want in cases:
            check(f"T2 {label}", got is not None and want in got, f"받은 사유 {got!r}")

        print("\n[3] 멀쩡한 파일")
        write_csv(fp, rows)
        s, out = call(load_price, 'QQQ')
        check("T3 그대로 읽힌다", len(s) == len(rows) and float(s.iloc[-1]) == rows[-1][1])
        check("T3 사유를 안 찍는다", out.strip() == '', out.strip())
        shuffled = [rows[-1]] + rows[:-1]
        write_csv(fp, shuffled)
        s, out = call(load_price, 'QQQ')
        check("T3 줄 순서가 섞여도 정렬 후 최대 날짜로 본다",
              len(s) == len(rows) and str(s.index.max().date()) == rows[-1][0])

        print("\n[4] 진짜 낡은 캐시 / 없는 파일")
        write_csv(fp, [((datetime.now().date() - timedelta(days=400 - i)).isoformat(),
                        100.0 + i) for i in range(30)])
        s, out = call(load_price, 'QQQ')
        check("T4 제외된다", len(s) == 0)
        check("T4 사유가 남는다", '낡았다' in out, out.strip())
        s, out = call(load_price, 'NOPE')
        check("T5 빈 시리즈 + 사유", len(s) == 0 and 'NOPE' in out and '파일이 없다' in out,
              out.strip())

        print("\n[5] 원자 저장")
        df = pd.DataFrame(rows, columns=['Date', 'Adj_Close'])
        apath = os.path.join(tmpdir, 'ATOM.csv')
        save(df, apath)
        leftovers = [f for f in os.listdir(tmpdir) if '.tmp' in f]
        check("T6 임시파일이 안 남는다", not leftovers, str(leftovers))
        check("T6 내용이 온전하다", len(pd.read_csv(apath)) == len(rows))
        names = {ns['_atomic_tmp'](apath) for _ in range(50)}
        check("T6 같은 프로세스 안에서도 임시이름이 안 겹친다", len(names) == 50, str(len(names)))

        print("\n[6] 동시 쓰기 — 네 프로세스가 같은 파일에")
        cpath = os.path.join(tmpdir, 'RACE.csv')
        sizes = [30, 31, 32, 33]
        ctx = mp.get_context('spawn')
        with ctx.Pool(4) as pool:
            pool.map(_writer, [(cpath, n, GENERAL) for n in sizes])
        check("T7 깨진 데가 없다", problem(cpath) is None, str(problem(cpath)))
        check("T7 길이가 쓴 것 중 하나와 정확히 같다", len(pd.read_csv(cpath)) in sizes,
              f"길이 {len(pd.read_csv(cpath))}")

        print("\n[7] 주식 입력 검사 — 집합으로 판정하고 개수로 판정하지 않는다")
        need = sorted(set(OFFENSE + DEFENSE + CANARY))
        idx = pd.to_datetime([d for d, _ in rows])
        full = {t: pd.Series(range(1, 301), index=pd.date_range(end=idx[-1], periods=300))
                for t in need}
        check("T8 다 있으면 통과", require(full) is None, str(require(full)))
        one_out = dict(full, QQQ=pd.Series(dtype=float))
        got = require(one_out)
        check("T8 하나 비면 그 티커를 가리킨다", got is not None and 'QQQ' in got, str(got))
        short = dict(full, VNQ=full['VNQ'].iloc[-100:])
        got = require(short)
        check("T8 이력이 모자라면 사유를 낸다", got is not None and '이력이 모자라다' in got,
              str(got))
        skew = dict(full)
        skew['BIL'] = pd.Series(full['BIL'].values,
                                index=full['BIL'].index - pd.Timedelta(days=1))
        got = require(skew)
        check("T8 기준일이 갈리면 사유를 낸다", got is not None and '기준일이 종목마다 다르다' in got,
              str(got))
        check("T8 뽑힌 종목 수로 판정하지 않는다",
              'len(prices)' not in (grab(GENERAL, ['require_stock_prices'])[0]
                                    ['require_stock_prices']))

        print("\n[8] 두 스크립트 정합·회귀")
        gwant, _ = grab(GENERAL, SHARED)
        pwant, psrc = grab(PERSONAL, SHARED)
        for n in SHARED:
            check(f"T9 {n} 본문이 같다", gwant[n] == pwant[n])
        for label, src in (('일반', gsrc), ('개인', psrc)):
            check(f"T10 {label}: 대상 경로 직접 쓰기가 없다",
                  'to_csv(fp' not in src and 'json.dump(cg_data, f)' not in src)
            dl = next(ast.get_source_segment(src, n) for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'download_required_data')
            check(f"T10 {label}: 가격 수집 경로에 사유 없는 except 가 없다",
                  'except:' not in dl and 'except: return pd.Series' not in src)
            check(f"T10 {label}: 고정 임시이름이 없다",
                  "+ '.tmp'" not in src and '".tmp"' not in src)
            check(f"T10 {label}: 주식 입력 검사를 실제로 부른다",
                  'require_stock_prices(prices)' in src)
            check(f"T11 {label}: 오늘 캐시도 내용 검사를 거친다",
                  '_cache_problem(fp)' in src.split('mtime == today_str')[1][:400])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAIL: {FAILED}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
