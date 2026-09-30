#!/usr/bin/env python3
"""현물 시총 목록 나이 제한 검증 (2026-09-30).

확인하는 것 넷.
  1. 응답 검증 — 형태 이상·행 부족·심볼 빈칸·급격한 축소를 캐시에 넣지 않는다.
  2. 나이 판정 — 신선/만료/시각 누락/손상/미래를 저장된 실제 성공 시각으로만 가른다.
  3. 막힌 결과 — 카나리 ON 이면 매매 차단(현금 목표를 내지 않는다), OFF 면 청산만 허용.
  4. 유니버스 캐시 재사용도 원본 시총 조회 시각으로 나이를 잰다.
  5. 만료 경보가 실행기의 카나리 플립 필터에 걸려 텔레그램에서 사라지지 않는다.

네트워크를 타지 않는다. 세션·봉 조회는 전부 가짜로 끼운다.
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coin_live_engine as cle  # noqa: E402

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def rows(n, prefix='C'):
    return [{'symbol': f'{prefix}{i}', 'market_cap': 10_000 - i} for i in range(n)]


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    """get() 호출마다 미리 넣어둔 응답을 하나씩 돌려준다."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        if not self._responses:
            raise RuntimeError('네트워크 차단')
        return self._responses.pop(0)


def write_cache(path, ts, data):
    with open(path, 'w') as f:
        json.dump({'ts': ts, 'data': data}, f)


def main():
    cle.time.sleep = lambda *_a, **_k: None  # 재시도 대기 제거
    now = datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc)

    # --- 1) 응답 검증 ---
    ok, why = cle.validate_cg_rows(rows(40))
    check(ok, f'T1: 정상 40행이 거부됨({why})')
    check(not cle.validate_cg_rows({'a': 1})[0], 'T1: 목록 아닌 응답이 통과')
    check(not cle.validate_cg_rows(rows(5))[0], 'T1: 행 5개가 통과')
    bad = rows(40)
    bad[7]['symbol'] = ''
    check(not cle.validate_cg_rows(bad)[0], 'T1: 심볼 빈 행이 통과')
    dup = [{'symbol': 'BTC', 'market_cap': 1} for _ in range(40)]
    check(not cle.validate_cg_rows(dup)[0], 'T1: 중복 심볼 40행이 통과')
    check(not cle.validate_cg_rows(rows(22), prev_count=40)[0],
          'T1: 40 → 22 급격한 축소가 통과')
    check(cle.validate_cg_rows(rows(30), prev_count=40)[0],
          'T1: 40 → 30 정상 변동이 거부됨')

    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, 'cg.json')

        # --- 2) 나이 판정 ---
        # 신선한 조회 성공 → 캐시 기록 + source=live
        sess = FakeSession([FakeResp(200, rows(40))])
        data, meta = cle.fetch_coingecko_top(sess, cache_path=cache, now_utc=now)
        check(len(data) == 40 and meta['source'] == 'live', 'T2: 정상 조회 결과 이상')
        check(os.path.exists(cache), 'T2: 캐시가 안 써짐')

        # 조회 실패 + 24시간 캐시 → 통과
        write_cache(cache, cle.to_utc_iso(now - timedelta(hours=24)), rows(40))
        sess = FakeSession([FakeResp(500) for _ in range(5)])
        data, meta = cle.fetch_coingecko_top(sess, cache_path=cache, retries=2, now_utc=now)
        check(meta['source'] == 'cache' and 23.9 < meta['age_h'] < 24.1,
              f'T2: 24시간 캐시 판정 이상({meta})')

        # 조회 실패 + 37시간 캐시 → 만료
        write_cache(cache, cle.to_utc_iso(now - timedelta(hours=37)), rows(40))
        sess = FakeSession([FakeResp(500) for _ in range(5)])
        try:
            cle.fetch_coingecko_top(sess, cache_path=cache, retries=2, now_utc=now)
            fails.append('T2: 37시간 캐시가 통과')
        except cle.MarketMetaUnavailable as e:
            check('37.0시간' in str(e), f'T2: 만료 사유에 나이가 없음({e})')

        # 시각 누락 / 손상 / 미래
        for label, ts in (('누락', ''), ('손상', 'not-a-time'),
                          ('미래', cle.to_utc_iso(now + timedelta(hours=5)))):
            write_cache(cache, ts, rows(40))
            sess = FakeSession([FakeResp(500) for _ in range(5)])
            try:
                cle.fetch_coingecko_top(sess, cache_path=cache, retries=2, now_utc=now)
                fails.append(f'T2: 시각 {label} 캐시가 통과')
            except cle.MarketMetaUnavailable:
                pass

        # 캐시 자체가 없으면 만료로 취급
        os.remove(cache)
        sess = FakeSession([FakeResp(500) for _ in range(5)])
        try:
            cle.fetch_coingecko_top(sess, cache_path=cache, retries=2, now_utc=now)
            fails.append('T2: 캐시 없음인데 통과')
        except cle.MarketMetaUnavailable:
            pass

        # 이상한 응답은 캐시를 덮어쓰지 않는다 — 옛 캐시가 살아남아 나이로 판정된다
        write_cache(cache, cle.to_utc_iso(now - timedelta(hours=10)), rows(40))
        sess = FakeSession([FakeResp(200, rows(3)), FakeResp(200, rows(3))])
        data, meta = cle.fetch_coingecko_top(sess, cache_path=cache, retries=2, now_utc=now)
        check(meta['source'] == 'cache', 'T2: 이상 응답이 live 로 채택됨')
        with open(cache) as f:
            kept = json.load(f)
        check(len(kept['data']) == 40, 'T2: 이상 응답이 캐시를 덮어씀')

    # --- 3) 막힌 결과 ---
    sma_p = cle.MEMBER_D_SMA42['sma_bars']
    n = sma_p + 60

    def btc_bars(last_close, base=100.0):
        idx = pd.date_range(end=cle.expected_last_closed_bar_ts('D', now),
                            periods=n, freq='D', tz='UTC')
        close = np.full(n, base, dtype=float)
        close[-1] = last_close
        return pd.DataFrame({'Open': close, 'High': close, 'Low': close,
                             'Close': close, 'Volume': np.ones(n)}, index=idx)

    orig_klines = cle.fetch_binance_klines
    try:
        # 카나리 OFF (직전 ON 에서 밴드 아래로) → 청산만
        cle.fetch_binance_klines = lambda *a, **k: btc_bars(80.0)
        state = {'members': {'D_SMA42': {'canary_on': True}}}
        res = cle._market_meta_blocked(FakeSession([]), state, now, '시험: 만료')
        check(res.liquidation_only is True, 'T3: 카나리 OFF 인데 청산 허용 안 함')
        check(res.combined_target == {'Cash': 1.0}, f'T3: OFF 목표 이상({res.combined_target})')
        check(res.all_fresh is True and res.any_new_bar is True, 'T3: OFF 인데 실행기가 스킵함')
        check(state == {'members': {'D_SMA42': {'canary_on': True}}}, 'T3: OFF 경로가 상태를 바꿈')

        # 카나리 ON → 매매 차단, 현금 목표를 내지 않는다
        cle.fetch_binance_klines = lambda *a, **k: btc_bars(130.0)
        state = {'members': {'D_SMA42': {'canary_on': True}}}
        res = cle._market_meta_blocked(FakeSession([]), state, now, '시험: 만료')
        check(res.liquidation_only is False, 'T3: 카나리 ON 인데 청산 허용')
        check(res.combined_target == {}, f'T3: ON 인데 목표를 냄({res.combined_target})')
        check(res.all_fresh is False, 'T3: ON 인데 실행기가 매매로 진행')

        # 봉을 못 얻으면 판정 불가 → 차단 (청산도 안 연다)
        cle.fetch_binance_klines = lambda *a, **k: pd.DataFrame()
        res = cle._market_meta_blocked(FakeSession([]), {}, now, '시험: 만료')
        check(res.liquidation_only is False and res.all_fresh is False,
              'T3: 봉 없음인데 청산/진행 허용')

        # 완료봉이 기대 봉과 다르면 판정 불가 → 차단
        stale = btc_bars(80.0)
        stale.index = stale.index - pd.Timedelta(days=3)
        cle.fetch_binance_klines = lambda *a, **k: stale
        res = cle._market_meta_blocked(FakeSession([]),
                                       {'members': {'D_SMA42': {'canary_on': True}}},
                                       now, '시험: 만료')
        check(res.liquidation_only is False and res.all_fresh is False,
              'T3: 낡은 봉으로 청산/진행 허용')
    finally:
        cle.fetch_binance_klines = orig_klines

    # --- 4) 유니버스 캐시 재사용도 원본 시각을 본다 ---
    calls = {'n': 0}

    def boom(*_a, **_k):
        calls['n'] += 1
        raise cle.MarketMetaUnavailable('시험: 원본 만료')

    orig_build = cle.build_dynamic_universe
    orig_canary = cle._independent_canary
    try:
        cle.build_dynamic_universe = boom
        cle._independent_canary = lambda *a, **k: (True, {'reason': '시험'})

        # 유니버스 캐시는 1시간 전(20시간 안)인데 원본 시총은 40시간 전
        state = {'universe_cache': {
            'ts': cle.to_utc_iso(now - timedelta(hours=1)),
            'universe': ['BTC', 'ETH', 'SOL'],
            'meta': {'cg_ts': cle.to_utc_iso(now - timedelta(hours=40))},
        }}
        res = cle.compute_live_targets(state, FakeSession([]), tempfile.gettempdir(),
                                       now_utc=now)
        check(calls['n'] == 1, 'T4: 원본이 40시간인데 갱신을 안 시도함')
        check(res.all_fresh is False and res.combined_target == {},
              'T4: 원본 만료인데 매매로 진행')

        # 원본 시각 자체가 없으면 만료로 본다
        calls['n'] = 0
        state = {'universe_cache': {
            'ts': cle.to_utc_iso(now - timedelta(hours=1)),
            'universe': ['BTC', 'ETH', 'SOL'],
            'meta': {},
        }}
        cle.compute_live_targets(state, FakeSession([]), tempfile.gettempdir(), now_utc=now)
        check(calls['n'] == 1, 'T4: 원본 시각 없음인데 캐시를 그대로 씀')
    finally:
        cle.build_dynamic_universe = orig_build
        cle._independent_canary = orig_canary

    # --- 5) 만료 경보가 실행기의 카나리 플립 필터에 안 걸린다 ---
    import executor_coin as ex  # noqa: E402
    flip_on = 'D_SMA42 카나리 ON 🟢 | BTC <금액> / SMA <금액> (1.111)'
    flip_off = 'D_SMA42 카나리 OFF 🔴 | BTC <금액> / SMA <금액> (0.889)'
    check(ex.is_canary_flip_alert(flip_on), 'T5: 플립 ON 알림이 안 걸러짐')
    check(ex.is_canary_flip_alert(flip_off), 'T5: 플립 OFF 알림이 안 걸러짐')

    orig_klines2 = cle.fetch_binance_klines
    try:
        cle.fetch_binance_klines = lambda *a, **k: btc_bars(80.0)
        liq = cle._market_meta_blocked(FakeSession([]),
                                       {'members': {'D_SMA42': {'canary_on': True}}},
                                       now, '시험: 만료')
        cle.fetch_binance_klines = lambda *a, **k: btc_bars(130.0)
        blk = cle._market_meta_blocked(FakeSession([]),
                                       {'members': {'D_SMA42': {'canary_on': True}}},
                                       now, '시험: 만료')
    finally:
        cle.fetch_binance_klines = orig_klines2
    for label, r in (('청산', liq), ('차단', blk)):
        for a in r.alerts:
            check(not ex.is_canary_flip_alert(a),
                  f'T5: {label} 경보가 플립 필터에 걸려 텔레그램에 안 나감 — {a}')

    print('# 현물 시총 목록 나이 제한 검증')
    print(f'  상한 {cle.MARKET_META_MAX_AGE_H:.0f}시간 · 최소 행 {cle.CG_MIN_ROWS} · '
          f'축소 허용 {cle.CG_SHRINK_RATIO:.0%}')
    print(f'  고정 목록 fallback 제거됨 = {not hasattr(cle, "HARDCODED_FALLBACK")}')
    if fails:
        print('FAIL:')
        for f in fails:
            print('  -', f)
        sys.exit(1)
    print('ALL PASS (T1~T5)')


if __name__ == '__main__':
    main()
