#!/usr/bin/env python3
"""상태·신호가 온전하지 않은 실행이 주문을 한 건도 내지 않는지 프로세스 수준에서 확인한다.

단위시험(test_stock_anchor_schedule.py)은 앵커 시간표까지만 본다. 여기서는 run_once 를
가짜 KIS 로 실제로 태워서 place_order 호출 횟수를 센다. 파괴적 반례에서 0건이어야 한다.
ai-debate run-20260828T051722Z 의 검증계획 항목.

파괴적 반례:
  · 스냅샷 하나가 비중을 잃음 → 합산 목표가 3분의 2로 축소돼 대량 매도가 나올 자리
  · 종목만 있고 비중이 빈 신호 / NaN / 음수 / 합이 안 맞는 비중
  · 상태파일 JSON 손상 → 콜드스타트로 오인하면 실재 보유를 무시하고 매매
양성 대조: 온전한 상태에서 앵커일이 오면 주문이 실제로 나간다(하니스가 주문을 볼 수 있음).
"""
import datetime as _dt
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade'))
import executor_stock as ex  # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ''))
    if not cond:
        FAILED.append(name)


class _FrozenDatetime(_dt.datetime):
    _today = _dt.date(2026, 8, 28)

    @classmethod
    def now(cls, tz=None):
        return _dt.datetime.combine(cls._today, _dt.time(23, 35))


class FakeAPI:
    """주문은 전부 place_order 한 곳을 지난다. 그 호출만 세면 된다."""

    def __init__(self, *args, **kwargs):
        self.orders = []
        # 체결된 매도는 잔고에서 빠진다. 집행기가 매도 뒤 잔고를 다시 읽어 "판 만큼
        # 줄었나" 를 보므로, 잔고가 고정이면 멀쩡한 매도도 안 비친 것으로 읽힌다.
        self.holdings = {'VNQ': 10, 'EEM': 10, 'SPY': 10}
        FakeAPI.last = self

    def cancel_all_pending(self):
        # 집행기는 결과 dict 를 보고 "미체결 없음" 과 "조회 실패" 를 가른다.
        return {'queried': True, 'found': 0, 'failed': 0}

    def get_balance(self):
        # VNQ/EEM/SPY 각 <금액> + 현금 7% = 목표(버퍼 반영)와 정확히 일치.
        # 이래야 드리프트가 저절로 발화하지 않아 시험하려는 경로만 남는다.
        # 총자산은 매도해도 안 바뀐다 — 주식이 현금으로 바뀔 뿐이다.
        return dict(self.holdings), 30000.0 / (1 - ex.CASH_BUFFER_DEFAULT), 1380.0

    def get_current_price(self, ticker, fresh_only=False):
        # fresh_only 는 완료 평가가 쓰는 칸이다. 대역이 그 칸을 안 받으면 집행기가
        # TypeError 로 읽어 평가를 통째로 막아 버려서, 이 파일이 보려는 경로가 아니라
        # "값을 못 매겼다" 가 결과를 정한다.
        return 1000.0

    def get_max_buy_qty(self, ticker, price):
        return 999

    def place_order(self, ticker, qty, price, side='buy', pre_send=None):
        # 집행기 계약: 'ok' / 'rejected' / 'unknown' / 'stale'. 참거짓이 아니다 — 접수
        # 여부를 모르는 것과 확정 거절을 가르지 않으면 중복 전송이 섞인다.
        # pre_send 는 전송 직전 계좌값 나이 검사다. 진짜 구현과 같은 자리에서 묻는다.
        if pre_send is not None and not pre_send():
            return 'stale'
        self.orders.append((side, ticker, qty))
        if side == 'sell':
            self.holdings[ticker] = max(0, self.holdings.get(ticker, 0) - qty)
        else:
            self.holdings[ticker] = self.holdings.get(ticker, 0) + qty
        return 'ok'


def make_signal(picks=('QQQ', 'SPY', 'EEM'), weights=None, risk_on=True):
    w = weights if weights is not None else {t: 1.0 / 3.0 for t in picks}
    stamp = _FrozenDatetime.now().strftime('%Y-%m-%d %H:%M')
    return {'stock': {'risk_on': risk_on,
                      'offense_picks': list(picks), 'offense_weights': dict(w),
                      'defense_picks': ['IEF', 'BIL'], 'defense_weights': {'IEF': .5, 'BIL': .5}},
            'meta': {'strategy_version': 'V25', 'updated_at': stamp, 'canary': {}}}


def run(state_obj, signal, today=_dt.date(2026, 8, 28), raw_state=None):
    """run_once 를 한 번 태우고 (주문목록, 알림목록, 저장된 상태) 를 돌려준다."""
    _FrozenDatetime._today = today
    # 신선도 도장은 실행일 기준으로 다시 찍는다(인자 평가 시점의 날짜가 아니라).
    signal.setdefault('meta', {})['updated_at'] = _FrozenDatetime.now().strftime('%Y-%m-%d %H:%M')
    alerts, saved = [], {}
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'kis_trade_state.json')
    if raw_state is not None:
        with open(path, 'w') as f:
            f.write(raw_state)
    elif state_obj is not None:
        with open(path, 'w') as f:
            json.dump(state_obj, f)

    keys = ('datetime', 'KISAPI', '_compute_signal_v25', 'send_telegram', 'save_json',
            'TRADE_STATE_FILE', 'time', '_read_alloc_transit_cap_ratio')
    old = {k: getattr(ex, k) for k in keys}
    try:
        ex.datetime = _FrozenDatetime
        ex.KISAPI = FakeAPI
        ex._compute_signal_v25 = lambda st: signal
        ex.send_telegram = lambda m: alerts.append(m)
        ex.save_json = lambda p, d, default=None: saved.update({'state': d})
        ex.TRADE_STATE_FILE = path
        # monotonic 은 계좌값 나이 측정용. 여기선 늙지 않게 고정해 둔다 — 나이 판정 자체는
        # tests/test_stock_balance_guard.py 의 [26] 이 가짜 시계로 본다.
        ex.time = types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0.0,
                                        monotonic=lambda: 0.0)
        ex._read_alloc_transit_cap_ratio = lambda sleeve: None
        FakeAPI.last = None
        ex.run_once(dry_run=False)
    finally:
        for k, v in old.items():
            setattr(ex, k, v)
    api = getattr(FakeAPI, 'last', None)
    return (api.orders if api else []), alerts, saved.get('state')


G, P = ex.SNAP_STAGGER_DAYS, ex.SNAP_PERIOD_DAYS
BASE = _dt.date(2026, 9, 15)


def healthy(base=BASE, weights=None):
    w = weights if weights is not None else {'VNQ': 1 / 3, 'EEM': 1 / 3, 'SPY': 1 / 3}
    return {'prev_risk_on': True,
            'snapshots': {str(i): {'picks': ['VNQ', 'EEM', 'SPY'], 'weights': dict(w),
                                   'last_rebal_date': (base - _dt.timedelta(days=P)).isoformat(),
                                   'next_rebal_date': (base + _dt.timedelta(days=i * G)).isoformat()}
                          for i in range(ex.N_SNAPS)}}


print("\n[1] 양성 대조 — 온전한 상태에서 앵커일이 오면 주문이 나간다")
orders, alerts, _ = run(healthy(), make_signal(), today=BASE)
check("주문이 실제로 나간다(하니스가 주문을 본다)", len(orders) > 0, str(orders))

print("\n[2] 스냅샷 하나가 비중을 잃음 — 주문 0건")
st = healthy()
st['snapshots']['1']['weights'] = {}
orders, alerts, saved = run(st, make_signal(), today=BASE)
check("주문 0건", orders == [], str(orders))
check("상태 이상을 알린다", any('상태 이상' in a for a in alerts), str(alerts))

print("\n[3] 스냅샷 둘이 비중을 잃음 — 주문 0건")
st = healthy()
st['snapshots']['1']['weights'] = {}
st['snapshots']['2'].pop('weights')
orders, _, _ = run(st, make_signal(), today=BASE)
check("주문 0건", orders == [], str(orders))

print("\n[4] 온전치 않은 비중은 앵커일에도 스냅샷에 안 들어간다 — 보유 유지")
for label, w in (('빈 비중', {}),
                 ('NaN', {'QQQ': float('nan'), 'SPY': 0.5}),
                 ('음수', {'QQQ': -0.5, 'SPY': 1.5}),
                 ('합 0.6', {'QQQ': 0.3, 'SPY': 0.3}),
                 ('합 1.5', {'QQQ': 0.75, 'SPY': 0.75}),
                 ('문자열', {'QQQ': 'x', 'SPY': 1.0})):
    orders, _, saved = run(healthy(), make_signal(weights=w), today=BASE)
    held = {t for t in ('VNQ', 'EEM', 'SPY')}
    liq = [o for o in orders if o[0] == 'sell' and o[1] in held and o[2] >= 10]
    check(f"전량 매도 없음 ({label})", liq == [], str(orders))
    check(f"스냅샷 비중 보존 ({label})",
          all((saved or {}).get('snapshots', {}).get(str(i), {}).get('weights')
              == {'VNQ': 1 / 3, 'EEM': 1 / 3, 'SPY': 1 / 3} for i in range(ex.N_SNAPS)),
          str((saved or {}).get('snapshots')))

print("\n[5] 드리프트 발화일에도 온전치 않은 비중은 리필되지 않는다")
st = healthy()
sig_bad = make_signal(weights={'QQQ': 0.3})
_swapped = ex.refill_snaps_fresh(sig_bad, st)
check("리필 거부", _swapped is False)
check("스냅샷 비중 보존",
      st['snapshots']['0']['weights'] == {'VNQ': 1 / 3, 'EEM': 1 / 3, 'SPY': 1 / 3},
      str(st['snapshots']['0']))

print("\n[6] 상태파일 JSON 손상 — 주문 0건, 콜드스타트로 오인하지 않는다")
orders, alerts, saved = run(None, make_signal(), today=BASE, raw_state='{"snapshots": {"0": ')
check("주문 0건", orders == [], str(orders))
check("손상을 알린다", any('손상' in a for a in alerts), str(alerts))
check("손상된 상태를 덮어쓰지 않는다", saved is None, str(saved))

print("\n[7] 상태파일 최상위가 dict 가 아님 — 주문 0건")
orders, alerts, _ = run(None, make_signal(), today=BASE, raw_state='[1, 2, 3]')
check("주문 0건", orders == [], str(orders))
check("손상을 알린다", any('손상' in a for a in alerts), str(alerts))

print("\n[8] 한 스냅만 69일 배수만큼 어긋난 시간표 — 잡아내 복원, 주문 0건")
st = healthy()
st['snapshots']['1']['next_rebal_date'] = (BASE + _dt.timedelta(days=G + P)).isoformat()
orders, alerts, saved = run(st, make_signal(), today=_dt.date(2026, 8, 28))
sched = [saved['snapshots'][str(i)]['next_rebal_date'] for i in range(ex.N_SNAPS)]
srt = sorted(_dt.date.fromisoformat(x) for x in sched)
check("주문 0건", orders == [], str(orders))
check("간격 23일로 복원", [(srt[i + 1] - srt[i]).days for i in range(2)] == [G, G], str(sched))
check("멀쩡한 둘(snap0·snap2)을 기준 삼는다", sched[0] == BASE.isoformat(), str(sched))

print("\n[9] 실제 서버 상태(2026-08-28) — 배포 당일 주문 0건")
real = {'prev_risk_on': True,
        'snapshots': {str(i): {'picks': ['EEM', 'VNQ', 'SPY'],
                               'weights': {'EEM': 1 / 3, 'VNQ': 1 / 3, 'SPY': 1 / 3},
                               'last_rebal_date': '2026-07-08'} for i in range(ex.N_SNAPS)}}
orders, alerts, saved = run(real, make_signal(), today=_dt.date(2026, 8, 28))
sched = [saved['snapshots'][str(i)]['next_rebal_date'] for i in range(ex.N_SNAPS)]
check("주문 0건", orders == [], str(orders))
check("시간표 09-15 / 10-08 / 10-31",
      sched == ['2026-09-15', '2026-10-08', '2026-10-31'], str(sched))
check("보유 종목 그대로", saved['snapshots']['0']['picks'] == ['EEM', 'VNQ', 'SPY'])

print("\n[10] 상태 전체 유실 — 셋을 다 채우고 목표 합이 1")
orders, alerts, saved = run({'prev_risk_on': True, 'snapshots': {}}, make_signal(),
                            today=_dt.date(2026, 8, 28))
merged = ex.merge_tranches(saved)
check("합산 목표 합이 1", abs(sum(merged.values()) - 1.0) < 1e-9, str(merged))
check("한 스냅만 채워 대량매도하지 않는다",
      all(saved['snapshots'][str(i)].get('weights') for i in range(ex.N_SNAPS)),
      str(saved['snapshots']))

print("\n[11] 스냅샷 키가 통째로 없음 — 주문 0건, 빈 껍데기를 만들지 않는다")
st = healthy()
del st['snapshots']['2']
orders, alerts, saved = run(st, make_signal(), today=BASE)
check("주문 0건", orders == [], str(orders))
check("상태 이상을 알린다", any('상태 이상' in a for a in alerts), str(alerts))
check("없는 snap 을 빈 껍데기로 만들지 않는다",
      '2' not in (saved or st)['snapshots'] or (saved or st)['snapshots']['2'].get('weights'),
      str((saved or st)['snapshots'].get('2')))

print("\n[12] 카나리 전환일에 신호 비중이 이상하면 전환을 소비하지 않는다")
st = healthy()
st['prev_risk_on'] = True
sig = make_signal(risk_on=False)
sig['stock']['defense_weights'] = {'IEF': 0.3}          # 합 0.3 — 온전치 않다
orders, alerts, saved = run(st, sig, today=_dt.date(2026, 8, 28))
check("주문 0건", orders == [], str(orders))
check("prev_risk_on 을 소비하지 않는다", (saved or st).get('prev_risk_on') is True,
      str((saved or st).get('prev_risk_on')))
check("보유(공격)를 그대로 둔다",
      (saved or st)['snapshots']['0']['picks'] == ['VNQ', 'EEM', 'SPY'],
      str((saved or st)['snapshots']['0']))
check("전환 보류를 알린다", any('보류' in a for a in alerts), str(alerts))

print("\n[13] 다음 날 신호가 정상이면 그때 방어로 전환한다")
orders, alerts, saved = run(st, make_signal(risk_on=False), today=_dt.date(2026, 8, 29))
check("prev_risk_on 이 넘어간다", (saved or {}).get('prev_risk_on') is False,
      str((saved or {}).get('prev_risk_on')))
check("방어 종목으로 교체", (saved or {})['snapshots']['0']['picks'] == ['IEF', 'BIL'],
      str((saved or {})['snapshots']['0']))

print("\n[14] 상태 모양이 틀림 — 예외 없이 주문 0건")
for label, raw in (('snapshots 가 list', '{"snapshots": [1, 2, 3]}'),
                   ('snapshots 가 문자열', '{"snapshots": "x"}'),
                   ('snap 이 list', '{"snapshots": {"0": [], "1": {}, "2": {}}}'),
                   ('snap 이 null', '{"snapshots": {"0": null, "1": {}, "2": {}}}')):
    orders, alerts, saved = run(None, make_signal(), today=BASE, raw_state=raw)
    check(f"주문 0건 ({label})", orders == [], str(orders))
    check(f"구조 이상을 알린다 ({label})",
          any('구조 이상' in a or '상태 이상' in a for a in alerts), str(alerts))

print("\n[15] 스냅샷 키가 남아돎 — merge 분모가 늘어 대량매도할 자리, 주문 0건")
st = healthy()
st['snapshots']['3'] = dict(st['snapshots']['0'])
orders, alerts, saved = run(st, make_signal(), today=BASE)
check("주문 0건", orders == [], str(orders))
check("구조 이상을 알린다", any('구조 이상' in a for a in alerts), str(alerts))
check("상태를 덮어쓰지 않는다", saved is None, str(saved))

print("\n[16] 종목과 비중이 어긋남 — 주문 0건")
st = healthy()
st['snapshots']['1']['picks'] = ['VNQ']          # 비중엔 EEM·SPY 도 있다
orders, alerts, _ = run(st, make_signal(), today=BASE)
check("주문 0건", orders == [], str(orders))
check("상태 이상을 알린다", any('어긋난다' in a for a in alerts), str(alerts))

print("\n[17] 결함 상태여도 카나리 방어 전환은 막지 않는다 — 방어 주문이 나간다")
st = healthy()
del st['snapshots']['2']                          # 결함: snap2 통째로 없음
st['prev_risk_on'] = True
orders, alerts, saved = run(st, make_signal(risk_on=False), today=_dt.date(2026, 8, 28))
check("방어 주문이 나간다", len(orders) > 0, str(orders))
check("위험자산을 판다", any(o[0] == 'sell' for o in orders), str(orders))
check("방어자산을 산다", any(o[0] == 'buy' and o[1] in ('IEF', 'BIL') for o in orders),
      str(orders))
check("전환을 확정한다", (saved or {}).get('prev_risk_on') is False,
      str((saved or {}).get('prev_risk_on')))

print("\n[18] 공격 전환은 세 스냅을 통째로 다시 쓴다 — 쪼그라든 목표가 남지 않는다")
st = healthy()
del st['snapshots']['2']
st['prev_risk_on'] = False
orders, alerts, saved = run(st, make_signal(risk_on=True), today=_dt.date(2026, 8, 28))
check("셋 다 채워진다", all((saved or {})['snapshots'][str(i)].get('weights')
                        for i in range(ex.N_SNAPS)), str((saved or {}).get('snapshots')))
merged = ex.merge_tranches(saved or {})
check("합산 목표 합이 1", abs(sum(merged.values()) - 1.0) < 1e-9, str(merged))
check("현금으로 새는 목표가 없다", 'Cash' not in merged or merged['Cash'] < 1e-9, str(merged))
check("전환을 확정한다", (saved or {}).get('prev_risk_on') is True,
      str((saved or {}).get('prev_risk_on')))
# 전환이 상태를 온전하게 되돌려 놓으므로, 결함 가드는 전환 뒤에는 걸릴 게 없다.
check("전환 뒤엔 결함이 남지 않는다", ex.snapshot_payload_defect(saved or {}) is None,
      str(ex.snapshot_payload_defect(saved or {})))

print("\n[19] 신호의 종목과 비중이 어긋남 — 주문 0건, 스냅샷 보존")
KEEP = {'VNQ': 1 / 3, 'EEM': 1 / 3, 'SPY': 1 / 3}
for label, sig in (
        ('앵커일 공격신호',
         make_signal(picks=('QQQ',), weights={'SPY': 1.0})),
        ('전량현금인데 종목이 남음',
         make_signal(picks=(), weights={'SPY': 0.5, 'Cash': 0.5})),
        ('picks 가 list 가 아님',
         make_signal(picks=(), weights={'QQQ': 1.0}))):
    sig['stock']['offense_picks'] = (list(sig['stock']['offense_picks'])
                                     if label != 'picks 가 list 가 아님' else 'QQQ')
    orders, alerts, saved = run(healthy(), sig, today=BASE)
    check(f"주문 0건 ({label})", orders == [], str(orders))
    check(f"스냅샷 보존 ({label})",
          all((saved or {}).get('snapshots', {}).get(str(i), {}).get('weights') == KEEP
              for i in range(ex.N_SNAPS)), str((saved or {}).get('snapshots')))

print("\n[20] 방어 신호가 어긋나면 카나리 전환도 하지 않는다")
st = healthy()
st['prev_risk_on'] = True
sig = make_signal(risk_on=False)
sig['stock']['defense_picks'] = ['IEF']
sig['stock']['defense_weights'] = {'BIL': 1.0}     # 종목과 비중이 다르다
orders, alerts, saved = run(st, sig, today=_dt.date(2026, 8, 28))
check("주문 0건", orders == [], str(orders))
check("prev_risk_on 을 소비하지 않는다", (saved or st).get('prev_risk_on') is True,
      str((saved or st).get('prev_risk_on')))
check("전환 보류를 알린다", any('보류' in a for a in alerts), str(alerts))

print("\n[21] 드리프트 리필도 어긋난 신호를 받지 않는다")
st = healthy()
bad = make_signal(picks=('QQQ',), weights={'SPY': 1.0})
check("리필 거부", ex.refill_snaps_fresh(bad, st) is False)
check("스냅샷 보존", st['snapshots']['0']['weights'] == KEEP, str(st['snapshots']['0']))

print("\n[21.5] picks 원소가 해시 불가능해도 죽지 않는다")
check("중첩 리스트 거부", ex._picks_weights_ok([['QQQ']], {'QQQ': 1.0}) is False)
check("숫자 티커 거부", ex._picks_weights_ok([1, 2], {'QQQ': 1.0}) is False)
orders, alerts, _ = run(healthy(), make_signal(picks=(), weights={'QQQ': 1.0}), today=BASE)
check("어긋난 신호는 앵커 보류를 알린다", any('보류' in a for a in alerts), str(alerts))

print("\n[22] 양성 대조 — 전량 현금 신호와 정상 신호는 통과한다")
check("전량 현금 통과", ex._picks_weights_ok([], {'Cash': 1.0}))
check("현금 섞인 정상 통과", ex._picks_weights_ok(['QQQ'], {'QQQ': 2 / 3, 'Cash': 1 / 3}))
check("정상 3종 통과", ex._picks_weights_ok(['A', 'B', 'C'],
                                        {'A': 1 / 3, 'B': 1 / 3, 'C': 1 / 3}))
orders, _, _ = run(healthy(), make_signal(), today=BASE)
check("앵커일 정상 신호는 주문이 나간다", len(orders) > 0, str(orders))

print()
if FAILED:
    print(f"{len(FAILED)} FAIL: {FAILED}")
    sys.exit(1)
print("ALL PASS")
