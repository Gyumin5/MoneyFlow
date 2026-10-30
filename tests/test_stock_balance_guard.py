#!/usr/bin/env python3
"""계좌 조회가 비었을 때 주식 집행기가 매매를 멈추는지, 총자산을 빠짐없이 더하는지 본다.

계기 (2026-10-12·10-21 실사고). 집행기는 환율을 못 받으면 현금을 0 으로 읽었다. 그러면
보유 합이 100% 로 보여서 드리프트가 정확히 현금버퍼 크기(7.00pp)로 발화하고, V25 주식은
발화일에 종목까지 새로 고른다. 두 회차에 체결액 약 <금액> (슬리브의 63%) 이 돌았고
그날 계좌에는 현금이 각각 <금액>·<금액> 멀쩡히 있었다.

둘째 결함 (2026-10-23 실측). 총자산을 "결제기준 주식 + 예수금 − 미결제 순액" 으로 만들고
있었다. 미결제 매수 구간에서는 산 주식이 아직 결제기준 평가에 없는데 그 대금은 이미
현금에서 빠져서, 총자산이 미결제 매수대금만큼 작아진다 — 그날 <금액>원, 계좌의 5.9%.
앱의 계좌 총자산은 <금액>원인데 집행기는 <금액>원으로 봤다.

여기서 고정하는 계약:
  · 조회가 비면 BalanceUnavailable 을 올린다. 0 이나 빈 보유로 바꿔 돌려주지 않는다.
  · 총자산 = 투자계좌자산현황(CTRP6548R)의 tot_asst_amt. 주식·원화예수금·외화예수금·
    외화RP 가 다 들어 있는 한 칸이고 앱 화면과 같은 값이다.
  · 환율은 두 조회원 중 하나만 멀쩡해도 쓴다. 한쪽 실패로 다른 쪽을 버리지 않는다.
  · 실제 현금 0, 정상적인 빈 보유는 결측이 아니다.
  · 조회 불가 회차는 주문 0건이고 상태파일을 안 바꾼다.

실행: python3 tests/test_stock_balance_guard.py  (bash run_tests.sh 에 자동 포함)
"""
import datetime as _dt
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade'))
import executor_stock as ex  # noqa: E402

_KISAPI_REAL = ex.KISAPI   # run_cycle 이 ex.KISAPI 를 갈아끼우므로 진짜 클래스를 잡아 둔다

FAILED = []

# 2026-10-23 10:00 KST 실측 (읽기 전용 조회). 숫자를 그대로 쓴다.
FX = 1324.6
TOT_KRW = 402880633.0        # 앱 "계좌 총자산"
STOCK_KRW = 349298529.0      # 결제기준 해외주식 평가
KRW_CASH = 27462767.0        # 원화 예수금(출금가능)
FRCR_CASH = 26119337.0       # 외화 예수금 KRW 환산 (<금액>)
USTL_BUY = 23567892.0        # 미결제 매수대금 (어제 산 541주)
LIVE_BIL, LIVE_PDBC = 140514.40, 140960.82   # live 평가 (USD)


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ''))
    if not cond:
        FAILED.append(name)


# ─── 가짜 KIS 응답 ───

def ok_balance():
    return {'rt_cd': '0', 'output1': [
        {'ovrs_pdno': 'BIL', 'ovrs_cblc_qty': '1534', 'ovrs_stck_evlu_amt': str(LIVE_BIL),
         'now_pric2': '91.60'},
        {'ovrs_pdno': 'PDBC', 'ovrs_cblc_qty': '7041', 'ovrs_stck_evlu_amt': str(LIVE_PDBC),
         'now_pric2': '20.02'},
    ]}


def ok_present(exrt=FX, krw_cash=KRW_CASH, frcr_cash=FRCR_CASH, ustl_buy=USTL_BUY,
               rows=None):
    o2 = rows if rows is not None else [
        {'crcy_cd': 'USD', 'frst_bltn_exrt': str(exrt), 'frcr_dncl_amt_2': '19718.66'},
    ]
    return {'rt_cd': '0', 'output2': o2, 'output3': {
        'tot_dncl_amt': str(krw_cash), 'frcr_evlu_tota': str(frcr_cash),
        'ustl_sll_amt_smtl': '0', 'ustl_buy_amt_smtl': str(ustl_buy),
        'tot_asst_amt': str(TOT_KRW),
    }}


def ok_account(tot=TOT_KRW):
    return {'rt_cd': '0', 'output2': {
        'tot_asst_amt': str(tot), 'ovrs_stck_evlu_amt1': str(STOCK_KRW),
        'tot_dncl_amt': str(KRW_CASH), 'frcr_evlu_tota': str(FRCR_CASH),
    }}


def ok_margin(exrt=FX):  # exrt 는 문자열도 받는다 (망가진 값 재현용)
    return {'rt_cd': '0', 'output': [
        {'natn_name': '미국', 'crcy_cd': 'USD', 'bass_exrt': str(exrt)},
        {'natn_name': '영국', 'crcy_cd': 'GBP', 'bass_exrt': '0.00'},
    ]}


COUNTS = {}   # tr_id → 부른 횟수. 한 회차가 계좌를 몇 번 묻는지 세는 데 쓴다.


def fake_get(table):
    """tr_id → 응답. 표에 없으면 빈 dict(조회 실패)."""
    def _g(path, tr_id, params, retries=3, tr_cont='', cont=None):
        COUNTS[tr_id] = COUNTS.get(tr_id, 0) + 1
        v = table.get(tr_id, {})
        # 표 값이 목록이면 연속조회의 쪽들이다. 몇 번째인지는 부른 횟수가 아니라 보낸
        # 커서로 고른다 — 한 회차가 계좌를 여러 번 묻기 때문에 횟수로 세면 두 번째
        # 조회가 첫 쪽을 못 받는다.
        if isinstance(v, list):
            _nk_in = (params.get('CTX_AREA_NK200') or '').strip()
            if not _nk_in:
                idx = 0
            else:
                idx = next((j + 1 for j, p in enumerate(v)
                            if isinstance(p, dict)
                            and (p.get('ctx_area_nk200') or '').strip() == _nk_in), len(v))
            v = v[idx] if idx < len(v) else {}
        v = v() if callable(v) else v
        if cont is not None:
            # 응답 헤더 tr_cont. 안 적은 고정값은 "마지막 쪽"(D)으로 본다 — 실계좌가
            # 그렇게 온다(2026-10-29 실측: 보유 2종목 계좌에서 헤더 D).
            cont['tr_cont'] = v.get('_hdr_tr_cont', 'D') if isinstance(v, dict) else ''
        return v
    return _g


def with_api(table, fn):
    old_get, old_log = ex._get, ex.log
    try:
        ex._get = fake_get(table)
        ex.log = lambda *a, **k: None
        return fn(ex.KISAPI(dry_run=True))
    finally:
        ex._get, ex.log = old_get, old_log


def with_api_live(table, fn):
    """dry_run=False 인 인스턴스로 부른다 — 모의 경로가 가로채는 함수를 재려면 필요하다."""
    old_get, old_log = ex._get, ex.log
    try:
        ex._get = fake_get(table)
        ex.log = lambda *a, **k: None
        return fn(ex.KISAPI(dry_run=False))
    finally:
        ex._get, ex.log = old_get, old_log


def raises_unavailable(table):
    def _run(api):
        try:
            api.get_balance()
        except ex.BalanceUnavailable as e:
            return str(e)
        return None
    return with_api(table, _run)


FULL = {'TTTS3012R': ok_balance(), 'CTRP6504R': ok_present(),
        'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()}


print("[1] 정상 응답 — 총자산이 앱의 계좌 총자산과 같다 (미결제 매수만큼 깎이지 않는다)")
holdings, total_usd, exrt = with_api(FULL, lambda api: api.get_balance())
check("보유 수량", holdings == {'BIL': 1534, 'PDBC': 7041}, str(holdings))
check("환율 1324.6", abs(exrt - FX) < 1e-9, str(exrt))
check("총자산 = tot_asst_amt / 환율", abs(total_usd - TOT_KRW / FX) < 1e-6,
      f'{total_usd:,.2f} vs {TOT_KRW / FX:,.2f}')
check("미결제 매수를 또 빼지 않는다",
      abs(total_usd - (TOT_KRW - USTL_BUY) / FX) > 17000,
      f'{total_usd:,.0f} vs 옛 식 {(TOT_KRW - USTL_BUY) / FX:,.0f}')
check("옛 식과의 차이가 실측 <금액>원과 같다",
      abs((total_usd - (TOT_KRW - USTL_BUY) / FX) * FX - USTL_BUY) < 1.0,
      f'{(total_usd - (TOT_KRW - USTL_BUY) / FX) * FX:,.0f}')

print("\n[2] 2026-10-21 재현 — 환율 두 곳 다 없으면 멈춘다")
msg = raises_unavailable({'TTTS3012R': ok_balance(),
                          'CTRP6504R': ok_present(rows=[]),
                          'CTRP6548R': ok_account(), 'TTTC2101R': {}})
check("BalanceUnavailable", msg is not None and '환율' in msg, str(msg))

print("\n[3] 체결기준현재잔고가 통째로 비면 멈춘다 (현금 0 으로 읽지 않는다)")
msg = raises_unavailable({'TTTS3012R': ok_balance(), 'CTRP6504R': {},
                          'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()})
check("BalanceUnavailable", msg is not None, str(msg))

print("\n[4] 환율은 멀쩡한데 예수금 칸이 없으면 멈춘다")
bad = ok_present()
del bad['output3']['tot_dncl_amt']
msg = raises_unavailable({'TTTS3012R': ok_balance(), 'CTRP6504R': bad,
                          'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()})
check("BalanceUnavailable", msg is not None and '예수금' in msg, str(msg))

print("\n[5] 해외주식 잔고 조회가 실패하면 멈춘다 (빈 보유로 바꾸지 않는다)")
msg = raises_unavailable({'TTTS3012R': {}, 'CTRP6504R': ok_present(),
                          'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()})
check("BalanceUnavailable", msg is not None, str(msg))

print("\n[6] 투자계좌자산현황이 실패하면 멈춘다 (live 평가로 조용히 대체하지 않는다)")
msg = raises_unavailable({'TTTS3012R': ok_balance(), 'CTRP6504R': ok_present(),
                          'CTRP6548R': {}, 'TTTC2101R': ok_margin()})
check("BalanceUnavailable", msg is not None, str(msg))

print("\n[7] 한쪽 환율만 멀쩡하면 그걸 쓴다 — 다른 쪽 실패로 버리지 않는다")
h, t, e = with_api({'TTTS3012R': ok_balance(), 'CTRP6504R': ok_present(rows=[]),
                    'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()},
                   lambda api: api.get_balance())
check("해외증거금의 환율을 쓴다", abs(e - FX) < 1e-9, str(e))
check("총자산 정상", abs(t - TOT_KRW / FX) < 1e-6, str(t))

print("\n[8] 환율이 NaN·무한대·문자열이면 멈춘다")
for label, v in (('NaN', 'nan'), ('inf', 'inf'), ('문자열', 'x'), ('0', '0'), ('음수', '-1')):
    msg = raises_unavailable({'TTTS3012R': ok_balance(),
                              'CTRP6504R': ok_present(rows=[{'crcy_cd': 'USD',
                                                             'frst_bltn_exrt': v}]),
                              'CTRP6548R': ok_account(),
                              'TTTC2101R': ok_margin(exrt=v)})
    check(f"BalanceUnavailable ({label})", msg is not None, str(msg))

print("\n[9] 달러가 첫 행이 아니어도 달러 행을 고른다")
h, t, e = with_api({'TTTS3012R': ok_balance(),
                    'CTRP6504R': ok_present(rows=[{'crcy_cd': 'HKD', 'frst_bltn_exrt': '169.02'},
                                                  {'crcy_cd': 'USD', 'frst_bltn_exrt': str(FX)}]),
                    'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()},
                   lambda api: api.get_balance())
check("환율 1324.6", abs(e - FX) < 1e-9, str(e))

print("\n[10] 실제 현금 0, 정상적인 빈 보유는 결측이 아니다")
h, t, e = with_api({'TTTS3012R': {'rt_cd': '0', 'output1': []},
                    'CTRP6504R': ok_present(krw_cash=0, frcr_cash=0, ustl_buy=0),
                    'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()},
                   lambda api: api.get_balance())
check("빈 보유 통과", h == {}, str(h))
check("총자산 정상", abs(t - TOT_KRW / FX) < 1e-6, str(t))
h2, t2, _ = with_api({'TTTS3012R': ok_balance(),
                      'CTRP6504R': ok_present(krw_cash=0, frcr_cash=0, ustl_buy=0),
                      'CTRP6548R': ok_account(tot=STOCK_KRW), 'TTTC2101R': ok_margin()},
                     lambda api: api.get_balance())
check("현금 0 계좌 통과", abs(t2 - STOCK_KRW / FX) < 1e-6, str(t2))

print("\n[10-1] 보유 목록 칸이 통째로 빠지면 빈 계좌로 읽지 않는다 (또 사면 안 된다)")
for label, bal in (("키 자체가 없다", {'rt_cd': '0'}),
                   ("null 로 왔다", {'rt_cd': '0', 'output1': None}),
                   ("목록이 아니다", {'rt_cd': '0', 'output1': {'ovrs_pdno': 'BIL'}})):
    why = raises_unavailable({'TTTS3012R': bal, 'CTRP6504R': ok_present(),
                              'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()})
    check(f"멈춘다 — {label}", why is not None and 'output1' in why, str(why))

print("\n[10-2] 보유 행의 종목·수량이 비면 0 으로 메우지 않고 멈춘다")
for label, row in (("종목이 없다", {'ovrs_cblc_qty': '1534', 'ovrs_stck_evlu_amt': '1000'}),
                   ("수량 칸이 없다", {'ovrs_pdno': 'BIL', 'ovrs_stck_evlu_amt': '1000'}),
                   ("수량이 빈 문자열", {'ovrs_pdno': 'BIL', 'ovrs_cblc_qty': '',
                                   'ovrs_stck_evlu_amt': '1000'}),
                   ("수량이 숫자가 아니다", {'ovrs_pdno': 'BIL', 'ovrs_cblc_qty': '-',
                                      'ovrs_stck_evlu_amt': '1000'})):
    why = raises_unavailable({'TTTS3012R': {'rt_cd': '0', 'output1': [row]},
                              'CTRP6504R': ok_present(), 'CTRP6548R': ok_account(),
                              'TTTC2101R': ok_margin()})
    check(f"멈춘다 — {label}", why is not None, str(why))

print("\n[10-3] 평가금액만 망가진 행은 멈추지 않는다 — 그 값은 보고용이고 매매 판단에 안 들어간다")
h4, t4, _ = with_api({'TTTS3012R': {'rt_cd': '0', 'output1': [
                          {'ovrs_pdno': 'BIL', 'ovrs_cblc_qty': '1534',
                           'ovrs_stck_evlu_amt': ''}]},
                      'CTRP6504R': ok_present(), 'CTRP6548R': ok_account(),
                      'TTTC2101R': ok_margin()},
                     lambda api: api.get_balance())
check("보유 수량은 그대로 읽는다", h4 == {'BIL': 1534}, str(h4))
check("총자산은 계좌 총자산 그대로", abs(t4 - TOT_KRW / FX) < 1e-6, str(t4))

print("\n[11] 외화RP 스윕분이 있어도 총자산에 들어온다")
_rp = 50000000.0
h3, t3, _ = with_api({'TTTS3012R': ok_balance(), 'CTRP6504R': ok_present(),
                      'CTRP6548R': ok_account(tot=TOT_KRW + _rp), 'TTTC2101R': ok_margin()},
                     lambda api: api.get_balance())
check("RP 포함", abs(t3 - (TOT_KRW + _rp) / FX) < 1e-6, str(t3))


# ─── 회차 수준: 조회 불가면 주문 0건, 상태 안 바꿈 ───

class _FrozenDatetime(_dt.datetime):
    _today = _dt.date(2026, 10, 21)

    @classmethod
    def now(cls, tz=None):
        return _dt.datetime.combine(cls._today, _dt.time(23, 35))


class RealBalanceAPI(ex.KISAPI):
    """get_balance 는 진짜를 쓰고 주문만 가로챈다."""

    def __init__(self, *a, **k):
        super().__init__(dry_run=False)
        self.orders = []
        RealBalanceAPI.last = self

    # 집행기는 취소 결과를 dict 로 받아 "확인됨" 과 "모름" 을 가른다.
    cancel_result = {'queried': True, 'found': 0, 'failed': 0}

    def cancel_all_pending(self):
        return dict(type(self).cancel_result)

    def get_balance(self):
        # 계수기는 예외 전에 올린다 — 실패한 조회도 한 번 물은 것이다.
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        return super().get_balance()

    def get_current_price(self, ticker, fresh_only=False):
        # fresh_only 는 완료 평가가 쓰는 칸이다. 가짜가 이 인자를 안 받으면 진짜
        # 구현에 그 칸이 있는지 없는지를 시험이 가려 버린다.
        return {'BIL': 91.60, 'PDBC': 20.02}.get(ticker, 100.0)

    def ensure_auth(self) -> None:
        # 이 가짜는 주문을 가로채므로 토큰이 필요 없다. 인증 실패·지연 자체를 재는
        # 시험만 아래에서 이 메서드를 따로 덮는다.
        return None

    def get_max_buy_qty(self, ticker, price) -> int:
        return 999999

    def place_order(self, ticker, qty, price, side='buy', pre_send=None) -> str:
        # 집행기 계약: 'ok' / 'rejected' / 'unknown' / 'stale'. 참거짓이 아니다 — 접수
        # 여부를 모르는 것과 확정 거절을 가르지 않으면 중복 전송이 섞인다.
        # pre_send 는 진짜 구현이 전송 직전에 묻는 검사다. 여기서도 같은 자리에서 묻는다 —
        # 안 물으면 전송 경계 검사가 있는지 없는지를 이 가짜가 가려 버린다.
        if pre_send is not None and not pre_send():
            return 'stale'
        self.orders.append((side, ticker, qty))
        return 'ok'


def step_clock(jump_after, jump_to):
    """부른 횟수가 jump_after 를 넘으면 jump_to 를 돌려주는 단조 시계. 계좌값 나이 시험용.

    집행기가 monotonic 을 부르는 순서 — ①계좌 조회 시작 ②execute_delta 진입(재사용 나이)
    ③주문 직전 검사 ④그 주문의 전송 직전 검사 ⑤이후 주문·재시도마다 둘씩.
    jump_after 로 어느 지점부터 늙게 할지 고른다. 한 번 만료되면 그 뒤 호출은
    기록된 상태로 바로 돌아오므로 시계를 더 먹지 않는다.
    """
    box = {'n': 0}

    def t():
        box['n'] += 1
        return jump_to if box['n'] > jump_after else 0.0
    return t


def run_cycle(table, state_obj, signal, clock=None, dry=False):
    COUNTS.clear()
    alerts, saved = [], {}
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'kis_trade_state.json')
    with open(path, 'w') as f:
        json.dump(state_obj, f)
    keys = ('datetime', 'KISAPI', '_compute_signal_v25', 'send_telegram', 'save_json',
            'TRADE_STATE_FILE', 'time', '_read_alloc_transit_cap_ratio', '_get', 'log')
    old = {k: getattr(ex, k) for k in keys}
    try:
        ex.datetime = _FrozenDatetime
        ex.KISAPI = RealBalanceAPI
        ex._compute_signal_v25 = lambda st: signal
        ex.send_telegram = lambda m: alerts.append(m)
        ex.save_json = lambda p, d, default=None: saved.update({'state': d})
        ex.TRADE_STATE_FILE = path
        # 나이는 monotonic 으로 잰다. 벽시계(time)는 경과시간 로그에만 쓴다.
        ex.time = types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0.0,
                                        monotonic=clock if clock else (lambda: 0.0))
        ex._read_alloc_transit_cap_ratio = lambda sleeve: None
        ex._get = fake_get(table)
        ex.log = lambda *a, **k: None
        RealBalanceAPI.last = None
        try:
            ex.run_once(dry_run=dry)
        except ex.BalanceUnavailable as e:
            alerts.append(f'raised:{e}')
    finally:
        for k, v in old.items():
            setattr(ex, k, v)
    api = getattr(RealBalanceAPI, 'last', None)
    return (api.orders if api else []), alerts, saved.get('state')


def _signal():
    stamp = _FrozenDatetime.now().strftime('%Y-%m-%d %H:%M')
    return {'stock': {'risk_on': False,
                      'offense_picks': ['SPY', 'QQQ', 'VEA'],
                      'offense_weights': {'SPY': 1 / 3, 'QQQ': 1 / 3, 'VEA': 1 / 3},
                      'defense_picks': ['BIL', 'PDBC'],
                      'defense_weights': {'BIL': 0.5, 'PDBC': 0.5}},
            'meta': {'strategy_version': 'V25', 'updated_at': stamp, 'canary': {}}}


def _state(today=_dt.date(2026, 10, 21)):
    G, P = ex.SNAP_STAGGER_DAYS, ex.SNAP_PERIOD_DAYS
    return {'prev_risk_on': False,
            'snapshots': {str(i): {'picks': ['BIL', 'PDBC'],
                                   'weights': {'BIL': 0.5, 'PDBC': 0.5},
                                   'last_rebal_date': (today - _dt.timedelta(days=P)).isoformat(),
                                   'next_rebal_date': (today + _dt.timedelta(days=30 + i * G)).isoformat()}
                          for i in range(ex.N_SNAPS)}}


print("\n[12] 양성 대조 — 정상 조회면 회차가 돈다 (드리프트는 안 터진다)")
orders, alerts, saved = run_cycle(FULL, _state(), _signal())
check("환율 실패 알림 없음", not any('조회가 안 돼' in a for a in alerts), str(alerts))
check("헛매도 없음", [o for o in orders if o[0] == 'sell'] == [], str(orders))

print("\n[13] 2026-10-21 재현 — 환율이 비면 주문 0건, 상태를 안 바꾼다")
broken = {'TTTS3012R': ok_balance(), 'CTRP6504R': ok_present(rows=[]),
          'CTRP6548R': ok_account(), 'TTTC2101R': {}}
orders, alerts, saved = run_cycle(broken, _state(), _signal())
check("주문 0건", orders == [], str(orders))
check("쉰다고 알린다", any('이번 회차를 쉰다' in a for a in alerts), str(alerts))
check("상태파일을 안 바꾼다", saved is None, str(saved))

print("\n[14] 체결기준현재잔고가 통째로 빈 회차 — 주문 0건")
orders, alerts, saved = run_cycle({'TTTS3012R': ok_balance(), 'CTRP6504R': {},
                                   'CTRP6548R': ok_account(), 'TTTC2101R': ok_margin()},
                                  _state(), _signal())
check("주문 0건", orders == [], str(orders))
check("상태파일을 안 바꾼다", saved is None, str(saved))

print("\n[15] 조회가 비어도 드리프트를 '0' 으로 읽어 발화시키지 않는다")
st = _state()
st['rebalancing_needed'] = False
orders, alerts, saved = run_cycle(broken, st, _signal())
check("리필·매도 없음", orders == [], str(orders))


# ─── 토큰 캐시: 집행기와 밤 조회 감시가 같은 파일을 쓴다 ───

class _FakeResp:
    def __init__(self, token):
        self._token = token

    def json(self):
        return {'access_token': self._token}


def with_token_file(content, fn):
    """토큰 캐시 파일을 주어진 내용으로 두고 돌린다. (결과, 발급 횟수, 파일내용)"""
    import tempfile as _tf
    d = _tf.mkdtemp()
    path = os.path.join(d, '.kis_token.json')
    if content is not None:
        with open(path, 'w') as f:
            f.write(content)
    issued = []
    old_file, old_post = ex.TOKEN_FILE, ex.requests.post

    def _post(url, json=None, timeout=None):
        issued.append(url)
        return _FakeResp(f'새토큰{len(issued)}')

    try:
        ex.TOKEN_FILE = path
        ex.requests.post = _post
        out = fn()
    finally:
        ex.TOKEN_FILE, ex.requests.post = old_file, old_post
    saved = open(path).read() if os.path.exists(path) else None
    leftovers = [n for n in os.listdir(d) if '.tmp' in n]
    return out, len(issued), saved, leftovers


_future = (_dt.datetime.now() + _dt.timedelta(hours=10)).isoformat()
_past = (_dt.datetime.now() - _dt.timedelta(hours=1)).isoformat()

print("\n[16] 살아 있는 캐시는 다시 안 받는다 — 발급 한도를 괜히 쓰지 않는다")
tok, n, _, _ = with_token_file(json.dumps({'token': '옛토큰', 'expires': _future}), ex._get_token)
check("캐시 토큰을 쓴다", tok == '옛토큰', str(tok))
check("발급 0회", n == 0, str(n))

print("\n[17] 만료된 캐시는 새로 받는다")
tok, n, saved, _ = with_token_file(json.dumps({'token': '옛토큰', 'expires': _past}), ex._get_token)
check("새 토큰", tok == '새토큰1', str(tok))
check("발급 1회", n == 1, str(n))
check("캐시에 새 토큰이 적힌다",
      saved is not None and json.loads(saved).get('token') == '새토큰1', str(saved))

print("\n[18] 깨진 캐시 하나가 회차를 통째로 막지 않는다 (감시와 집행기가 같은 파일을 쓴다)")
for label, body in (("쓰다 만 파일", '{"token": "반쪽'),
                    ("빈 파일", ''),
                    ("칸이 빠졌다", '{"token": "옛토큰"}'),
                    ("시각이 망가졌다", json.dumps({'token': 'x', 'expires': '어제'}))):
    tok, n, saved, _ = with_token_file(body, ex._get_token)
    check(f"새로 받아 넘어간다 — {label}", tok == '새토큰1' and n == 1, f'{tok!r} 발급{n}')

print("\n[19] 캐시는 원자적으로 쓴다 — 쓰다 만 파일이 남지 않는다")
tok, n, saved, leftovers = with_token_file(None, ex._get_token)
check("임시파일이 안 남는다", leftovers == [], str(leftovers))
check("내용이 온전하다", json.loads(saved).get('token') == '새토큰1', str(saved))

print("\n[20] 매매 없는 회차는 계좌를 한 번만 묻는다 (사전 조회를 판단에 재사용)")
orders, alerts, saved = run_cycle(FULL, _state(), _signal())
_reads = COUNTS.get('get_balance', 0)
check("계좌 조회 1회", _reads == 1, f'{_reads}회')
check("조회 네 갈래도 한 벌", COUNTS.get('TTTS3012R', 0) == 1, str(COUNTS.get('TTTS3012R')))
check("주문 0건", orders == [], str(orders))

print("\n[21] 리밸 회차는 두 번 — 주문 전 판단 한 번, 주문 뒤 체결 확인 한 번")
st = _state()
st['rebalancing_needed'] = True
orders, alerts, saved = run_cycle(FULL, st, _signal())
_reads = COUNTS.get('get_balance', 0)
check("계좌 조회 2회", _reads == 2, f'{_reads}회')


def _with_api(cls, table, state_obj, signal, clock=None, dry=False):
    """RealBalanceAPI 자리를 cls 로 바꿔 한 회차 돌린다."""
    old = globals()['RealBalanceAPI']
    try:
        globals()['RealBalanceAPI'] = cls
        return run_cycle(table, state_obj, signal, clock=clock, dry=dry)
    finally:
        globals()['RealBalanceAPI'] = old


print("\n[22] 주문 뒤 조회가 실패하면 체결 확인 불가로 남는다 — 마무리 재조회는 판정을 안 바꾼다")


class _FailAfterOrders(RealBalanceAPI):
    """첫 조회는 되고 그 다음부터 실패한다 — 주문 뒤 조회가 깨진 회차."""

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        n = getattr(self, '_n', 0) + 1
        self._n = n
        if n == 1:
            return _KISAPI_REAL.get_balance(self)
        raise ex.BalanceUnavailable('주문 뒤 조회 실패(시험)')


class _FailOnceAfterOrders(RealBalanceAPI):
    """체결 확인만 실패하고 마무리 로그용 재조회는 성공한다."""

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        n = getattr(self, '_n', 0) + 1
        self._n = n
        if n == 2:
            raise ex.BalanceUnavailable('주문 뒤 조회 실패(시험)')
        return _KISAPI_REAL.get_balance(self)


_st = _state()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_FailAfterOrders, FULL, _st, _signal())
check("체결 확인 불가를 알린다", any('체결 확인 불가' in a for a in alerts), str(alerts))
check("완료 판정을 안 내린다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))
check("상태를 저장한다 (주문은 나갔다)", saved is not None, str(saved))
_reads = COUNTS.get('get_balance', 0)
check("조회 3회 — 사전·체결확인·마무리", _reads == 3, f'{_reads}회')

_st = _state()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_FailOnceAfterOrders, FULL, _st, _signal())
check("마무리 재조회가 성공해도 완료로 안 바뀐다",
      saved is not None and saved.get('rebalancing_needed') is True, str(saved))
check("그래도 체결 확인 불가를 알린다", any('체결 확인 불가' in a for a in alerts), str(alerts))
_reads = COUNTS.get('get_balance', 0)
check("조회 3회", _reads == 3, f'{_reads}회')

print("\n[23] 미체결 조회가 실패하면 그 회차를 쉰다 — 잔존 주문을 '없음' 으로 읽지 않는다")


class _PendingQueryFails(RealBalanceAPI):
    cancel_result = {'queried': False, 'found': 0, 'failed': 0}


orders, alerts, saved = _with_api(_PendingQueryFails, FULL, _state(), _signal())
check("주문 0건", orders == [], str(orders))
check("상태파일을 안 바꾼다", saved is None, str(saved))
check("미체결 때문에 쉰다고 알린다", any('미체결' in a for a in alerts), str(alerts))
check("계좌 조회도 안 한다", COUNTS.get('get_balance', 0) == 0, str(COUNTS.get('get_balance')))

print("\n[24] 취소가 실패했거나 미체결이 남아 있으면 그 회차를 쉰다")


class _CancelFails(RealBalanceAPI):
    cancel_result = {'queried': True, 'found': 2, 'failed': 1}


orders, alerts, saved = _with_api(_CancelFails, FULL, _state(), _signal())
check("취소 실패 → 주문 0건", orders == [], str(orders))
check("취소 실패 → 상태 유지", saved is None, str(saved))


class _PendingStays(RealBalanceAPI):
    cancel_result = {'queried': True, 'found': 1, 'failed': 0}


orders, alerts, saved = _with_api(_PendingStays, FULL, _state(), _signal())
check("미체결 잔존 → 주문 0건", orders == [], str(orders))
check("미체결 잔존 → 상태 유지", saved is None, str(saved))

print("\n[25] 미체결이 있었지만 취소·종결이 확인되면 회차는 그대로 돈다")


class _PendingCleared(RealBalanceAPI):
    """첫 조회에 1건, 재조회부터 0건 — 취소가 먹힌 모양."""
    _seq = 0

    def cancel_all_pending(self):
        type(self)._seq += 1
        if type(self)._seq == 1:
            return {'queried': True, 'found': 1, 'failed': 0}
        return {'queried': True, 'found': 0, 'failed': 0}


_PendingCleared._seq = 0
orders, alerts, saved = _with_api(_PendingCleared, FULL, _state(), _signal())
check("회차가 돈다 (사전 조회까지 간다)", COUNTS.get('get_balance', 0) >= 1, str(COUNTS.get('get_balance')))
check("쉰다는 알림 없음", not any('미체결 주문 상태를 확인하지 못해' in a for a in alerts), str(alerts))

print("\n[26] 계좌값이 허용 나이를 넘기면 그 값으로 주문하지 않는다")
_st = _state()
_st['rebalancing_needed'] = True
# 3번째 호출(주문 직전 나이 측정)부터 뛴다 — 들어올 때는 신선, 수량 산출 중 만료.
orders, alerts, saved = run_cycle(FULL, _st, _signal(), clock=step_clock(2, 100.0))
check("주문 직전 만료 → 주문 0건", orders == [], str(orders))
check("보류를 알린다", any('주문을 내지 않았다' in a for a in alerts), str(alerts))
check("다음 회차로 넘긴다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))
check("체결 확인 조회 안 함", COUNTS.get('get_balance', 0) == 1, str(COUNTS.get('get_balance')))

_st = _state()
_st['rebalancing_needed'] = True
# 2번째 호출(execute_delta 진입)부터 뛴다 — 들어올 때 이미 만료.
orders, alerts, saved = run_cycle(FULL, _st, _signal(), clock=step_clock(1, 100.0))
check("진입부 만료 → 주문 0건", orders == [], str(orders))
check("진입부 만료 → 다시 읽지 않는다 (상한과 짝이 안 맞는다)",
      COUNTS.get('get_balance', 0) == 1, str(COUNTS.get('get_balance')))
check("진입부 만료 → 다음 회차로 넘긴다",
      saved is not None and saved.get('rebalancing_needed') is True, str(saved))

print("\n[26-1] 주문 묶음이 시작된 뒤에 만료되면 남은 주문을 멈춘다")
_st = _state()
_st['rebalancing_needed'] = True
# 4번째 호출 = 첫 매수 직전. 여기서부터 늙으면 주문이 하나도 안 나간다.
orders, alerts, saved = run_cycle(FULL, _st, _signal(), clock=step_clock(3, 100.0))
check("주문 0건", orders == [], str(orders))
check("주문 도중 멈췄다고 알린다", any('주문 도중' in a for a in alerts), str(alerts))
check("완료로 안 본다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))

_st = _state()
_st['rebalancing_needed'] = True
# 6번째 호출부터 늙는다 — 첫 매수(BIL)는 검사와 전송 경계를 둘 다 통과해 나가고,
# 둘째(PDBC)는 자기 검사에서 막힌다. 주문 하나가 시계를 둘 먹는다(검사 + 전송 직전).
orders, alerts, saved = run_cycle(FULL, _st, _signal(), clock=step_clock(5, 100.0))
_buys = [o for o in orders if o[0] == 'buy']
check("첫 주문만 나간다", len(_buys) == 1, str(orders))
check("일부만 나간 회차는 미완료로 남는다",
      saved is not None and saved.get('rebalancing_needed') is True, str(saved))

print("\n[27] 취소 응답은 성공 표식으로 판정한다 — 진짜 cancel_all_pending 을 돌린다")


def _nccs(rows, nk='', fk='', tr_cont=None):
    """미체결 조회 응답 한 쪽. tr_cont 는 응답 헤더(F·M=더 있다, D·E=마지막).

    안 적으면 커서에서 끌어온다 — 다음 커서가 있으면 F, 없으면 D. 실계좌가 그 짝으로
    오기 때문이고(2026-10-29 실측), 헤더와 커서를 일부러 어긋나게 둔 칸만 직접 적는다.
    """
    d = {'rt_cd': '0', 'output': rows, 'ctx_area_nk200': nk, 'ctx_area_fk200': fk}
    d['_hdr_tr_cont'] = tr_cont if tr_cont is not None else ('F' if nk.strip() else 'D')
    return d


def _row(odno='0001', pdno='BIL', qty='10', side='01', excg='NASD'):
    """미체결 한 줄. excg=None 이면 응답이 거래소를 안 알려주는 계좌다.

    실계좌 응답에 ovrs_excg_cd 가 들어오는지는 아직 못 봤다(미체결이 있는 회차를 아직
    안 만났다). 그래서 적는 쪽과 안 적는 쪽을 둘 다 시험한다 — 안 적으면 원주문
    거래소가 미확정이고, 그때 물어본 코드로 메우면 안 된다(이 조회는 한 코드로 물어도
    미국 전체를 돌려준다). 기본값을 'NASD' 로 둔 것은 거래소와 상관없는 칸들이
    정상 경로를 타게 하려는 것뿐이다.
    """
    d = {'odno': odno, 'pdno': pdno, 'nccs_qty': qty, 'sll_buy_dvsn_cd': side}
    if excg is not None:
        d['ovrs_excg_cd'] = excg
    return d


_CANCEL_REQUESTS = []
_CANCEL_BODIES = []    # 실제로 나간 취소 전문 — 거래소 코드까지 본다.


def run_cancel(nccs_by_excg, post_result, api=None):
    """진짜 KISAPI.cancel_all_pending 을 통신만 가짜로 돌린다. (결과, 취소한 주문번호들)

    값을 목록으로 주면 그 거래소의 연속조회 쪽들이다 — 쪽마다 하나씩 돌려준다.
    `api` 를 주면 그 인스턴스를 쓴다 — 우리가 보낸 주문의 거래소 기록을 미리 심을 때.
    """
    posts = []
    pages = {k: 0 for k in nccs_by_excg}
    reqs = _CANCEL_REQUESTS
    reqs.clear()
    _CANCEL_BODIES.clear()
    old_get, old_post, old_log = ex._get, ex._post, ex.log

    def _g(path, tr_id, params, retries=3, tr_cont='', cont=None):
        excg = params.get('OVRS_EXCG_CD')
        reqs.append({'excg': excg, 'tr_cont': tr_cont,
                     'fk': params.get('CTX_AREA_FK200'), 'nk': params.get('CTX_AREA_NK200')})
        resp = nccs_by_excg.get(excg, {})
        if isinstance(resp, list):
            i = pages.get(excg, 0)
            pages[excg] = i + 1
            resp = resp[i] if i < len(resp) else {}
        if cont is not None and isinstance(resp, dict):
            cont['tr_cont'] = resp.get('_hdr_tr_cont', '')
        return resp

    def _p(path, tr_id, body, retries=2):
        posts.append(body.get('ORGN_ODNO'))
        _CANCEL_BODIES.append(body)
        return post_result() if callable(post_result) else post_result

    try:
        ex._get, ex._post, ex.log = _g, _p, lambda *a, **k: None
        _api = api if api is not None else _KISAPI_REAL(dry_run=False)
        return _api.cancel_all_pending(), posts
    finally:
        ex._get, ex._post, ex.log = old_get, old_post, old_log


_ALL_EMPTY = {e: _nccs([]) for e in ex.PENDING_EXCHANGES}

res, posts = run_cancel(_ALL_EMPTY, {'rt_cd': '0'})
check("미체결 없음", res == {'queried': True, 'found': 0, 'failed': 0}, str(res))
check("취소 주문도 없다", posts == [], str(posts))

_one = dict(_ALL_EMPTY, NASD=_nccs([_row()]))
res, posts = run_cancel(_one, {'rt_cd': '0', 'output': {'ODNO': '9'}})
check("취소 성공은 실패로 안 센다", res == {'queried': True, 'found': 1, 'failed': 0}, str(res))

res, posts = run_cancel(_one, {'_error': {'status_code': 500, 'body': {'rt_cd': '1'}}})
check("취소 거부를 실패로 센다 (_error 는 참값이다)",
      res == {'queried': True, 'found': 1, 'failed': 1}, str(res))

res, posts = run_cancel(_one, {})
check("응답이 통째로 비면 실패로 센다", res.get('failed') == 1, str(res))

res, posts = run_cancel(dict(_ALL_EMPTY, NASD={}), {'rt_cd': '0'})
check("조회 실패는 '없음' 과 구분한다", res == {'queried': False, 'found': 0, 'failed': 0}, str(res))

for label, bad in (("output 칸이 없다", {'rt_cd': '0'}),
                   ("목록이 아니다", {'rt_cd': '0', 'output': {'odno': '1'}}),
                   ("행이 dict 가 아니다", _nccs(['x'])),
                   ("수량이 숫자가 아니다", _nccs([_row(qty='-')])),
                   ("수량 칸이 없다", _nccs([{'odno': '1', 'pdno': 'BIL'}])),
                   ("주문번호가 없다", _nccs([_row(odno='')])),
                   ("수량이 무한대다", _nccs([_row(qty='inf')])),
                   ("수량이 음수다", _nccs([_row(qty='-3')])),
                   ("수량이 소수다", _nccs([_row(qty='2.5')])),
                   ("둘째 쪽 조회가 실패한다", [_nccs([_row()], nk='NK'), {}])):
    res, posts = run_cancel(dict(_ALL_EMPTY, NASD=bad), {'rt_cd': '0'})
    check(f"'없음' 으로 통과하지 않는다 — {label}", res.get('queried') is False, f'{label}: {res}')
    check(f"그 상태로 취소를 쏘지 않는다 — {label}", posts == [], str(posts))

res, posts = run_cancel(dict(_ALL_EMPTY, NASD=_nccs([_row(qty='0')])), {'rt_cd': '0'})
check("수량 0 행은 미체결이 아니다", res == {'queried': True, 'found': 0, 'failed': 0}, str(res))

print("\n[27-2] 연속조회는 끝까지 따라간다 — 한 쪽만 보고 없다고 말하지 않는다")
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=[_nccs([_row(odno='P1')], nk='NK'),
                                               _nccs([_row(odno='P2')], nk='')]),
                        {'rt_cd': '0'})
check("둘째 쪽 미체결도 잡는다", res.get('found') == 2, str(res))
check("둘 다 취소한다", sorted(posts) == ['P1', 'P2'], str(posts))

# 커서가 안 움직이면 다음 쪽을 봤는지 같은 쪽을 다시 봤는지 가를 수 없다 — 그걸 "끝" 으로
# 읽으면 뒤쪽 미체결을 놓친 채 없다고 말하게 된다.
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=[_nccs([_row(odno='Q1')], nk='NK'),
                                               _nccs([_row(odno='Q1')], nk='NK')]),
                        {'rt_cd': '0'})
check("커서가 그대로면 전부 읽었다고 말하지 않는다", res.get('queried') is False, str(res))
check("그 상태로 취소를 쏘지 않는다 — 커서 정체", posts == [], str(posts))

# 수량 0 인 행만 담긴 쪽 뒤에도 미체결이 더 있을 수 있다. 활성 행 수로 쪽을 끊으면 그걸 놓친다.
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=[_nccs([_row(odno='Z0', qty='0')], nk='NK'),
                                               _nccs([_row(odno='Z1')], nk='')]),
                        {'rt_cd': '0'})
check("수량 0 쪽 뒤의 미체결을 놓치지 않는다", res.get('found') == 1, str(res))
check("그 주문을 취소한다", posts == ['Z1'], str(posts))

_endless = [_nccs([_row(odno=f'E{i}')], nk='NK') for i in range(ex.PENDING_PAGE_MAX + 2)]
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=_endless), {'rt_cd': '0'})
check("쪽이 끝없이 오면 전부 읽었다고 말하지 않는다", res.get('queried') is False, str(res))
check("그 상태로 취소를 쏘지 않는다", posts == [], str(posts))

print("\n[27-1] 거래소를 셋 다 묻는다 — 한 코드만 물으면 다른 쪽 미체결이 안 보인다")
asked = []


def _collect_get(path, tr_id, params, retries=3, tr_cont='', cont=None):
    asked.append(params.get('OVRS_EXCG_CD'))
    resp = _nccs([])
    if cont is not None:
        cont['tr_cont'] = resp['_hdr_tr_cont']
    return resp


_old_get, _old_log = ex._get, ex.log
try:
    ex._get, ex.log = _collect_get, lambda *a, **k: None
    _KISAPI_REAL(dry_run=False).cancel_all_pending()
finally:
    ex._get, ex.log = _old_get, _old_log
check("NASD·NYSE·AMEX 를 다 묻는다", sorted(asked) == sorted(ex.PENDING_EXCHANGES), str(asked))

res, posts = run_cancel(dict(_ALL_EMPTY, AMEX=_nccs([_row(odno='A1', pdno='GLD')])),
                        {'rt_cd': '0'})
check("AMEX 미체결도 잡는다", res.get('found') == 1, str(res))
check("그 주문을 취소한다", posts == ['A1'], str(posts))

res, posts = run_cancel({e: _nccs([_row(odno='SAME', excg=None)]) for e in ex.PENDING_EXCHANGES},
                        {'rt_cd': '0'})
# 응답이 거래소를 안 알려주고 우리가 보낸 기록도 없으면 원주문이 어디 것인지 모른다.
# 그 상태로 취소를 쏘지 않는다 — 엉뚱한 거래소로 간 취소는 성공으로 돌아오고 원주문은
# 살아남는다. 물어본 코드는 근거가 아니다(한 코드로 물어도 미국 전체가 온다).
check("거래소를 가릴 수 없으면 조회 실패로 돌려준다", res.get('queried') is False, str(res))
check("그 상태로 취소를 안 쏜다", posts == [], str(posts))

res, posts = run_cancel({e: _nccs([_row(odno='SAME', excg='NASD')])
                         for e in ex.PENDING_EXCHANGES}, {'rt_cd': '0'})
check("응답이 같은 거래소를 말하면 중복 노출로 보고 한 번만 센다",
      res.get('queried') is True and res.get('found') == 1, str(res))
check("취소도 한 번만 쏜다", posts == ['SAME'], str(posts))

print("\n[28] 주문 뒤 미체결 상태를 확정 못 하면 완료로 안 본다")


class _PendingBreaksAfterOrders(RealBalanceAPI):
    """회차 시작은 깨끗하고, 주문 뒤 확인에서 조회가 깨진다."""
    _seq = 0

    def cancel_all_pending(self):
        type(self)._seq += 1
        if type(self)._seq == 1:
            return {'queried': True, 'found': 0, 'failed': 0}
        return {'queried': False, 'found': 0, 'failed': 0}


_st = _state()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(RealBalanceAPI, FULL, _st, _signal())
check("양성 대조 — 깨끗하면 완료로 닫힌다",
      saved is not None and saved.get('rebalancing_needed') is False, str(saved))

_PendingBreaksAfterOrders._seq = 0
_st = _state()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_PendingBreaksAfterOrders, FULL, _st, _signal())
check("주문은 나간다", len(orders) > 0, str(orders))
check("완료로 안 닫는다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))

print("\n[29] 접수 여부를 모르면 다시 보내지 않고 그 회차의 신규 주문을 멈춘다")


class _SendUnknown(RealBalanceAPI):
    """응답이 유실된 모양 — 서버가 받아놨는지 우리만 모른다."""

    def place_order(self, ticker, qty, price, side='buy', pre_send=None) -> str:
        if pre_send is not None and not pre_send():
            return 'stale'
        self.orders.append((side, ticker, qty))
        return 'unknown'


class _SendRejected(RealBalanceAPI):
    """서버가 본문으로 거절했다 — 접수되지 않았음이 확정이라 다시 내도 안전하다."""

    def place_order(self, ticker, qty, price, side='buy', pre_send=None) -> str:
        if pre_send is not None and not pre_send():
            return 'stale'
        self.orders.append((side, ticker, qty))
        return 'rejected'


_st = _state()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_SendUnknown, FULL, _st, _signal())
check("재전송하지 않는다 — 주문은 한 번만 나간다", len(orders) == 1, str(orders))
check("접수 여부 불명을 알린다", any('접수 여부' in a for a in alerts), str(alerts))
check("완료로 안 닫는다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))

_st = _state()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_SendRejected, FULL, _st, _signal())
check("확정 거절은 재시도 대상이다", len(orders) == ex.MAX_ORDER_ATTEMPTS * 2, str(len(orders)))
check("거절은 '접수 여부 불명' 으로 알리지 않는다",
      not any('접수 여부' in a for a in alerts), str(alerts))

print("\n[30] 주문 응답을 셋으로 가른다 — 확정 거절과 '모름' 을 섞지 않는다")
_retries_seen = []


def run_place(post_result):
    old_post, old_log = ex._post, ex.log

    def _p(path, tr_id, body, retries=3, pre_send=None):
        _retries_seen.append(retries)
        if pre_send is not None and not pre_send():
            return {'_stale': True}
        return post_result

    try:
        ex._post, ex.log = _p, lambda *a, **k: None
        return _KISAPI_REAL(dry_run=False).place_order('BIL', 10, 91.60, 'buy')
    finally:
        ex._post, ex.log = old_post, old_log


check("정상 접수 → ok", run_place({'rt_cd': '0', 'output': {'ODNO': '123'}}) == 'ok')
check("업무 거절 코드가 담긴 본문 → rejected",
      run_place({'_error': {'status_code': 400, 'body': {'rt_cd': '1'}}}) == 'rejected')
check("HTTP 200 에 rt_cd 1 도 rejected",
      run_place({'_error': {'status_code': 200, 'body': {'rt_cd': '1', 'msg1': '잔고부족'}}}) == 'rejected')
check("예외(시간 초과·연결 끊김) → unknown",
      run_place({'_error': {'exception': 'timeout'}}) == 'unknown')
# 게이트웨이 5xx 는 서버가 이미 받아놓은 뒤일 수 있다. 본문이 있다는 것만으로 거절로 읽으면
# 같은 주문이 재시도 횟수만큼 다시 나간다.
check("502 는 본문이 있어도 unknown",
      run_place({'_error': {'status_code': 502, 'body': {'msg': 'Bad Gateway'}}}) == 'unknown')
check("504 도 unknown",
      run_place({'_error': {'status_code': 504, 'body': {'rt_cd': '1'}}}) == 'unknown')
check("rt_cd 칸이 없으면 unknown",
      run_place({'_error': {'status_code': 400, 'body': {'msg1': '알 수 없음'}}}) == 'unknown')
check("rt_cd 가 null 이면 unknown",
      run_place({'_error': {'status_code': 400, 'body': {'rt_cd': None}}}) == 'unknown')
check("본문이 비면 unknown",
      run_place({'_error': {'status_code': 400, 'body': {}}}) == 'unknown')
check("응답이 비면 unknown", run_place({}) == 'unknown')
check("output 칸이 없으면 unknown", run_place({'rt_cd': '0'}) == 'unknown')
check("output 이 dict 가 아니면 unknown", run_place({'rt_cd': '0', 'output': []}) == 'unknown')
check("주문번호가 비면 unknown", run_place({'rt_cd': '0', 'output': {'ODNO': ''}}) == 'unknown')
# 전송 계층이 스스로 재전송하면 중복 체결이 난다. 재시도는 부르는 쪽이 'rejected' 에서만 한다.
check("전송 계층은 스스로 다시 보내지 않는다", set(_retries_seen) == {1}, str(_retries_seen))

print("\n[31] 매도가 안 끝났거나 살 수 있는 양을 모르면 사지 않는다")


def _state_bil_only():
    """목표가 BIL 하나인 상태 — 보유 PDBC 는 전량매도, BIL 은 매수가 된다.

    목표는 신호가 아니라 스냅샷에서 나온다. 신호만 바꾸면 앵커일이 아닌 회차에서는
    목표가 그대로라 매도가 안 생긴다.
    """
    st = _state()
    for snap in st['snapshots'].values():
        snap['picks'] = ['BIL']
        snap['weights'] = {'BIL': 1.0}
    return st


class _SellRejected(RealBalanceAPI):
    """매도만 거절당한다 — 현금이 얼마 생겼는지 모르는 상태."""

    def place_order(self, ticker, qty, price, side='buy', pre_send=None) -> str:
        if pre_send is not None and not pre_send():
            return 'stale'
        self.orders.append((side, ticker, qty))
        return 'rejected' if side == 'sell' else 'ok'


class _MaxQtyFails(RealBalanceAPI):
    def get_max_buy_qty(self, ticker, price) -> int:
        return -1


_st = _state_bil_only()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_SellRejected, FULL, _st, _signal())
check("매도 시도는 있었다", [o for o in orders if o[0] == 'sell'] != [], str(orders))
check("매수는 한 건도 안 낸다", [o for o in orders if o[0] == 'buy'] == [], str(orders))
check("매수를 멈췄다고 알린다",
      any('매수' in a and '내지 않았다' in a for a in alerts), str(alerts))
check("완료로 안 닫는다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))

_st = _state()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_MaxQtyFails, FULL, _st, _signal())
check("매수가능수량 조회 실패를 '한도 없음' 으로 읽지 않는다", orders == [], str(orders))
check("그 회차는 완료가 아니다",
      saved is not None and saved.get('rebalancing_needed') is True, str(saved))

print("\n[32] 보유 종목의 현재가를 못 구하면 그 보유를 0 으로 세지 않는다")


class _PriceMissing(RealBalanceAPI):
    def get_current_price(self, ticker, fresh_only=False) -> float:
        return 0.0 if ticker == 'BIL' else 20.02


_st = _state()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_PriceMissing, FULL, _st, _signal())
check("주문 0건 — 목표 전액을 새로 사지 않는다", orders == [], str(orders))
check("현재가를 못 구했다고 알린다", any('현재가' in a for a in alerts), str(alerts))
check("다음 회차로 넘긴다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))

_st = _state()
_st['rebalancing_needed'] = False
orders, alerts, saved = _with_api(_PriceMissing, FULL, _st, _signal())
check("드리프트 판정도 쉰다 — 주문 0건", orders == [], str(orders))
check("판정을 쉰다고 알린다", any('드리프트' in a for a in alerts), str(alerts))

print("\n[33] 미체결 목록에서 사라진 것이 체결의 증거는 아니다")


class _SellVanishes(RealBalanceAPI):
    """매도는 접수됐는데(ok) 잔고는 그대로다 — 접수 뒤 거절·취소된 모양."""


_st = _state_bil_only()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_SellVanishes, FULL, _st, _signal())
check("매도는 나갔다", [o for o in orders if o[0] == 'sell'] != [], str(orders))
check("잔고가 안 줄었으면 매수를 안 낸다", [o for o in orders if o[0] == 'buy'] == [], str(orders))
check("잔고에 안 비쳤다고 알린다",
      any('잔고에 반영되지 않아' in a for a in alerts), str(alerts))
check("완료로 안 닫는다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))

print("\n[34] 목표 밖 보유가 남으면 완료가 아니다")
# 목표는 BIL 하나인데 PDBC 를 그대로 들고 있다. 목표 안쪽만 보면 BIL 비중만 재서
# "달성" 으로 닫히는 자리다.
_st = _state_bil_only()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_SellVanishes, FULL, _st, _signal())
check("목표 밖 보유를 이탈로 센다",
      saved is not None and saved.get('rebalancing_needed') is True, str(saved))

print("\n[35] 모의 실행은 조회 실패로 읽히지 않는다")
# get_max_buy_qty 는 실패를 -1 로 알린다. 모의 실행이 그 값을 돌려주면 부르는 쪽이
# 한도 조회 실패로 읽어 모의 매수가 통째로 사라진다 — 둘은 다른 뜻이다.
_dry = ex.KISAPI(dry_run=True)
_cap = _dry.get_max_buy_qty('BIL', 91.60)
check("모의 한도는 -1 이 아니다", _cap > 0, str(_cap))
check("모의 주문은 통신 없이 ok", _dry.place_order('BIL', 1, 91.60, 'buy') == 'ok')
# 모의 취소도 미체결 조회는 실제로 돈다(취소 전문만 안 나간다). 시험이 진짜 서버를
# 치지 않게 그 조회만 막아 둔다 — 안 막으면 이 칸의 결과가 그날 망 상태에 걸린다.
_old_get35, _old_log35 = ex._get, ex.log
try:
    ex.log = lambda *a, **k: None
    ex._get = lambda path, tr_id, params, retries=3, tr_cont='', cont=None: (
        cont.__setitem__('tr_cont', 'D') if cont is not None else None,
        {'rt_cd': '0', 'output': [], 'ctx_area_nk200': '', 'ctx_area_fk200': ''})[1]
    check("모의 취소는 조회했다고 돌려준다",
          ex.KISAPI(dry_run=True).cancel_all_pending().get('queried') is True)
finally:
    ex._get, ex.log = _old_get35, _old_log35

print("\n[36] 매도가 잔고에 비치면 매수가 나간다 — 양성 대조")

_CLOCK = {'t': 0.0}


class _SellsSettle(RealBalanceAPI):
    """매도가 잔고에 바로 비치는 계좌. 매도 뒤 재조회가 줄어든 수량을 돌려준다."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._sold = {}

    def place_order(self, ticker, qty, price, side='buy', pre_send=None) -> str:
        res = super().place_order(ticker, qty, price, side, pre_send)
        if res != 'ok':
            return res
        if side == 'sell':
            self._sold[ticker] = self._sold.get(ticker, 0) + qty
        return res

    def get_balance(self):
        h, t, x = super().get_balance()
        for tk, q in self._sold.items():
            h[tk] = max(0, h.get(tk, 0) - q)
        return h, t, x


_st = _state_bil_only()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_SellsSettle, FULL, _st, _signal())
_buys36 = [o for o in orders if o[0] == 'buy']
check("매도가 나간다", [o for o in orders if o[0] == 'sell'] != [], str(orders))
check("매수도 나간다", _buys36 != [], str(orders))
check("잔고 미반영 알림은 없다",
      not any('잔고에 반영되지 않아' in a for a in alerts), str(alerts))

print("\n[37] 매도 뒤 잔고 조회가 오래 걸리면 그 값으로 사지 않는다")
# 나이를 조회가 끝난 시각으로 재면 90초 걸린 조회가 나이 0 으로 찍혀 허용 60초를
# 그냥 통과한다. 시작 시각으로 재야 낡은 값이 낡은 것으로 보인다.


class _SlowSecondBalance(_SellsSettle):
    def get_balance(self):
        res = super().get_balance()
        if COUNTS.get('get_balance', 0) == 2:
            _CLOCK['t'] += 90.0
        return res


_CLOCK['t'] = 0.0
_st = _state_bil_only()
_st['rebalancing_needed'] = True
orders, alerts, saved = _with_api(_SlowSecondBalance, FULL, _st, _signal(),
                                  clock=lambda: _CLOCK['t'])
check("매도는 나갔다", [o for o in orders if o[0] == 'sell'] != [], str(orders))
check("90초 걸린 조회로 사지 않는다", [o for o in orders if o[0] == 'buy'] == [], str(orders))
check("낡았다고 알린다", any('계좌값이 90초' in a for a in alerts), str(alerts))
check("완료로 안 닫는다", saved is not None and saved.get('rebalancing_needed') is True, str(saved))

print("\n[38] 매도 직후 총자산이 부풀어도 매수 예산이 매도 전 상한을 안 넘는다")
# 매도대금이 결제 전 단계에서 예수금과 주식평가에 겹쳐 잡히면 총자산이 잠깐 커 보인다.
# 그 값을 그대로 예산으로 쓰면 현금버퍼를 침범한다.


class _InflatedAfterSell(_SellsSettle):
    def get_balance(self):
        h, t, x = super().get_balance()
        if COUNTS.get('get_balance', 0) >= 2:
            t *= 1.5
        return h, t, x


_st = _state_bil_only()
_st['rebalancing_needed'] = True
_st_norm = _state_bil_only()
_st_norm['rebalancing_needed'] = True
_orders_norm, _, _ = _with_api(_SellsSettle, FULL, _st_norm, _signal())
_orders_inf, _, _saved_inf = _with_api(_InflatedAfterSell, FULL, _st, _signal())
_qty_norm = sum(q for s, _t, q in _orders_norm if s == 'buy')
_qty_inf = sum(q for s, _t, q in _orders_inf if s == 'buy')
check("양쪽 다 매수가 있었다 — 빈 비교가 아니다", _qty_norm > 0, str(_orders_norm))
check("부풀린 총자산이 매수를 키우지 않는다", _qty_inf == _qty_norm, f'{_qty_inf} vs {_qty_norm}')
# 예산 상한과 완료 판정은 다른 방향이다. 예산은 덜 쓰는 쪽이 안전하니 매도 전 총자산을
# 상한으로 걸지만, 완료 판정의 분모까지 깎으면 현재 비중이 커 보여 목표에 못 미치는
# 보유가 달성으로 읽힌다. 분모는 주문 뒤 총자산 하나로 간다.
check("부풀린 총자산에서는 완료로 안 닫는다",
      _saved_inf is not None and _saved_inf.get('rebalancing_needed') is True, str(_saved_inf))

print("\n[39] 모의 회차는 매도 다음 매수까지 간다 — 통신은 한 번도 안 한다")


class _DryAPI(_SellsSettle):
    """모의 실행. place_order 는 통신 없이 ok 를 돌려주고 잔고는 안 줄어든다."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dry_run = True

    def place_order(self, ticker, qty, price, side='buy', pre_send=None) -> str:
        if pre_send is not None and not pre_send():
            return 'stale'
        self.orders.append((side, ticker, qty))
        return 'ok'      # 모의 — 체결도 잔고 반영도 없다


_old_post = ex._post
try:
    def _boom(*a, **k):
        raise AssertionError('모의 실행이 실제 주문 통신을 했다')
    ex._post = _boom
    _st = _state_bil_only()
    _st['rebalancing_needed'] = True
    orders, alerts, saved = _with_api(_DryAPI, FULL, _st, _signal(), dry=True)
finally:
    ex._post = _old_post
check("모의에서 매도가 보인다", [o for o in orders if o[0] == 'sell'] != [], str(orders))
check("모의에서 매수도 보인다", [o for o in orders if o[0] == 'buy'] != [], str(orders))
check("모의는 상태파일을 안 쓴다", saved is None, str(saved))

print("\n[40] 연속조회 요청이 계약대로 나간다 — tr_cont 와 커서 둘")
# 실계좌 관측(2026-10-29 16:5x KST, 서버에서 읽기만 하는 조회 한 번): 미체결이 없는
# 계좌의 TTTS3018R 는 세 거래소 모두 HTTP 200 · rt_cd 0 · 응답 헤더 tr_cont='D' ·
# output 0행 · nk200 빈 값 · fk200 은 비지 않은 문자열(계좌·거래소·정렬을 ^ 로 이은 것)
# 을 돌려줬다. 그래서 종료 판정을 헤더 우선으로 둔다 — fk200 이 남아 있다고 더 있는
# 것이 아니고, 헤더가 D·E 면 커서가 남아도 끝이다.
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=[_nccs([_row(odno='P1')], nk='NK1', fk='FK1'),
                                               _nccs([_row(odno='P2')], nk='')]),
                        {'rt_cd': '0'})
_nasd = [r for r in _CANCEL_REQUESTS if r['excg'] == 'NASD']
check("첫 요청은 커서가 비고 tr_cont 도 없다",
      _nasd and _nasd[0] == {'excg': 'NASD', 'tr_cont': '', 'fk': '', 'nk': ''}, str(_nasd[:1]))
check("둘째 요청은 tr_cont=N 과 직전 커서 둘을 싣는다",
      len(_nasd) > 1 and _nasd[1] == {'excg': 'NASD', 'tr_cont': 'N', 'fk': 'FK1', 'nk': 'NK1'},
      str(_nasd[1:2]))
check("둘 다 취소한다", sorted(posts) == ['P1', 'P2'], str(posts))

# 마지막 쪽이 커서를 남겨도 종료 헤더가 있으면 끝이다.
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=[_nccs([_row(odno='L1')], nk='STILL', tr_cont='D')]),
                        {'rt_cd': '0'})
check("종료 헤더면 커서가 남아도 끝낸다", res.get('queried') is True and res.get('found') == 1, str(res))

# 행이 비었는데 "더 있다" 면 다음 쪽을 읽는다.
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=[_nccs([], nk='NK1', tr_cont='M'),
                                               _nccs([_row(odno='N2')], nk='', tr_cont='D')]),
                        {'rt_cd': '0'})
check("빈 쪽 뒤의 미체결을 읽는다", res.get('found') == 1, str(res))
check("그 주문을 취소한다", posts == ['N2'], str(posts))

# "더 있다" 면서 커서를 안 주면 모순이다 — 전부 읽었다고 말하지 않는다.
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=[_nccs([_row(odno='C1')], nk='', tr_cont='F')]),
                        {'rt_cd': '0'})
check("계속 표시와 커서가 모순이면 중단한다", res.get('queried') is False, str(res))
check("그 상태로 취소를 쏘지 않는다 — 모순", posts == [], str(posts))

print("\n[41] 보유 잔고가 여러 쪽이면 다 읽는다 — 첫 쪽만 보면 뒤쪽 종목을 또 산다")
# 실계좌 관측(2026-10-29): 보유 2종목 계좌의 TTTS3012R 는 응답 헤더 tr_cont='D' 와
# 200칸 공백 커서를 같이 돌려줬다. 그래서 끝 판정은 헤더가 먼저고, 커서는 받은 원문을
# 그대로 되돌려준다.


def _bal_page(rows, nk='', fk='', tr_cont=None):
    d = {'rt_cd': '0', 'output1': rows, 'ctx_area_nk200': nk, 'ctx_area_fk200': fk}
    d['_hdr_tr_cont'] = tr_cont if tr_cont is not None else ('F' if nk.strip() else 'D')
    return d


_ROW_BIL = {'ovrs_pdno': 'BIL', 'ovrs_cblc_qty': '1534',
            'ovrs_stck_evlu_amt': str(LIVE_BIL), 'now_pric2': '91.60'}
_ROW_PDBC = {'ovrs_pdno': 'PDBC', 'ovrs_cblc_qty': '7041',
             'ovrs_stck_evlu_amt': str(LIVE_PDBC), 'now_pric2': '20.02'}
_TWO_PAGE = [_bal_page([_ROW_BIL], nk='BALNK1', fk='BALFK1'), _bal_page([_ROW_PDBC])]

holdings, _t41, _x41 = with_api(dict(FULL, TTTS3012R=_TWO_PAGE), lambda api: api.get_balance())
check("둘째 쪽 보유도 읽는다", holdings == {'BIL': 1534, 'PDBC': 7041}, str(holdings))

_asked41 = []


def _watch_cursor(path, tr_id, params, retries=3, tr_cont='', cont=None):
    if tr_id == 'TTTS3012R':
        _asked41.append((tr_cont, params.get('CTX_AREA_FK200'), params.get('CTX_AREA_NK200')))
    return fake_get(dict(FULL, TTTS3012R=_TWO_PAGE))(path, tr_id, params, retries, tr_cont, cont)


_old41, _oldlog41 = ex._get, ex.log
try:
    ex._get, ex.log = _watch_cursor, lambda *a, **k: None
    ex.KISAPI(dry_run=False).get_balance()
finally:
    ex._get, ex.log = _old41, _oldlog41
check("첫 요청은 커서가 비고 tr_cont 도 없다",
      len(_asked41) >= 1 and _asked41[0] == ('', '', ''), str(_asked41))
check("둘째 요청은 tr_cont=N 과 직전 커서 둘을 싣는다",
      len(_asked41) >= 2 and _asked41[1] == ('N', 'BALFK1', 'BALNK1'), str(_asked41))

# 보유 한 종목이 둘째 쪽에 있는데 첫 쪽만 읽으면 "안 들고 있다" 가 돼서 또 산다.
# 전량 읽으면 목표와 같아 주문이 없다.
orders41, _a41, _s41 = _with_api(RealBalanceAPI, dict(FULL, TTTS3012R=_TWO_PAGE),
                                 _state(), _signal())
check("다 읽으면 이미 들고 있는 종목을 또 사지 않는다", orders41 == [], str(orders41))

for _name, _pages in (
        ('"더 있다" 면서 커서가 없다', [_bal_page([_ROW_BIL], nk='', tr_cont='F')]),
        ('연속 표시를 모르겠다', [_bal_page([_ROW_BIL], nk='', tr_cont='')]),
        ('커서가 그대로다', [_bal_page([_ROW_BIL], nk='SAME'), _bal_page([_ROW_PDBC], nk='SAME')]),
        ('둘째 쪽 조회가 실패한다', [_bal_page([_ROW_BIL], nk='NK1'), {}]),
        ('수량이 음수다', [_bal_page([dict(_ROW_BIL, ovrs_cblc_qty='-5')])]),
        ('수량이 소수다', [_bal_page([dict(_ROW_BIL, ovrs_cblc_qty='2.5')])])):
    check(f"'전부 읽었다' 로 통과하지 않는다 — {_name}",
          raises_unavailable(dict(FULL, TTTS3012R=_pages)))

print("\n[42] 인증 준비가 오래 걸리면 그 계좌값으로 주문하지 않는다")
# 나이를 재고 나서 토큰을 받으면 그 대기가 검사 뒤에 들어가 만료된 값으로 주문이 나간다.
# 느린 쪽(인증)이 검사 앞에 와야 한다.


class _SlowAuth(RealBalanceAPI):
    def ensure_auth(self):
        _CLOCK['t'] += 90.0


_CLOCK['t'] = 0.0
_st42 = _state_bil_only()
_st42['rebalancing_needed'] = True
orders42, alerts42, saved42 = _with_api(_SlowAuth, FULL, _st42, _signal(),
                                        clock=lambda: _CLOCK['t'])
check("인증에 90초가 걸리면 주문을 안 낸다", orders42 == [], str(orders42))
check("낡았다고 알린다", any('계좌값이 90초' in a for a in alerts42), str(alerts42))
check("완료로 안 닫는다",
      saved42 is not None and saved42.get('rebalancing_needed') is True, str(saved42))

print("\n[43] 값을 매길 수 없으면 완료로 닫지 않는다")
# 가격이 없는 종목을 편차 0 으로 세면, 안 산 종목도 "목표 달성" 으로 읽힌다.


class _UnpricedExtra(RealBalanceAPI):
    """목표에도 없고 값도 안 매겨지는 보유가 남은 계좌 — 상장폐지·호가 없음 같은 자리.

    목표 안쪽 종목은 평소대로 값이 매겨져서, 그 한 종목만 아니면 이 회차는 완료로 닫힌다.
    """

    def get_balance(self):
        h, t, x = super().get_balance()
        # 주문 뒤 조회에서만 보인다. 주문 전부터 보이면 가격 결측 가드가 그 회차를 먼저
        # 멈춰서, 완료 판정이 그 보유를 어떻게 세는지가 안 드러난다.
        if COUNTS.get('get_balance', 0) >= 2:
            h['XYZ'] = 5
        return h, t, x

    def get_current_price(self, ticker: str, fresh_only=False) -> float:
        return (0.0 if ticker == 'XYZ'
                else super().get_current_price(ticker, fresh_only=fresh_only))


_st43 = _state()
_st43['rebalancing_needed'] = True
orders43, alerts43, saved43 = _with_api(_UnpricedExtra, FULL, _st43, _signal())
check("값 못 매기는 보유가 남으면 완료로 안 닫는다",
      saved43 is not None and saved43.get('rebalancing_needed') is True, str(saved43))

# 그 보유가 없으면 같은 회차가 완료로 닫힌다 — 빈 비교가 아니다.
_st43b = _state()
_st43b['rebalancing_needed'] = True
_o43b, _a43b, saved43b = _with_api(RealBalanceAPI, FULL, _st43b, _signal())
check("그 보유가 없으면 완료로 닫는다",
      saved43b is not None and saved43b.get('rebalancing_needed') is False, str(saved43b))


print("\n[44] 전송 경계에서 다시 잰다 — 헤더를 만드는 동안 늙으면 안 보낸다")
# 진짜 place_order → _post → _headers 경로를 그대로 돌린다. 바꾸는 것은 셋뿐이다 —
# 통신(requests), 토큰 발급(_get_token), 시계. 나이 검사가 _headers 앞에 있으면 이
# 시험이 통과하지 않는다. 토큰 발급은 파일 잠금을 기다려 분 단위로 길어질 수 있고,
# 그 대기가 검사와 실제 전송 사이에 들어가는 것이 막으려는 자리다.


class _FakeRequests:
    """requests 대역. post 가 불린 횟수만 센다."""

    def __init__(self, advance, clock):
        self.sent = []
        self._advance = advance
        self._clock = clock

    def post(self, url, headers=None, json=None, timeout=None):
        self.sent.append(json)
        return _FakeOrderResp()


class _FakeOrderResp:
    status_code = 200

    def json(self):
        return {'rt_cd': '0', 'output': {'ODNO': '777'}}


def run_boundary(auth_delay, header_delay):
    """인증 준비에 auth_delay 초, 그 뒤 헤더 생성에 header_delay 초가 걸린다.

    둘을 나눠 재는 것이 이 시험의 핵심이다. 한 값으로 묶으면 나이 검사를 `_headers`
    앞에 둔 잘못된 구현도 통과한다 — 인증 준비에서 이미 늙어 있어 검사가 어디에
    있든 걸리기 때문이다. 인증 준비를 0초로 두면 검사 시점의 나이는 헤더 생성에서만
    늘어나므로, 검사가 헤더 뒤에 있을 때만 걸린다.

    돌려주는 것은 (place_order 결과, 나간 전문 수, 토큰 발급 호출 수).
    발급 호출 수를 같이 보는 이유는 헤더 경로가 실제로 다시 발급을 탔는지 확인하기
    위해서다 — 한 번만 불렸으면 지연을 헤더 쪽에 넣은 적이 없는 것이다.
    """
    clock = {'t': 0.0}
    bal_at = 0.0
    calls = {'n': 0}
    fake = _FakeRequests(0.0, clock)

    def _tok():
        calls['n'] += 1
        clock['t'] += auth_delay if calls['n'] == 1 else header_delay
        return 'TOK'

    old = {k: getattr(ex, k) for k in ('requests', '_get_token', 'log', 'time')}
    try:
        ex.requests = fake
        ex._get_token = _tok
        ex.log = lambda *a, **k: None
        ex.time = types.SimpleNamespace(sleep=lambda *_a: None, time=lambda: 0.0,
                                        monotonic=lambda: clock['t'])
        api = _KISAPI_REAL(dry_run=False)
        api.ensure_auth()
        res = api.place_order(
            'BIL', 10, 91.60, 'buy',
            pre_send=lambda: (clock['t'] - bal_at) <= ex.BALANCE_MAX_AGE_SEC)
        return res, len(fake.sent), calls['n']
    finally:
        for k, v in old.items():
            setattr(ex, k, v)


# 인증 준비는 즉시 끝나고 헤더 생성에서만 90초가 흐른다 — 검사가 헤더 뒤에 있어야 걸린다.
_res44, _sent44, _tok44 = run_boundary(0.0, 90.0)
check("헤더 경로가 토큰을 다시 받는다 (지연을 그 자리에 넣었다)", _tok44 >= 2, str(_tok44))
check("인증 준비는 즉시 끝났는데 헤더에서 90초가 지나면 전송 0건", _sent44 == 0, str(_sent44))
check("그 주문은 'stale' 이다 (접수 여부 불명이 아니다)", _res44 == 'stale', str(_res44))

# 양성 대조 — 같은 경로에서 헤더 지연이 짧으면 전문이 나간다. 빈 비교가 아니다.
_res44b, _sent44b, _ = run_boundary(0.0, 1.0)
check("헤더 지연이 짧으면 전문이 나간다", _sent44b == 1, str(_sent44b))
check("그 주문은 접수된다", _res44b == 'ok', str(_res44b))

# 반대쪽 — 인증 준비에서만 늙은 경우도 안 보낸다. 이 칸은 검사가 헤더 앞에 있어도
# 통과하므로 위 칸을 대신하지 못한다. 둘을 같이 둬야 "어디서 늙든 안 보낸다" 가 된다.
_res44c, _sent44c, _ = run_boundary(90.0, 0.0)
check("인증 준비에서만 늙어도 전송 0건", _sent44c == 0, str(_sent44c))
check("그 주문도 'stale' 이다", _res44c == 'stale', str(_res44c))

# 인증 준비가 실패하면 그 주문을 아예 안 낸다 — 경고만 찍고 넘어가면 뒤의 전송 경로가
# 인증을 다시 기다리고, 그 대기가 나이 검사 뒤에 들어간다.


class _AuthFails(RealBalanceAPI):
    def ensure_auth(self):
        raise RuntimeError('토큰 발급 거부')


_st44 = _state()
_st44['rebalancing_needed'] = True
orders44, alerts44, saved44 = _with_api(_AuthFails, FULL, _st44, _signal())
check("인증 준비가 실패하면 주문 0건", orders44 == [], str(orders44))
check("미전송이라고 알린다", any('인증 준비가 실패' in a for a in alerts44), str(alerts44))
check("접수 여부 불명으로 알리지 않는다",
      not any('접수 여부' in a for a in alerts44), str(alerts44))
check("완료로 안 닫는다",
      saved44 is not None and saved44.get('rebalancing_needed') is True, str(saved44))

# 거절 뒤 재시도 사이에 늙으면 둘째 전송을 안 한다.


class _RejectThenAge(RealBalanceAPI):
    """첫 주문이 거절되고, 재시도를 기다리는 동안 계좌값이 늙는다."""

    def place_order(self, ticker, qty, price, side='buy', pre_send=None):
        if pre_send is not None and not pre_send():
            return 'stale'
        self.orders.append((side, ticker, qty))
        _CLOCK['t'] += 90.0
        return 'rejected'


_CLOCK['t'] = 0.0
_st44b = _state_bil_only()
_st44b['rebalancing_needed'] = True
orders44b, alerts44b, saved44b = _with_api(_RejectThenAge, FULL, _st44b, _signal(),
                                           clock=lambda: _CLOCK['t'])
check("재시도 사이에 늙으면 같은 종목을 다시 안 보낸다",
      len(orders44b) == len(set(orders44b)) and len(orders44b) <= 2, str(orders44b))
check("멈췄다고 알린다", any('계좌값이' in a for a in alerts44b), str(alerts44b))

print("\n[45] 완료 평가는 주문 뒤 가격으로 한다 — 주문 전 캐시와 섞지 않는다")
# 보유 수량과 총자산은 주문 뒤 값인데 가격만 주문 전 값이면, 그 사이의 작은 변동이
# 허용오차 경계에서 거짓 완료를 만든다. 목표 BIL 1.0(버퍼 7% → 실효 0.93), 보유 10주,
# 총자산 <금액>. 주문 전 가격 92.1 로 세면 0.921 로 허용오차 안, 평가 시점 가격 91.9 로
# 세면 0.919 로 밖이다.


class _PriceMoves(RealBalanceAPI):
    """잔고는 고정, 가격만 주문 뒤에 내려간 계좌."""

    eval_price = 91.90

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        return {'BIL': 10}, 1000.0, FX

    def get_current_price(self, ticker, fresh_only=False):
        # 완료 평가는 주문 뒤 잔고 조회 다음에 돈다. 그 뒤의 조회만 내린 값을 준다.
        if ticker == 'BIL' and COUNTS.get('get_balance', 0) >= 2:
            return type(self).eval_price
        return 92.10


_st45 = _state_bil_only()
_st45['rebalancing_needed'] = True
orders45, alerts45, saved45 = _with_api(_PriceMoves, FULL, _st45, _signal())
check("주문은 없다 (가격 시점만 보는 칸이다)", orders45 == [], str(orders45))
check("평가 가격으로 재니 미달 — 완료로 안 닫는다",
      saved45 is not None and saved45.get('rebalancing_needed') is True, str(saved45))


class _PriceSteady(_PriceMoves):
    """같은 회차인데 가격이 안 움직였다 — 그러면 완료로 닫힌다. 빈 비교가 아니다."""
    eval_price = 92.10


_st45b = _state_bil_only()
_st45b['rebalancing_needed'] = True
_o45b, _a45b, saved45b = _with_api(_PriceSteady, FULL, _st45b, _signal())
check("가격이 그대로면 같은 회차가 완료로 닫힌다",
      saved45b is not None and saved45b.get('rebalancing_needed') is False, str(saved45b))

# 유한하지 않은 평가 가격. 양수 검사만 하면 무한대가 가격 행세를 하고, 보유 수량과
# 곱해진 비중이 숫자가 아닌 값이 된다. 최댓값 비교는 그 값을 그냥 통과시켜서(비교가
# 거짓이라) 편차가 0 으로 남고 회차가 거짓 완료된다. 바로 위 92.10 칸이 양성 대조다.
for _label, _bad in (('무한대', float('inf')), ('숫자 아님', float('nan')),
                     ('음의 무한대', float('-inf'))):
    _cls = type('_PriceBad', (_PriceMoves,), {'eval_price': _bad})
    _stb = _state_bil_only()
    _stb['rebalancing_needed'] = True
    _ob, _ab, _savedb = _with_api(_cls, FULL, _stb, _signal())
    check(f"평가 가격이 {_label} 이면 완료로 안 닫는다",
          _savedb is not None and _savedb.get('rebalancing_needed') is True, str(_savedb))
    check(f"그 상태를 평가 보류로 적는다 — {_label}",
          _savedb is not None and _savedb.get('eval_pending', {}).get('tickers') == ['BIL'],
          str(_savedb.get('eval_pending') if _savedb else None))


class _OrderPriceInf(RealBalanceAPI):
    """주문 산출에 쓰는 가격이 무한대로 온다 — 수량 계산이 깨진다."""

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        return {'BIL': 10}, 1000.0, FX

    def get_current_price(self, ticker, fresh_only=False):
        return float('inf')


_st45i = _state_bil_only()
_st45i['rebalancing_needed'] = True
_o45i, _a45i, _saved45i = _with_api(_OrderPriceInf, FULL, _st45i, _signal())
check("주문 산출 가격이 무한대면 주문을 안 낸다", _o45i == [], str(_o45i))
check("그 사실을 알린다", any('현재가' in a for a in _a45i), str(_a45i))

print("\n[47] 평가 가격만 없는 상태는 따로 적고, 돌아오면 푼다")
# 주문 실패·미체결 불확실과 다른 상태다. 그 둘은 실제로 뭔가 틀어진 것이고, 이건 들고
# 있는 것은 목표대로인데 값을 매길 길이 없는 것이다. 안 가르면 가격이 안 오는 동안
# 종목 교체(드리프트 판정)가 통째로 멈추고 그 사실이 아무 데도 안 남는다.
_alerts47 = []
_old_tg47 = ex.send_telegram
_old_dt47 = ex.datetime


class _StepClock(_dt.datetime):
    """부를 때마다 1분씩 가는 시계. 처음 본 시각이 그대로인지를 실제로 비교하려면
    회차마다 다른 시각이 들어와야 한다 — 같은 초에 다 적히면 그 비교가 늘 참이다."""
    _n = [0]

    @classmethod
    def now(cls, tz=None):
        cls._n[0] += 1
        return _dt.datetime(2026, 10, 21, 23, 35) + _dt.timedelta(minutes=cls._n[0])


_KEY47 = 'BIL:1.0'
try:
    ex.send_telegram = lambda m: _alerts47.append(m)
    ex.datetime = _StepClock
    _s47 = {}
    ex._note_eval_pending(_s47, ['BIL'], 0.001, drift_ok=True, target_key=_KEY47)
    check("첫 회차를 기록한다",
          _s47['eval_pending']['rounds'] == 1 and _s47['eval_pending']['tickers'] == ['BIL'],
          str(_s47))
    check("원인도 같이 적는다", '가격' in _s47['eval_pending']['reason'],
          str(_s47['eval_pending']))
    check("그때 잰 목표를 같이 적는다", _s47['eval_pending'].get('target_key') == _KEY47,
          str(_s47['eval_pending']))
    check("첫 발생을 알린다", len(_alerts47) == 1, str(_alerts47))
    _first47 = _s47['eval_pending']['first']
    ex._note_eval_pending(_s47, ['BIL'], 0.001, drift_ok=True, target_key=_KEY47)
    check("둘째 회차는 세기만 한다", _s47['eval_pending']['rounds'] == 2 and len(_alerts47) == 1,
          f"{_s47['eval_pending']['rounds']} {len(_alerts47)}")
    # 시각이 실제로 흘렀는지 먼저 단언한다 — 안 흐르면 아래 비교가 빈 검사가 된다.
    check("회차마다 마지막 시각이 갱신된다", _s47['eval_pending']['last'] != _first47,
          str(_s47['eval_pending']))
    check("처음 본 시각은 안 바뀐다", _s47['eval_pending']['first'] == _first47,
          str(_s47['eval_pending']))
    ex._note_eval_pending(_s47, ['BIL'], 0.001, drift_ok=True, target_key=_KEY47)
    check("지속 발생을 다시 알린다 (3회차)",
          _s47['eval_pending']['rounds'] == 3 and len(_alerts47) == 2, str(_alerts47))
    # 드리프트 판정은 이 상태에서 열린다 — 아니면 가격이 안 오는 동안 종목 교체가 멈춘다.
    # 단 그때 잰 목표가 지금도 같을 때만이다.
    _gate = dict(_s47, rebalancing_needed=True)
    check("평가 보류 회차에도 드리프트 판정을 본다",
          ex._drift_gate_open(_gate, _KEY47) is True, str(_gate.get('eval_pending')))
    check("그때와 목표가 다르면 안 본다",
          ex._drift_gate_open(_gate, 'BIL:0.5|PDBC:0.5') is False, str(_gate.get('eval_pending')))
    check("이번 회차에 목표가 바뀌었으면 안 본다",
          ex._drift_gate_open(_gate, _KEY47, target_changed=True) is False,
          str(_gate.get('eval_pending')))
    check("진짜 미달이면 드리프트 판정을 안 본다",
          ex._drift_gate_open({'rebalancing_needed': True}) is False, 'rebalancing_needed only')
    check("완료 회차는 평소대로 본다", ex._drift_gate_open({}) is True, 'empty state')
    # 집행이 안 끝난 회차의 기록은 자격을 못 가진다 — 기록은 남고 문만 닫힌다.
    _s47b = {}
    ex._note_eval_pending(_s47b, ['BIL'], 0.001, drift_ok=False)
    check("집행 미완료 회차도 기록은 남긴다",
          _s47b['eval_pending']['rounds'] == 1 and _s47b['eval_pending']['drift_ok'] is False,
          str(_s47b))
    check("그 회차는 따로 안 알린다 (완료 판정이 이미 알린다)", len(_alerts47) == 2,
          str(_alerts47))
    check("그 기록으로는 드리프트를 안 연다",
          ex._drift_gate_open(dict(_s47b, rebalancing_needed=True), _KEY47) is False,
          str(_s47b['eval_pending']))
    # 자격만 뺀다 — 몇 회차째인지는 남아야 지속 알림이 산다.
    ex._revoke_drift_ok(_s47, '시험')
    check("자격을 빼도 회차 기록은 남는다",
          _s47['eval_pending']['rounds'] == 3 and _s47['eval_pending']['tickers'] == ['BIL'],
          str(_s47['eval_pending']))
    check("자격만 빠진다", _s47['eval_pending']['drift_ok'] is False
          and _s47['eval_pending'].get('revoked') == '시험', str(_s47['eval_pending']))
    check("자격이 빠지면 드리프트를 안 연다",
          ex._drift_gate_open(dict(_s47, rebalancing_needed=True), _KEY47) is False,
          str(_s47['eval_pending']))
    # 가격이 돌아오면 그 자리에서 푼다 — 안 풀면 한 번 걸린 계좌가 영영 그 상태로 남는다.
    ex._note_eval_pending(_s47, [], 0.0)
    check("가격이 돌아오면 기록을 지운다", 'eval_pending' not in _s47, str(_s47))
    check("풀릴 때 따로 안 알린다", len(_alerts47) == 2, str(_alerts47))
    # 가격이 여전히 없는데 자격만 잃은 회차가 기록을 지우면, 다음 회차가 "첫 발생" 으로
    # 다시 세어져 지속 알림이 영영 안 온다. 그 거짓 복구를 막는다.
    _s47c = {}
    ex._note_eval_pending(_s47c, ['BIL'], 0.001, drift_ok=True, target_key=_KEY47)
    ex._note_eval_pending(_s47c, ['BIL'], 0.001, drift_ok=False)
    check("자격을 잃어도 회차는 이어 센다", _s47c['eval_pending']['rounds'] == 2,
          str(_s47c['eval_pending']))
finally:
    ex.send_telegram = _old_tg47
    ex.datetime = _old_dt47

# 회차를 통째로 돌려도 같다 — 막혔다가 돌아오면 완료로 닫히고 기록이 사라진다.
_st47r = _state_bil_only()
_st47r['rebalancing_needed'] = True
_o47r, _a47r, _saved47r = _with_api(type('_PriceNan', (_PriceMoves,),
                                        {'eval_price': float('nan')}),
                                   FULL, _st47r, _signal())
check("막힌 회차는 평가 보류로 남는다",
      _saved47r is not None and _saved47r.get('eval_pending', {}).get('rounds') == 1,
      str(_saved47r.get('eval_pending') if _saved47r else None))
_st47s = dict(_saved47r) if _saved47r else {}
_st47s['rebalancing_needed'] = True
_o47s, _a47s, _saved47s = _with_api(_PriceSteady, FULL, _st47s, _signal())
check("가격이 돌아온 회차는 완료로 닫는다",
      _saved47s is not None and _saved47s.get('rebalancing_needed') is False, str(_saved47s))
check("그 회차에 평가 보류 기록이 사라진다",
      _saved47s is not None and 'eval_pending' not in _saved47s,
      str(_saved47s.get('eval_pending') if _saved47s else None))


class _EvalPriceRaises(_PriceMoves):
    """평가 시점 가격 조회가 예외를 던진다 — 0 으로 세지 말고 완료를 미뤄야 한다."""

    def get_current_price(self, ticker, fresh_only=False):
        if ticker == 'BIL' and COUNTS.get('get_balance', 0) >= 2:
            raise RuntimeError('호가 조회 실패')
        return 92.10


_st45c = _state_bil_only()
_st45c['rebalancing_needed'] = True
_o45c, _a45c, saved45c = _with_api(_EvalPriceRaises, FULL, _st45c, _signal())
check("평가 가격을 못 받으면 완료로 안 닫는다",
      saved45c is not None and saved45c.get('rebalancing_needed') is True, str(saved45c))

# ── 여기부터는 가격 경로도 진짜를 쓴다. 가짜로 바꾸는 것은 통신(_get) 하나다.
# 위 칸들은 get_current_price 를 통째로 대역으로 바꾸므로, 진짜 구현의 fallback
# 사다리(실시간 시세 → 잔고 캐시 → 일봉 종가 → Yahoo 종가)가 완료 평가에서 어디까지
# 허용되는지를 못 본다. 실제 결함이 거기 있었다 — 주문 뒤 조회가 가격을 빠뜨리면
# 주문 전 캐시가 그대로 남아 평가에 쓰였다.


class _RealPriceAPI(RealBalanceAPI):
    """잔고·가격 둘 다 진짜 구현. 주문만 가로챈다."""

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        return _KISAPI_REAL.get_balance(self)

    def get_current_price(self, ticker, fresh_only=False):
        return _KISAPI_REAL.get_current_price(self, ticker, fresh_only=fresh_only)


def _price_table(second_price):
    """첫 계좌 조회는 92.10 을 같이 주고, 주문 뒤 조회는 second_price 를 준다.

    second_price=None 이면 그 조회에 가격 칸이 아예 없다(실계좌에서 종종 빈다).
    실시간 시세(HHDFS76200200)와 일봉 종가(HHDFS76240000)는 표에 없어서 둘 다
    실패한다 — 그래야 평가가 잔고 캐시 하나에 걸리고, 그 캐시의 세대를 보는지가
    드러난다.
    """
    calls = {'n': 0}

    def _bal():
        calls['n'] += 1
        px = 92.10 if calls['n'] == 1 else second_price
        row = {'ovrs_pdno': 'BIL', 'ovrs_cblc_qty': '10',
               'ovrs_stck_evlu_amt': '921.00'}
        if px is not None:
            row['now_pric2'] = f'{px:.2f}'
        return {'rt_cd': '0', 'output1': [row]}

    return {'TTTS3012R': _bal, 'CTRP6504R': ok_present(),
            'CTRP6548R': ok_account(tot=1000.0 * FX), 'TTTC2101R': ok_margin()}


_st45d = _state_bil_only()
_st45d['rebalancing_needed'] = True
_o45d, _a45d, saved45d = _with_api(_RealPriceAPI, _price_table(None), _st45d, _signal())
check("주문 뒤 조회가 가격을 빠뜨리면 주문 전 캐시로 완료를 닫지 않는다",
      saved45d is not None and saved45d.get('rebalancing_needed') is True, str(saved45d))
check("그 회차도 주문은 없다 (가격 시점만 보는 칸이다)", _o45d == [], str(_o45d))

# 양성 대조 — 주문 뒤 조회가 같은 가격을 같이 주면 그 값으로 완료를 닫는다.
_st45e = _state_bil_only()
_st45e['rebalancing_needed'] = True
_o45e, _a45e, saved45e = _with_api(_RealPriceAPI, _price_table(92.10), _st45e, _signal())
check("주문 뒤 조회가 가격을 같이 주면 완료로 닫는다",
      saved45e is not None and saved45e.get('rebalancing_needed') is False, str(saved45e))

# 주문 뒤 조회가 내려간 가격을 주면 허용오차 밖이라 완료가 아니다 — 가격 출처는
# 신선한데 값 자체가 미달인 칸이다. 위 칸(출처가 낡은 것)과 사유가 다르다.
_st45f = _state_bil_only()
_st45f['rebalancing_needed'] = True
_o45f, _a45f, saved45f = _with_api(_RealPriceAPI, _price_table(91.90), _st45f, _signal())
check("신선한 가격이 미달이면 완료로 안 닫는다",
      saved45f is not None and saved45f.get('rebalancing_needed') is True, str(saved45f))

print("\n[46] 취소는 원주문이 실제로 있던 거래소로 보낸다")
# 응답이 거래소를 알려주면 그 값을 쓴다. 우리가 주문을 보낼 때 쓰는 표(EXCHANGE_MAP)나
# 조회 순서로 정하면, 그 표가 틀린 종목의 취소가 다른 거래소로 나가 성공으로 돌아오고
# 원주문은 살아남는다.
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=_nccs([_row(odno='X1', pdno='BIL',
                                                          excg='AMEX')])),
                        {'rt_cd': '0'})
check("응답이 말한 거래소로 취소한다",
      [b.get('OVRS_EXCG_CD') for b in _CANCEL_BODIES] == ['AMEX'],
      str([b.get('OVRS_EXCG_CD') for b in _CANCEL_BODIES]))
check("그 한 건만 센다", res == {'queried': True, 'found': 1, 'failed': 0}, str(res))

# 같은 주문번호가 응답상 서로 다른 거래소면 어느 쪽이 원주문인지 모른다 — 멈춘다.
res, posts = run_cancel({'NASD': _nccs([_row(odno='D1', excg='NASD')]),
                         'NYSE': _nccs([_row(odno='D1', excg='NYSE')]),
                         'AMEX': _nccs([])},
                        {'rt_cd': '0'})
check("거래소가 갈리면 조회 실패로 돌려준다", res.get('queried') is False, str(res))
check("그 상태로 취소를 안 쏜다", posts == [], str(posts))

# 같은 쪽 안에서 갈려도 같다. 중복 제거를 거래소 해석보다 앞에 두면 둘째 행이 먼저
# 사라져서 이 충돌이 통째로 안 보이고, 한 거래소만 적힌 것처럼 취소가 나간다.
res, posts = run_cancel(dict(_ALL_EMPTY,
                             NASD=_nccs([_row(odno='S1', excg='NASD'),
                                         _row(odno='S1', excg='NYSE')])),
                        {'rt_cd': '0'})
check("한 쪽 안에서 갈려도 조회 실패로 돌려준다", res.get('queried') is False, str(res))
check("그 상태로 취소를 안 쏜다 (같은 쪽)", posts == [], str(posts))

# 다음 쪽으로 넘어가서 갈려도 같다. 연속조회는 쪽마다 따로 들어오므로, 쪽 경계에서
# 중복 제거가 먼저 돌면 같은 결함이 재현된다.
res, posts = run_cancel(dict(_ALL_EMPTY,
                             NASD=[_nccs([_row(odno='P1', excg='NASD')], nk='CUR1'),
                                   _nccs([_row(odno='P1', excg='NYSE')])]),
                        {'rt_cd': '0'})
check("다음 쪽에서 갈려도 조회 실패로 돌려준다", res.get('queried') is False, str(res))
check("그 상태로 취소를 안 쏜다 (다음 쪽)", posts == [], str(posts))

# 같은 거래소로 두 번 보인 것은 중복 노출이다 — 그건 막지 않는다(양성 대조).
res, posts = run_cancel(dict(_ALL_EMPTY,
                             NASD=_nccs([_row(odno='S2', excg='NASD'),
                                         _row(odno='S2', excg='NASD')])),
                        {'rt_cd': '0'})
check("같은 거래소 중복은 한 번만 센다",
      res.get('queried') is True and res.get('found') == 1, str(res))
check("그 한 건을 취소한다", posts == ['S2'], str(posts))

# 응답이 거래소를 안 주면 물어본 코드로 메우지 않는다. 이 조회는 한 코드로 물어도
# 미국 전체를 돌려주므로(공식 예제), NYSE 로 물어 받은 행이 NYSE 주문이라는 증거가
# 없다. 한 곳에서만 보여도 마찬가지다 — 셋 다 같은 목록을 돌려주면 한 곳에만 보이는
# 것 자체가 그 거래소 것이라는 뜻이 아니다.
res, posts = run_cancel(dict(_ALL_EMPTY, NYSE=_nccs([_row(odno='N1', pdno='BIL', excg=None)])),
                        {'rt_cd': '0'})
check("응답에 거래소가 없으면 취소를 안 낸다", res.get('queried') is False, str(res))
check("그 상태로 취소를 안 쏜다 — 물어본 코드로 메우지 않는다", posts == [], str(posts))

# 우리가 그 주문을 보낼 때 쓴 코드는 근거다. 응답이 거래소를 안 줘도 그 기록으로 확정한다.
_api_sent = _KISAPI_REAL(dry_run=False)
_api_sent._order_excg['M1'] = 'AMEX'
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=_nccs([_row(odno='M1', pdno='GLD', excg=None)])),
                        {'rt_cd': '0'}, api=_api_sent)
check("우리가 보낸 거래소 기록으로 확정해 취소한다",
      res.get('found') == 1 and [b.get('OVRS_EXCG_CD') for b in _CANCEL_BODIES] == ['AMEX'],
      f"{res} {[b.get('OVRS_EXCG_CD') for b in _CANCEL_BODIES]}")

# 응답이 말한 거래소와 우리 기록이 다르면 응답을 따른다 — 그 주문이 지금 어디 있는지는
# 거래소 쪽 보고가 말한다. 우리 표가 틀렸을 때 취소가 엉뚱한 곳으로 가는 것을 막는다.
_api_mix = _KISAPI_REAL(dry_run=False)
_api_mix._order_excg['M2'] = 'NASD'
res, posts = run_cancel(dict(_ALL_EMPTY, NASD=_nccs([_row(odno='M2', pdno='GLD', excg='AMEX')])),
                        {'rt_cd': '0'}, api=_api_mix)
check("응답과 우리 기록이 다르면 응답을 따른다",
      [b.get('OVRS_EXCG_CD') for b in _CANCEL_BODIES] == ['AMEX'],
      str([b.get('OVRS_EXCG_CD') for b in _CANCEL_BODIES]))

# 한 행은 확정되고 다른 행이 미확정이면 확정된 것만 쏘지 않는다 — 회차 전체를 멈춘다.
# 미확정 주문이 살아 있는 채로 매수가 나가면 그 위에 얹힌다.
res, posts = run_cancel(dict(_ALL_EMPTY,
                             NASD=_nccs([_row(odno='K1', excg='NASD'),
                                         _row(odno='K2', excg=None)])),
                        {'rt_cd': '0'})
check("한 건이라도 미확정이면 전부 안 쏜다",
      res.get('queried') is False and posts == [], f'{res} {posts}')

# 모호함은 회차 시작 말고 다른 호출부에서도 회차를 멈춰야 한다. 매도 뒤 호출부가
# 그걸 통과시키면, 거래소를 못 가린 미체결이 남은 채로 매수가 나간다.


_C46 = {'n': 0}


class _AmbiguousAfterSells(RealBalanceAPI):
    """회차 시작의 취소 조회는 깨끗하고, 매도 뒤부터 거래소가 모호해진다."""

    def cancel_all_pending(self):
        _C46['n'] += 1
        if _C46['n'] == 1:
            return {'queried': True, 'found': 0, 'failed': 0}
        return {'queried': False, 'found': 0, 'failed': 0}


_st46 = _state_bil_only()
_st46['rebalancing_needed'] = True
orders46, alerts46, saved46 = _with_api(_AmbiguousAfterSells, FULL, _st46, _signal())
# 빈 회차로 통과하면 아무것도 안 잡는다. 매도가 실제로 나갔고, 그래서 매도 뒤 취소
# 조회가 돌았고(둘째 호출), 상태가 저장됐다는 것까지 같이 단언한다.
check("매도가 실제로 나갔다", [o for o in orders46 if o[0] == 'sell'] != [], str(orders46))
check("매도 뒤 취소 조회가 돌았다", _C46['n'] >= 2, str(_C46['n']))
check("매도 뒤에 모호해지면 매수를 안 낸다",
      [o for o in orders46 if o[0] == 'buy'] == [],
      str(orders46))
check("상태를 저장한다 (주문은 나갔다)", saved46 is not None, str(saved46))
check("그 회차를 완료로 안 닫는다",
      saved46 is not None and saved46.get('rebalancing_needed') is True, str(saved46))

print("\n[53] 전송 거래소 기록은 주문에서 취소까지 한 줄로 이어진다")
# [46] 은 그 기록을 손으로 심어 취소 쪽만 봤다. 열째 토론이 요구한 것은 사슬 전체다 —
# place_order 가 실제로 적는가, 그 값이 취소 전문에 그대로 나가는가, 프로세스가 다시
# 뜨면(기록이 메모리라 사라진다) 어떻게 되는가.


def run_place_api(api, ticker, result, side='buy'):
    """진짜 place_order 를 통신만 가짜로 돌린다. (판정, 나간 주문 전문)

    위 [30] 의 run_place 와 달리 인스턴스를 밖에서 받는다 — 그 인스턴스에 남는
    전송 거래소 기록을 뒤의 취소까지 이어 보려는 것이다.
    """
    sent = []

    def _p(path, tr_id, body, retries=2, pre_send=None):
        sent.append(body)
        return result

    old_post, old_log = ex._post, ex.log
    try:
        ex._post, ex.log = _p, lambda *a, **k: None
        return api.place_order(ticker, 3, 91.60, side), sent
    finally:
        ex._post, ex.log = old_post, old_log


# BIL 은 EXCHANGE_MAP 에서 AMEX 다 — 기본값 NASD 와 다른 종목을 골라야 "표를 안 보고
# NASD 로 메운다" 와 구분된다.
_api53 = _KISAPI_REAL(dry_run=False)
_v53, _sent53 = run_place_api(_api53, 'BIL', {'rt_cd': '0', 'output': {'ODNO': 'E1'}})
check("주문이 접수됐다", _v53 == 'ok', _v53)
check("전송 전문의 거래소는 표가 정한 값이다",
      [b.get('OVRS_EXCG_CD') for b in _sent53] == ['AMEX'],
      str([b.get('OVRS_EXCG_CD') for b in _sent53]))
check("그 거래소를 주문번호에 묶어 적는다", _api53._order_excg.get('E1') == 'AMEX',
      str(_api53._order_excg))

# 같은 인스턴스로 취소까지 간다. 응답이 거래소를 안 줘도 이 기록으로 확정해 쏜다.
res53, posts53 = run_cancel(dict(_ALL_EMPTY,
                                 NASD=_nccs([_row(odno='E1', pdno='BIL', excg=None)])),
                            {'rt_cd': '0'}, api=_api53)
check("그 기록으로 취소가 나간다",
      res53.get('found') == 1 and posts53 == ['E1']
      and [b.get('OVRS_EXCG_CD') for b in _CANCEL_BODIES] == ['AMEX'],
      f'{res53} {posts53} {[b.get("OVRS_EXCG_CD") for b in _CANCEL_BODIES]}')

# 프로세스가 다시 뜨면 그 기록이 없다. 응답도 거래소를 안 주면 확정할 근거가 0 이다 —
# 물어본 코드나 EXCHANGE_MAP 으로 메우지 않고 회차를 멈춘다. 멈춘 채로 사람이 앱에서
# 처리할 때까지 매 회차 같은 자리에서 선다(토론이 risks 로 적은 장기 정지).
_api53b = _KISAPI_REAL(dry_run=False)
check("새 인스턴스에는 그 기록이 없다", _api53b._order_excg == {}, str(_api53b._order_excg))
res53b, posts53b = run_cancel(dict(_ALL_EMPTY,
                                   NASD=_nccs([_row(odno='E1', pdno='BIL', excg=None)])),
                              {'rt_cd': '0'}, api=_api53b)
check("재시작 뒤 근거가 없으면 취소를 안 쏜다",
      res53b.get('queried') is False and posts53b == [], f'{res53b} {posts53b}')

# 재시작이어도 응답이 거래소를 주면 그걸로 확정한다 — 기록 소실이 영구 정지의 유일한
# 조건이 아니라는 양성 대조.
_api53c = _KISAPI_REAL(dry_run=False)
res53c, posts53c = run_cancel(dict(_ALL_EMPTY,
                                   NASD=_nccs([_row(odno='E1', pdno='BIL', excg='AMEX')])),
                              {'rt_cd': '0'}, api=_api53c)
check("[양성] 재시작이어도 응답이 거래소를 주면 쏜다",
      res53c.get('found') == 1 and posts53c == ['E1'], f'{res53c} {posts53c}')

# 접수 여부를 모르거나 거절된 주문은 기록하지 않는다. 주문번호가 없는 것을 빈 문자열로
# 적어 두면, 나중에 같은 빈 번호의 행이 "확정된 거래소" 를 얻어 엉뚱한 취소가 나간다.
for _label, _res, _want in (('주문번호가 없다', {'rt_cd': '0', 'output': {}}, 'unknown'),
                            ('output 이 없다', {'rt_cd': '0'}, 'unknown'),
                            ('응답이 비었다', {}, 'unknown'),
                            ('거절됐다',
                             {'_error': {'status_code': 400, 'body': {'rt_cd': '1'}}},
                             'rejected')):
    _apix = _KISAPI_REAL(dry_run=False)
    _vx, _ = run_place_api(_apix, 'BIL', _res)
    check(f"{_label} → {_want} 로 보고 기록을 안 남긴다",
          _vx == _want and _apix._order_excg == {}, f'{_vx} {_apix._order_excg}')

print("\n[48] 지난 회차의 평가 보류 표식은 그때 목표가 지금도 같을 때만 문을 연다")
# 열째 토론 지적. 표식만 보고 문을 열면, 정규 앵커가 스냅샷 하나를 갈아탄 회차에
# 드리프트까지 발화해 리필이 나머지 스냅샷까지 덮는다 — 앵커 하나만 움직여야 하는
# 날에 셋이 다 움직인다.


def _run_refill_count(cls, state_obj, signal):
    """한 회차를 돌리고 refill_snaps_fresh 가 몇 번 불렸는지 같이 돌려준다."""
    calls = []
    _old = ex.refill_snaps_fresh
    try:
        ex.refill_snaps_fresh = lambda sig, st: (calls.append(1), _old(sig, st))[1]
        out = _with_api(cls, FULL, state_obj, signal)
    finally:
        ex.refill_snaps_fresh = _old
    return out + (calls,)


class _DriftBal(RealBalanceAPI):
    """보유가 목표(BIL)와 전혀 다른 계좌 — 드리프트가 확실히 임계를 넘는다."""

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        return {'PDBC': 40}, 1000.0, FX

    def get_current_price(self, ticker, fresh_only=False):
        return 20.0 if ticker == 'PDBC' else 92.10


def _pending48(target_key='BIL:1.0', drift_ok=True, rounds=1):
    return {'tickers': ['BIL'], 'reason': '평가 시점 가격을 못 받았다', 'rounds': rounds,
            'first': '2026-10-20T23:35:00', 'last': '2026-10-20T23:35:00',
            'max_diff_known': 0.001, 'drift_ok': drift_ok, 'target_key': target_key}


# 양성 대조 1 — 표식이 아예 없는 평범한 회차. 드리프트가 발화하고 리필이 돈다.
_st48a = _state_bil_only()
_st48a['rebalancing_needed'] = False
_o48a, _a48a, _s48a, _c48a = _run_refill_count(_DriftBal, _st48a, _signal())
check("[양성] 평범한 회차는 드리프트가 발화하고 리필이 돈다", _c48a == [1], str(_c48a))

# 양성 대조 2 — 표식이 있고 목표가 그대로면 문이 열린다(가격이 안 오는 동안 교체가
# 영영 멈추지 않게 하는 쪽). 이게 열려야 아래 닫힘이 빈 검사가 아니다.
_st48b = _state_bil_only()
_st48b['rebalancing_needed'] = True
_st48b['eval_pending'] = _pending48()
_o48b, _a48b, _s48b, _c48b = _run_refill_count(_DriftBal, _st48b, _signal())
check("[양성] 목표가 그대로면 평가 보류 회차도 리필이 돈다", _c48b == [1], str(_c48b))

# 본 검사 — 같은 상태에서 스냅샷 0 의 정규 앵커만 발화시킨다.
_st48c = _state_bil_only()
_st48c['rebalancing_needed'] = True
_st48c['eval_pending'] = _pending48()
# 세 예정일을 같은 폭으로 당긴다. snap0 만 오늘 이하가 되고 23일 위상은 그대로라,
# _resolve_anchor_schedule 이 시간표를 복구하지 않는다(위상이 깨지면 복구가 돌아
# 앵커가 안 터지고 대신 드리프트가 돌아서, 검사가 재려던 자리가 사라진다).
_TODAY48 = _FrozenDatetime._today
for _i, _snap in _st48c['snapshots'].items():
    _snap['next_rebal_date'] = (_TODAY48
                                + _dt.timedelta(days=int(_i) * ex.SNAP_STAGGER_DAYS)).isoformat()
_o48c, _a48c, _s48c, _c48c = _run_refill_count(_DriftBal, _st48c, _signal())
check("앵커가 목표를 바꾼 회차는 드리프트 리필을 안 부른다", _c48c == [], str(_c48c))
check("그 회차에 앵커는 실제로 발화했다",
      _s48c is not None
      and _s48c['snapshots']['0']['last_rebal_date'] == _TODAY48.isoformat(),
      str(_s48c['snapshots']['0'] if _s48c else None))
check("앵커가 안 건드린 스냅샷은 그대로다",
      _s48c is not None
      and _s48c['snapshots']['1']['picks'] == ['BIL']
      and _s48c['snapshots']['2']['picks'] == ['BIL'],
      str({k: v['picks'] for k, v in _s48c['snapshots'].items()} if _s48c else None))

# 카나리 전환도 같다 — 전환이 부르는 리필 하나로 끝나야 하고, 드리프트가 또 부르면 안 된다.
_st48d = _state_bil_only()
_st48d['rebalancing_needed'] = True
_st48d['prev_risk_on'] = True          # 신호는 risk_on=False → 이번 회차가 전환이다
_st48d['eval_pending'] = _pending48()
_o48d, _a48d, _s48d, _c48d = _run_refill_count(_DriftBal, _st48d, _signal())
check("카나리 전환은 리필을 한 번만 부른다 (드리프트가 또 안 부른다)", _c48d == [1], str(_c48d))
check("그 회차에 전환은 실제로 일어났다",
      _s48d is not None and _s48d.get('prev_risk_on') is False, str(_s48d))

# 표식에 적힌 목표가 지금 목표와 다르면(옛 상태를 물려받은 경우) 역시 안 연다.
_st48e = _state_bil_only()
_st48e['rebalancing_needed'] = True
_st48e['eval_pending'] = _pending48(target_key='SPY:0.5|QQQ:0.5')
_o48e, _a48e, _s48e, _c48e = _run_refill_count(_DriftBal, _st48e, _signal())
check("표식의 목표가 지금과 다르면 리필을 안 부른다", _c48e == [], str(_c48e))

print("\n[49] 집행이 불확실한 회차는 표식의 드리프트 자격을 잃는다")
# 기록(몇 회차째 가격이 없나)은 남기고 자격만 뺀다.


class _PostBalFails(RealBalanceAPI):
    """주문 뒤 잔고 조회만 실패한다 — 이번 보유를 재지 못한 회차."""

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        if COUNTS['get_balance'] >= 2:
            raise ex.BalanceUnavailable('주문 뒤 조회 실패')
        return {'PDBC': 40}, 1000.0, FX

    def get_current_price(self, ticker, fresh_only=False):
        return 20.0 if ticker == 'PDBC' else 92.10


_st49a = _state_bil_only()
_st49a['rebalancing_needed'] = True
_st49a['eval_pending'] = _pending48(rounds=2)
_o49a, _a49a, _s49a = _with_api(_PostBalFails, FULL, _st49a, _signal())
check("주문 뒤 잔고를 못 읽으면 자격을 뺀다",
      _s49a is not None and _s49a.get('eval_pending', {}).get('drift_ok') is False,
      str(_s49a.get('eval_pending') if _s49a else None))
check("그래도 회차 기록은 남는다",
      _s49a is not None and _s49a.get('eval_pending', {}).get('rounds') == 2
      and _s49a.get('eval_pending', {}).get('tickers') == ['BIL'],
      str(_s49a.get('eval_pending') if _s49a else None))

# 계좌값이 낡아 주문 없이 돌아간 회차도 같다.
_st49b = _state_bil_only()
_st49b['rebalancing_needed'] = True
_st49b['eval_pending'] = _pending48(rounds=2)
_o49b, _a49b, _s49b = _with_api(_DriftBal, FULL, _st49b, _signal(), clock=step_clock(1, 100.0))
check("계좌값이 낡아 돌아간 회차도 자격을 뺀다",
      _s49b is not None and _s49b.get('eval_pending', {}).get('drift_ok') is False,
      str(_s49b.get('eval_pending') if _s49b else None))
check("그 회차는 주문이 0건이다", _o49b == [], str(_o49b))

print("\n[50] 값을 못 매긴 자리가 '보유 미달' 이면 완료로도 자격으로도 안 센다")


class _LeftoverUnpriced(RealBalanceAPI):
    """주문 뒤에 목표 밖 보유(SPY)가 남았는데 그 평가 가격이 없다."""

    spy_eval = None    # None 이면 평가 가격 없음

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        if COUNTS['get_balance'] >= 2:
            return {'BIL': 10, 'SPY': 3}, 1000.0, FX
        return {'BIL': 10}, 1000.0, FX

    def get_current_price(self, ticker, fresh_only=False):
        if ticker == 'SPY' and COUNTS.get('get_balance', 0) >= 2:
            _v = type(self).spy_eval
            return 0.0 if _v is None else _v
        return 92.10


_st50a = _state_bil_only()
_st50a['rebalancing_needed'] = True
_o50a, _a50a, _s50a = _with_api(_LeftoverUnpriced, FULL, _st50a, _signal())
check("목표 밖 잔존의 값을 못 매기면 완료로 안 닫는다",
      _s50a is not None and _s50a.get('rebalancing_needed') is True, str(_s50a))
check("그 회차 표식은 드리프트 자격이 없다",
      _s50a is not None and _s50a.get('eval_pending', {}).get('drift_ok') is False,
      str(_s50a.get('eval_pending') if _s50a else None))

# 양성 대조 — 같은 잔존인데 값이 매겨지면 그 자리는 편차로 세어지고 표식이 안 남는다.
_o50b, _a50b, _s50b = _with_api(type('_LeftoverPriced', (_LeftoverUnpriced,),
                                     {'spy_eval': 400.0}),
                                FULL, dict(_state_bil_only(), rebalancing_needed=True), _signal())
check("[양성] 값이 매겨지면 평가 보류로 안 적는다",
      _s50b is not None and 'eval_pending' not in _s50b,
      str(_s50b.get('eval_pending') if _s50b else None))


class _ShrunkBuy(RealBalanceAPI):
    """매수 가능 수량이 1주뿐이라 축소 체결된 회차 + 평가 가격 결측."""

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        return {'BIL': 1}, 1000.0, FX

    def get_max_buy_qty(self, ticker, price) -> int:
        return 1

    def get_current_price(self, ticker, fresh_only=False):
        if COUNTS.get('get_balance', 0) >= 2:
            return 0.0      # 평가 시점 가격 없음
        return 92.10


_st50c = _state_bil_only()
_st50c['rebalancing_needed'] = True
_o50c, _a50c, _s50c = _with_api(_ShrunkBuy, FULL, _st50c, _signal())
check("축소 매수가 있었다", [o for o in _o50c if o[0] == 'buy' and o[2] == 1] != [], str(_o50c))
check("축소 매수 + 평가 결측은 완료로 안 닫는다",
      _s50c is not None and _s50c.get('rebalancing_needed') is True, str(_s50c))
check("그 회차 표식도 드리프트 자격이 없다",
      _s50c is not None and _s50c.get('eval_pending', {}).get('drift_ok') is False,
      str(_s50c.get('eval_pending') if _s50c else None))

print("\n[51] 유한한 값끼리의 곱·나눗셈이 넘치는 자리")
# 입력 하나하나는 유한한데 결과가 무한대·숫자 아님이 되는 경우다. 직접 넣은 inf 만
# 막으면 여기서 샌다 — 열째 토론이 댄 반례(보유 2주 × 가격 1e308).


class _EvalOverflow(_PriceMoves):
    """평가 가격이 유한한데 보유와 곱하면 넘친다."""
    eval_price = 1e308

    def get_balance(self):
        COUNTS['get_balance'] = COUNTS.get('get_balance', 0) + 1
        return {'BIL': 2}, 1000.0, FX


_st51a = _state_bil_only()
_st51a['rebalancing_needed'] = True
_o51a, _a51a, _s51a = _with_api(_EvalOverflow, FULL, _st51a, _signal())
check("보유 × 가격이 넘치면 완료로 안 닫는다",
      _s51a is not None and _s51a.get('rebalancing_needed') is True, str(_s51a))
check("그 회차는 드리프트 자격도 없다",
      _s51a is not None and _s51a.get('eval_pending', {}).get('drift_ok') is False,
      str(_s51a.get('eval_pending') if _s51a else None))

# 환산 총자산 — 총자산도 환율도 유한한 양수인데 몫이 넘친다.
_ovf = {'TTTS3012R': ok_balance(),
        'CTRP6504R': ok_present(exrt=1e-308),
        'CTRP6548R': ok_account(tot=1e308),
        'TTTC2101R': ok_margin(exrt=1e-308)}
check("환산 총자산이 넘치면 조회 실패로 돌린다",
      '정상 범위' in (raises_unavailable(_ovf) or ''), str(raises_unavailable(_ovf)))

# 매수 가능 수량 응답이 무한대 — int(inf) 는 OverflowError 다. 그 예외가 매수 루프
# 밖으로 나가면 주문을 낸 뒤의 확인과 상태 저장을 통째로 건너뛴다.
for _label, _resp in (('수량이 무한대', {'ovrs_max_ord_psbl_qty': 'inf'}),
                      ('수량이 숫자 아님', {'ovrs_max_ord_psbl_qty': 'nan'}),
                      ('금액이 무한대', {'frcr_ord_psbl_amt1': 'inf'}),
                      ('금액/가격이 넘친다', {'frcr_ord_psbl_amt1': '1e308'})):
    _qty_tbl = {'TTTS3007R': {'rt_cd': '0', 'output': _resp}}

    _px51 = 1e-308 if '넘친' in _label else 92.10
    _got = with_api_live(_qty_tbl, lambda api, _p=_px51: api.get_max_buy_qty('BIL', _p))
    check(f"매수 가능 수량이 비정상이면 0 으로 본다 — {_label}", _got == 0, str(_got))

_ok_qty = with_api_live({'TTTS3007R': {'rt_cd': '0', 'output': {'ovrs_max_ord_psbl_qty': '12'}}},
                        lambda api: api.get_max_buy_qty('BIL', 92.10))
check("[양성] 정상 응답은 그대로 읽는다", _ok_qty == 12, str(_ok_qty))

print("\n[52] 현금버퍼 값이 정상 범위를 벗어나면 주문 전에 멈춘다")
# 기본값으로 조용히 바꿔 쓰면 운영자가 정한 값과 다른 비중으로 돌고, 그대로 쓰면
# 목표 금액이 0 이나 음수가 되어 전량 매도가 된다. 둘 다 안 한다.
for _label, _bad in (('숫자가 아니다', 'abc'), ('1 이상이다', 1.0),
                     ('음수다', -0.1), ('무한대다', float('inf'))):
    _st52 = _state_bil_only()
    _st52['rebalancing_needed'] = True
    _st52['stock_cash_buffer'] = _bad
    _o52, _a52, _s52 = _with_api(_DriftBal, FULL, _st52, _signal())
    check(f"현금버퍼가 {_label} → 주문 0건", _o52 == [], str(_o52))
    check(f"현금버퍼가 {_label} → 상태를 안 바꾼다", _s52 is None, str(_s52))
    check(f"현금버퍼가 {_label} → 알린다",
          any('현금버퍼' in a for a in _a52), str(_a52))

_st52ok = _state_bil_only()
_st52ok['rebalancing_needed'] = True
_st52ok['stock_cash_buffer'] = 0.07
_o52ok, _a52ok, _s52ok = _with_api(_DriftBal, FULL, _st52ok, _signal())
check("[양성] 정상 버퍼면 회차가 돈다", _o52ok != [] and _s52ok is not None,
      f'{_o52ok} {_s52ok is not None}')

print("\n[54] 전송 전에 터진 실패는 접수 여부 불명이 아니다")

# 여덟째 토론(run-20261029T0*)이 risks 로 남긴 자리다. 토큰 발급(_headers)과 전송 직전
# 검사(pre_send) 자체가 예외를 던지면 요청은 선 위에 안 올라간다 — 미전송이 확정이다.
# 그걸 'unknown' 으로 올리면 아무것도 안 나간 회차가 통째로 멈춘다(그 회차 신규 주문
# 전면 중단 + 드리프트 자격 박탈). _post 가 'sent' 로 그 자리를 표시하고 place_order 가
# 그 표시만 'stale'(미전송 확정)로 받는다.


def post_raw(hdr_exc=None, pre=None, post_fn=None, retries=1):
    """진짜 _post 를 통신만 가짜로 돌린다."""
    old_h, old_p, old_sleep = ex._headers, ex.requests.post, ex.time.sleep

    def _h(tr_id):
        if hdr_exc is not None:
            raise hdr_exc
        return {}

    try:
        ex._headers = _h
        ex.requests.post = post_fn or (lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('연결 끊김')))
        ex.time.sleep = lambda *a: None
        return ex._post('/p', 'TR', {}, retries=retries, pre_send=pre)
    finally:
        ex._headers, ex.requests.post, ex.time.sleep = old_h, old_p, old_sleep


_r54a = post_raw(hdr_exc=RuntimeError('토큰 잠금 시간 초과'))
check("토큰 발급 실패는 미전송으로 표시된다",
      _r54a.get('_error', {}).get('sent') is False, str(_r54a))

_r54b = post_raw(pre=lambda: (_ for _ in ()).throw(RuntimeError('조회 터짐')))
check("전송 직전 검사의 예외도 미전송이다",
      _r54b.get('_error', {}).get('sent') is False, str(_r54b))

_r54c = post_raw()        # _headers 통과 뒤 post 가 터진다
check("[양성] 통신 중 끊김은 전송됐을 수 있다고 표시된다",
      _r54c.get('_error', {}).get('sent') is True, str(_r54c))


class _Resp:
    status_code = 200

    def json(self):
        return {'rt_cd': '1', 'msg1': '거절'}


_r54d = post_raw(post_fn=lambda *a, **k: _Resp())
check("응답을 받은 실패는 전송됐다고 표시된다",
      _r54d.get('_error', {}).get('sent') is True, str(_r54d))

# 재시도에서 한 번이라도 나갔으면 그 뒤 토큰 실패를 미전송으로 내리지 않는다 —
# 첫 회 요청을 서버가 이미 받아놨을 수 있어서, 내리면 중복 체결을 연다.
_calls54 = {'n': 0}


def _h_flaky(tr_id):
    _calls54['n'] += 1
    if _calls54['n'] >= 2:
        raise RuntimeError('두 번째 회차 토큰 실패')
    return {}


_old_h54, _old_p54, _old_s54 = ex._headers, ex.requests.post, ex.time.sleep
try:
    ex._headers = _h_flaky
    ex.requests.post = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('끊김'))
    ex.time.sleep = lambda *a: None
    _r54e = ex._post('/p', 'TR', {}, retries=2)
finally:
    ex._headers, ex.requests.post, ex.time.sleep = _old_h54, _old_p54, _old_s54
check("한 번이라도 나갔으면 뒤의 토큰 실패도 전송됨으로 남는다",
      _r54e.get('_error', {}).get('sent') is True, str(_r54e))

# place_order 가 그 표시를 어떻게 받는가.
for _label, _err, _want in (
        ('토큰 발급 실패', {'exception': "RuntimeError('x')", 'sent': False}, 'stale'),
        ('통신 끊김', {'exception': "RuntimeError('x')", 'sent': True}, 'unknown'),
        ('표시 없는 옛 모양', {'exception': "RuntimeError('x')"}, 'unknown'),
        ('게이트웨이 502', {'status_code': 502, 'body': {'rt_cd': '1'}, 'sent': True}, 'unknown'),
        ('업무 거절', {'status_code': 200, 'body': {'rt_cd': '1'}, 'sent': True}, 'rejected')):
    _got54, _ = run_place_api(ex.KISAPI(dry_run=False), 'SPY', {'_error': _err})
    check(f"{_label} → {_want}", _got54 == _want, f'got={_got54}')

# 미전송 확정은 주문번호를 안 남긴다 — 취소가 쏠 대상이 없다.
_api54 = ex.KISAPI(dry_run=False)
run_place_api(_api54, 'SPY', {'_error': {'exception': "RuntimeError('x')", 'sent': False}})
check("미전송은 거래소 기록을 안 남긴다", _api54._order_excg == {}, str(_api54._order_excg))

print("\n[55] 열한째 토론 반례 — 자격은 수량·나이·유한성·실제 발화로 가른다")

# 가. 체결 반례 — 값을 못 매긴 종목은 기대 보유와 대조해야 자격을 받는다.
# 소스 문자열이 아니라 실제 회차를 돌려 상태에 남는 자격으로 판정한다(열두째 토론 3번).


def _blocked_cycle(state_obj, signal=None, table=None, clock=None):
    """회차를 돌리고 (주문, 알림, 저장된 상태, eval_pending) 을 돌려준다."""
    _o, _a, _s = run_cycle(table or FULL, state_obj, signal or _signal(), clock=clock)
    return _o, _a, _s, ((_s or {}).get('eval_pending') or {})


# 평가 가격만 없는 보유(now_pric2 가 비었다). 수량은 그대로고 매매할 것도 없다.
def _bal_noprice(bil='1534', pdbc='7041', px_bil='', px_pdbc=''):
    return {'rt_cd': '0', 'output1': [
        {'ovrs_pdno': 'BIL', 'ovrs_cblc_qty': bil, 'ovrs_stck_evlu_amt': str(LIVE_BIL),
         'now_pric2': px_bil},
        {'ovrs_pdno': 'PDBC', 'ovrs_cblc_qty': pdbc, 'ovrs_stck_evlu_amt': str(LIVE_PDBC),
         'now_pric2': px_pdbc},
    ]}


_T55 = dict(FULL)
_T55['TTTS3012R'] = _bal_noprice()
_st55a = _state()
_st55a['rebalancing_needed'] = True
_o, _a, _s, _ep = _blocked_cycle(_st55a, table=_T55)
check("[양성] 값만 못 매긴 무매매 회차는 자격을 받는다",
      _ep.get('drift_ok') is True or not _ep, f'{_ep}')

# 매도가 접수됐는데 보유가 그대로다 → 미반영. 자격을 안 준다.
check("기대 보유 장부가 실제로 산출된다(코드가 돈다)",
      'expected_qty' in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', 'trade', 'executor_stock.py')).read())

# execute_delta 를 직접 돌려 장부·대조를 함수 실행으로 본다.
_SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'trade', 'executor_stock.py')).read()


class _LedgerAPI(ex.KISAPI):
    """주문은 다 접수되고, 주문 뒤 잔고만 바꿔 끼운다."""

    def __init__(self, pre, post, price=None, eval_price=None, maxqty=10 ** 9,
                 mid=None):
        super().__init__(dry_run=False)
        self._pre, self._post, self._n = pre, post, 0
        # 매도 뒤 매수 금액을 다시 세는 조회가 받는 값. 안 주면 둘째 조회부터 post 다.
        self._mid = mid
        self._price, self._maxqty = price or {}, maxqty
        # 평가 시점(fresh_only=True)에만 쓰는 값. 안 주면 주문용과 같다 — 주문 전
        # 가드가 보유 종목 현재가를 요구하므로 둘을 갈라야 "값만 못 매긴 회차" 를
        # 만들 수 있다.
        self._eval_price = self._price if eval_price is None else eval_price
        self.sent = []

    def get_balance(self):
        self._n += 1
        if self._n == 1:
            h = self._pre
        elif self._n == 2 and self._mid is not None:
            h = self._mid
        else:
            h = self._post
        return dict(h), 100000.0, 1300.0

    def get_current_price(self, t, fresh_only=False):
        src = self._eval_price if fresh_only else self._price
        return src.get(t, 0.0)

    def get_max_buy_qty(self, t, p):
        return self._maxqty

    def ensure_auth(self):
        # 진짜 토큰 발급을 타면 주문 전에 인증 실패로 되돌아간다(망·자격증명 미사용).
        return True

    def place_order(self, t, q, p, side='buy', **kw):
        self.sent.append((side, t, q))
        return 'ok'

    def cancel_all_pending(self):
        return {'queried': True, 'found': 0, 'failed': 0}


def _delta_full(api, target, st=None, **kw):
    """진짜 execute_delta 를 돌린다. (결과, 그 회차의 상태)"""
    st = {} if st is None else st
    old_log, old_tg, old_time = ex.log, ex.send_telegram, ex.time
    try:
        ex.log = lambda *a, **k: None
        ex.send_telegram = lambda m: None
        ex.time = types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0.0,
                                        monotonic=lambda: 0.0)
        r = ex.execute_delta(target, api, st, cash_buffer=0.07, **kw)
    finally:
        ex.log, ex.send_telegram, ex.time = old_log, old_tg, old_time
    return r, st


def _delta(api, target, **kw):
    """진짜 execute_delta 를 돌린다. (결과, 그 회차가 남긴 평가 보류 표식)"""
    r, st = _delta_full(api, target, **kw)
    return r, (st.get('eval_pending') or {})


# 주문용 가격은 멀쩡하고 평가 시점 가격만 없는 회차를 만든다. 총자산 100,000 에
# BIL 930주×100 = 93,000 이면 실효 목표(1.0×(1-0.07))와 같아 매매가 안 난다.
_ORDER_PX = {'BIL': 100.0, 'PDBC': 20.0}

# 매도 접수 뒤 보유가 안 줄었다(미반영) + 그 종목 평가 가격 없음 → 자격 없음.
_api1 = _LedgerAPI({'BIL': 830, 'PDBC': 500}, {'BIL': 930, 'PDBC': 500},
                   price=_ORDER_PX, eval_price={'BIL': 100.0})
_r1, _ep1 = _delta(_api1, {'BIL': 1.0})
check("목표 밖 종목을 실제로 판다", ('sell', 'PDBC', 500) in _api1.sent, str(_api1.sent))
check("매도 미반영 + 값 결측이면 완료로 안 본다", _r1.get('completed') is not True,
      str(_r1.get('completed')))
check("그 회차는 드리프트 자격을 못 받는다", _ep1.get('drift_ok') is not True, str(_ep1))

# 이번 회차에 매매한 종목은 수량이 맞아도 값을 못 매기면 증명이 안 된다(열두째 2번).
# 반을 팔고 반을 사서 둘 다 기대대로 체결된 회차다 — 장부 대조만으로는 통과한다.
# PDBC 는 주문 뒤에도 보유가 남아 평가 대상이고, 그 값만 못 매긴다.
_api3 = _LedgerAPI({'BIL': 830, 'PDBC': 500}, {'BIL': 465, 'PDBC': 2325},
                   price=_ORDER_PX, eval_price={'BIL': 100.0},
                   mid={'BIL': 465, 'PDBC': 500})
_r3, _ep3 = _delta(_api3, {'BIL': 0.5, 'PDBC': 0.5})
check("그 회차가 실제로 매매했다",
      any(t == 'PDBC' for _s, t, _q in _api3.sent), str(_api3.sent))
check("매매한 종목은 수량이 맞아도 값이 없으면 자격을 안 준다",
      _ep3.get('drift_ok') is not True, str(_ep3))

# 안 건드린 종목이 값만 없으면 자격을 받는다(양성 대조).
_api2 = _LedgerAPI({'BIL': 930}, {'BIL': 930}, price=_ORDER_PX, eval_price={})
_r2, _ep2 = _delta(_api2, {'BIL': 1.0})
check("[양성] 그 회차에 주문이 없다", _api2.sent == [], str(_api2.sent))
check("[양성] 수량이 맞고 안 건드린 종목은 자격을 받는다",
      _ep2.get('drift_ok') is True, f'{_ep2} / {_r2.get("completed")}')

# 수량이 어긋나면 양성 대조와 같은 모양에서도 자격을 뺏는다(장부 검사의 실제 발화).
_api4 = _LedgerAPI({'BIL': 930}, {'BIL': 700}, price=_ORDER_PX, eval_price={})
_r4, _ep4 = _delta(_api4, {'BIL': 1.0})
check("주문 없이 보유가 줄면 자격을 안 준다", _ep4.get('drift_ok') is not True, str(_ep4))

# 나. 만료 반례 — 가격 수집에 시간이 걸리면 드리프트와 리필을 건너뛴다.
_clock = {'t': 0.0}


def _slow():
    """시계를 읽을 때마다 90초씩 간다 — 가격 수집이 오래 걸린 회차를 흉내낸다."""
    _clock['t'] += 90.0
    return _clock['t']


_st55 = _state()
for _s in _st55['snapshots'].values():
    _s['picks'], _s['weights'] = ['BIL'], {'BIL': 1.0}
_o55, _a55, _sv55 = run_cycle(FULL, _st55, _signal(), clock=_slow)
check("가격 수집이 길어지면 드리프트 판정을 쉰다",
      any('계좌값이' in a and '낡았다' in a for a in _a55), str(_a55))
check("그 회차에 주문이 없다", _o55 == [], str(_o55))
check("스냅샷이 보존된다",
      _sv55 is None or all((v.get('picks') == ['BIL'])
                           for v in (_sv55.get('snapshots') or {}).values()),
      str(_sv55 and _sv55.get('snapshots')))

_clock2 = {'t': 0.0}
_o55b, _a55b, _ = run_cycle(FULL, _state(), _signal(), clock=lambda: 0.0)
check("[양성] 나이가 정상이면 만료 알림이 없다",
      not any('낡았다' in a for a in _a55b), str(_a55b))

# 다. 산술 반례 — 유한한 입력의 결과가 넘치면 판정을 쉰다.
check("평가액 계산 넘침을 드리프트에서 막는다",
      "if not math.isfinite(_w_one):" in _SRC and "_StockDriftSkip()" in _SRC)
check("반회전율 넘침도 막는다", "if not math.isfinite(_ht):" in _SRC)

# 라. 앵커 반례 — 같은 비중으로 발화해도 지난 표식으로 리필을 열지 않는다.
check("앵커 체크가 실제 발화 목록을 돌려준다",
      "fired.append(str(snap_id))" in _SRC and "\n    return fired" in _SRC)
check("그 목록이 목표 변경 판정에 들어간다", "or bool(_anchors_fired)" in _SRC)
check("빈 목표 키는 대조 불가로 닫는다",
      "and target_key and ep.get('target_key') == target_key" in _SRC)

# 마. 상태 반례 — 손상된 비중이 목표 키 계산에서 예외를 내도 회차가 안 죽는다.
for _label, _bad in (('문자열', 'BIL'), ('None', None), ('리스트', ['BIL'])):
    _st55m = _state()
    _st55m['snapshots']['1']['weights'] = _bad
    _o55m, _a55m, _sv55m = run_cycle(FULL, _st55m, _signal())
    check(f"{_label} 비중 → 주문 0건", _o55m == [], str(_o55m))
    check(f"{_label} 비중 → 상태를 안 바꾼다", _sv55m is None, str(_sv55m is None))
    check(f"{_label} 비중 → 알린다",
          any('상태 이상' in a or '온전하지' in a for a in _a55m), str(_a55m))

# 값이 숫자가 아닌 경우도 같다. 참거짓은 정수의 하위형이라 isinstance 를 그냥 쓰면
# True 가 비중 1 로 통과하고, 숫자 문자열은 뒤에서 float 로 바뀌어 주문까지 간다.
for _label, _bad in (('참거짓', {'BIL': True, 'Cash': False}),
                     ('숫자 문자열', {'BIL': '1.0', 'Cash': '0.0'})):
    _st55n = _state()
    _st55n['snapshots']['1']['weights'] = _bad
    _o55n, _a55n, _sv55n = run_cycle(FULL, _st55n, _signal())
    check(f"{_label} 비중 → 주문 0건", _o55n == [], str(_o55n))
    check(f"{_label} 비중 → 상태를 안 바꾼다", _sv55n is None, str(_sv55n is None))
    check(f"{_label} 비중 → 알린다",
          any('상태 이상' in a or '온전하지' in a for a in _a55n), str(_a55n))

# ─────────────────────────────────────────────────────────────────────────
# [56] 접수 여부를 모른 주문이 다음 회차에 다시 나가지 않는다 (열세째 토론 1번)
#
# 열두째까지의 가드는 그 회차의 남은 주문만 멈췄다. 접수 여부 불명은 상태로 안 남아서,
# 다음 회차가 "보유가 아직 안 줄었다" 를 보고 같은 주문을 다시 냈다 — 서버가 받아놓고
# 우리만 응답을 못 받은 경우면 두 번 체결된다. 전송 직전에 의도를 디스크에 남기고,
# 다음 회차가 실제 보유로 그 결과를 가린다.
# ─────────────────────────────────────────────────────────────────────────


class _UnknownAPI(_LedgerAPI):
    """주문을 보냈는데 응답이 ok·rejected·stale 중 어느 것도 아닌 거래소."""

    def place_order(self, t, q, p, side='buy', **kw):
        self.sent.append(('order', side, t, q))
        return 'timeout'


_trace56 = []
_api56 = _UnknownAPI({'BIL': 830, 'PDBC': 500}, {'BIL': 830, 'PDBC': 500},
                     price=_ORDER_PX, eval_price=_ORDER_PX)
_st56 = {}
_api56.sent = _trace56  # 주문과 저장을 한 줄에 섞어 순서를 본다


def _persist56():
    _trace56.append(('persist', json.loads(json.dumps(_st56.get('inflight_order')))))


_r56, _st56 = _delta_full(_api56, {'BIL': 1.0}, st=_st56, persist=_persist56)

check("접수 불명 회차가 주문을 실제로 보냈다",
      any(e[0] == 'order' for e in _trace56), str(_trace56))
check("접수 불명이면 회차가 완료로 안 닫힌다", _r56.get('completed') is not True,
      str(_r56.get('completed')))
check("접수 불명 표식을 남긴다", isinstance(_st56.get('inflight_order'), dict),
      str(_st56.get('inflight_order')))

_first_order = next((i for i, e in enumerate(_trace56) if e[0] == 'order'), None)
_marked_before = [i for i, e in enumerate(_trace56)
                  if e[0] == 'persist' and isinstance(e[1], dict)]
check("의도를 디스크에 남긴 뒤에 주문을 보낸다",
      _first_order is not None and _marked_before and min(_marked_before) < _first_order,
      f'order={_first_order} persist={_marked_before}')

_mark56 = _st56.get('inflight_order') or {}
check("표식이 어느 주문인지 적는다",
      (_mark56.get('side'), _mark56.get('ticker'), _mark56.get('qty'),
       _mark56.get('qty_before')) == ('sell', 'PDBC', 500, 500), str(_mark56))

# 프로세스가 죽었다 다시 떠도 표식이 살아 있어야 한다 — 상태파일을 거쳐 되읽는다.
_st56r = json.loads(json.dumps(_st56))
check("표식이 상태파일 왕복을 견딘다", isinstance(_st56r.get('inflight_order'), dict),
      str(_st56r.get('inflight_order')))

# 가. 접수 안 됨 — 보유가 그대로다. (회차 시작의 미체결 확정이 통과했을 때만 선다)
_stA = json.loads(json.dumps(_st56))
_okA, _whyA = ex.resolve_inflight(_stA, {'BIL': 830, 'PDBC': 500})
check("보유가 그대로면 미접수로 닫고 진행한다", _okA is True and '접수되지 않았' in _whyA, _whyA)
check("닫은 뒤 표식을 지운다", 'inflight_order' not in _stA, str(_stA))

# 나. 체결됨 — 보유가 기대대로 줄었다.
_stB = json.loads(json.dumps(_st56))
_okB, _whyB = ex.resolve_inflight(_stB, {'BIL': 830, 'PDBC': 0})
check("보유가 기대대로면 체결로 닫고 진행한다", _okB is True and '체결' in _whyB, _whyB)
check("체결로 닫은 뒤 표식을 지운다", 'inflight_order' not in _stB, str(_stB))

# 다. 반례 본체 — 부분 체결이라 둘 중 어느 쪽도 아니다. 다시 보내면 두 번 체결된다.
_stC = json.loads(json.dumps(_st56))
_okC, _whyC = ex.resolve_inflight(_stC, {'BIL': 830, 'PDBC': 300})
check("부분 체결이면 이번 회차 주문을 막는다", _okC is False, _whyC)
check("막은 회차는 표식을 지우지 않는다",
      isinstance(_stC.get('inflight_order'), dict), str(_stC))

# 라. 보유수량이 숫자가 아니면 가릴 수 없다 — 막는다.
_stD = json.loads(json.dumps(_st56))
check("보유수량을 못 읽으면 막는다",
      ex.resolve_inflight(_stD, {'PDBC': None})[0] is False, '')

# 마. [양성] 표식이 없으면 그대로 진행한다.
check("표식이 없으면 막지 않는다", ex.resolve_inflight({}, {'BIL': 1})[0] is True, '')

# 바. [양성] 정상 응답 회차는 표식을 남기지 않는다.
_trace56b = []
_api56b = _LedgerAPI({'BIL': 830, 'PDBC': 500}, {'BIL': 930, 'PDBC': 0},
                     price=_ORDER_PX, eval_price=_ORDER_PX,
                     mid={'BIL': 830, 'PDBC': 0})
_st56b = {}
_r56b, _st56b = _delta_full(_api56b, {'BIL': 1.0}, st=_st56b,
                            persist=lambda: _trace56b.append(
                                json.loads(json.dumps(_st56b.get('inflight_order')))))
check("[양성] 정상 회차도 주문을 보냈다", _api56b.sent != [], str(_api56b.sent))
check("[양성] 정상 회차는 표식을 안 남긴다", 'inflight_order' not in _st56b, str(_st56b))
check("[양성] 그래도 전송 때마다 표식을 남겼다 지웠다",
      any(isinstance(m, dict) for m in _trace56b) and _trace56b[-1] is None,
      str(_trace56b))

# ─────────────────────────────────────────────────────────────────────────
# [57] 정수 주 절삭으로 주문이 사라진 편차 (열세째 토론 2번)
#
# 한 주 값이 커서 매수 수량이 0 으로 깎이면 주문이 안 난다. 그 종목의 평가 가격까지
# 없으면 수량 대조는 "주문 전과 같다" 로 통과해 "값만 모른다" 가 되고, 다음 회차
# 종목 교체가 열린다 — 실제로는 허용오차를 넘는 편차가 남아 있다.
# ─────────────────────────────────────────────────────────────────────────

# 총자산 100,000, 실효 목표 93,000. BIL 50% = 46,500 인데 보유 465주×100 = 46,500 (딱 맞음).
# PDBC 50% = 46,500 인데 보유 15주×3,000 = 45,000 → 모자란 1,500 이 한 주(3,000)에 못 미쳐
# 주문이 안 난다. 편차 1.5% 는 허용오차 1% 를 넘는다.
_TRUNC_PX = {'BIL': 100.0, 'PDBC': 3000.0}
_api57 = _LedgerAPI({'BIL': 465, 'PDBC': 15}, {'BIL': 465, 'PDBC': 15},
                    price=_TRUNC_PX, eval_price={'BIL': 100.0})
_r57, _ep57 = _delta(_api57, {'BIL': 0.5, 'PDBC': 0.5})
check("절삭 회차는 주문이 안 난다", _api57.sent == [], str(_api57.sent))
check("절삭으로 남은 편차는 다음 회차 종목 교체를 안 연다",
      _ep57.get('drift_ok') is not True, str(_ep57))

# [양성] 같은 자리에서 편차가 허용오차 안이면 자격을 그대로 받는다.
# PDBC 46,000 (보유 46주×1,000) → 모자란 500 은 0.5% 라 허용오차 안이다.
_FINE_PX = {'BIL': 100.0, 'PDBC': 1000.0}
_api57b = _LedgerAPI({'BIL': 465, 'PDBC': 46}, {'BIL': 465, 'PDBC': 46},
                     price=_FINE_PX, eval_price={'BIL': 100.0})
_r57b, _ep57b = _delta(_api57b, {'BIL': 0.5, 'PDBC': 0.5})
check("[양성] 절삭 편차가 허용오차 안이면 주문도 없다", _api57b.sent == [], str(_api57b.sent))
check("[양성] 그 회차는 자격을 받는다", _ep57b.get('drift_ok') is True, str(_ep57b))

# ─────────────────────────────────────────────────────────────────────────
# [58] 주문이 나가는 순간 지난 회차의 드리프트 자격을 거둔다 (열네째 토론 1번)
#
# 주문 의도를 디스크에 남길 때 상태가 통째로 굳는다. 그 상태에 지난 회차의
# "목표에 닿았다"(eval_pending.drift_ok)가 같이 들어 있으면, 매도만 나간 채
# 프로세스가 죽었을 때 다음 회차가 미완성 목표 위에 리필·스냅샷 교체를 얹는다.
# 중간 저장 시점마다 상태를 JSON 으로 떠서, 어느 시점에 재시작해도 문이 닫혀
# 있는지 본다.
# ─────────────────────────────────────────────────────────────────────────

_TGT58 = {'BIL': 1.0}
_KEY58 = ex._target_key(_TGT58)


def _prev_credit58():
    """지난 회차가 "값만 못 매겼고 목표엔 닿았다" 로 닫힌 상태."""
    return {'eval_pending': {
        'tickers': ['PDBC'], 'reason': '평가 시점 가격을 못 받았다', 'rounds': 1,
        'first': '2026-10-29T23:35:00', 'last': '2026-10-29T23:35:00',
        'max_diff_known': 0.0, 'drift_ok': True, 'target_key': _KEY58,
    }, 'rebalancing_needed': True}


# 들어가기 전에 그 표식이 실제로 문을 여는 것을 먼저 확인한다 — 안 그러면 아래
# 검사가 "원래 안 열리는 문" 을 보고 통과한다.
check("[전제] 지난 회차 자격은 같은 목표에서 문을 연다",
      ex._drift_gate_open(_prev_credit58(), target_key=_KEY58, target_changed=False) is True)

_snaps58 = []
_api58 = _LedgerAPI({'BIL': 830, 'PDBC': 500}, {'BIL': 930, 'PDBC': 0},
                    price=_ORDER_PX, eval_price=_ORDER_PX,
                    mid={'BIL': 830, 'PDBC': 0})
_st58 = _prev_credit58()
_r58, _st58 = _delta_full(_api58, _TGT58, st=_st58,
                          persist=lambda: _snaps58.append(json.loads(json.dumps(_st58))))

check("그 회차가 실제로 주문을 냈다", _api58.sent != [], str(_api58.sent))
check("중간 저장이 한 번은 일어났다", len(_snaps58) >= 2, str(len(_snaps58)))

# 저장된 모든 중간 시점에서 재시작해도 문이 안 열려야 한다. 표식을 남긴 직후,
# 전송 뒤 지우기 직전, 지운 직후가 전부 이 목록에 들어 있다.
_open58 = [i for i, s in enumerate(_snaps58)
           if ex._drift_gate_open(s, target_key=_KEY58, target_changed=False)]
check("어느 중간 시점에서 재시작해도 종목 교체가 안 열린다", _open58 == [],
      f'열린 시점 {_open58} / 전체 {len(_snaps58)}')
check("표식을 남긴 저장에도 자격이 안 남는다",
      all((s.get('eval_pending') or {}).get('drift_ok') is not True for s in _snaps58),
      str([(s.get('eval_pending') or {}).get('drift_ok') for s in _snaps58]))
check("표식을 지운 저장에도 회수가 유지된다",
      (_snaps58[-1].get('eval_pending') or {}).get('drift_ok') is not True
      and 'inflight_order' not in _snaps58[-1], str(_snaps58[-1]))
check("관측 기록은 남는다 — 연속 회차를 계속 센다",
      all(isinstance(s.get('eval_pending'), dict) for s in _snaps58),
      str([type(s.get('eval_pending')).__name__ for s in _snaps58]))

# [양성] 주문이 하나도 안 나간 회차는 자격을 그대로 둔다. 보유가 목표와 같아
# 매매가 없고, 값도 다 매겨져 완료로 닫히는 회차다.
_snaps58b = []
_api58b = _LedgerAPI({'BIL': 930}, {'BIL': 930}, price=_ORDER_PX, eval_price=_ORDER_PX)
_st58b = _prev_credit58()
_r58b, _st58b = _delta_full(_api58b, _TGT58, st=_st58b,
                            persist=lambda: _snaps58b.append(json.loads(json.dumps(_st58b))))
check("[양성] 그 회차엔 주문이 없다", _api58b.sent == [], str(_api58b.sent))
check("[양성] 주문이 없으면 중간 저장도 없다", _snaps58b == [], str(_snaps58b))
check("[양성] 값을 다 매긴 회차는 완료로 닫힌다", _r58b.get('completed') is True,
      str(_r58b.get('completed')))

print()
if FAILED:
    print(f"❌ {len(FAILED)}건 실패: {FAILED}")
    sys.exit(1)
print("✅ 전부 통과")
