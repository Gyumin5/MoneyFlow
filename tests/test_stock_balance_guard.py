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
    def _g(path, tr_id, params, retries=3):
        COUNTS[tr_id] = COUNTS.get(tr_id, 0) + 1
        v = table.get(tr_id, {})
        return v() if callable(v) else v
    return _g


def with_api(table, fn):
    old_get, old_log = ex._get, ex.log
    try:
        ex._get = fake_get(table)
        ex.log = lambda *a, **k: None
        return fn(ex.KISAPI(dry_run=True))
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

    def get_current_price(self, ticker):
        return {'BIL': 91.60, 'PDBC': 20.02}.get(ticker, 100.0)

    def get_max_buy_qty(self, ticker, price) -> int:
        return 999999

    def place_order(self, ticker, qty, price, side='buy') -> str:
        # 집행기 계약: 'ok' / 'rejected' / 'unknown'. 참거짓이 아니다 — 접수 여부를
        # 모르는 것과 확정 거절을 가르지 않으면 중복 전송이 섞인다.
        self.orders.append((side, ticker, qty))
        return 'ok'


def step_clock(jump_after, jump_to):
    """부른 횟수가 jump_after 를 넘으면 jump_to 를 돌려주는 단조 시계. 계좌값 나이 시험용.

    집행기가 monotonic 을 부르는 순서 — ①계좌 조회 시작 ②execute_delta 진입(재사용 나이)
    ③주문 직전 ④이후 주문·재시도마다 하나씩. jump_after 로 어느 지점부터 늙게 할지 고른다.
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
# 5번째 호출부터 늙는다 — 첫 매수(BIL)는 나가고 둘째(PDBC)는 막힌다.
orders, alerts, saved = run_cycle(FULL, _st, _signal(), clock=step_clock(4, 100.0))
_buys = [o for o in orders if o[0] == 'buy']
check("첫 주문만 나간다", len(_buys) == 1, str(orders))
check("일부만 나간 회차는 미완료로 남는다",
      saved is not None and saved.get('rebalancing_needed') is True, str(saved))

print("\n[27] 취소 응답은 성공 표식으로 판정한다 — 진짜 cancel_all_pending 을 돌린다")


def _nccs(rows, nk='', fk='', tr_cont=''):
    """미체결 조회 응답 한 쪽. tr_cont 는 응답 헤더(F·M=더 있다, D·E=마지막)."""
    d = {'rt_cd': '0', 'output': rows, 'ctx_area_nk200': nk, 'ctx_area_fk200': fk}
    d['_hdr_tr_cont'] = tr_cont
    return d


def _row(odno='0001', pdno='BIL', qty='10', side='01'):
    return {'odno': odno, 'pdno': pdno, 'nccs_qty': qty, 'sll_buy_dvsn_cd': side}


_CANCEL_REQUESTS = []


def run_cancel(nccs_by_excg, post_result):
    """진짜 KISAPI.cancel_all_pending 을 통신만 가짜로 돌린다. (결과, 취소한 주문번호들)

    값을 목록으로 주면 그 거래소의 연속조회 쪽들이다 — 쪽마다 하나씩 돌려준다.
    """
    posts = []
    pages = {k: 0 for k in nccs_by_excg}
    reqs = _CANCEL_REQUESTS
    reqs.clear()
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
        return post_result() if callable(post_result) else post_result

    try:
        ex._get, ex._post, ex.log = _g, _p, lambda *a, **k: None
        return _KISAPI_REAL(dry_run=False).cancel_all_pending(), posts
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
    return _nccs([])


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

res, posts = run_cancel({e: _nccs([_row(odno='SAME')]) for e in ex.PENDING_EXCHANGES},
                        {'rt_cd': '0'})
check("같은 주문이 여러 코드에 보이면 한 번만 센다", res.get('found') == 1, str(res))
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

    def place_order(self, ticker, qty, price, side='buy') -> str:
        self.orders.append((side, ticker, qty))
        return 'unknown'


class _SendRejected(RealBalanceAPI):
    """서버가 본문으로 거절했다 — 접수되지 않았음이 확정이라 다시 내도 안전하다."""

    def place_order(self, ticker, qty, price, side='buy') -> str:
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

    def _p(path, tr_id, body, retries=3):
        _retries_seen.append(retries)
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

    def place_order(self, ticker, qty, price, side='buy') -> str:
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
    def get_current_price(self, ticker) -> float:
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
check("모의 취소는 조회했다고 돌려준다",
      ex.KISAPI(dry_run=True).cancel_all_pending().get('queried') is not None)

print("\n[36] 매도가 잔고에 비치면 매수가 나간다 — 양성 대조")

_CLOCK = {'t': 0.0}


class _SellsSettle(RealBalanceAPI):
    """매도가 잔고에 바로 비치는 계좌. 매도 뒤 재조회가 줄어든 수량을 돌려준다."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._sold = {}

    def place_order(self, ticker, qty, price, side='buy') -> str:
        res = super().place_order(ticker, qty, price, side)
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

print("\n[39] 모의 회차는 매도 다음 매수까지 간다 — 통신은 한 번도 안 한다")


class _DryAPI(_SellsSettle):
    """모의 실행. place_order 는 통신 없이 ok 를 돌려주고 잔고는 안 줄어든다."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dry_run = True

    def place_order(self, ticker, qty, price, side='buy') -> str:
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

print()
if FAILED:
    print(f"❌ {len(FAILED)}건 실패: {FAILED}")
    sys.exit(1)
print("✅ 전부 통과")
