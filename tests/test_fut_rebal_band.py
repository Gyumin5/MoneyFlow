"""선물 체결 밴드 0.01 → 0.05 (BT 정합) 검증 — 2026-08-25.

배경: 채택 BT(backtest_futures_v25._execute_rebalance)는 수량 ±5% 밴드에서만 체결하는데
라이브는 명목 ±1% 였다. 회전이 1.69배(연 264회 대 156회)라 실비용이 BT 가정보다 컸다.
측정(reports/2026-08-25-fut-rebal-band.html): 좁혀도 수익은 동률이고 낙폭만 악화
(Cal 7.03 대 7.57, MDD -41.0% 대 -38.3%), 비용 5배 스트레스 전 구간 열위.

ai-debate run-20260825T013824Z 는 조건부 GO 였고, 배포 전 조건이 아래다.
  (1) 밴드를 타지 않아야 하는 경로가 실제로 안 타는지 호출 경로 확인
      — 목표비중 0 전량청산 / 미보유 신규진입 / 카나리 OFF 청산 / 레버리지 변경
  (2) 경계·부분체결·라운딩·MIN_NOTIONAL·L 하향을 고정 스냅샷으로 재생
  (3) 주문 상태 검증을 경제적 완료 판정과 분리 (거절·실패가 밴드 안에 묻히면 안 됨)
  (4) 라이브 편차 정의와 BT 밴드 정의가 경계에서 같은 판정을 주는지

네트워크·라이브 state 미사용. 주문은 가짜 클라이언트가 받아 기록만 한다.
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_rebal_band.py
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'trade'))
import auto_trade_binance as a  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


CONSTRAINTS = {
    'step_size': 0.001, 'min_qty': 0.001, 'min_notional': 5.0,
    'tick_size': 0.01, 'qty_precision': 3,
}
PRICE = 100.0
PV = 100_000.0          # equity
BUF = 0.02              # CASH_BUFFER
TGT_W = 0.2666666666    # 코인 1종 목표비중
TGT_LEV = 3


def target_notional(w=TGT_W, lev=TGT_LEV, pv=PV):
    return pv * (1 - BUF) * w * lev


class FakeClient:
    """주문을 받아 적기만 하는 클라이언트."""

    def __init__(self, positions):
        self.positions = positions
        self.orders = []
        self.fail_symbols = set()

    def futures_symbol_ticker(self, symbol=None):
        return {'symbol': symbol, 'price': str(PRICE)}

    def futures_create_order(self, **kw):
        if kw.get('symbol') in self.fail_symbols:
            raise a.BinanceAPIException(_FakeResp(), 400, '{"code":-2019,"msg":"Margin is insufficient"}')
        self.orders.append(kw)
        return {'orderId': 1, 'status': 'NEW'}


class _FakeResp:
    status_code = 400
    text = '{"code":-2019,"msg":"Margin is insufficient"}'

    def json(self):
        return {'code': -2019, 'msg': 'Margin is insufficient'}


def pos(coin, notional, lev=TGT_LEV, price=PRICE):
    qty = notional / price
    return {
        'qty': qty, 'qty_raw': f'{qty:.3f}', 'symbol': coin + 'USDT',
        'isolated': False, 'entry_price': price, 'mark_price': price,
        'pnl': 0.0, 'liquidation_price': 0.0, 'notional': notional,
        'leverage': float(lev), 'real_notional': notional / lev,
        'weight': notional / PV, 'real_weight': notional / lev / PV,
    }


def with_fakes(positions):
    """execute_rebalance / needs_rebalance 가 부르는 외부 접점을 가짜로 바꾼다."""
    cli = FakeClient(positions)
    a.CASH_BUFFER = BUF
    a.get_symbol_constraints = lambda client, symbol: dict(CONSTRAINTS)
    a.get_current_positions = lambda client: (positions, PV, True)
    a.cancel_stop_orders = lambda client, symbols: None
    a._v25_touch_reset()
    return cli


# ── 1. 밴드 값과 정의 ────────────────────────────────────────────────────────
print("[1] 밴드 상수")

check("DELTA_THRESHOLD = 0.05 (BT 정합)", a.DELTA_THRESHOLD == 0.05, f"({a.DELTA_THRESHOLD})")

_src = open(os.path.join(_HERE, '..', 'trade', 'auto_trade_binance.py')).read()
check("밴드는 매도 게이트에 쓰인다", 'elif delta_pct < -DELTA_THRESHOLD:' in _src)
check("밴드는 매수 게이트에 쓰인다", 'if delta_pct <= DELTA_THRESHOLD:' in _src)
check("밴드는 needs_rebalance 에 쓰인다", 'if abs(delta_pct) > DELTA_THRESHOLD:' in _src)
_n_band = len(re.findall(r'\bDELTA_THRESHOLD\b', _src))
check("밴드 사용처는 그 세 곳뿐", _n_band == 4, f"(정의 1 + 사용 3 = 4 여야 하는데 {_n_band})")

# BT 는 수량 기준(delta_qty vs holdings×5%), 라이브는 명목 기준(delta_notional/current_notional).
# 같은 가격이면 두 비율은 대수적으로 동일하다 — 경계에서 같은 판정을 주는지 수치로 확인.
same = True
for cur_qty, tgt_qty in [(100.0, 94.9), (100.0, 95.1), (100.0, 104.9), (100.0, 105.1),
                         (37.5, 35.0), (0.123, 0.13)]:
    bt_fire = abs(tgt_qty - cur_qty) > cur_qty * 0.05
    live_fire = abs(tgt_qty * PRICE - cur_qty * PRICE) / (cur_qty * PRICE) > 0.05
    if bt_fire != live_fire:
        same = False
check("BT 수량밴드와 라이브 명목밴드는 같은 판정 (가격이 약분됨)", same)


# ── 2. 경계 판정 — needs_rebalance ──────────────────────────────────────────
print("\n[2] 경계 판정 (needs_rebalance)")

TN = target_notional()
for delta, expect in [(-0.049, False), (-0.051, True), (0.049, False), (0.051, True),
                      (0.0, False)]:
    # current = target / (1+delta) 이면 (target-current)/current = delta
    cur_notional = TN / (1 + delta)
    p = {'AAA': pos('AAA', cur_notional)}
    cli = with_fakes(p)
    got = a.needs_rebalance(cli, {'AAA': TGT_W, 'CASH': 1 - TGT_W}, p, PV, {'AAA': TGT_LEV})
    check(f"delta {delta:+.1%} → 리밸필요={expect}", got is expect, f"(got={got})")


# ── 3. 경계 판정 — 실제 주문 생성 ───────────────────────────────────────────
print("\n[3] 경계 판정 (execute_rebalance 주문)")


def run_exec(cur_notional, tgt_w=TGT_W, lev=TGT_LEV, coin='AAA', extra_pos=None,
             liquidation_only=False, fail=()):
    p = {}
    if cur_notional > 0:
        p[coin] = pos(coin, cur_notional, lev)
    if extra_pos:
        p.update(extra_pos)
    cli = with_fakes(p)
    cli.fail_symbols = set(fail)
    oa, ea = [], []
    tgt = {coin: tgt_w, 'CASH': max(0.0, 1 - tgt_w)} if tgt_w > 0 else {'CASH': 1.0}
    a.execute_rebalance(cli, tgt, PV, {coin: lev} if tgt_w > 0 else {},
                        order_alerts=oa, error_alerts=ea,
                        liquidation_only=liquidation_only)
    return cli, oa, ea


cli, oa, _ = run_exec(TN / (1 - 0.049))   # 목표보다 4.9% 초과 보유
check("초과 4.9% → 주문 없음", cli.orders == [], f"({cli.orders})")
cli, oa, _ = run_exec(TN / (1 - 0.051))   # 5.1% 초과
check("초과 5.1% → 매도 1건", len(cli.orders) == 1 and cli.orders[0]['side'] == 'SELL',
      f"({cli.orders})")
check("그 매도는 reduceOnly", cli.orders and cli.orders[0].get('reduceOnly') == 'true')
cli, oa, _ = run_exec(TN / (1 + 0.049))   # 4.9% 미달
check("미달 4.9% → 주문 없음", cli.orders == [], f"({cli.orders})")
cli, oa, _ = run_exec(TN / (1 + 0.051))   # 5.1% 미달
check("미달 5.1% → 매수 1건", len(cli.orders) == 1 and cli.orders[0]['side'] == 'BUY',
      f"({cli.orders})")

# 옛 밴드(1%)였다면 주문이 나갔을 구간이 이제 안 나간다 — 변경의 실제 효과
cli, _, _ = run_exec(TN / (1 + 0.03))
check("미달 3% → 주문 없음 (옛 1% 밴드였다면 매수했을 구간)", cli.orders == [], f"({cli.orders})")


# ── 4. 밴드를 타지 않아야 하는 경로 (ai-debate 조건 1) ──────────────────────
print("\n[4] 밴드 무관 경로")

# (a) 목표비중 0 → 밴드와 무관하게 전량청산
cli, oa, _ = run_exec(0, tgt_w=0, extra_pos={'BBB': pos('BBB', 1000.0)})
check("목표비중 0 보유 → 전량청산 주문", len(cli.orders) == 1
      and cli.orders[0]['side'] == 'SELL' and cli.orders[0].get('reduceOnly') == 'true',
      f"({cli.orders})")

# (b) 미보유 + 목표 있음 → 밴드 계산 자체를 안 타고 신규 진입
cli, oa, _ = run_exec(0)
check("미보유 → 신규 진입 매수", len(cli.orders) == 1 and cli.orders[0]['side'] == 'BUY',
      f"({cli.orders})")
check("신규 진입은 reduceOnly 아님", cli.orders and 'reduceOnly' not in cli.orders[0])

# (c) 카나리 OFF 전량청산(liquidation_only) — 매수 경로 미진입 + 전부 reduceOnly
cli, oa, ea = run_exec(0, tgt_w=0, extra_pos={'BBB': pos('BBB', 50_000.0)},
                       liquidation_only=True)
check("카나리 OFF → reduceOnly 매도만", cli.orders
      and all(o['side'] == 'SELL' and o.get('reduceOnly') == 'true' for o in cli.orders),
      f"({cli.orders})")
check("카나리 OFF 경로에 매수 없음", not any(o['side'] == 'BUY' for o in cli.orders))

# (d) 레버리지·마진 변경은 execute_rebalance 밖(사전 단계)이라 밴드와 무관 — 소스로 고정
_lev_block = _src[_src.index('V25 prep'):_src.index('V25 prep') + 1500]
check("레버리지 준비 구간에 밴드 판정 없음", 'DELTA_THRESHOLD' not in _lev_block)
check("L↓ 사전매도는 목표를 new_lev/cur_lev 비율로 잡는다(밴드 아님)",
      'target_notional = current_notional * (new_lev / cur_lev)' in _src)


# ── 5. L 하향 후 잔차 (ai-debate 조건 2 — 08-24 실측 재생) ──────────────────
print("\n[5] L 하향 후 잔차 — 08-24 실측 재생")

# 08-24 실측: 사전매도로 4→3 배 만들며 명목 $70,702→$53,027(qty 25% 감소).
# 그 뒤 현재 $57,853 대 새 목표 $58,927 = +1.9%. 1% 밴드에선 매수했고(실제 그랬다),
# 5% 밴드에선 스킵한다. 목표 미달 1.9% 를 하루 안고 가는 게 BT 와 같은 처분이다.
_cur, _tgt = 57853.47, 58927.16
_delta = (_tgt - _cur) / _cur
check(f"08-24 잔차 {_delta:+.1%} 는 5% 밴드 안", abs(_delta) <= 0.05, f"({_delta:+.3%})")
check(f"08-24 잔차는 옛 1% 밴드는 넘었다(그래서 주문이 나갔다)", abs(_delta) > 0.01)
cli, _, _ = run_exec(_cur, tgt_w=_tgt / (PV * (1 - BUF) * TGT_LEV))
check("같은 상황 재생 → 주문 없음", cli.orders == [], f"({cli.orders})")


# ── 6. 거래소 최소단위 (밴드는 넘지만 주문은 못 냄) ─────────────────────────
print("\n[6] 최소주문·라운딩")

_saved = CONSTRAINTS['min_notional']
CONSTRAINTS['min_notional'] = 5000.0  # 비현실적으로 큰 최소주문
cli, _, _ = run_exec(TN / (1 + 0.06))  # 밴드는 넘음
check("밴드 초과라도 거래소 최소주문 미달이면 매수 스킵", cli.orders == [], f"({cli.orders})")
CONSTRAINTS['min_notional'] = _saved

CONSTRAINTS['step_size'], CONSTRAINTS['min_qty'] = 1000.0, 1000.0
cli, _, _ = run_exec(TN / (1 - 0.10))
check("스텝 라운딩으로 수량 0 이 되면 주문 없음", cli.orders == [], f"({cli.orders})")
CONSTRAINTS['step_size'], CONSTRAINTS['min_qty'] = 0.001, 0.001


# ── 7. 주문 실패는 완료 판정과 분리 (ai-debate 조건 3) ──────────────────────
print("\n[7] 주문 실패 처리")

cli, oa, ea = run_exec(TN / (1 - 0.10), fail={'AAAUSDT'})
check("주문 거절 → error_alerts 에 ORDER FAILED", any(m.startswith('ORDER FAILED') for m in ea),
      f"({ea})")
check("주문 거절 → order_alerts 는 비어 있다", oa == [], f"({oa})")
check("주문 거절도 계좌 변경으로 표시(접수 가능성)", a._v25_account_changed())

check("완료 판정이 주문 실패를 별도로 본다",
      "_order_failed = any(str(m).startswith('ORDER FAILED') for m in error_alerts)" in _src)
check("완료 판정 조건에 주문 실패가 OR 로 들어간다",
      re.search(r'if needs_rebalance\([^)]*\) or _order_failed:', _src) is not None)


# ── 8. 회귀 — 기존 불변식이 살아 있다 ──────────────────────────────────────
print("\n[8] 회귀")

check("liquidation_only 불변식(비-reduceOnly 감지 시 주문 없이 중단) 유지",
      'liquidation_only 인데 비-reduceOnly/매수 주문 감지' in _src)
check("매도 먼저·매수 나중 정렬 유지",
      "sorted(trades, key=lambda x: 0 if x[0] == 'SELL' else 1)" in _src)
check("상시 무결성 검사는 여전히 체결 전 포지션으로",
      '_v25_standing_check(positions_before, _sbase)' in _src
      and '_v25_standing_check(positions_after' not in _src)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
