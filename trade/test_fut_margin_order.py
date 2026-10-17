"""선물 체결층의 증거금 재표시 자리를 반례로 확인한다 (2026-10-18).

토론 run-20261017T204756Z 의 비판자가 낸 반례다. 재표시를 매도보다 먼저 돌리면
현금이 모자라 덜 묶인 상태에서 매도가 또 비례 차감해, 매도 뒤 남는 증거금이
라이브보다 적어진다. 4배에서 3배로 내리는 경우 수량은 같은데 증거금이 63.75 대
82.5 로 갈리고 현금이 62.5 대 2.5 로 갈린다 — 다음 봉의 드리프트와 주문이 달라진다.

여기서는 엔진 본문의 _execute_rebalance 를 소스에서 그대로 떼어내 바깥 상태만
가짜로 채워 돌린다. 식을 베껴 쓰지 않는다 — 베끼면 엔진이 아니라 사본을 재는 것이다.

실행: cd /home/gmoh/mon/251229 && bash run_tests.sh
"""
from __future__ import annotations

import ast
import os
import sys
import textwrap

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "strategies", "cap_defend", "backtest_futures_v25.py")


def _extract() -> str:
    src = open(ENGINE).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_rebalance":
            return ast.get_source_segment(src, node) or ""
    raise SystemExit("_execute_rebalance 를 못 찾았다")


def _make_runner(state: dict):
    """떼어낸 함수를 바깥 상태만 가짜로 채운 틀 안에서 되살린다."""
    body = textwrap.indent(textwrap.dedent(_extract()), "    ")
    outer = (
        "def _mk(st):\n"
        "    capital = st['capital']\n"
        "    trade_count = 0\n"
        "    holdings = st['holdings']\n"
        "    margins = st['margins']\n"
        "    entry_prices = st['entry_prices']\n"
        "    entry_levs = st['entry_levs']\n"
        "    entry_bar_index = st['entry_bar_index']\n"
        "    CASH_BUFFER = st['cash_buffer']\n"
        "    tx_cost = st['tx_cost']\n"
        "    SLIPPAGE_MAP = st['slippage']\n"
        "    def _port_val(d):\n"
        "        return st['pv']\n"
        "    def _get_fill_price(c, d):\n"
        "        return st['px'].get(c, 0.0)\n"
        "    def _cur_lev(d, c):\n"
        "        return st['lev'].get(c, 0.0)\n"
        "    def _get_bar_index(c, d):\n"
        "        return 0\n"
        f"{body}\n"
        "    def _go(tw, date):\n"
        "        _execute_rebalance(tw, date)\n"
        "        return dict(capital=capital, holdings=dict(holdings),\n"
        "                    margins=dict(margins), entry_levs=dict(entry_levs))\n"
        "    return _go\n"
    )
    ns: dict = {}
    exec(compile(outer, "<extracted>", "exec"), ns)
    return ns["_mk"](state)


def case_lev_down():
    """4배 → 3배. 라이브는 먼저 비율만큼 팔고 레버리지를 바꾼다."""
    coins = ["AAA", "BBB", "CCC"]
    st = dict(
        capital=2.5,
        holdings={c: 3.3 for c in coins},
        margins={c: 82.5 for c in coins},          # 3.3 × 100 / 4
        entry_prices={c: 100.0 for c in coins},
        entry_levs={c: 4.0 for c in coins},
        entry_bar_index={c: 0 for c in coins},
        cash_buffer=0.01,
        tx_cost=0.0,
        slippage={c: 0.0 for c in coins},
        pv=250.0,
        px={c: 100.0 for c in coins},
        lev={c: 3.0 for c in coins},
    )
    go = _make_runner(st)
    out = go({c: 1.0 / 3 for c in coins}, "2026-01-01")
    live_margin = {c: out["holdings"][c] * 100.0 / 3.0 for c in coins}
    return out, live_margin


def case_lev_up():
    """2배 → 4배. 풀린 개시증거금이 살 수 있는 현금이 되는지."""
    coins = ["AAA", "BBB"]
    st = dict(
        capital=0.0,
        holdings={c: 1.0 for c in coins},
        margins={c: 50.0 for c in coins},          # 1 × 100 / 2
        entry_prices={c: 100.0 for c in coins},
        entry_levs={c: 2.0 for c in coins},
        entry_bar_index={c: 0 for c in coins},
        cash_buffer=0.01,
        tx_cost=0.0,
        slippage={c: 0.0 for c in coins},
        pv=100.0,
        px={c: 100.0 for c in coins},
        lev={c: 4.0 for c in coins},
    )
    go = _make_runner(st)
    out = go({c: 0.5 for c in coins}, "2026-01-01")
    return out


def main() -> int:
    out, live = case_lev_down()
    print("=== 4배 → 3배 (반례) ===")
    for c in sorted(out["holdings"]):
        print(f"  {c}: 수량 {out['holdings'][c]:.4f}  증거금 {out['margins'][c]:.4f}"
              f"  라이브식 {live[c]:.4f}  차이 {out['margins'][c]-live[c]:+.4f}")
    print(f"  현금 {out['capital']:.4f}  (토론 반례에서 엔진 62.5 대 라이브 2.5)")
    ok_m = all(abs(out["margins"][c] - live[c]) < 1e-6 for c in out["margins"])
    ok_c = abs(out["capital"] - 2.5) < 1e-6
    print(f"  증거금 일치 {ok_m} · 현금 일치 {ok_c}")

    out2 = case_lev_up()
    print("\n=== 2배 → 4배 (풀린 증거금이 쓰이나) ===")
    for c in sorted(out2["holdings"]):
        print(f"  {c}: 수량 {out2['holdings'][c]:.4f}  증거금 {out2['margins'][c]:.4f}"
              f"  레버리지 {out2['entry_levs'][c]:.0f}")
    print(f"  현금 {out2['capital']:.4f}")
    want_qty = 100.0 * 0.5 * 0.99 * 4 / 100.0   # pv × w × (1-버퍼) × L ÷ 가격
    ok_up = all(abs(out2["holdings"][c] - want_qty) < 1e-6 for c in out2["holdings"])
    print(f"  목표 수량 {want_qty:.4f} 도달 {ok_up}")

    if ok_m and ok_c and ok_up:
        print("\nALL PASS (증거금 재표시 자리 = 매도와 매수 사이)")
        return 0
    print("\nFAILED — 재표시 자리가 라이브와 어긋난다")
    return 1


if __name__ == "__main__":
    sys.exit(main())
