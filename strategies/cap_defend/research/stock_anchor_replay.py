"""주식 앵커 스태거 복원용 반사실 재생 (read-only).

물음: 2026-07-08 스태거 붕괴가 없었다면, 오늘 주식 슬리브는 무엇을 들고 있어야 하나.

방법: 복원된 시간표(09-15 / 10-08 / 10-31)에서 69일씩 거꾸로 세면 각 snap 의 직전
앵커일이 나온다 — snap0 07-08, snap1 07-31, snap2 08-23. 그 앞은 04-30 / 05-23 / 06-15.
06-15 상태로 시작해 매 거래일 라이브 규칙(카나리 플립 → 앵커 → 드리프트 리필 → 체결)을
그대로 돌린다. 신호 계산은 라이브와 같은 함수(stock_strategy_v25)를 쓰고, 신호일은
직전 거래일(T-1), 체결은 당일 종가로 본다.

라이브 state 를 읽지도 쓰지도 않는다. 출력만 낸다.
실행: cd /home/gmoh/mon/251229 && python3 strategies/cap_defend/research/stock_anchor_replay.py
"""
import os
import sys
from datetime import date, timedelta

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
import stock_strategy_v25 as ss  # noqa: E402

SNAP_PERIOD_DAYS = 69
SNAP_STAGGER_DAYS = 23
N_SNAPS = 3
CASH_BUFFER = 0.07
DRIFT_THRESHOLD = 0.05

# 복원된 시간표에서 69 를 빼서 얻은 직전 앵커일. 그 앞 회차가 재생 시작점이다.
LAST_ANCHORS = {0: date(2026, 7, 8), 1: date(2026, 7, 31), 2: date(2026, 8, 23)}
PREV_ANCHORS = {k: v - timedelta(days=SNAP_PERIOD_DAYS) for k, v in LAST_ANCHORS.items()}
END = date(2026, 8, 28)          # 마지막 확정 종가일(= 어제 기준 라이브가 본 날)


def fetch_prices():
    import yfinance as yf
    tickers = sorted(set(ss.OFFENSIVE_STOCK_UNIVERSE + ss.DEFENSIVE_STOCK_UNIVERSE
                         + [ss.STOCK_CANARY_TICKER]))
    out = {}
    for t in tickers:
        df = yf.download(t, start='2024-08-01', end=(END + timedelta(days=1)).isoformat(),
                         progress=False, auto_adjust=True)
        s = df['Close'].squeeze().dropna()
        s.index = pd.to_datetime(s.index)
        out[t] = s[s.index.date <= END]
    return out


def signal_on(prices, sig_day, prev_risk_on):
    """sig_day 종가까지만 보고 신호를 만든다(= 그 다음 거래일에 체결할 신호)."""
    sl = {t: s[s.index.date <= sig_day] for t, s in prices.items()}
    off_picks, off_w, _ = ss.compute_offense(sl)
    def_picks, def_w, _ = ss.compute_defense(sl)
    risk_on, _ = ss.eem_canary(sl.get(ss.STOCK_CANARY_TICKER), prev_risk_on)
    return {
        'risk_on': bool(risk_on),
        'picks': list(off_picks) if risk_on else list(def_picks),
        'weights': dict(off_w) if risk_on else dict(def_w),
    }


def merged_target(snaps):
    m = {}
    for s in snaps.values():
        for tk, w in (s.get('weights') or {}).items():
            m[tk] = m.get(tk, 0.0) + w / N_SNAPS
    return m


def with_buffer(target):
    out = {k: v * (1.0 - CASH_BUFFER) for k, v in target.items()}
    out['Cash'] = out.get('Cash', 0.0) + max(0.0, 1.0 - sum(out.values()))
    return out


def half_turnover(cur, tgt):
    keys = set(cur) | set(tgt)
    return sum(abs(cur.get(k, 0.0) - tgt.get(k, 0.0)) for k in keys) / 2


def main():
    prices = fetch_prices()
    days = sorted({d.date() for d in prices[ss.STOCK_CANARY_TICKER].index})
    start = PREV_ANCHORS[2]                                   # 2026-06-15
    tdays = [d for d in days if d >= start]
    print(f"거래일 {tdays[0]} ~ {tdays[-1]} ({len(tdays)}일)")

    # ── 초기 상태: 각 snap 을 자기 직전 앵커일 신호로 채운다 ──
    snaps, prev_risk_on = {}, None
    for sid in range(N_SNAPS):
        a = PREV_ANCHORS[sid]
        sig_day = max(d for d in days if d < a)
        sig = signal_on(prices, sig_day, prev_risk_on)
        prev_risk_on = sig['risk_on']
        snaps[sid] = {'picks': sig['picks'], 'weights': sig['weights'],
                      'next': a + timedelta(days=SNAP_PERIOD_DAYS)}
        print(f"  init snap{sid}: 앵커 {a} (신호 {sig_day}) → {sig['picks']} "
              f"next={snaps[sid]['next']}")

    holdings = with_buffer(merged_target(snaps))
    events = []

    for i, d in enumerate(tdays):
        if i == 0:
            continue
        prev_d = tdays[i - 1]
        # 시세 변동으로 보유 비중이 흐른다 (현금은 그대로)
        grown = {}
        for k, w in holdings.items():
            if k == 'Cash':
                grown[k] = w
                continue
            s = prices.get(k)
            r = 1.0
            if s is not None:
                try:
                    r = float(s[s.index.date <= d].iloc[-1]) / float(s[s.index.date <= prev_d].iloc[-1])
                except Exception:
                    r = 1.0
            grown[k] = w * r
        tot = sum(grown.values())
        holdings = {k: v / tot for k, v in grown.items()}

        sig = signal_on(prices, prev_d, prev_risk_on)
        need = False

        if prev_risk_on is not None and sig['risk_on'] != prev_risk_on:
            for sid in range(N_SNAPS):
                snaps[sid]['picks'] = list(sig['picks'])
                snaps[sid]['weights'] = dict(sig['weights'])
            events.append((d, '카나리 플립', f"{prev_risk_on}→{sig['risk_on']} 전 snap 교체"))
            need = True
        prev_risk_on = sig['risk_on']

        for sid in range(N_SNAPS):
            if d >= snaps[sid]['next']:
                before = list(snaps[sid]['picks'])
                snaps[sid]['picks'] = list(sig['picks'])
                snaps[sid]['weights'] = dict(sig['weights'])
                snaps[sid]['next'] = snaps[sid]['next'] + timedelta(days=SNAP_PERIOD_DAYS)
                events.append((d, f'앵커 snap{sid}', f"{before} → {sig['picks']}"))
                need = True

        if not need:
            ht = half_turnover(holdings, with_buffer(merged_target(snaps)))
            if ht >= DRIFT_THRESHOLD:
                before = {sid: list(snaps[sid]['picks']) for sid in range(N_SNAPS)}
                for sid in range(N_SNAPS):
                    snaps[sid]['picks'] = list(sig['picks'])
                    snaps[sid]['weights'] = dict(sig['weights'])
                events.append((d, '드리프트 리필', f"ht={ht*100:.2f}pp, {before} → {sig['picks']}"))
                need = True

        if need:
            holdings = with_buffer(merged_target(snaps))

    print("\n=== 사건 ===")
    for d, kind, detail in events:
        print(f"  {d} {kind}: {detail}")

    print("\n=== 오늘 있어야 할 상태 ===")
    for sid in range(N_SNAPS):
        print(f"  snap{sid}: {snaps[sid]['picks']} next={snaps[sid]['next']}")
    tgt = with_buffer(merged_target(snaps))
    print("  합산 목표(현금완충 7% 반영):")
    for k, v in sorted(tgt.items(), key=lambda x: -x[1]):
        print(f"    {k:6s} {v*100:6.2f}%")

    print("\n=== 실제 보유 (2026-08-29 조회) 와 비교 ===")
    actual = {'SPY': 0.3026, 'EEM': 0.3018, 'VNQ': 0.2935, 'Cash': 0.1021}
    for k in sorted(set(tgt) | set(actual), key=lambda x: -(tgt.get(x, 0) + actual.get(x, 0))):
        print(f"    {k:6s} 목표 {tgt.get(k, 0)*100:6.2f}%  실제 {actual.get(k, 0)*100:6.2f}%  "
              f"차이 {(tgt.get(k, 0)-actual.get(k, 0))*100:+6.2f}pp")
    print(f"  half_turnover = {half_turnover(actual, tgt)*100:.2f}pp")


if __name__ == '__main__':
    main()
