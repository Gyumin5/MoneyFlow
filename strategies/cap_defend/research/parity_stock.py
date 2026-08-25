"""주식 라이브 선정함수 vs 채택 BT 선정함수 일별 정합 검증 (읽기전용, 2026-08-25).

코인 현물·선물에는 이미 같은 하니스가 있다(parity_spot.py / parity_fut.py, 둘 다 100% 일치).
주식만 없었다. 2026-06-06 에 라이브 Z-랭킹 모멘텀이 순수 252일(V15 잔재)로 들어가 채택 BT 와
종목 선정이 16.8% 날 어긋난 사고가 있었고, 그때는 replay-diff 로 뒤늦게 발견했다.
같은 종류의 어긋남을 상시로 잡을 기준선을 만든다.

방법: 같은 가격 시계열을 양쪽에 주입한다.
  라이브 = stock_strategy_v25.compute_strategy(각 날짜까지 자른 가격, 전날 risk_on)
  BT    = bt_stock_coin_v3.precompute 의 Z랭킹 + bt_stock_mom3.fresh_pick_3mom(30/72/230)
          + bt_stock_single_snap.picks_to_target(cap) / select_off
스냅 머신·드리프트·체결은 둘 다 끈다(그 층은 이 검증 대상이 아니다). 하루치 선정만 비교한다.
현금버퍼는 BT 쪽 0 으로 두고 비교한다 — 라이브는 executor 가 버퍼를 곱하므로 비중 정의가
"위험자산 내 비율"로 같아진다.

판정 (결과 보기 전 고정):
  카나리 불일치일 0, 공격 국면 picks 불일치일 0 이어야 통과.
  방어 국면 비중식 차이는 별도로 센다 — 라이브는 1/n 균등, BT 는 cap 1/3 + 현금.
  이건 코드를 읽으면 바로 보이는 구조 차이라 "발견"이 아니라 "확인"이다. 몇 날이나
  실제로 갈리는지가 필요한 값이다.

실행: python3 strategies/cap_defend/research/parity_stock.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/gmoh/mon/251229'
CAP = os.path.join(ROOT, 'strategies', 'cap_defend')
sys.path.insert(0, CAP)
sys.path.insert(0, os.path.join(CAP, 'research'))

from stock_engine import load_prices                      # noqa: E402
import stock_strategy_v25 as live                          # noqa: E402
from bt_stock_coin_v3 import precompute, OFF_R7, DEF_TICKERS, CASH_KEY  # noqa: E402
from bt_stock_mom3 import fresh_pick_3mom                  # noqa: E402
from bt_stock_single_snap import picks_to_target, select_off  # noqa: E402

START = '2017-01-01'
END = '2026-08-24'
MS, MID, ML = live.MOM_SHORT, live.MOM_MID, live.MOM_LONG   # 30 / 72 / 230
OUT = os.path.join(ROOT, 'state', 'parity', 'stock_result.json')


def norm(w):
    """현금 제외 위험자산 비중만 남기고 소수 6자리로 정규화."""
    return {k: round(float(v), 6) for k, v in w.items()
            if k != live.CASH_ASSET and k != CASH_KEY and abs(v) > 1e-9}


def main():
    tickers = sorted(set(OFF_R7) | set(DEF_TICKERS))
    print(f"[가격] {len(tickers)}종 로딩: {tickers}", flush=True)
    pm = load_prices(tickers, start='2005-01-01')
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    pdf = pdf.dropna(how='all')
    print(f"  {pdf.index[0].date()} ~ {pdf.index[-1].date()} ({len(pdf)}일)", flush=True)

    ranked, mom_off, mom_def, canary = precompute(pdf, [MS, MID, ML], [126])

    dates = pdf.index[(pdf.index >= START) & (pdf.index <= END)]
    print(f"[비교] {dates[0].date()} ~ {dates[-1].date()} ({len(dates)}일)", flush=True)

    # 라이브 카나리 초기상태를 BT 와 맞춘다 (창 시작 이전 상태 승계)
    before = pdf.index[pdf.index < dates[0]]
    prev_risk_on = bool(canary.at[before[-1]]) if len(before) else False

    can_mismatch = []
    on_mismatch = []
    off_weight_diff = []
    on_days = off_days = 0
    series = {c: pdf[c] for c in pdf.columns}

    for i, d in enumerate(dates):
        sl = {c: s.loc[:d] for c, s in series.items()}
        res = live.compute_strategy(sl, prev_risk_on)
        live_on = bool(res['risk_on'])
        bt_on = bool(canary.at[d])

        if live_on != bt_on:
            can_mismatch.append({'date': str(d.date()), 'live': live_on, 'bt': bt_on})

        if bt_on:
            on_days += 1
            bt_picks = fresh_pick_3mom(ranked.at[d], mom_off[MS].loc[d],
                                       mom_off[MID].loc[d], mom_off[ML].loc[d])
            bt_w = norm(picks_to_target(bt_picks, 0.0, 'cap'))
        else:
            off_days += 1
            bt_picks = []
            bt_w = norm(select_off(d, mom_def, 0.0, 'cap'))

        live_w = norm(res['weights'])

        if live_on == bt_on:
            if bt_on:
                if set(live_w) != set(bt_w) or live_w != bt_w:
                    on_mismatch.append({'date': str(d.date()),
                                        'live': live_w, 'bt': bt_w})
            else:
                if set(live_w) != set(bt_w):
                    off_weight_diff.append({'date': str(d.date()), 'kind': 'picks',
                                            'live': live_w, 'bt': bt_w})
                elif live_w != bt_w:
                    off_weight_diff.append({'date': str(d.date()), 'kind': 'weights',
                                            'live': live_w, 'bt': bt_w})

        prev_risk_on = live_on
        if (i + 1) % 500 == 0:
            print(f"  … {i+1}/{len(dates)}일", flush=True)

    n = len(dates)
    off_pick_diff = [x for x in off_weight_diff if x['kind'] == 'picks']
    off_w_only = [x for x in off_weight_diff if x['kind'] == 'weights']
    res = {
        'window': [str(dates[0].date()), str(dates[-1].date())],
        'days': n,
        'on_days': on_days,
        'off_days': off_days,
        'canary_mismatch_days': len(can_mismatch),
        'canary_mismatch_first10': can_mismatch[:10],
        'offense_mismatch_days': len(on_mismatch),
        'offense_mismatch_first10': on_mismatch[:10],
        'defense_pick_diff_days': len(off_pick_diff),
        'defense_pick_diff_first10': off_pick_diff[:10],
        'defense_weight_only_diff_days': len(off_w_only),
        'defense_weight_only_first10': off_w_only[:10],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)

    print(f"\n비교일 {n} (공격 {on_days} / 방어 {off_days})")
    print(f"  카나리 불일치일: {len(can_mismatch)}")
    print(f"  공격 국면 선정·비중 불일치일: {len(on_mismatch)}")
    print(f"  방어 국면 종목 불일치일: {len(off_pick_diff)}")
    print(f"  방어 국면 비중만 다른 날: {len(off_w_only)}")
    ok = not can_mismatch and not on_mismatch and not off_pick_diff
    print(f"\n판정: {'PASS' if ok else 'FAIL'} "
          f"(공격 정합률 {100*(on_days-len(on_mismatch))/max(on_days,1):.4f}%)")
    print(f"  결과: {OUT}")


if __name__ == '__main__':
    main()
