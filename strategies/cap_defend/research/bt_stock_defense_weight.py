"""방어 국면 비중식 차이(라이브 1/n 균등 vs 채택 BT cap 1/3+현금)의 성과 영향 측정.

계기: parity_stock.py (2026-08-25) 로 라이브 주식 선정이 채택 BT 와 공격 국면 1631일 100% 일치,
카나리 0 불일치, 방어 종목 0 불일치임을 확인했다. 유일하게 갈리는 곳이 방어 국면 비중식이다.
방어 픽이 3개 미만인 날(792 방어일 중 310일) 라이브는 뽑힌 것에 1/n 로 몰고, BT 는 1/3 씩만
주고 남은 슬롯을 현금으로 둔다. 즉 라이브가 방어자산을 더 많이 든다.

이 차이가 성과를 얼마나 바꾸는지를 잰다. 바꾸는 축은 방어 비중식 하나뿐이고 나머지는 채택 설정
그대로다(3-mom 30/72/230, 스냅 3 × 간격 69 × 스태거 23, 드리프트 5pp, 현금버퍼 7%, 거래비용 0.1%).

판정 (결과 보기 전 고정):
  두 방식의 윈도우 rank-sum 평균 순위 차이가 0.1 미만이면 무차별 — 문서에 구조 차이로만 적고 둔다.
  BT 방식(cap+현금)이 유의하게 낫다면 라이브를 BT 에 맞춘다(방어도 cap 1/3).
  라이브 방식이 유의하게 낫다면 BT 를 라이브에 맞추고 채택 성과표를 다시 낸다.
  어느 쪽이든 코드 변경은 별도 승인 사항이다 — 이 스크립트는 측정만 한다.

실행: python3 strategies/cap_defend/research/bt_stock_defense_weight.py
"""
from __future__ import annotations
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')

from stock_engine import load_prices                       # noqa: E402
from bt_stock_coin_v3 import precompute, half_t, CASH_KEY, TX  # noqa: E402
from bt_stock_mom3 import fresh_pick_3mom                   # noqa: E402
from bt_stock_single_snap import picks_to_target, DEF_TICKERS  # noqa: E402
from bt_stock_mom_grid import window_rank_sum_multi         # noqa: E402

MS, MID, ML = 30, 72, 230
THR = 0.05
CASH_BUF = 0.07
SNAP_INT = 69
N_SNAPS = 3
STAGGER = 23
START = "2017-01-01"
END = "2026-08-24"


def select_off_mode(d, mom_def, cash_buf, def_mode):
    """def_mode: 'cap' (채택 BT) | 'ew' (라이브 stock_strategy_v25.compute_defense)."""
    scores = []
    for t in DEF_TICKERS:
        r = mom_def[126].at[d, t] if t in mom_def[126].columns else np.nan
        if pd.notna(r) and r > 0:
            scores.append((t, r))
    scores.sort(key=lambda x: -x[1])
    picks = [t for t, _ in scores[:3]]
    return picks_to_target(picks, cash_buf, def_mode)


def run(pdf, ranked, mom_off, mom_def, canary, anchor, def_mode):
    sim_dates = pdf.index[(pdf.index >= START) & (pdf.index <= END)]
    if len(sim_dates) < 50:
        return None
    snaps = [{'phase': (anchor + k * STAGGER) % SNAP_INT, 'target': {CASH_KEY: 1.0}}
             for k in range(N_SNAPS)]
    holdings = {CASH_KEY: 1.0}
    prev_can = bool(canary.iloc[0]) if len(canary) else False

    def refresh(d, can_now):
        for s in snaps:
            if can_now:
                picks = fresh_pick_3mom(ranked.at[d], mom_off[MS].loc[d],
                                        mom_off[MID].loc[d], mom_off[ML].loc[d])
                s['target'] = picks_to_target(picks, CASH_BUF, 'cap')
            else:
                s['target'] = select_off_mode(d, mom_def, CASH_BUF, def_mode)

    def merge():
        agg = {}
        for s in snaps:
            for k, v in s['target'].items():
                agg[k] = agg.get(k, 0.0) + v / N_SNAPS
        return agg

    equity = []
    for i, d in enumerate(sim_dates):
        if i > 0:
            prev_d = sim_dates[i - 1]
            for k in list(holdings.keys()):
                if k == CASH_KEY or k not in pdf.columns:
                    continue
                p_prev = pdf.at[prev_d, k]
                p_now = pdf.at[d, k]
                if pd.notna(p_prev) and pd.notna(p_now) and p_prev > 0:
                    holdings[k] = holdings[k] * (p_now / p_prev)
        can_now = bool(canary.at[d]) if d in canary.index else prev_can
        for s in snaps:
            if (i - s['phase']) >= 0 and (i - s['phase']) % SNAP_INT == 0:
                if can_now:
                    picks = fresh_pick_3mom(ranked.at[d], mom_off[MS].loc[d],
                                            mom_off[MID].loc[d], mom_off[ML].loc[d])
                    s['target'] = picks_to_target(picks, CASH_BUF, 'cap')
                else:
                    s['target'] = select_off_mode(d, mom_def, CASH_BUF, def_mode)
        if can_now != prev_can:
            refresh(d, can_now)
            prev_can = can_now
        target = merge()
        total = sum(holdings.values())
        if total <= 0:
            holdings = {CASH_KEY: 1.0}
            total = 1.0
        cur_w = {k: v / total for k, v in holdings.items()}
        ht = half_t(cur_w, target)
        if ht >= THR:
            refresh(d, can_now)
            target = merge()
            pv = total * (1 - TX * ht)
            holdings = {k: pv * w for k, w in target.items() if w > 0}
        equity.append(sum(holdings.values()))
        prev_can = can_now
    return pd.Series(equity, index=sim_dates).dropna()


def stats(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    peak = eq.cummax()
    mdd = float((eq / peak - 1).min())
    ret = eq.pct_change().dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    return dict(CAGR=cagr, MDD=mdd, Cal=cagr / abs(mdd) if mdd < 0 else 0.0, Sharpe=sharpe)


def main():
    t0 = time.time()
    tickers = sorted(set(('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ')) | set(DEF_TICKERS))
    pm = load_prices(tickers, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [MS, MID, ML], [126])
    print(f"데이터 준비 {time.time()-t0:.1f}s", flush=True)

    sums = defaultdict(float)
    wins = defaultdict(int)
    n_all = 0
    agg = defaultdict(list)
    for anchor in range(11):
        eqs = {}
        for mode, tag in (('cap', 'BT_cap13_cash'), ('ew', 'LIVE_ew_1n')):
            eq = run(pdf, ranked, mom_off, mom_def, canary, anchor, mode)
            if eq is not None:
                eqs[tag] = eq
                agg[tag].append(stats(eq))
        if len(eqs) < 2:
            continue
        rs = window_rank_sum_multi(eqs)
        if rs is None:
            continue
        s, w, n = rs
        for k, v in s.items():
            sums[k] += v
        for k, v in w.items():
            wins[k] += v
        n_all += n
        print(f"  anchor {anchor} 완료 ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n윈도우 {n_all}개 (11 앵커 합산)")
    print(f"  {'방식':<18} {'평균순위':>8} {'승률':>7} {'CAGR':>8} {'MDD':>8} {'Calmar':>7} {'Sharpe':>7}")
    for tag in ('BT_cap13_cash', 'LIVE_ew_1n'):
        rows = agg[tag]
        print(f"  {tag:<18} {sums[tag]/max(n_all,1):>8.3f} {wins[tag]/max(n_all,1)*100:>6.1f}% "
              f"{np.mean([r['CAGR'] for r in rows])*100:>7.2f}% "
              f"{np.mean([r['MDD'] for r in rows])*100:>7.2f}% "
              f"{np.mean([r['Cal'] for r in rows]):>7.3f} "
              f"{np.mean([r['Sharpe'] for r in rows]):>7.3f}")
    gap = abs(sums['BT_cap13_cash'] - sums['LIVE_ew_1n']) / max(n_all, 1)
    print(f"\n평균순위 차이 {gap:.3f} → {'무차별(0.1 미만)' if gap < 0.1 else '유의차'}")
    print(f"총 소요 {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
