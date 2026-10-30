"""합성 모형을 실제 운영으로 바꿔 비중을 다시 겨룬다 (2026-10-30).

왜 재나. 앞선 측정 셋(alloc_mix_window_rs · alloc_mix_ratio · alloc_mix_ratio_edge)은
`bt_coin_param_resweep.build_alloc` 로 합성했는데 그건 날마다 비중을 목표로 되돌리는
고정비중 합성이다. 실제 운영은 그렇지 않다 — 트리거(반회전율 20%p 또는 상대부족 20% +
그 슬리브 카나리 ON)가 켜질 때만 사람이 송금한다. 날마다 되돌리는 쪽은 변동성 큰 슬리브에
유리하고, 선물은 레버리지 2~4배라 그 효과를 가장 크게 받는다.

그래서 history #92 가 "창 순위합으로는 자산배분을 못 고른다" 로 닫힐 때 재개 조건을
이렇게 적었다 — 합성 모형을 실제 운영으로 바꿔 재서 선물 쪽 단조가 꺾이는지 본다.
꺾이면 창 순위합을 자산배분 비교에 다시 쓸 수 있다.

바꾼 것은 합성 한 축뿐이다. 슬리브 백테스트·앵커·비용 조건·창 길이·이동폭은 그대로 두고,
같은 비중 열두 칸을 두 합성 모형으로 각각 재서 나란히 찍는다.

실제 운영 모형:
  - 세 슬리브 자본이 각자 수익률로 따로 자란다 (날마다 안 되돌린다).
  - 날마다 트리거를 본다. T1 = 반회전율 sum|목표-현재|/2 >= 0.20.
    T3U_can = max (목표-현재)/목표 >= 0.20 이고 그 슬리브 카나리가 ON.
  - 발화하면 그날 목표 비중으로 되돌린다 (송금 지연 0일 — 낙관 쪽 끝).
  - 송금 수수료는 0 으로 둔다. 기존 모형도 0 이라 두 모형의 차이가 합성 방식 하나로 남는다.

카나리는 이렇게 읽는다:
  - 주식 = EEM 카나리 원본 (bt_stock_coin_v3.precompute 가 주는 그 시리즈).
  - 현물·선물 = 그날 목표에 현금 아닌 비중이 있나 (엔진 _trace 의 target).
    카나리 ON 인데 건강한 코인이 하나도 없어 전액 현금인 날은 OFF 로 읽힌다 — 드문 자리고
    그 날은 어차피 송금해도 살 게 없다.

실행: python3 alloc_mix_real_rebal.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (이름, 주식, 현물, 선물). 앞 네 칸은 제약(현물 >= 선물) 안 4차와 같은 칸이고,
# 뒤 네 칸은 기존 모형에서 1위로 올라오던 선물 쏠림 끝 칸이다. 단조가 꺾이는지 보려면
# 같은 표 안에 둘 다 있어야 한다.
MIXES = [
    ('현행 60/25/15', 0.60, 0.25, 0.15),
    ('80/10/10', 0.80, 0.10, 0.10),
    ('70/15/15', 0.70, 0.15, 0.15),
    ('60/20/20', 0.60, 0.20, 0.20),
    ('50/25/25', 0.50, 0.25, 0.25),
    ('70/18/12', 0.70, 0.18, 0.12),
    ('60/24/16', 0.60, 0.24, 0.16),
    ('70/10/20', 0.70, 0.10, 0.20),
    ('70/5/25', 0.70, 0.05, 0.25),
    ('70/0/30', 0.70, 0.00, 0.30),
    ('60/13/27', 0.60, 0.1333, 0.2667),
    ('60/0/40', 0.60, 0.00, 0.40),
]

BASE = MIXES[0][0]      # 직접 승률의 기준선. 이름을 바꾸면 여기만 따라온다.
HT_THR = 0.20           # T1 반회전율 (recommend_personal.REBAL_HT_THRESHOLD)
T3U_REL = 0.20          # T3U_can 상대부족 (recommend_personal.REBAL_T3U_REL)


def canary_from_trace(trace: list) -> pd.Series:
    """엔진 _trace 에서 '그날 코인을 들고 있었나' 를 뽑는다. 전액 현금이면 False."""
    idx, val = [], []
    for row in trace:
        tgt = row['target']
        risk = sum(v for k, v in tgt.items() if k.upper() != 'CASH')
        idx.append(row['date'])
        val.append(risk > 1e-9)
    return pd.Series(val, index=pd.DatetimeIndex(idx))


def build_alloc_real(eq_st, eq_sp, eq_fu, can_st, can_sp, can_fu, w,
                     ht_thr: float = HT_THR, t3u_rel: float = T3U_REL):
    """트리거 발화 때만 목표로 되돌리는 합성. 반환 = (자산곡선, 발화 횟수).

    ht_thr·t3u_rel 기본값은 라이브 값(0.20·0.20)이다. 비중마다 자기 최적 타이밍을
    찾을 때만 다른 값을 넘긴다 — alloc_mix_trigger_opt.py 가 그렇게 쓴다.
    """
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    common = common.sort_values()
    rets = np.column_stack([
        eq_st.loc[common].pct_change().fillna(0).to_numpy(),
        eq_sp.loc[common].pct_change().fillna(0).to_numpy(),
        eq_fu.loc[common].pct_change().fillna(0).to_numpy(),
    ])
    tgt = np.array(w, dtype=float)
    cans = np.column_stack([
        can_st.reindex(common).ffill().fillna(False).to_numpy().astype(bool),
        can_sp.reindex(common).ffill().fillna(False).to_numpy().astype(bool),
        can_fu.reindex(common).ffill().fillna(False).to_numpy().astype(bool),
    ])
    cap = tgt.copy()
    out = np.empty(len(common))
    fires = 0
    for i in range(len(common)):
        cap = cap * (1.0 + rets[i])
        total = cap.sum()
        if total <= 0:
            out[i:] = 0.0
            break
        out[i] = total
        cur = cap / total
        fire = abs(tgt - cur).sum() / 2.0 >= ht_thr
        if not fire:
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(tgt > 0, (tgt - cur) / np.where(tgt > 0, tgt, 1.0), 0.0)
            if ((rel >= t3u_rel) & cans[i]).any():
                fire = True
        if fire:
            cap = tgt * total
            fires += 1
    return pd.Series(out, index=common), fires


def table(label: str, sums, wins, head, total_n, cal_rows, cagr_rows, mdd_rows,
          fire_rows=None, anchors=0):
    print()
    print(f'=== {label} (앵커 {anchors}곳 합산, 창 {total_n}개)')
    hdr = (f'  {"비중":>16} {"평균순위":>9} {"창 1위":>8} {"현행 직접승":>11}'
           f' {"칼마":>7} {"연복리":>8} {"낙폭":>8}')
    if fire_rows is not None:
        hdr += f' {"송금/년":>8}'
    print(hdr)
    for name, avg in sorted(((k, sums[k] / total_n) for k in sums), key=lambda x: x[1]):
        line = (f'  {name:>16} {avg:>9.3f} {wins[name]:>7}회'
                f' {head[name] / total_n * 100:>10.1f}%'
                f' {float(np.mean(cal_rows[name])):>7.2f}'
                f' {float(np.mean(cagr_rows[name])):>7.1f}%'
                f' {float(np.mean(mdd_rows[name])):>7.1f}%')
        if fire_rows is not None:
            line += f' {float(np.mean(fire_rows[name])):>8.1f}'
        print(line)


def main() -> int:
    import bt_coin_param_resweep as R
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal

    t0 = time.time()
    print('코인 일봉 로드 중...')
    R._bars, R._funding = load_data('D')
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f'  심볼 {len(R._bars)}종 ({time.time() - t0:.0f}초)')

    print('주식 카나리(EEM) 로드 중...')
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pdf = pd.DataFrame(load_prices(ALL_TICKERS, start="2005-01-01"))
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    _r, _mo, _md, can_st = precompute(pdf, [30, 72, 230], [42, 63, 126])
    can_st = can_st.astype(bool)
    print(f'  주식 카나리 {len(can_st)}일, ON 비율 {can_st.mean() * 100:.1f}%'
          f' ({time.time() - t0:.0f}초)')
    print(f'  창 길이 {R.WIN_SIZES} · 이동폭 {R.STRIDES}')

    for stress in (False, True):
        label = '비용 5배' if stress else '기본 비용'
        agg = {}
        for mode in ('daily', 'real'):
            agg[mode] = dict(sums=defaultdict(float), wins=defaultdict(int),
                             head=defaultdict(int), n=0,
                             cal=defaultdict(list), cagr=defaultdict(list),
                             mdd=defaultdict(list), fire=defaultdict(list))
        anchors = 0
        for i in range(7):
            st = R.run_stock(i, stress)
            tr_sp: list = []
            sp = R.run_spot({}, R.PH_SPOT[i], stress, trace=tr_sp)
            tr_fu: list = []
            fu = R.run_fut({}, R.PH_FUT[i], stress, trace=tr_fu)
            if st is None or sp is None or fu is None:
                continue
            can_sp = canary_from_trace(tr_sp)
            can_fu = canary_from_trace(tr_fu)

            eqs = {'daily': {}, 'real': {}}
            for name, ws, wsp, wfu in MIXES:
                eqs['daily'][name] = R.build_alloc(st, sp, fu, w=(ws, wsp, wfu))
                eq, fires = build_alloc_real(st, sp, fu, can_st, can_sp, can_fu,
                                             (ws, wsp, wfu))
                eqs['real'][name] = eq
                yrs = (eq.index[-1] - eq.index[0]).days / 365.25
                agg['real']['fire'][name].append(fires / yrs)

            for mode in ('daily', 'real'):
                A = agg[mode]
                for name, eq in eqs[mode].items():
                    m = R.metrics(eq)
                    A['cal'][name].append(m['Cal'])
                    A['cagr'][name].append(m['CAGR'])
                    A['mdd'][name].append(m['MDD'])
                s, w, n = R.window_rs(eqs[mode])
                for k, v in s.items():
                    A['sums'][k] += v
                for k, v in w.items():
                    A['wins'][k] += v
                A['n'] += n
                # 창별 직접 승: 같은 창에서 현행보다 칼마가 높았나
                idx = sorted(set.intersection(*[set(e.index) for e in eqs[mode].values()]))
                for size in R.WIN_SIZES:
                    for stride in R.STRIDES:
                        for j in range(0, len(idx) - size, stride):
                            d0, d1 = idx[j], idx[j + size - 1]
                            cal = {}
                            for k, e in eqs[mode].items():
                                seg = e.loc[d0:d1].dropna()
                                if len(seg) < 30:
                                    cal = {}
                                    break
                                yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                                c = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                                md = float((seg / seg.cummax() - 1).min())
                                cal[k] = c / abs(md) if md < 0 else 0.0
                            if not cal:
                                continue
                            base = cal[BASE]
                            for k, v in cal.items():
                                if v > base:
                                    A['head'][k] += 1
            anchors += 1
            print(f'  [{label}] 앵커 {i} 끝 ({time.time() - t0:.0f}초)')

        if not agg['real']['n']:
            print(f'=== {label}: 창을 못 만들었다')
            continue
        A = agg['daily']
        table(f'{label} — 날마다 되돌림 (기존 모형)', A['sums'], A['wins'], A['head'],
              A['n'], A['cal'], A['cagr'], A['mdd'], None, anchors)
        A = agg['real']
        table(f'{label} — 트리거 발화 때만 송금 (실제 운영)', A['sums'], A['wins'],
              A['head'], A['n'], A['cal'], A['cagr'], A['mdd'], A['fire'], anchors)

    print()
    print(f'끝 ({time.time() - t0:.0f}초)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
