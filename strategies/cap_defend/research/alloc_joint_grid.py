"""자산배분 비중 × 리밸런싱 조건 합동 격자 — 작업 하나(앵커 1곳 × 비용 1조건) (2026-10-31).

운영자 지시(텔레그램 9440·9441): 제약은 주식 >= 60%, 현물 > 선물. 그 안에서 비중과
리밸런싱 조건의 최적 조합을 찾되 격자 설계는 먼저 토론해서 정하라.

설계는 ai-debate run-20261031T005336Z 가 확정했다(1라운드 합의, 참여 셋 전부 성공).
원문 /home/gmoh/claude-work/ai-debate/run-20261031T005336Z. 확정된 것:

  비중 40칸 = 주식 60·65·70·75·80·85·90·95% × 현물:선물 1:0·5:1·3:1·5:3·11:9.
    상한을 안 둔 이유는 운영자 제약이 하한만 줬기 때문이다 — 80 에서 끊으면 안 받은
    제한을 더하는 것이다. 계산한 비중은 정수로 반올림하지 않는다.
  조건 56칸 = 반회전율 10·15·20·25·30·40·50%p·꺼짐 × 상대부족 10·15·20·25·30·40%·꺼짐.
    꺼짐을 넣는 이유는 두 트리거 각각의 단독 효과와 무송금 기준선을 봐야 해서다.
  합동 2,240 조합. 앵커 7곳 × 비용 2조건 = 14 작업으로 나눠 병렬 계산한다.

토론이 더 제안한 학습·후행 분할은 안 쓴다. 운영자가 홀드아웃·워크포워드·연도별 순위를
금지하고 창 기반 순위합만 쓰라고 이미 정했고(메모리 feedback_no_holdout), 실제로 공통
구간이 1,467 영업일이라 1,008일을 후행으로 떼면 학습 쪽에 504일짜리 창 하나가 안 들어가
창 수가 0 이 된다. 선택 운은 남은 셋으로 본다 — 이웃 평지, 두 비용 조건, 앵커 7곳.

이 파일은 그 14 작업 중 하나다. 집계와 판정은 alloc_joint_agg.py 가 한다.

왜 새로 짰나(기존 alloc_mix_trigger_opt.py 를 안 늘렸나): 조합이 420 에서 2,240 으로
늘면 기존 경로가 못 쓴다. bt_coin_param_resweep.window_rs 는 조합마다 pandas 슬라이스를
떠서 창 441개를 돌므로 98만 번의 .loc 가 되고, build_alloc_real 도 조합마다 날짜
1,500회를 따로 돈다. 단일 프로세스 두 시간이 넘는다. 그래서 둘 다 조합 축으로
벡터화했다 — 날짜 루프 한 번이 2,240 조합을 동시에 전진시키고, 창 순위도 (조합 × 날짜)
행렬 한 장으로 잰다. 결과 정의는 기존과 같다(아래 재현 검사 참조).

실행: python3 alloc_joint_grid.py --anchor 0 --stress 0 --out <경로.npz>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- 격자 정의 (토론 확정분. 여기를 고치면 집계기의 재현 검사가 걸린다) -------------

STOCK_AXIS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# (이름, 현물 몫, 선물 몫). 코인 전체를 이 비로 가른다. 전부 현물은 1:0.
COIN_RATIOS = [('1:0', 1, 0), ('5:1', 5, 1), ('3:1', 3, 1), ('5:3', 5, 3),
               ('7:5', 7, 5), ('11:9', 11, 9), ('21:19', 21, 19)]
# 7:5 와 21:19 는 2026-10-31 토론 뒤에 더했다. 그 전 격자는 11:9 에서 끝났는데 최적이
# 거기 붙어서, 봉우리인지 격자 끝인지 구분이 안 됐다 — 상대부족 축에서 이미 같은 일을
# 겪었다. 21:19 는 현물 52.5 대 선물 47.5 로, "현물이 선물보다 크게" 라는 운영자 제약이
# 허용하는 거의 끝이다. 1:1 은 그 제약을 깨므로 안 넣는다.

# 꺼짐은 np.inf 로 둔다 — 임계에 영원히 안 닿으므로 그 트리거가 없는 것과 같다.
GRID_HT = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, np.inf]
# 상대부족 아래쪽 둘(2.5·5%)은 1회차 측정 뒤에 더했다. 토론 확정 격자의 바닥이 10% 였는데
# 최적이 거기 붙어서, 봉우리인지 벽인지 구분이 안 됐다 — 격자 끝 문제는 이 측정에서 이미
# 한 번 겪었다. 0 은 안 넣는다: 상대부족 0 은 "카나리가 켜진 슬리브가 목표에 모자라면
# 언제나" 라 매일 송금하는 것과 같고, 그건 조건이 아니라 다른 정책이다.
GRID_T3U = [0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, np.inf]
# 바닥 넷(0.5·1·1.5·2%)은 3회차에 더했다. 운영자가 송금 횟수 제한을 풀어서(텔레그램 9450)
# 2.5% 아래에 봉우리가 있는지 봐야 한다. 0% 는 안 넣는다 — 조건이 아니라 매일 되돌리는
# 다른 정책이라 같은 축에서 못 겨룬다.

LIVE_HT, LIVE_T3U = 0.20, 0.20     # recommend_personal.REBAL_HT_THRESHOLD / REBAL_T3U_REL
LIVE_MIX = (0.60, 0.25, 0.15)      # 현행 자산배분. 격자의 '60/5:3' 칸과 같은 자리다.


def build_mixes(stock_axis=None, ratios=None,
                allow_equal: bool = False) -> list[tuple[str, float, float, float]]:
    """제약(주식 >= 60, 현물 > 선물)을 만족하는 비중 40칸.

    축을 인자로 주면 그만큼만 만든다 — 운영자가 특정 비중을 지목해 물을 때 쓴다.
    `allow_equal` 은 현물 > 선물 걸름을 끈다. 기본 격자에는 안 쓴다(그 제약이 운영자
    것이라서). 손으로 비를 지목한 호출은 그 지목 자체가 제약 해제라 이 길로 온다.

    현행 60/25/15 는 따로 더하지 않는다 — 주식 60 의 5:3 이 바로 그 칸이라(코인 40 을
    5:3 으로 가르면 25/15) 더하면 같은 자리가 두 번 들어간다. 그러면 그 칸이 자기
    복제본과 동률을 이뤄 창마다 평균순위가 0.5 씩 밀리고, 경쟁자 수도 56 늘어 모든
    순위합이 함께 흔들린다. 하필 운영자가 가장 궁금해하는 칸이라 이름만 붙여 둔다.
    """
    out = []
    for st in (STOCK_AXIS if stock_axis is None else stock_axis):
        coin = 1.0 - st
        for rname, a, b in (COIN_RATIOS if ratios is None else ratios):
            sp = coin * a / (a + b)
            fu = coin - sp
            if not allow_equal and not (sp > fu or (fu == 0.0 and sp >= 0.0)):
                continue
            name = f'{st * 100:.0f}/{rname}'
            if np.allclose((st, sp, fu), LIVE_MIX):
                name += ' (현행)'
            out.append((name, st, sp, fu))
    return out


def build_combos(mixes):
    """(비중 이름, 주식, 현물, 선물, 반회전율, 상대부족) 전체 조합."""
    combos = []
    for mname, ws, wsp, wfu in mixes:
        for ht in GRID_HT:
            for t3u in GRID_T3U:
                combos.append((mname, ws, wsp, wfu, ht, t3u))
    return combos


# --- 합성 (조합 축 벡터화) ---------------------------------------------------------

def build_alloc_real_many(eq_st, eq_sp, eq_fu, can_st, can_sp, can_fu,
                          tgt: np.ndarray, ht: np.ndarray, t3u: np.ndarray,
                          delay: int = 0, transfer_cost: float = 0.0):
    """모든 조합을 동시에 전진시킨다. 반환 = (공통 날짜, 자산곡선 (K,N), 발화수 (K,)).

    transfer_cost 는 계좌 사이로 옮기는 금액에 물리는 비율이다. 0 이면 공짜 송금이고
    그게 기본값이다 — 기존 결과를 그대로 재현하기 위해서다. 슬리브 안의 매매비용은
    각 슬리브 엔진이 이미 물리지만 계좌 간 이동(환전 스프레드·출금 수수료)은 아무도
    안 물린다. 이 값은 그 빠진 자리를 메운다. 무는 금액은 그날 실제로 옮긴 몫
    sum|목표금액 - 현재금액|/2 이고, 총자산에서 깎은 뒤 목표 비중으로 다시 나눈다.

    delay 는 발화일과 실제 송금 사이의 영업일 수다. 0 이면 그날 바로 되돌린다.
    라이브는 알림을 받고 사람이 송금하므로 0 이 아니고, 결정 규칙(Decisions In Force)이
    5영업일 안에 보내라고 적어 둔 값이 5 다. 지연 중에는 다시 무장하지 않는다 —
    송금 한 번이 끝나기 전에 또 보내지는 않기 때문이다. 발화수는 무장한 횟수를 센다
    (= 사람이 받는 알림 수 = 송금 횟수).

    정의는 alloc_mix_real_rebal.build_alloc_real 과 같다 — 세 슬리브 자본이 각자 자라고,
    반회전율 >= ht 이거나 (상대부족 >= t3u 이고 그 슬리브 카나리 ON) 이면 그날 목표로
    되돌린다. 다른 것은 조합을 하나씩 도는 대신 축으로 쌓은 것뿐이다.

    목표가 0 인 슬리브의 상대부족은 -inf 로 둔다. 목표가 0 이면 그 슬리브는 모자랄 수
    없는데 0/0 을 그냥 0 으로 두면 t3u 가 0 일 때 발화한다 — 지금 격자에 0 은 없지만
    분모가 0 인 칸(선물 1:0)이 실제로 있으므로 뜻을 명시해 둔다.
    """
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index).sort_values()
    rets = np.column_stack([
        eq_st.loc[common].pct_change().fillna(0).to_numpy(),
        eq_sp.loc[common].pct_change().fillna(0).to_numpy(),
        eq_fu.loc[common].pct_change().fillna(0).to_numpy(),
    ])
    cans = np.column_stack([
        can_st.reindex(common).ffill().fillna(False).to_numpy().astype(bool),
        can_sp.reindex(common).ffill().fillna(False).to_numpy().astype(bool),
        can_fu.reindex(common).ffill().fillna(False).to_numpy().astype(bool),
    ])
    K, N = tgt.shape[0], len(common)
    cap = tgt.copy()
    out = np.empty((K, N))
    fires = np.zeros(K, dtype=np.int64)
    pos = tgt > 0
    t3u_col = t3u[:, None]
    # 남은 영업일. -1 = 무장 안 됨. delay=0 이면 무장과 집행이 같은 날이라 안 쓴다.
    pend = np.full(K, -1, dtype=np.int64)
    for i in range(N):
        cap *= (1.0 + rets[i])
        total = cap.sum(axis=1)
        out[:, i] = total
        with np.errstate(divide='ignore', invalid='ignore'):
            cur = cap / total[:, None]
        fire = np.abs(tgt - cur).sum(axis=1) / 2.0 >= ht
        rel = np.where(pos, (tgt - cur) / np.where(pos, tgt, 1.0), -np.inf)
        fire |= ((rel >= t3u_col) & cans[i][None, :]).any(axis=1)
        def _settle(sel):
            """sel 인 행을 목표 비중으로 되돌린다. 옮긴 몫에 송금비용을 물린다."""
            t = total[sel]
            if transfer_cost > 0.0:
                moved = np.abs(tgt[sel] * t[:, None] - cap[sel]).sum(axis=1) / 2.0
                t = t - moved * transfer_cost
            cap[sel] = tgt[sel] * t[:, None]

        if delay <= 0:
            if fire.any():
                _settle(fire)
                fires += fire
            continue
        # 무장은 쉬고 있을 때만 — 이미 보내는 중이면 또 보내지 않는다.
        arm = fire & (pend < 0)
        fires += arm
        pend[arm] = delay
        due = pend == 0
        if due.any():
            _settle(due)
        pend[pend >= 0] -= 1
    return common, out, fires


# --- 창 순위 (조합 축 벡터화) ------------------------------------------------------

def _avg_ranks(score: np.ndarray) -> np.ndarray:
    """높을수록 좋은 점수를 내림차순 순위로. 동률은 평균순위."""
    K = score.shape[0]
    order = np.argsort(-score, kind='stable')
    ssorted = score[order]
    ranks = np.empty(K)
    s = 0
    while s < K:
        e = s
        while e + 1 < K and ssorted[e + 1] == ssorted[s]:
            e += 1
        ranks[order[s:e + 1]] = (s + e) / 2.0 + 1.0
        s = e + 1
    return ranks


def window_ranksum(curves: np.ndarray, dates: pd.DatetimeIndex,
                   win_sizes, strides, lo: int = 0, hi: int | None = None,
                   detail: list | None = None):
    """창마다 순위를 매겨 합산. 두 기준을 같은 창으로 함께 잰다.

    반환 = (칼마 순위합 (K,), 칼마 1위 횟수 (K,),
            연복리 순위합 (K,), 연복리 1위 횟수 (K,), 창 수).

    detail 에 리스트를 주면 창마다 (창길이, 이동폭, 시작, 칼마등수, 연복리등수,
    창 연복리, 창 낙폭)를 그 리스트에 쌓는다. 합계만으로는 못 보는 것을 보려고 둔
    통로다 — 창 길이별로 답이 뒤집히는지, 같은 창에서 후보끼리 짝지으면 누가 이기는지,
    불리한 창이 언제이고 그때 실제 낙폭이 얼마였는지. 안 주면 아무것도 안 쌓아
    기존 경로가 그대로다.

    칼마 정의는 bt_coin_param_resweep.window_rs 와 같다 — 창 안 연복리를 낙폭
    절대값으로 나눈 값으로 순위를 매기고 동률은 평균순위를 준다. 연복리 기준은
    그 분자만 쓴다(운영자 지시 9451 — 칼마만 보지 말고 수익률 자체도 본다).
    두 기준이 같은 창·같은 동률 규칙을 쓰므로 순위를 나란히 읽을 수 있다.
    lo/hi 는 창의 시작 위치를 그 구간 안으로 제한한다.
    """
    K, N = curves.shape
    hi = N if hi is None else hi
    days = dates.to_numpy().astype('datetime64[D]').astype(np.int64)
    rs = np.zeros(K)
    wins = np.zeros(K, dtype=np.int64)
    rs_cagr = np.zeros(K)
    wins_cagr = np.zeros(K, dtype=np.int64)
    n = 0
    for size in win_sizes:
        for stride in strides:
            for i in range(lo, hi - size, stride):
                j = i + size - 1
                if j >= hi:
                    break
                seg = curves[:, i:j + 1]
                yrs = (days[j] - days[i]) / 365.25
                if yrs <= 0:
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    cagr = (seg[:, -1] / seg[:, 0]) ** (1.0 / yrs) - 1.0
                    mdd = (seg / np.maximum.accumulate(seg, axis=1) - 1.0).min(axis=1)
                    cal = np.where(mdd < 0, cagr / np.abs(mdd), 0.0)
                cal = np.nan_to_num(cal, nan=-1e9, posinf=-1e9, neginf=-1e9)
                cg = np.nan_to_num(cagr, nan=-1e9, posinf=-1e9, neginf=-1e9)
                r_cal = _avg_ranks(cal)
                r_cg = _avg_ranks(cg)
                rs += r_cal
                wins += cal == cal.max()
                rs_cagr += r_cg
                wins_cagr += cg == cg.max()
                if detail is not None:
                    # 등수와 함께 그 창의 원지표도 남긴다. 등수만으로는 "어느 창에서
                    # 얼마나 나빴나" 를 못 말한다 — 2026-10-31 토론이 불리한 창을
                    # 보고할 때 날짜·절대 낙폭·연복리를 같이 내라고 한 자리다.
                    detail.append((size, stride, i,
                                   r_cal.astype(np.float32), r_cg.astype(np.float32),
                                   cg.astype(np.float32), mdd.astype(np.float32)))
                n += 1
    return rs, wins, rs_cagr, wins_cagr, n


def curve_metrics(curves: np.ndarray, dates: pd.DatetimeIndex):
    """칼마·샤프·연복리·낙폭을 조합 축으로 한 번에."""
    days = dates.to_numpy().astype('datetime64[D]').astype(np.int64)
    yrs = (days[-1] - days[0]) / 365.25
    with np.errstate(divide='ignore', invalid='ignore'):
        cagr = (curves[:, -1] / curves[:, 0]) ** (1.0 / yrs) - 1.0
        mdd = (curves / np.maximum.accumulate(curves, axis=1) - 1.0).min(axis=1)
        rets = curves[:, 1:] / curves[:, :-1] - 1.0
        sd = rets.std(axis=1, ddof=1)
        sharpe = np.where(sd > 0, rets.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(252), 0.0)
        cal = np.where(mdd < 0, cagr / np.abs(mdd), 0.0)
    return (np.nan_to_num(cal), np.nan_to_num(sharpe),
            np.nan_to_num(cagr) * 100, np.nan_to_num(mdd) * 100, yrs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', type=int, required=True)
    ap.add_argument('--stress', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--win-detail', action='store_true',
                    help='창별 등수를 따로 저장한다(<out>.win.npz). 짧은 구간 대응 분석용')
    ap.add_argument('--delay', type=int, default=0,
                    help='발화일과 실제 송금 사이의 영업일 수. 0 = 그날 바로(기본, 옛 결과 재현용)')
    ap.add_argument('--transfer-cost', type=float, default=0.0,
                    help='계좌 사이로 옮기는 금액에 물리는 비율(예: 0.002 = 0.2%%). '
                         '0 = 공짜 송금(기본, 옛 결과 재현용). 환전 스프레드·출금 수수료 자리다')
    ap.add_argument('--stocks', default='',
                    help='주식 축을 퍼센트로 지목한다(예: 60,70). 비우면 기본 8칸')
    ap.add_argument('--ratios', default='',
                    help='코인 현물:선물 비를 지목한다(예: 1:1,5:3). 비우면 기본 7칸. '
                         '지목하면 "현물 > 선물" 걸름을 끈다 — 지목 자체가 제약 해제다')
    a = ap.parse_args()
    stock_axis = [float(x) / 100 for x in a.stocks.split(',')] if a.stocks else None
    ratios = None
    if a.ratios:
        ratios = []
        for tok in a.ratios.split(','):
            x, y = tok.split(':')
            ratios.append((tok, int(x), int(y)))

    import bt_coin_param_resweep as R
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    from alloc_mix_real_rebal import canary_from_trace

    t0 = time.time()
    stress = bool(a.stress)
    R._bars, R._funding = load_data('D')
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)

    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pdf = pd.DataFrame(load_prices(ALL_TICKERS, start="2005-01-01"))
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    _r, _mo, _md, can_st = precompute(pdf, [30, 72, 230], [42, 63, 126])
    can_st = can_st.astype(bool)
    print(f'[{a.anchor}/{a.stress}] 입력 준비 ({time.time() - t0:.0f}초)', flush=True)

    st = R.run_stock(a.anchor, stress)
    tr_sp: list = []
    sp = R.run_spot({}, R.PH_SPOT[a.anchor], stress, trace=tr_sp)
    tr_fu: list = []
    fu = R.run_fut({}, R.PH_FUT[a.anchor], stress, trace=tr_fu)
    if st is None or sp is None or fu is None:
        print(f'[{a.anchor}/{a.stress}] 슬리브를 못 만들었다')
        return 1
    can_sp = canary_from_trace(tr_sp)
    can_fu = canary_from_trace(tr_fu)
    print(f'[{a.anchor}/{a.stress}] 슬리브 셋 ({time.time() - t0:.0f}초)', flush=True)

    mixes = build_mixes(stock_axis, ratios, allow_equal=bool(ratios))
    combos = build_combos(mixes)
    tgt = np.array([[c[1], c[2], c[3]] for c in combos], dtype=float)
    ht = np.array([c[4] for c in combos], dtype=float)
    t3u = np.array([c[5] for c in combos], dtype=float)

    dates, curves, fires = build_alloc_real_many(st, sp, fu, can_st, can_sp, can_fu,
                                                 tgt, ht, t3u, delay=a.delay,
                                                 transfer_cost=a.transfer_cost)
    print(f'[{a.anchor}/{a.stress}] 조합 {len(combos)}개 합성 ({time.time() - t0:.0f}초)',
          flush=True)

    cal, sharpe, cagr, mdd, yrs = curve_metrics(curves, dates)
    det: list | None = [] if a.win_detail else None
    rs, wins, rs_cagr, wins_cagr, n = window_ranksum(curves, dates, R.WIN_SIZES, R.STRIDES,
                                                     detail=det)
    print(f'[{a.anchor}/{a.stress}] 창 {n} ({time.time() - t0:.0f}초)', flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        mix=np.array([c[0] for c in combos]),
        ht=ht, t3u=t3u,
        w_st=tgt[:, 0], w_sp=tgt[:, 1], w_fu=tgt[:, 2],
        rs=rs, wins=wins, rs_cagr=rs_cagr, wins_cagr=wins_cagr, n=n,
        cal=cal, sharpe=sharpe, cagr=cagr, mdd=mdd,
        fires_per_year=fires / yrs,
        anchor=a.anchor, stress=a.stress, delay=a.delay,
        transfer_cost=a.transfer_cost,
        d0=str(dates[0].date()), d1=str(dates[-1].date()), ndays=len(dates),
    )
    print(f'[{a.anchor}/{a.stress}] 저장 {a.out} ({time.time() - t0:.0f}초)', flush=True)
    if det is not None:
        wp = a.out.replace('.npz', '') + '.win.npz'
        np.savez_compressed(
            wp,
            size=np.array([d[0] for d in det], dtype=np.int32),
            stride=np.array([d[1] for d in det], dtype=np.int32),
            start=np.array([d[2] for d in det], dtype=np.int32),
            rank_cal=np.stack([d[3] for d in det]),
            rank_cg=np.stack([d[4] for d in det]),
            cagr_w=np.stack([d[5] for d in det]),
            mdd_w=np.stack([d[6] for d in det]),
            # 실제 거래일 배열을 그대로 남긴다. 휴일 때문에 d0 에서 영업일을 세어
            # 되만들면 어긋난다(2026-10-31 실측 2026-06-29 대 2026-09-18).
            days=np.array([str(d.date()) for d in dates]),
            anchor=a.anchor, stress=a.stress, delay=a.delay,
        transfer_cost=a.transfer_cost,
        )
        print(f'[{a.anchor}/{a.stress}] 창별 등수 {wp} ({len(det)}창)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
