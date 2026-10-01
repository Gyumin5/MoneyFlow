"""선물 리필의 탈락조건과 채움순서를 바꿔 본다 — 끔까지 포함해 60/25/15 로 잰다 (2026-10-01).

운영자 물음 셋.
  ⑴ 탈락조건을 이동평균 교차로 바꾸면 어떤가. 18일선이 127일선 아래로 내려가면 탈락.
     다른 기간 조합도 같이 본다.
  ⑵ 채움순서가 왜 모멘텀순인가. 시총순으로도 본다.
  ⑶ 끄는 것도 본다 — 켜는 게 좋아서 들어가 있을 텐데 그게 맞는지.
  셋을 따로 잰 뒤 좋은 것끼리 조합한다(조합 팔은 2차 실행에서 넣는다).

재는 자. 결정층은 합성 60/25/15 윈도우 순위합이고 앵커 7곳 각각에서 낸다. 비용 5배
스트레스를 같은 축에 붙인다. 주식·현물 슬리브는 이 축이 아니라 고정 팔이다 —
선물만 바뀐다.

기준선 주의. bt_coin_param_resweep 의 FUT_BASE 는 문턱 0.03 이다(2026-09-22 측정 당시 값).
지금 라이브는 0.015 이므로 여기서는 0.015 로 덮어쓴다. 그래서 캐시도 따로 쓴다.

2차 (2026-10-01, 토론 run-20261001T120336Z 뒤). 1차의 "모멘텀순 전체풀" 은 라이브와 다르다 —
라이브는 채움 후보에서 BTC 를 뺀다. 그래서 진짜 라이브 팔을 넣고, 풀 넓이와 BTC 취급을
정렬 축과 분리해 다시 잰다.

채택 기준을 결과 보기 전에 못박는다. 아래 넷을 다 만족할 때만 채택 후보로 올린다.
  ① 실제 운영 기준선(넓은풀·모멘텀순·BTC제외) 대비 두 비용 조건 모두 순위평균이 낮다.
  ② 이긴 앵커가 두 조건 각각 5/7 이상이다.
  ③ 합성 최대낙폭이 기준선 대비 0.5%포인트 넘게 나빠지지 않는다.
  ④ BTC 포함·제외 두 변형이 모두 기준선을 이긴다(한쪽만 이기면 BTC 덕이라 기각).
  좁은풀과 넓은풀의 순위평균 차가 0.2 이내로 실질 동률이면 단순한 좁은풀을 고른다.
  시총순은 범주형 축이라 인접값 고원을 못 본다 — 대신 순위 교란 안정성으로 대신한다
  (시총 순위 30일 지연, 인접쌍 교환). 교란에서 기준선에 지면 순위 정보가 아니라
  특정 배열에 얹힌 것으로 보고 기각한다.

엔진은 환경변수로만 바뀌어 있고 측정이 끝나면 되돌린다. 정본에 실험 코드를 남기지 않는다.
  REFILL_FAIL_MODE = mom2 | smacross,  REFILL_FAIL_SMA_S / _L
  REFILL_ORDER     = bt | alpha | mom | mcap
  REFILL_POOL      = fresh | wide,  REFILL_EXCL_BTC = 0 | 1
  REFILL_MCAP_LAG  = N일 지연,  REFILL_MCAP_SWAP = 1 이면 인접쌍 교환
  DRIFT_HEALTH_MODE = refill | off   (기존 토글)

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_refill_variants.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402  (하니스·순위합·주식 슬리브 재사용)

OUT = "/home/gmoh/mon/251229/state/20261001-futrefill"
EQDIR = os.path.join(OUT, "eq")

DRIFT_FUT = 0.015  # 라이브 현행 (2026-09-23 배포)

CUR = "현행(둘 다 음수)"

# (이름, 환경변수 덮어쓰기)
ARMS_FAIL: list[tuple[str, dict]] = [
    ("끔", dict(DRIFT_HEALTH_MODE="off")),
    (CUR, {}),
    ("교차 18/127", dict(REFILL_FAIL_MODE="smacross",
                        REFILL_FAIL_SMA_S="18", REFILL_FAIL_SMA_L="127")),
    ("교차 20/120", dict(REFILL_FAIL_MODE="smacross",
                        REFILL_FAIL_SMA_S="20", REFILL_FAIL_SMA_L="120")),
    ("교차 20/60", dict(REFILL_FAIL_MODE="smacross",
                       REFILL_FAIL_SMA_S="20", REFILL_FAIL_SMA_L="60")),
    ("교차 10/120", dict(REFILL_FAIL_MODE="smacross",
                        REFILL_FAIL_SMA_S="10", REFILL_FAIL_SMA_L="120")),
    ("교차 30/90", dict(REFILL_FAIL_MODE="smacross",
                       REFILL_FAIL_SMA_S="30", REFILL_FAIL_SMA_L="90")),
    ("교차 50/200", dict(REFILL_FAIL_MODE="smacross",
                        REFILL_FAIL_SMA_S="50", REFILL_FAIL_SMA_L="200")),
]

ARMS_ORDER: list[tuple[str, dict]] = [
    (CUR, {}),
    ("알파벳 전체풀", dict(REFILL_ORDER="alpha")),
    ("모멘텀순 전체풀", dict(REFILL_ORDER="mom")),
    ("시총순 전체풀", dict(REFILL_ORDER="mcap")),
]

LIVE = "넓은풀 모멘텀순 BTC제외"   # 실제 운영 규칙 = 2차의 기준선

# 2차: 풀 넓이 · BTC 취급 · 정렬을 분리한다.
ARMS_POOL: list[tuple[str, dict]] = [
    (CUR, {}),                                   # 현행 백테스트 (좁은풀·알파벳·BTC포함)
    (LIVE, dict(REFILL_ORDER="mom", REFILL_POOL="wide", REFILL_EXCL_BTC="1")),
    ("넓은풀 모멘텀순 BTC포함", dict(REFILL_ORDER="mom", REFILL_POOL="wide")),
    ("넓은풀 시총순 BTC포함", dict(REFILL_ORDER="mcap", REFILL_POOL="wide")),
    ("넓은풀 시총순 BTC제외", dict(REFILL_ORDER="mcap", REFILL_POOL="wide",
                                 REFILL_EXCL_BTC="1")),
    ("좁은풀 시총순 BTC포함", dict(REFILL_ORDER="mcap", REFILL_POOL="fresh")),
    ("좁은풀 시총순 BTC제외", dict(REFILL_ORDER="mcap", REFILL_POOL="fresh",
                                 REFILL_EXCL_BTC="1")),
    # 순위 교란 안정성(범주형 축이라 인접값 고원을 못 보는 것의 대체).
    ("넓은풀 시총순 지연30일", dict(REFILL_ORDER="mcap", REFILL_POOL="wide",
                                 REFILL_MCAP_LAG="30")),
    ("넓은풀 시총순 인접교환", dict(REFILL_ORDER="mcap", REFILL_POOL="wide",
                                 REFILL_MCAP_SWAP="1")),
]

# 3차 (2026-10-01, 토론 run-20261001T125747Z 가 정의한 정책 변종).
# 공통 규칙은 엔진 쪽 REFILL_POLICY 분기에 들어 있다 — 데이터 없는 보유는 유지,
# 신규 후보는 두 모멘텀 양수, 후보 풀은 get_mcap 목록에서 BTC 제외, 비중 상한 1/3 준수,
# 모자란 몫은 현금. DRIFT_CASH_REFILL 은 전 팔에서 끈다.
B_ARM = "B 모멘텀순 상한준수"
ARMS_POLICY: list[tuple[str, dict]] = [
    (B_ARM, dict(REFILL_POLICY="bmom")),
    ("V1 탈락비중 현금화", dict(REFILL_POLICY="v1cash")),
    ("V2 넓은풀 시총순 상한", dict(REFILL_POLICY="v2mcap")),
    ("V3 건강풀 모멘텀순", dict(REFILL_POLICY="v3volmom")),
    ("V4 탈락스냅 재선정", dict(REFILL_POLICY="v4snap")),
    ("V5 전스냅 재선정", dict(REFILL_POLICY="v5all")),
    (CUR, {}),
]

# 쌍대 비교 목록 — 팔 구성에 따라 순위가 흔들리는 것을 없애려고 둘씩만 넣고 잰다.
PAIRS: list[tuple[str, str]] = [
    ("V1 탈락비중 현금화", B_ARM),
    ("V2 넓은풀 시총순 상한", B_ARM),
    ("V3 건강풀 모멘텀순", B_ARM),
    ("V4 탈락스냅 재선정", B_ARM),
    ("V5 전스냅 재선정", B_ARM),
    (CUR, B_ARM),
    ("V1 탈락비중 현금화", "V2 넓은풀 시총순 상한"),
    ("V3 건강풀 모멘텀순", "V2 넓은풀 시총순 상한"),
    ("V4 탈락스냅 재선정", "V2 넓은풀 시총순 상한"),
    ("V5 전스냅 재선정", "V2 넓은풀 시총순 상한"),
]

AXES: dict[str, list[tuple[str, dict]]] = {
    "정책 변종": ARMS_POLICY,
}

ENV_KEYS = ("DRIFT_HEALTH_MODE", "REFILL_FAIL_MODE",
            "REFILL_FAIL_SMA_S", "REFILL_FAIL_SMA_L", "REFILL_ORDER",
            "REFILL_POOL", "REFILL_EXCL_BTC", "REFILL_MCAP_LAG",
            "REFILL_MCAP_SWAP", "REFILL_POLICY", "DRIFT_CASH_REFILL")
ENV_DEFAULT = dict(DRIFT_HEALTH_MODE="refill", REFILL_FAIL_MODE="mom2",
                   REFILL_FAIL_SMA_S="18", REFILL_FAIL_SMA_L="127",
                   REFILL_ORDER="bt", REFILL_POOL="fresh", REFILL_EXCL_BTC="0",
                   REFILL_MCAP_LAG="0", REFILL_MCAP_SWAP="0",
                   REFILL_POLICY="", DRIFT_CASH_REFILL="off")


def set_env(over: dict) -> None:
    for k in ENV_KEYS:
        os.environ[k] = over.get(k, ENV_DEFAULT[k])


def _key(name: str, anchor: int, stress: bool) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return f"{safe}__a{anchor}{'__s5' if stress else ''}"


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(os.path.join(EQDIR, f"{kind}__{key}.csv"), header=["eq"])


def _load(kind: str, key: str):
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def run_fut_arm(over_env: dict, anchor: int, stress: bool, want_counts=False):
    from backtest_futures_v25 import run as fbt_run
    set_env(over_env)
    kw = dict(R.FUT_BASE)
    kw["drift_threshold"] = DRIFT_FUT
    m = fbt_run(R._bars, R._funding, interval="D", leverage=R._k2,
                tx_cost=R.TX_FUT * (R.STRESS if stress else 1.0), maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=5,
                phase_offset_bars=R.PH_FUT[anchor], health_mode="mom2vol",
                start_date=R.START, end_date=R.END, **kw)
    set_env({})
    if want_counts:
        return m
    return m.get("_equity")


def ensure_spot(anchor: int, stress: bool) -> pd.Series:
    """현물은 이 축이 아니다 — 현행 한 벌만 돌려 모든 팔에 공유한다."""
    k = _key("현물현행", anchor, stress)
    eq = _load("sp", k)
    if eq is None:
        os.environ["DRIFT_HEALTH_MODE"] = "refill"
        eq = R.run_spot({}, R.PH_SPOT[anchor], stress)
        _save("sp", k, eq)
    return eq


def report(axis: str, arms, stress: bool, stock_eq, spot_eq) -> None:
    names = [a[0] for a in arms]
    rank = {n: [] for n in names}
    comp = {n: [] for n in names}      # 합성 지표 dict 목록
    fut = {n: [] for n in names}       # 선물 단독 지표 dict 목록

    for a in R.ANCHORS:
        curves = {}
        for name in names:
            fu = _load("fu", _key(name, a, stress))
            if fu is None:
                continue
            curves[name] = R.build_alloc(stock_eq[a], spot_eq[(a, stress)], fu)
            comp[name].append(R.metrics(curves[name]))
            fut[name].append(R.metrics(fu))
        if len(curves) < 2:
            continue
        sums, _w, n = R.window_rs(curves)
        for name in curves:
            rank[name].append(sums[name] / max(n, 1))

    ref = LIVE if LIVE in rank and rank[LIVE] else CUR
    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== {axis} · {tag} · 앵커 {len(R.ANCHORS)}곳 · 합성 60/25/15 ===")
    print(f"  기준선 = {ref}")
    print(f"{'팔':<24}{'순위평균':>9}{'중앙':>7}{'최악':>7}{'기준대비':>9}"
          f"{'합성Cal':>8}{'합성Sharpe':>11}{'합성CAGR':>10}{'합성MDD':>9}"
          f"{'선물Cal':>8}{'선물CAGR':>10}{'선물MDD':>9}{'이긴앵커':>9}")
    base = rank.get(ref, [])
    base_mean = float(np.mean(base)) if base else float("nan")
    rows = sorted((float(np.mean(r)), n, r) for n, r in rank.items() if r)
    for mean_r, name, r in rows:
        c = comp[name]
        f = fut[name]
        wins = ("" if name == ref or not base
                else f"{sum(1 for x, y in zip(r, base) if x < y)}/{len(base)}")
        mark = ("  <- 기준(라이브)" if name == ref
                else "  <- 현행 백테스트" if name == CUR else "")
        print(f"{name:<24}{mean_r:>9.3f}{float(np.median(r)):>7.3f}{max(r):>7.3f}"
              f"{mean_r-base_mean:>+9.3f}"
              f"{np.mean([x['Cal'] for x in c]):>8.2f}"
              f"{np.mean([x['Sharpe'] for x in c]):>11.2f}"
              f"{np.mean([x['CAGR'] for x in c]):>9.1f}%"
              f"{np.mean([x['MDD'] for x in c]):>8.2f}%"
              f"{np.mean([x['Cal'] for x in f]):>8.2f}"
              f"{np.mean([x['CAGR'] for x in f]):>9.1f}%"
              f"{np.mean([x['MDD'] for x in f]):>8.2f}%"
              f"{wins:>9}{mark}")


def pairwise(pairs, stress: bool, stock_eq, spot_eq) -> None:
    """팔 둘만 넣은 순위합. 팔 구성에 따른 순위 이동을 없앤다."""
    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== 쌍대 비교 · {tag} · 앵커 {len(R.ANCHORS)}곳 · 합성 60/25/15 ===")
    print("낮은 쪽이 우세. 이김 = 후보의 창 순위평균이 기준보다 낮은 앵커 수.")
    print(f"{'후보':<22}{'기준':<22}{'후보순위':>9}{'기준순위':>9}{'차':>8}{'이김':>6}"
          f"{'합성Cal':>8}{'기준Cal':>8}{'합성MDD':>9}{'기준MDD':>9}"
          f"{'선물MDD':>9}{'기준선물MDD':>12}")
    for a, b in pairs:
        ra, rb, ma, mb, fa, fb = [], [], [], [], [], []
        for an in R.ANCHORS:
            ea = _load("fu", _key(a, an, stress))
            eb = _load("fu", _key(b, an, stress))
            if ea is None or eb is None:
                continue
            ca = R.build_alloc(stock_eq[an], spot_eq[(an, stress)], ea)
            cb = R.build_alloc(stock_eq[an], spot_eq[(an, stress)], eb)
            sums, _w, n = R.window_rs({a: ca, b: cb})
            ra.append(sums[a] / max(n, 1))
            rb.append(sums[b] / max(n, 1))
            ma.append(R.metrics(ca))
            mb.append(R.metrics(cb))
            fa.append(R.metrics(ea))
            fb.append(R.metrics(eb))
        if not ra:
            continue
        wins = sum(1 for x, y in zip(ra, rb) if x < y)
        print(f"{a:<22}{b:<22}{float(np.mean(ra)):>9.3f}{float(np.mean(rb)):>9.3f}"
              f"{float(np.mean(ra))-float(np.mean(rb)):>+8.3f}{wins:>4}/{len(ra)}"
              f"{np.mean([x['Cal'] for x in ma]):>8.2f}{np.mean([x['Cal'] for x in mb]):>8.2f}"
              f"{np.mean([x['MDD'] for x in ma]):>8.2f}%{np.mean([x['MDD'] for x in mb]):>8.2f}%"
              f"{np.mean([x['MDD'] for x in fa]):>8.2f}%{np.mean([x['MDD'] for x in fb]):>11.2f}%")


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    os.makedirs(R.EQDIR, exist_ok=True)
    t0 = time.time()
    set_env({})

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...")
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    print("주식 슬리브(고정 팔) 준비 중...")
    stock_eq = {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if os.path.exists(p):
            stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = R.run_stock(a, False)
            stock_eq[a].to_csv(p, header=["eq"])
    print(f"  완료 ({time.time()-t0:.0f}초)", flush=True)

    print("현물 슬리브(고정 팔) 준비 중...")
    spot_eq = {}
    for a in R.ANCHORS:
        for s in (False, True):
            spot_eq[(a, s)] = ensure_spot(a, s)
    print(f"  완료 ({time.time()-t0:.0f}초)", flush=True)

    uniq = {}
    for _axis, arms in AXES.items():
        for name, env in arms:
            uniq[name] = env
    todo = [(n, e, a, s) for n, e in uniq.items()
            for a in R.ANCHORS for s in (False, True)]
    print(f"\n선물 계산 조합 {len(todo)}건 (이미 있는 것은 건너뜀)", flush=True)
    done = 0
    for name, env, a, s in todo:
        k = _key(name, a, s)
        if _load("fu", k) is None:
            _save("fu", k, run_fut_arm(env, a, s))
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  전부 완료 ({time.time()-t0:.0f}초)", flush=True)

    for axis, arms in AXES.items():
        for stress in (False, True):
            report(axis, arms, stress, stock_eq, spot_eq)

    for stress in (False, True):
        pairwise(PAIRS, stress, stock_eq, spot_eq)

    print("\n=== 리필이 실제로 바꾼 사건 (앵커 7곳 전부 · 기본 비용 · 선물 단독) ===")
    print("탈락일 = 리필이 코인을 하나라도 떨어뜨린 날. 채움 = (날짜,스냅,코인) 건수.")
    print("서로 다른 날 = 앵커를 합쳐 중복을 뺀 날짜 수. 겹치는 창은 같은 사건을 여러 번 센다.")
    fills_by_arm = {}
    print(f"  {'팔':<24}{'탈락일':>8}{'탈락수':>8}{'채움':>7}{'서로다른날':>11}"
          f"{'상한위반':>9}{'최대비중':>9}{'BTC·ETH비중':>12}  상위 채움코인")
    import json
    for name, env in uniq.items():
        cpath = os.path.join(OUT, f"counts2__{_key(name, 0, False)}.json")
        if os.path.exists(cpath):
            with open(cpath) as fh:
                blob = json.load(fh)
            days, drops = blob["days"], blob["drops"]
            viol, maxw = blob["viol"], blob["maxw"]
            allf = [tuple(x) for x in blob["fills"]]
        else:
            days = drops = viol = 0
            maxw = 0.0
            allf = []
            for a in R.ANCHORS:
                m = run_fut_arm(env, a, False, want_counts=True)
                days += int(m.get('RefillDays', 0))
                drops += int(m.get('RefillDrops', 0))
                viol += int(m.get('RefillCapViol', 0))
                maxw = max(maxw, float(m.get('RefillMaxW', 0.0)))
                allf += list(m.get('RefillFills', []))
            with open(cpath, "w") as fh:
                json.dump({"days": days, "drops": drops, "viol": viol,
                           "maxw": maxw, "fills": allf}, fh)
        fills_by_arm[name] = allf
        hist = {}
        for _d, _si, _c in allf:
            hist[_c] = hist.get(_c, 0) + 1
        top = ", ".join(f"{c}×{n}" for c, n in
                        sorted(hist.items(), key=lambda x: (-x[1], x[0]))[:5])
        be = sum(n for c, n in hist.items() if c in ('BTC', 'ETH'))
        share = f"{100*be/len(allf):.0f}%" if allf else "-"
        print(f"  {name:<24}{days:>8}{drops:>8}{len(allf):>7}"
              f"{len({d for d, _s, _c in allf}):>11}{viol:>9}{maxw:>9.3f}"
              f"{share:>12}  {top}", flush=True)

    if LIVE in fills_by_arm:
        base_set = {(d, si, c) for d, si, c in fills_by_arm[LIVE]}
        base_days = {d for d, _s, _c in fills_by_arm[LIVE]}
        print("\n  기준선(라이브) 대비 선택이 실제로 갈린 정도:")
        for name, f in fills_by_arm.items():
            if name == LIVE:
                continue
            s = {(d, si, c) for d, si, c in f}
            dd = {d for d, _s, _c in f}
            print(f"    {name:<24} 갈린 채움 {len(s ^ base_set):>4}건 · "
                  f"갈린 날 {len(dd ^ base_days):>3}일 · 공통 날 {len(dd & base_days):>3}일")

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
