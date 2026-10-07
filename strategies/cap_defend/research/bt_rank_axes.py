"""선정 순위 축 — 1차 측정 (2026-10-07, ai-debate run-20261007T013223Z 명세 그대로).

운영자 지시: "헬스체크 통과한 시총상위권만 뽑는데 … 다양한 순위를 기준으로 다 테스트해봐줄래
순위 조합도 좋고 순위로 나온 리스트 합집합도 가능하고".

지금 규칙은 헬스 통과 집합을 원시 시총 내림차순으로 훑어 앞 3종을 자르는 것이다. 바꾸는 것은
"무엇으로 줄 세워 3종을 고르는가" 하나뿐이다. 헬스 게이트·스냅 수·간격·드리프트·리필·
greedy 흡수·상한은 전부 고정한다.

설계의 핵심 하나. 새 순위대로 뽑은 목록을 그대로 greedy 흡수에 넣으면 안 된다. 흡수는 앞뒤
두 종목의 단기 모멘텀을 비교해 앞이 크거나 같으면 뒤를 버리는데, 모멘텀 내림차순이면 그
조건이 항상 참이라 1종까지 버린다(실측: 해당 1,245일 전부 100%, 현금 2/3 고정). 그래서 모든
팔은 "새 축으로 3종을 고르고 → 시총순으로 다시 세우고 → 기존 흡수" 순서로 돈다. 이러면
종목 수 분포가 기준선과 거의 같아진다(1종 19.5% 대 18.2%, 2종 64.4% 대 62.5%).

팔 (현물 6 / 선물 7):
  B0 시총       원시 시총 내림차순 앞 3종 — 현행 기준선
  S단기         단기 모멘텀(선물 18봉·현물 20봉) 큰 순
  S장기         127봉 모멘텀 큰 순
  S효율         127봉 모멘텀 / 90봉 일간 변동성 큰 순
  C시총효율     시총 순위 + 효율 순위 작은 순
  U교대3        시총 목록과 효율 목록을 번갈아 읽어 중복 빼고 앞 3종 (합집합, 최대 3종·상한 1/3)
  S거래대금     종가×거래량 30봉 평균 큰 순 — 선물만. 현물은 라이브가 업비트 봉이라 입력 불일치.

재는 자: 자산배분 60/25/15 합성의 겹치는 창 순위합. 앵커 7곳 × 비용 2조건.
판정은 후보와 B0 둘만 넣은 쌍대 비교로 한다(여러 팔을 같이 넣으면 팔 구성에 따라 순위가 흔들린다).
전체 팔을 같이 넣은 표는 보고용이다.

채택 게이트 ①② — 결과를 보기 전에 못박았다:
  ① G_c = (평균_a R_B0 - 평균_a R_후보) / 평균_a R_B0 가 두 비용 모두 0.02 이상이고,
     앵커별로 R_후보 < R_B0 인 곳이 각각 5/7 이상.
  ② 모든 앵커·두 비용에서 합성 최대낙폭 악화 0.5%포인트 이하, 바뀐 슬리브 단독 악화 2%포인트 이하.
게이트 ③~⑧(40종 풀, 교란 둘, 노출 대조, 위약 5벌, 구현 검증, 보고)은 통과 후보에만 적용하며
bt_rank_axes_confirm.py 가 맡는다. 슬리브별 확증 후보는 최대 하나다.

구현 검증 ⑦ 선행 통과: 신규 선택 함수의 b0 가 정본 경로와 자산곡선 최대괴리 0.000e+00
(선물 최종자산 14166951.295769 동일, 현물 144925.815547 동일).

실행 단위는 bt_coin_param_resweep(R) 의 run_spot/run_fut/metrics/window_rs/build_alloc 를
그대로 빌려 쓴다. 기준선이 그 하니스와 한 글자도 다르면 안 되기 때문이다. 드리프트만
라이브 현행값으로 덮어쓴다 — R.FUT_BASE 는 2026-09-22 당시의 0.03 을 들고 있다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_rank_axes.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261007-rankaxes"
EQDIR = os.path.join(OUT, "eq")

DRIFT_FUT = 0.015   # 라이브 현행. R.FUT_BASE 의 0.03 을 그대로 쓰지 않는다.
DRIFT_SPOT = 0.10   # 라이브 현행 (= R.SPOT_BASE 와 같지만 명시해 둔다).

SP_OVER = dict(drift_threshold=DRIFT_SPOT)
FU_OVER = dict(drift_threshold=DRIFT_FUT)

B0 = "B0 시총"
ARMS_SPOT: list[tuple[str, str]] = [
    (B0, "b0"),
    ("S단기", "smom"),
    ("S장기", "slong"),
    ("S효율", "seff"),
    ("C시총효율", "cmix"),
    ("U교대3", "ualt"),
]
ARMS_FUT: list[tuple[str, str]] = ARMS_SPOT + [("S거래대금", "sdvol")]


def set_env(mode: str = "", lag: int = 0, swap: bool = False,
            seed: int = 0, pool40: bool = False) -> None:
    os.environ["SELECT_RANK"] = mode
    os.environ["SELECT_RANK_LAG"] = str(lag)
    os.environ["SELECT_RANK_SWAP"] = "1" if swap else "0"
    os.environ["SELECT_RANK_SEED"] = str(seed)
    os.environ["SELECT_POOL40"] = "1" if pool40 else "0"


def _key(sleeve: str, name: str, anchor: int, stress: bool, tag: str = "") -> str:
    safe = name.replace("/", "_").replace(" ", "_")
    return f"{sleeve}__{safe}__a{anchor}{'__s5' if stress else ''}{tag}"


def _save(key: str, eq) -> None:
    eq.to_csv(os.path.join(EQDIR, f"{key}.csv"), header=["eq"])


def _load(key: str):
    p = os.path.join(EQDIR, f"{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def ensure(sleeve: str, name: str, mode: str, anchor: int, stress: bool,
           tag: str = "", **env):
    k = _key(sleeve, name, anchor, stress, tag)
    eq = _load(k)
    if eq is None:
        set_env(mode, **env)
        try:
            if sleeve == "fu":
                eq = R.run_fut(FU_OVER, R.PH_FUT[anchor], stress)
            else:
                eq = R.run_spot(SP_OVER, R.PH_SPOT[anchor], stress)
        finally:
            set_env()
        _save(k, eq)
    return eq


def pairwise_rank(cand, base, other, stock, sleeve):
    """후보와 B0 둘만 넣은 쌍대 순위합. 바뀌지 않는 슬리브는 B0 로 고정한다."""
    if sleeve == "fu":
        eq_a = R.build_alloc(stock, other, cand)
        eq_b = R.build_alloc(stock, other, base)
    else:
        eq_a = R.build_alloc(stock, cand, other)
        eq_b = R.build_alloc(stock, base, other)
    sums, _wins, n = R.window_rs({"A": eq_a, "B": eq_b})
    return sums["A"] / n, sums["B"] / n, eq_a, eq_b


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"데이터 로드 {time.time() - t0:.0f}s · 종목 {len(R._bars)}", flush=True)

    # 주식 슬리브는 이 실험의 축이 아니다. 기존 하니스 캐시를 그대로 빌려 쓴다.
    stock = {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if os.path.exists(p):
            stock[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock[a] = R.run_stock(a, False)
            stock[a].to_csv(p, header=["eq"])
    print(f"주식 슬리브 7벌 준비 {time.time() - t0:.0f}s", flush=True)

    total = (len(ARMS_SPOT) + len(ARMS_FUT)) * len(R.ANCHORS) * 2
    done = 0
    for sleeve, arms in (("sp", ARMS_SPOT), ("fu", ARMS_FUT)):
        for name, mode in arms:
            for a in R.ANCHORS:
                for st in (False, True):
                    ensure(sleeve, name, mode, a, st)
                    done += 1
            print(f"  {sleeve} {name} 14벌 완료 ({done}/{total}, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    lines = []
    results = {}
    for sleeve, arms in (("sp", ARMS_SPOT), ("fu", ARMS_FUT)):
        other = "fu" if sleeve == "sp" else "sp"
        label = "현물" if sleeve == "sp" else "선물"
        lines.append(f"\n==== {label} 슬리브 — 후보 대 B0 쌍대 ====")
        for name, _mode in arms:
            if name == B0:
                continue
            row: dict = {"sleeve": label, "arm": name}
            for st in (False, True):
                ck = "5배" if st else "기본"
                ra, rb, wins = [], [], 0
                dmdd_comp, dmdd_sl = [], []
                for a in R.ANCHORS:
                    cand = _load(_key(sleeve, name, a, st))
                    base = _load(_key(sleeve, B0, a, st))
                    oth = _load(_key(other, B0, a, st))
                    x, y, eq_a, eq_b = pairwise_rank(cand, base, oth, stock[a], sleeve)
                    ra.append(x)
                    rb.append(y)
                    wins += 1 if x < y else 0
                    dmdd_comp.append(abs(R.metrics(eq_a)["MDD"])
                                     - abs(R.metrics(eq_b)["MDD"]))
                    dmdd_sl.append(abs(R.metrics(cand)["MDD"])
                                   - abs(R.metrics(base)["MDD"]))
                g = (np.mean(rb) - np.mean(ra)) / np.mean(rb)
                row[ck] = dict(R_cand=float(np.mean(ra)), R_b0=float(np.mean(rb)),
                               G=float(g), wins=wins,
                               mdd_worst_comp=float(max(dmdd_comp)),
                               mdd_worst_sleeve=float(max(dmdd_sl)))
                lines.append(
                    f"{name:<10} {ck:<3} 순위 후보 {np.mean(ra):.3f} / B0 {np.mean(rb):.3f}"
                    f" · 개선율 {g * 100:+.2f}% · 이긴 앵커 {wins}/7"
                    f" · 합성낙폭 악화 최대 {max(dmdd_comp):+.2f}%p"
                    f" · 슬리브단독 악화 최대 {max(dmdd_sl):+.2f}%p")
            g1, g2 = row["기본"]["G"], row["5배"]["G"]
            w1, w2 = row["기본"]["wins"], row["5배"]["wins"]
            m1 = max(row["기본"]["mdd_worst_comp"], row["5배"]["mdd_worst_comp"])
            m2 = max(row["기본"]["mdd_worst_sleeve"], row["5배"]["mdd_worst_sleeve"])
            gate1 = bool(g1 >= 0.02 and g2 >= 0.02 and w1 >= 5 and w2 >= 5)
            gate2 = bool(m1 <= 0.5 and m2 <= 2.0)
            row["gate1"], row["gate2"] = gate1, gate2
            lines.append(f"{'':<10} 게이트① {'통과' if gate1 else '탈락'}"
                         f" · 게이트② {'통과' if gate2 else '탈락'}")
            results[f"{label}/{name}"] = row

    # 보고용 전체 팔 동시 비교 (판정 근거 아님).
    for sleeve, arms in (("sp", ARMS_SPOT), ("fu", ARMS_FUT)):
        other = "fu" if sleeve == "sp" else "sp"
        label = "현물" if sleeve == "sp" else "선물"
        for st in (False, True):
            ck = "5배" if st else "기본"
            tot = {name: 0.0 for name, _ in arms}
            for a in R.ANCHORS:
                eqd = {}
                for name, _m in arms:
                    cand = _load(_key(sleeve, name, a, st))
                    oth = _load(_key(other, B0, a, st))
                    eqd[name] = (R.build_alloc(stock[a], oth, cand) if sleeve == "fu"
                                 else R.build_alloc(stock[a], cand, oth))
                s, _w, n = R.window_rs(eqd)
                for name in tot:
                    tot[name] += s[name] / n / len(R.ANCHORS)
            lines.append(f"\n[보고용] {label} {ck} 전체 팔 순위평균 — " + " · ".join(
                f"{k} {v:.3f}" for k, v in sorted(tot.items(), key=lambda kv: kv[1])))
            results[f"보고용/{label}/{ck}"] = {k: float(v) for k, v in tot.items()}

    lines.append("\n==== 슬리브 단독 성과 (앵커 0, 기본 비용) ====")
    for sleeve, arms in (("sp", ARMS_SPOT), ("fu", ARMS_FUT)):
        label = "현물" if sleeve == "sp" else "선물"
        for name, _m in arms:
            eq = _load(_key(sleeve, name, 0, False))
            mt = R.metrics(eq)
            lines.append(f"{label} {name:<10} Cal {mt['Cal']:.2f} · Sharpe {mt['Sharpe']:.2f}"
                         f" · CAGR {mt['CAGR']:.1f}% · MDD {mt['MDD']:.1f}%")
            results[f"성과/{label}/{name}"] = {k: float(v) for k, v in mt.items()}

    txt = "\n".join(lines)
    print(txt, flush=True)
    with open(os.path.join(OUT, "result.txt"), "w") as f:
        f.write(txt + f"\n\n총 {time.time() - t0:.0f}s\n")
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n총 {time.time() - t0:.0f}s · 저장 {OUT}", flush=True)


if __name__ == "__main__":
    main()
