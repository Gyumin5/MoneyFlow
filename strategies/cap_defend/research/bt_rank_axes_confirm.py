"""선정 순위 축 — 확증 검사 (2026-10-07, ai-debate run-20261007T013223Z 게이트 ③~⑥).

1차 측정(bt_rank_axes.py) 결과 게이트 ①② 를 둘 다 통과한 팔은 현물 U교대3 하나뿐이다.
선물은 일곱 팔 전부 탈락했으므로 명세의 중단 조건대로 B0(시총)를 유지하고, 확증은 돌리지 않는다.
확증 후보는 슬리브당 하나로 못박혀 있어 차점자(현물 C시총효율)로 내려가지 않는다.

여기서 재는 넷:
  ③ 풀 폭      후보와 B0 를 모두 원시 시총 상위 40종으로 제한해 재생한다. 제한 풀에서도
                두 비용 각각 순위평균 우위 + 5/7 앵커 승리, G_c 가 전체 풀 값의 50% 이상,
                그리고 게이트 ② 의 낙폭 한도를 그대로 통과해야 한다.
  ④ 교란 둘    지연형(점수만 s-5 봉까지) · 경계형(흡수 전 최종 순서의 3위↔4위 교환).
                각 변형이 B0 보다 순위평균이 낮고 5/7 이상 이기며 G_c 가 원형의 50% 이상,
                낙폭 한도 통과. 교란이 실제로 아무것도 안 바꿨으면 안정성 근거로 치지 않는다
                (현물 경계 교란 가능일 52.9% — probe_rank_select_compare.py 로 선행 확인).
  ⑤ 노출 대조  각 정규 스냅의 총 배정 비중 e_A·e_0 에 대해 e*=min 으로 양쪽을 줄여
                스냅별 현금 비중을 맞춘 뒤 두 곡선을 재생한다. 개별 상한은 안 올린다.
                이 대조에서도 두 비용 순위평균 우위 + 각각 5/7 승리를 요구한다.
  ⑥ 위약       시드 0~4 의 고정 해시 무작위 우선순위. 후보의 B0 대비 개선율이 두 비용 모두
                위약 다섯 중 최대보다 커야 하고, 각 위약과의 직접 쌍대에서도 이겨야 한다.
                다중비교 방어 대조이지 유의확률·거짓양성률 추정이 아니다.

노출 대조의 엔진 지원: unified_backtest.run 이 EXPO_REC 이면 (bar_i, 스냅번호, 총배정) 을
적고 EXPO_APPLY 면 그 표의 값으로 비중을 줄인다. 기록은 앵커당 한 번만 뜬다 — 스냅 배정
시점과 배정 비중은 가격·카나리·앵커 위상만으로 정해져서 비용 배수에 안 흔들린다. 그래서
기본비용에서 기록해 두 비용에 같이 쓴다. 실제로 같은지는 기록 회차 자산곡선이 1차 캐시와
최대괴리 0 인지로 같이 확인한다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_rank_axes_confirm.py
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
import bt_rank_axes as X  # noqa: E402

OUT = X.OUT
EQDIR = X.EQDIR
EXPODIR = os.path.join(OUT, "expo")

CAND_NAME, CAND_MODE = "U교대3", "ualt"
B0 = X.B0
SLEEVE, OTHER = "sp", "fu"
PLACEBO_SEEDS = [0, 1, 2, 3, 4]

# R.metrics 의 MDD 는 이미 퍼센트 단위다(mdd*100). 한도도 같은 단위로 쓴다.
MDD_COMP_LIMIT = 0.5   # 합성 최대낙폭 악화 0.5%포인트
MDD_SLEEVE_LIMIT = 2.0  # 슬리브 단독 악화 2%포인트


def ensure_expo(name: str, mode: str, anchor: int, stress: bool, tag: str,
                rec: str = "", apply: str = ""):
    """EXPO_REC / EXPO_APPLY 를 켠 채로 한 벌 돌린다. 캐시는 ensure 와 같은 규칙."""
    k = X._key(SLEEVE, name, anchor, stress, tag)
    eq = X._load(k)
    # 기록 회차는 산출물이 자산곡선이 아니라 기록 파일이다 — 그게 없으면 캐시를 안 믿는다.
    if eq is not None and (not rec or os.path.exists(rec)):
        return eq
    os.environ["EXPO_REC"] = rec
    os.environ["EXPO_APPLY"] = apply
    X.set_env(mode)
    try:
        eq = R.run_spot(X.SP_OVER, R.PH_SPOT[anchor], stress)
    finally:
        X.set_env()
        os.environ["EXPO_REC"] = ""
        os.environ["EXPO_APPLY"] = ""
    X._save(k, eq)
    return eq


def read_expo(path: str) -> dict:
    tbl = {}
    with open(path) as f:
        for ln in f:
            a, b, c = ln.split()
            tbl[(int(a), int(b))] = float(c)
    return tbl


def compare(get_cand, get_base, stock, *, need_mdd=True):
    """후보 대 기준선 쌍대 평가. 두 비용 각각 순위평균·승수·낙폭 악화를 돌려준다."""
    out = {}
    for st in (False, True):
        ck = "5배" if st else "기본"
        ra, rb, wins = [], [], 0
        d_comp, d_sl = [], []
        for a in R.ANCHORS:
            cand = get_cand(a, st)
            base = get_base(a, st)
            oth = X._load(X._key(OTHER, B0, a, st))
            x, y, eq_a, eq_b = X.pairwise_rank(cand, base, oth, stock[a], SLEEVE)
            ra.append(x)
            rb.append(y)
            wins += 1 if x < y else 0
            if need_mdd:
                d_comp.append(abs(R.metrics(eq_a)["MDD"]) - abs(R.metrics(eq_b)["MDD"]))
                d_sl.append(abs(R.metrics(cand)["MDD"]) - abs(R.metrics(base)["MDD"]))
        g = (np.mean(rb) - np.mean(ra)) / np.mean(rb)
        out[ck] = dict(R_cand=float(np.mean(ra)), R_base=float(np.mean(rb)),
                       G=float(g), wins=wins,
                       dmdd_comp=float(max(d_comp)) if d_comp else 0.0,
                       dmdd_sleeve=float(max(d_sl)) if d_sl else 0.0)
    return out


def fmt(res: dict, title: str, lines: list) -> None:
    for ck in ("기본", "5배"):
        r = res[ck]
        lines.append(f"{title:<16}{ck:<4} 순위 후보 {r['R_cand']:.3f} / 기준 {r['R_base']:.3f}"
                     f" · 개선율 {r['G'] * 100:+.2f}% · 이긴 앵커 {r['wins']}/7"
                     f" · 합성낙폭 {r['dmdd_comp']:+.2f}%p"
                     f" · 슬리브낙폭 {r['dmdd_sleeve']:+.2f}%p")


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    os.makedirs(EXPODIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"데이터 로드 {time.time() - t0:.0f}s", flush=True)

    stock = {}
    for a in R.ANCHORS:
        stock[a] = pd.read_csv(os.path.join(R.EQDIR, f"st__a{a}.csv"),
                               index_col=0, parse_dates=True)["eq"]

    base_eq = {(a, st): X._load(X._key(SLEEVE, B0, a, st))
               for a in R.ANCHORS for st in (False, True)}
    cand_eq = {(a, st): X._load(X._key(SLEEVE, CAND_NAME, a, st))
               for a in R.ANCHORS for st in (False, True)}
    missing = [k for k, v in list(base_eq.items()) + list(cand_eq.items()) if v is None]
    if missing:
        raise SystemExit(f"1차 캐시 누락 {missing} — bt_rank_axes.py 를 먼저 돌려라")

    lines: list[str] = []
    verdict: dict = {}

    # 원형 G (게이트 ③④ 의 50% 기준선)
    orig = compare(lambda a, st: cand_eq[(a, st)], lambda a, st: base_eq[(a, st)], stock)
    lines.append("==== 원형 (1차 재확인) ====")
    fmt(orig, CAND_NAME, lines)
    G0 = {ck: orig[ck]["G"] for ck in ("기본", "5배")}

    # ── ③ 풀 폭 40종 ──
    print("③ 풀 폭 40종", flush=True)
    for a in R.ANCHORS:
        for st in (False, True):
            X.ensure(SLEEVE, CAND_NAME, CAND_MODE, a, st, "__p40", pool40=True)
            X.ensure(SLEEVE, B0, "b0", a, st, "__p40", pool40=True)
    g3 = compare(lambda a, st: X._load(X._key(SLEEVE, CAND_NAME, a, st, "__p40")),
                 lambda a, st: X._load(X._key(SLEEVE, B0, a, st, "__p40")), stock)
    lines.append("\n==== ③ 풀 폭 — 양쪽 모두 원시 시총 앞 40종 ====")
    fmt(g3, CAND_NAME + " 40종", lines)
    ok3 = all(g3[ck]["G"] > 0 and g3[ck]["wins"] >= 5
              and g3[ck]["G"] >= 0.5 * G0[ck]
              and g3[ck]["dmdd_comp"] <= MDD_COMP_LIMIT
              and g3[ck]["dmdd_sleeve"] <= MDD_SLEEVE_LIMIT for ck in ("기본", "5배"))
    verdict["③ 풀 폭"] = ok3

    # ── ④ 교란 둘 ──
    print("④ 교란 둘", flush=True)
    for a in R.ANCHORS:
        for st in (False, True):
            X.ensure(SLEEVE, CAND_NAME, CAND_MODE, a, st, "__lag5", lag=5)
            X.ensure(SLEEVE, CAND_NAME, CAND_MODE, a, st, "__swap", swap=True)
    g4 = {}
    for tag, label in (("__lag5", "지연 5봉"), ("__swap", "경계 3↔4")):
        r = compare(lambda a, st, _t=tag: X._load(X._key(SLEEVE, CAND_NAME, a, st, _t)),
                    lambda a, st: base_eq[(a, st)], stock)
        g4[label] = r
        lines.append(f"\n==== ④ 교란 — {label} ====")
        fmt(r, CAND_NAME + " " + label, lines)
    ok4 = all(r[ck]["G"] > 0 and r[ck]["wins"] >= 5 and r[ck]["G"] >= 0.5 * G0[ck]
              and r[ck]["dmdd_comp"] <= MDD_COMP_LIMIT
              and r[ck]["dmdd_sleeve"] <= MDD_SLEEVE_LIMIT
              for r in g4.values() for ck in ("기본", "5배"))
    verdict["④ 교란"] = ok4

    # ── ⑤ 노출 대조 ──
    print("⑤ 노출 대조", flush=True)
    rec_dev = []
    for a in R.ANCHORS:
        pc = os.path.join(EXPODIR, f"cand_a{a}.txt")
        pb = os.path.join(EXPODIR, f"b0_a{a}.txt")
        ps = os.path.join(EXPODIR, f"star_a{a}.txt")
        if not os.path.exists(ps):
            e1 = ensure_expo(CAND_NAME, CAND_MODE, a, False, "__rec", rec=pc)
            e2 = ensure_expo(B0, "b0", a, False, "__rec", rec=pb)
            # 기록 회차는 비중을 안 건드린다 — 1차 캐시와 같아야 한다.
            rec_dev.append(float(np.max(np.abs(e1.values - cand_eq[(a, False)].values))))
            rec_dev.append(float(np.max(np.abs(e2.values - base_eq[(a, False)].values))))
            tc, tb = read_expo(pc), read_expo(pb)
            with open(ps, "w") as f:
                for k in sorted(set(tc) | set(tb)):
                    f.write(f"{k[0]} {k[1]} {min(tc.get(k, 1.0), tb.get(k, 1.0))!r}\n")
        for st in (False, True):
            ensure_expo(CAND_NAME, CAND_MODE, a, st, "__xm", apply=ps)
            ensure_expo(B0, "b0", a, st, "__xm", apply=ps)
    if rec_dev:
        lines.append(f"\n기록 회차 자산곡선 최대괴리 {max(rec_dev):.3e} (0 이어야 한다)")
    g5 = compare(lambda a, st: X._load(X._key(SLEEVE, CAND_NAME, a, st, "__xm")),
                 lambda a, st: X._load(X._key(SLEEVE, B0, a, st, "__xm")), stock)
    lines.append("\n==== ⑤ 노출 대조 — 스냅별 총 배정을 작은 쪽에 맞춤 ====")
    fmt(g5, CAND_NAME + " 노출맞춤", lines)
    ok5 = all(g5[ck]["G"] > 0 and g5[ck]["wins"] >= 5 for ck in ("기본", "5배"))
    verdict["⑤ 노출 대조"] = ok5

    # ── ⑥ 위약 ──
    print("⑥ 위약 5벌", flush=True)
    for s in PLACEBO_SEEDS:
        for a in R.ANCHORS:
            for st in (False, True):
                X.ensure(SLEEVE, f"위약{s}", "placebo", a, st, f"__pb{s}", seed=s)
    lines.append("\n==== ⑥ 위약 대조 ====")
    pb_g = {}
    for s in PLACEBO_SEEDS:
        r = compare(lambda a, st, _s=s: X._load(X._key(SLEEVE, f"위약{_s}", a, st, f"__pb{_s}")),
                    lambda a, st: base_eq[(a, st)], stock, need_mdd=False)
        pb_g[s] = r
        lines.append(f"위약{s} 대 B0        기본 개선율 {r['기본']['G'] * 100:+.2f}%"
                     f" (승 {r['기본']['wins']}/7) · 5배 {r['5배']['G'] * 100:+.2f}%"
                     f" (승 {r['5배']['wins']}/7)")
    best_pb = {ck: max(pb_g[s][ck]["G"] for s in PLACEBO_SEEDS) for ck in ("기본", "5배")}
    lines.append(f"위약 최대 개선율 — 기본 {best_pb['기본'] * 100:+.2f}%"
                 f" · 5배 {best_pb['5배'] * 100:+.2f}%"
                 f" / 후보 {G0['기본'] * 100:+.2f}% · {G0['5배'] * 100:+.2f}%")
    direct_ok = True
    for s in PLACEBO_SEEDS:
        r = compare(lambda a, st: cand_eq[(a, st)],
                    lambda a, st, _s=s: X._load(X._key(SLEEVE, f"위약{_s}", a, st, f"__pb{_s}")),
                    stock, need_mdd=False)
        win = all(r[ck]["R_cand"] < r[ck]["R_base"] for ck in ("기본", "5배"))
        direct_ok = direct_ok and win
        lines.append(f"후보 대 위약{s} 직접   기본 {r['기본']['R_cand']:.3f}/{r['기본']['R_base']:.3f}"
                     f" · 5배 {r['5배']['R_cand']:.3f}/{r['5배']['R_base']:.3f}"
                     f" · {'우위' if win else '열위'}")
    ok6 = direct_ok and all(G0[ck] > best_pb[ck] for ck in ("기본", "5배"))
    verdict["⑥ 위약"] = ok6

    lines.append("\n==== 판정 ====")
    for k, v in verdict.items():
        lines.append(f"{k:<12} {'통과' if v else '탈락'}")
    lines.append(f"현물 {CAND_NAME} 확증 {'전부 통과' if all(verdict.values()) else '탈락'}")
    lines.append(f"선물 — 게이트 ①② 통과 팔 없음. B0(시총) 유지, 확증 미실시.")
    lines.append(f"\n총 {time.time() - t0:.0f}s")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "confirm.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, "confirm.json"), "w") as f:
        json.dump({"orig": orig, "pool40": g3, "perturb": g4, "expo": g5,
                   "placebo_G": pb_g, "placebo_best": best_pb,
                   "verdict": verdict}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
