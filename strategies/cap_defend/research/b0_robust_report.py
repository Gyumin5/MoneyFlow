"""현행 선정 규칙의 우연 여부 — 판정 (2026-10-07).

bt_b0_robust.py 가 쌓아 둔 자산곡선만 읽는다. 백테스트를 다시 돌리지 않는다.
문턱은 전부 b0_robust_prereg.md 에 결과 보기 전에 고정한 값이고 여기서 바꾸지 않는다.

내는 것.
  ① H0-R — 현행이 고정형·갱신형 무작위 선정과 단순 대조 둘을 이겼는가.
  ② A(E)·rho(E) — 코인을 빼면 무작위 대비 이득이 얼마나 남는가.
  ③ 의존 확정 검사 — 보존율·샤프·최대낙폭까지 같이 본다.
  ④ 기간 집중 — 창 크기별 승률, 최장 연패, 상위 10퍼센트 창 제거 민감도.
  ⑤ 검증 — 내 창별 순위합이 기존 window_rs 와 같은지.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 b0_robust_report.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402
import bt_b0_robust as B  # noqa: E402

OUT = B.OUT
CAL_FLOOR = 0.10          # 상대 개선율 분모 바닥
A_EMPTY_MIN = 0.10        # A(공집합) 이 이보다 작으면 잔존율 판정 불충분
RHO_SINGLE = 0.50
RHO_GROUP = 1.0 / 3.0
RHO_DEP = 0.25
PRESERVE_DEP = 0.70
SHARPE_DROP = 0.20
MDD_RISE = 3.0            # 퍼센트포인트. R.metrics 의 MDD 는 이미 퍼센트다.
WIN_MIN_ROBUST = 0.55
WIN_MIN_H0R = 0.60
ANCHOR_WIN_MIN = 5
STREAK_MAX = 4
TRIM_FRAC = 0.10
TRIM_WIN_MIN = 0.55
TRIM_CAL_MIN = 0.05

PLACEBO_5 = B.PLACEBO_5
PFIX = [f"pfix:{i}" for i in range(B.N_SEED)]
PREF = [f"pref:{i}" for i in range(B.N_SEED)]

_stock: dict = {}


def stock(a):
    if a not in _stock:
        import pandas as pd
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        _stock[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return _stock[a]


def comp(a, stress, sp_arm, sp_ex, fu_arm, fu_ex):
    """합성 60/25/15. 빠진 곡선이 있으면 None."""
    sp = B.load(B.key("sp", sp_arm, sp_ex, a, stress))
    fu = B.load(B.key("fu", fu_arm, fu_ex, a, stress))
    if sp is None or fu is None:
        return None
    return R.build_alloc(stock(a), sp, fu)


def variant(sleeve, arm, ex, a, stress, both=False):
    """바뀌는 슬리브만 갈고 나머지는 B0 무제외로 고정. both 면 양쪽에 같은 제외."""
    if both:
        return comp(a, stress, arm, ex, arm, ex)
    if sleeve == "sp":
        return comp(a, stress, arm, ex, "b0", [])
    return comp(a, stress, "b0", [], arm, ex)


def win_cals(eq_dict):
    """창마다 각 곡선의 칼마. window_rs 와 같은 창 집합·같은 식을 쓴다."""
    common = None
    for s in eq_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    common = sorted(common)
    rows = []
    for size in R.WIN_SIZES:
        for stride in R.STRIDES:
            for i in range(0, len(common) - size, stride):
                d0, d1 = common[i], common[i + size - 1]
                cals, bad = {}, False
                for k, s in eq_dict.items():
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30:
                        bad = True
                        break
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0:
                        bad = True
                        break
                    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                    mdd = float((seg / seg.cummax() - 1).min())
                    cals[k] = cagr / abs(mdd) if mdd < 0 else 0.0
                if not bad:
                    rows.append(dict(size=size, stride=stride, start=d0, cals=cals))
    return rows


def pair_R(eq_a, eq_b):
    """쌍대 순위합의 창당 평균. 낮을수록 우세."""
    sums, _w, n = R.window_rs({"A": eq_a, "B": eq_b})
    return sums["A"] / n, sums["B"] / n


def win_rate_by_size(rows, name_a, name_b):
    """창 크기별 승률. 동률 0.5승."""
    out = {}
    for size in R.WIN_SIZES:
        sel = [r for r in rows if r["size"] == size]
        if not sel:
            out[size] = float("nan")
            continue
        w = 0.0
        for r in sel:
            ca, cb = r["cals"][name_a], r["cals"][name_b]
            w += 1.0 if ca > cb else (0.5 if ca == cb else 0.0)
        out[size] = w / len(sel)
    return out


def median_ctrl_rows(rows, b0_name, ctrl_names):
    """대조군 중앙값을 CTRL 이라는 이름의 한 곡선처럼 붙인다."""
    for r in rows:
        r["cals"]["CTRL"] = st.median([r["cals"][c] for c in ctrl_names])
    return rows


def rel_improve(cal_a, cal_b):
    return (cal_a - cal_b) / max(abs(cal_b), CAL_FLOOR)


# ───────────────────────── ① H0-R ─────────────────────────

def h0r(sleeve, lines):
    lab = "현물" if sleeve == "sp" else "선물"
    res = {}
    for cost_name, stress in (("기본", False), ("5배", True)):
        # 쌍대 앵커 승리
        pair_wins = {}
        for arm in PFIX + PREF + ["top1", "fullpool"]:
            w = 0
            for a in R.ANCHORS:
                ea = variant(sleeve, "b0", [], a, stress)
                eb = variant(sleeve, arm, [], a, stress)
                if ea is None or eb is None:
                    continue
                ra, rb = pair_R(ea, eb)
                w += 1 if ra < rb else 0
            pair_wins[arm] = w
        # 모형별 중앙값 대조
        fam = {}
        for fname, members in (("고정형", PFIX), ("갱신형", PREF)):
            wr_acc = {s: [] for s in R.WIN_SIZES}
            rels = []
            for a in R.ANCHORS:
                d = {"B0": variant(sleeve, "b0", [], a, stress)}
                for m in members:
                    e = variant(sleeve, m, [], a, stress)
                    if e is not None:
                        d[m] = e
                if d["B0"] is None or len(d) < 2:
                    continue
                rows = median_ctrl_rows(win_cals(d), "B0", [m for m in members if m in d])
                wr = win_rate_by_size(rows, "B0", "CTRL")
                for s in R.WIN_SIZES:
                    wr_acc[s].append(wr[s])
                cb = R.metrics(d["B0"])["Cal"]
                cc = st.median([R.metrics(d[m])["Cal"] for m in members if m in d])
                rels.append(rel_improve(cb, cc))
            fam[fname] = dict(
                win_rate={s: float(np.mean(v)) if v else float("nan")
                          for s, v in wr_acc.items()},
                rel_median=float(st.median(rels)) if rels else float("nan"),
                n_pass_pair=sum(1 for m in members
                                if pair_wins.get(m, 0) >= ANCHOR_WIN_MIN),
                n=len(members))
        res[cost_name] = dict(pair_wins=pair_wins, fam=fam)
        lines.append(f"\n-- {lab} · {cost_name} 비용 --")
        for fname in ("고정형", "갱신형"):
            f = fam[fname]
            wr = " / ".join(f"{s}봉 {f['win_rate'][s]*100:.1f}%" for s in R.WIN_SIZES)
            lines.append(f"  {fname} 위약 {f['n']}벌 — 앵커 5/7 이상 이긴 상대 "
                         f"{f['n_pass_pair']}/{f['n']} · 중앙값 대조 창 승률 {wr}"
                         f" · 칼마 상대개선 앵커중앙 {f['rel_median']*100:+.1f}%")
        for simple in ("top1", "fullpool"):
            rels = []
            for a in R.ANCHORS:
                ea = variant(sleeve, "b0", [], a, stress)
                eb = variant(sleeve, simple, [], a, stress)
                if ea is not None and eb is not None:
                    rels.append(rel_improve(R.metrics(ea)["Cal"], R.metrics(eb)["Cal"]))
            res[cost_name][simple] = dict(wins=pair_wins.get(simple, 0),
                                          rel_median=float(st.median(rels)) if rels else float("nan"))
            lines.append(f"  단순 대조 {simple:9s} — 앵커 승 {pair_wins.get(simple,0)}/7"
                         f" · 칼마 상대개선 앵커중앙 {st.median(rels)*100:+.1f}%")
    # 판정
    ok = True
    why = []
    for cost_name in ("기본", "5배"):
        for fname in ("고정형", "갱신형"):
            f = res[cost_name]["fam"][fname]
            if f["n_pass_pair"] < 18:
                ok = False
                why.append(f"{cost_name}·{fname} 앵커 5/7 이상이 {f['n_pass_pair']}/19")
            if any(f["win_rate"][s] < WIN_MIN_H0R for s in R.WIN_SIZES):
                ok = False
                why.append(f"{cost_name}·{fname} 창 승률 60% 미만")
            if not (f["rel_median"] >= 0.05):
                ok = False
                why.append(f"{cost_name}·{fname} 칼마 상대개선 5% 미만")
        for simple in ("top1", "fullpool"):
            s = res[cost_name][simple]
            if s["wins"] < ANCHOR_WIN_MIN or not (s["rel_median"] >= 0.05):
                ok = False
                why.append(f"{cost_name}·{simple} 미달")
    res["통과"] = ok
    res["사유"] = why
    lines.append(f"  → {lab} H0-R {'기각(현행 우위)' if ok else '기각 실패'}"
                 + ("" if ok else " — " + "; ".join(why[:4])))
    return res


# ───────────────────────── ② 제외 ─────────────────────────

def a_of(sleeve, ex, stress, both=False):
    """A(E) — 같은 제외 위약 5벌과의 창별 합성 칼마 차이. 창별 중앙값 → 전체 중앙값."""
    per = []
    for a in R.ANCHORS:
        d = {"B0": variant(sleeve, "b0", ex, a, stress, both)}
        for m in PLACEBO_5:
            e = variant(sleeve, m, ex, a, stress, both)
            if e is not None:
                d[m] = e
        if d["B0"] is None or len(d) < 2:
            continue
        for r in win_cals(d):
            diffs = [r["cals"]["B0"] - r["cals"][m] for m in PLACEBO_5 if m in r["cals"]]
            if diffs:
                per.append(st.median(diffs))
    return float(st.median(per)) if per else float("nan")


def excl_block(sleeve, name, ex, is_group, a_empty, lines, both=False):
    # both=True 는 두 슬리브에 같은 제외를 거는 측정이라 슬리브 축이 없다. 한 줄로만 적고
    # 분모도 같은 조건(두 슬리브 위약)의 A(공집합)을 쓴다 — 한쪽 슬리브 분모를 쓰면
    # 같은 측정이 현물·선물에서 다른 판정으로 갈린다.
    lab = "두 슬리브" if both else ("현물" if sleeve == "sp" else "선물")
    row = dict(sleeve=lab, name=name, ex=list(ex), group=is_group, both=both)
    for cost_name, stress in (("기본", False), ("5배", True)):
        A = a_of(sleeve, ex, stress, both)
        rho = A / a_empty[cost_name] if a_empty[cost_name] not in (0, None) else float("nan")
        # 같은 제외 위약 5벌 상대 앵커 승리
        pw = {}
        for m in PLACEBO_5:
            w = 0
            for a in R.ANCHORS:
                ea = variant(sleeve, "b0", ex, a, stress, both)
                eb = variant(sleeve, m, ex, a, stress, both)
                if ea is None or eb is None:
                    continue
                ra, rb = pair_R(ea, eb)
                w += 1 if ra < rb else 0
            pw[m] = w
        n_pass = sum(1 for m in PLACEBO_5 if pw[m] >= ANCHOR_WIN_MIN)
        # 창 크기별 승률 — 위약 5벌 중앙값 대조
        wr_acc = {s: [] for s in R.WIN_SIZES}
        preserve, dsharpe, dmdd = [], [], []
        for a in R.ANCHORS:
            d = {"B0": variant(sleeve, "b0", ex, a, stress, both)}
            for m in PLACEBO_5:
                e = variant(sleeve, m, ex, a, stress, both)
                if e is not None:
                    d[m] = e
            base = variant(sleeve, "b0", [], a, stress)
            if d["B0"] is None or base is None or len(d) < 2:
                continue
            rows = median_ctrl_rows(win_cals(d), "B0",
                                    [m for m in PLACEBO_5 if m in d])
            wr = win_rate_by_size(rows, "B0", "CTRL")
            for s in R.WIN_SIZES:
                wr_acc[s].append(wr[s])
            mE, m0 = R.metrics(d["B0"]), R.metrics(base)
            preserve.append(mE["Cal"] / m0["Cal"] if m0["Cal"] else float("nan"))
            dsharpe.append(m0["Sharpe"] - mE["Sharpe"])
            dmdd.append(abs(mE["MDD"]) - abs(m0["MDD"]))
        wr_mean = {s: float(np.mean(v)) if v else float("nan") for s, v in wr_acc.items()}
        rho_thr = RHO_GROUP if is_group else RHO_SINGLE
        robust = (rho >= rho_thr and n_pass >= 4
                  and all(wr_mean[s] >= WIN_MIN_ROBUST for s in R.WIN_SIZES))
        n_pres_bad = sum(1 for p in preserve if p <= PRESERVE_DEP)
        dep = ((rho < RHO_DEP or A <= 0) and n_pres_bad >= ANCHOR_WIN_MIN
               and (st.median(dsharpe) >= SHARPE_DROP or st.median(dmdd) >= MDD_RISE))
        row[cost_name] = dict(A=A, rho=rho, n_pass=n_pass, pair_wins=pw,
                              win_rate=wr_mean, robust=bool(robust), dep=bool(dep),
                              preserve_med=float(st.median(preserve)) if preserve else float("nan"),
                              n_preserve_bad=n_pres_bad,
                              dsharpe_med=float(st.median(dsharpe)) if dsharpe else float("nan"),
                              dmdd_med=float(st.median(dmdd)) if dmdd else float("nan"))
        c = row[cost_name]
        wr = "/".join(f"{c['win_rate'][s]*100:.0f}" for s in R.WIN_SIZES)
        lines.append(f"  {lab} {name:12s} {cost_name:3s} A={A:+.3f} rho={rho:+.2f} "
                     f"위약승 {n_pass}/5 창승률 {wr} 보존 {c['preserve_med']:.2f} "
                     f"d샤프 {c['dsharpe_med']:+.2f} d낙폭 {c['dmdd_med']:+.1f}%p "
                     f"→ {'견고' if robust else ('의존' if dep else '사이')}")
    # 두 비용 모두 만족해야 한다
    row["판정"] = ("견고" if row["기본"]["robust"] and row["5배"]["robust"]
                 else "의존" if row["기본"]["dep"] and row["5배"]["dep"]
                 else "비용취약" if row["기본"]["robust"] and not row["5배"]["robust"]
                 else "부분의존/불충분")
    return row


# ───────────────────────── ④ 기간 집중 ─────────────────────────

def period_block(sleeve, lines):
    lab = "현물" if sleeve == "sp" else "선물"
    out = {}
    for cost_name, stress in (("기본", False), ("5배", True)):
        streaks, trims, wins, cals = [], [], [], []
        for a in R.ANCHORS:
            d = {"B0": variant(sleeve, "b0", [], a, stress)}
            for m in PFIX + PREF:
                e = variant(sleeve, m, [], a, stress)
                if e is not None:
                    d[m] = e
            if d["B0"] is None or len(d) < 2:
                continue
            rows = median_ctrl_rows(win_cals(d), "B0",
                                    [m for m in PFIX + PREF if m in d])
            sel = sorted([r for r in rows if r["size"] == 504 and r["stride"] == 63],
                         key=lambda r: r["start"])
            seq = [1 if r["cals"]["B0"] > r["cals"]["CTRL"] else 0 for r in sel]
            best = cur = 0
            for x in seq:
                cur = 0 if x else cur + 1
                best = max(best, cur)
            streaks.append(best)
            diffs = [(r["cals"]["B0"] - r["cals"]["CTRL"]) for r in rows]
            k = int(np.ceil(len(diffs) * TRIM_FRAC))
            order = sorted(range(len(diffs)), key=lambda i: (-diffs[i],
                                                             str(rows[i]["start"]),
                                                             rows[i]["size"]))
            drop = set(order[:k])
            keep = [diffs[i] for i in range(len(diffs)) if i not in drop]
            w = sum(1.0 if x > 0 else (0.5 if x == 0 else 0.0) for x in keep) / len(keep)
            trims.append((w, float(st.median(keep))))
            wins.append(w)
            cals.append(float(st.median(keep)))
        out[cost_name] = dict(
            streak_max=max(streaks) if streaks else None,
            trim_win=float(np.mean(wins)) if wins else float("nan"),
            trim_cal=float(np.mean(cals)) if cals else float("nan"))
        o = out[cost_name]
        ok = (o["streak_max"] is not None and o["streak_max"] <= STREAK_MAX
              and o["trim_win"] >= TRIM_WIN_MIN and o["trim_cal"] >= TRIM_CAL_MIN)
        o["통과"] = bool(ok)
        lines.append(f"  {lab} {cost_name:3s} 최장 연패 {o['streak_max']} "
                     f"· 상위10% 제거 승률 {o['trim_win']*100:.1f}% "
                     f"· 칼마차 중앙 {o['trim_cal']:+.3f} → {'통과' if ok else '미달'}")
    return out


# ───────────────────────── ⑤ 검증 ─────────────────────────

def verify(lines):
    """내 창별 순위합이 기존 window_rs 와 같은가."""
    a = variant("sp", "b0", [], 0, False)
    b = variant("sp", "pfix:0", [], 0, False)
    rows = win_cals({"A": a, "B": b})
    sa = sb = 0.0
    for r in rows:
        ca, cb = r["cals"]["A"], r["cals"]["B"]
        if ca == cb:
            sa += 1.5
            sb += 1.5
        elif ca > cb:
            sa += 1
            sb += 2
        else:
            sa += 2
            sb += 1
    sums, _w, n = R.window_rs({"A": a, "B": b})
    ok = (len(rows) == n and abs(sa - sums["A"]) < 1e-9 and abs(sb - sums["B"]) < 1e-9)
    lines.append(f"  창 수 {len(rows)} 대 {n} · 순위합 {sa:.1f}/{sb:.1f} 대 "
                 f"{sums['A']:.1f}/{sums['B']:.1f} → {'일치' if ok else '불일치'}")
    return ok


def main() -> None:
    lines = []
    report: dict = {}

    lines.append("==== ⑤ 창별 순위합 검증 ====")
    report["verify"] = verify(lines)

    lines.append("\n==== ① H0-R — 현행이 무작위 선정보다 나은가 ====")
    for sleeve in ("sp", "fu"):
        report[f"h0r_{sleeve}"] = h0r(sleeve, lines)

    lines.append("\n==== ② 제외 — A(E)·잔존율 ====")
    excl = []
    for sleeve in ("sp", "fu"):
        a_empty = {c: a_of(sleeve, [], s) for c, s in (("기본", False), ("5배", True))}
        lab = "현물" if sleeve == "sp" else "선물"
        lines.append(f"\n-- {lab} · A(공집합) 기본 {a_empty['기본']:+.3f} "
                     f"· 5배 {a_empty['5배']:+.3f} --")
        for c in ("기본", "5배"):
            if not (a_empty[c] >= A_EMPTY_MIN):
                lines.append(f"  !! {lab} {c} A(공집합)이 {A_EMPTY_MIN} 미만 "
                             f"— 잔존율 판정은 불충분으로 적는다")
        for coin in B.TARGETS[sleeve]:
            excl.append(excl_block(sleeve, coin, [coin], False, a_empty, lines))
        for gname, g in B.GROUPS[sleeve]:
            excl.append(excl_block(sleeve, gname, g, True, a_empty, lines))

    # 두 슬리브 동시 제외 — 슬리브 축이 없으므로 한 번만 잰다. 분모도 같은 조건이다.
    a_both = {c: a_of("sp", [], s, True) for c, s in (("기본", False), ("5배", True))}
    lines.append(f"\n-- 두 슬리브 동시 · A(공집합) 기본 {a_both['기본']:+.3f} "
                 f"· 5배 {a_both['5배']:+.3f} --")
    for c in ("기본", "5배"):
        if not (a_both[c] >= A_EMPTY_MIN):
            lines.append(f"  !! 두 슬리브 {c} A(공집합)이 {A_EMPTY_MIN} 미만 "
                         f"— 잔존율 판정은 불충분으로 적는다")
    for gname, g in B.COMMON:
        excl.append(excl_block("sp", gname, g, True, a_both, lines, both=True))
    report["excl"] = excl

    lines.append("\n==== ④ 기간 집중 ====")
    for sleeve in ("sp", "fu"):
        report[f"period_{sleeve}"] = period_block(sleeve, lines)

    lines.append("\n==== 요약 ====")
    for sleeve in ("sp", "fu"):
        lab = "현물" if sleeve == "sp" else "선물"
        h = report[f"h0r_{sleeve}"]
        lines.append(f"  {lab} H0-R {'기각 — 지정 귀무 대비 우위' if h['통과'] else '기각 실패 — 추가 가치 미확인'}")
    for r in excl:
        lines.append(f"  {r['sleeve']} {r['name']:12s} {r['판정']}")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n저장 {OUT}/report.txt · report.json")


if __name__ == "__main__":
    main()
