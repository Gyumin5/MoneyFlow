"""채택 결정층을 앵커 7곳에서 11곳으로 바꾸면 판정이 뒤집히나 (2026-10-15).

물음은 운영자가 준 그대로다 — 더 좋으면 바꾸고 아니면 접는다. 더 좋은지를
"기존 판정을 새 채점법으로 다시 매겨 뒤집히는 게 하나라도 있나" 로 잰다.

배경. 2026-09-22 토론(history.md 같은 날 '논문 기반 후보 3건')이 앵커 한 곳
순위합을 폐기하라고 했고, 같은 날 만든 bt_coin_param_resweep.py 가 이미 앵커
일곱 곳 평균으로 돌고 있다. 그러니 남은 물음은 일곱이냐 열하나냐 하나다.

재는 대상은 그 뒤로 쌓인 두 스윕의 판정 전부다.
  · 코인 열 축 (bt_coin_param_resweep.py, 2026-09-22)
  · 주식 아홉 축 (bt_stock_param_resweep.py, 2026-09-25)

재는 법. 이 스크립트는 아무것도 새로 계산하지 않는다. 체크포인트(eq/*.csv)만
읽어서 같은 축·같은 팔을 앵커 0~6 평균과 앵커 0~10 평균으로 두 번 매긴다.
앞 일곱은 두 채점법에 똑같이 들어가므로, 차이는 뒤에 덧붙인 넷에서만 온다.

판정 기준은 이 저장소가 쓰던 것 그대로다 — 두 비용 조건 모두에서 현행보다
순위평균이 낮고 이긴 앵커가 과반인 팔만 후보로 본다. 그 통과 집합이 일곱과
열하나에서 다르면 뒤집힌 것이고, 같으면 바꿀 이유가 없다.

주의(토론이 못박은 것): 앵커 열하나는 스태거로 강하게 겹쳐 독립 표본이 아니다.
그래서 "몇 곳에서 이겼나" 를 승률로 읽지 않는다. 여기서 과반 조건을 쓰는 것은
새 문턱을 세우려는 게 아니라 옛 판정이 쓴 자를 그대로 대보기 위해서다.

먼저 할 것: ANCHORS_N=11 로 두 스윕을 돌려 앵커 7~10 체크포인트를 채운다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_anchor11_compare.py            # 코인 + 주식
      python3 bt_anchor11_compare.py coin       # 코인만
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

os.environ["ANCHORS_N"] = "11"
import bt_coin_param_resweep as R  # noqa: E402

A7 = list(range(7))
A11 = list(range(11))


def _read(path: str) -> pd.Series | None:
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def _stock_fixed(anchors: list[int]) -> dict[int, pd.Series]:
    """코인 축을 잴 때 합성에 쓰는 고정 주식 팔."""
    eq = {}
    for a in anchors:
        s = _read(os.path.join(R.EQDIR, f"st__a{a}.csv"))
        if s is None:
            raise SystemExit(f"주식 고정 팔 곡선이 없다 (앵커 {a}) — "
                             f"먼저 ANCHORS_N=11 python3 bt_coin_param_resweep.py")
        eq[a] = s
    return eq


def rank_by_anchor(arm_names, build, anchors):
    """앵커마다 팔들을 윈도우 순위합으로 매긴다. 반환 = {팔: [앵커별 평균순위]}."""
    out: dict[str, list[float]] = {n: [] for n in arm_names}
    for a in anchors:
        comps = {}
        for name in arm_names:
            eq = build(name, a)
            if eq is not None:
                comps[name] = eq
        if len(comps) < 2:
            continue
        sums, _, n = R.window_rs(comps)
        for name in comps:
            out[name].append(sums[name] / max(n, 1))
    return out


def gate_pass(base_cost: dict, stress_cost: dict, idx: list[int] | None = None):
    """두 비용 조건 모두 현행보다 낮은 순위평균 + 이긴 앵커 과반인 팔.

    idx 를 주면 그 앵커 자리만 골라서 매긴다(부분집합 안정성 측정용).
    """
    def pick(xs):
        return xs if idx is None else [xs[i] for i in idx if i < len(xs)]

    cur_a, cur_b = pick(base_cost.get(R.CUR, [])), pick(stress_cost.get(R.CUR, []))
    if not cur_a or not cur_b:
        return frozenset()
    ok = set()
    for name in base_cost:
        if name == R.CUR:
            continue
        ra, rb = pick(base_cost[name]), pick(stress_cost.get(name, []))
        if len(ra) != len(cur_a) or len(rb) != len(cur_b):
            continue
        if float(np.mean(ra)) >= float(np.mean(cur_a)):
            continue
        if float(np.mean(rb)) >= float(np.mean(cur_b)):
            continue
        wa = sum(1 for x, y in zip(ra, cur_a) if x < y)
        wb = sum(1 for x, y in zip(rb, cur_b) if x < y)
        if wa * 2 > len(ra) and wb * 2 > len(rb):
            ok.add(name)
    return frozenset(ok)


def compare_axis(axis, arm_names, builders, flipped):
    """builders = {(stress, anchor_count): build(name, anchor) -> eq|None}."""
    cache = {}
    for stress in (False, True):
        for anchors, tag in ((A7, 7), (A11, 11)):
            cache[(stress, tag)] = rank_by_anchor(
                arm_names, builders[stress], anchors)

    pass7 = gate_pass(cache[(False, 7)], cache[(True, 7)])
    pass11 = gate_pass(cache[(False, 11)], cache[(True, 11)])
    same = pass7 == pass11

    print(f"\n=== {axis} ===")
    print(f"{'팔':<16}{'기본7':>8}{'기본11':>8}{'차':>7}"
          f"{'5배7':>8}{'5배11':>8}{'차':>7}{'이긴앵커7':>10}{'이긴앵커11':>11}")
    cur7, cur11 = cache[(False, 7)].get(R.CUR, []), cache[(False, 11)].get(R.CUR, [])
    rows = []
    for name in arm_names:
        r7, r11 = cache[(False, 7)][name], cache[(False, 11)][name]
        t7, t11 = cache[(True, 7)][name], cache[(True, 11)][name]
        if not r7 or not r11:
            continue
        w7 = w11 = "-"
        if name != R.CUR and cur7 and cur11:
            w7 = "%d/7" % sum(1 for x, y in zip(r7, cur7) if x < y)
            w11 = "%d/11" % sum(1 for x, y in zip(r11, cur11) if x < y)
        rows.append((float(np.mean(r11)), name, float(np.mean(r7)),
                     float(np.mean(t7)), float(np.mean(t11)), w7, w11))
    for m11, name, m7, n7, n11, w7, w11 in sorted(rows):
        mark = "  <- 현행" if name == R.CUR else ""
        print(f"{name:<16}{m7:>8.3f}{m11:>8.3f}{m11-m7:>+7.3f}"
              f"{n7:>8.3f}{n11:>8.3f}{n11-n7:>+7.3f}{w7:>10}{w11:>11}{mark}")
    print(f"  게이트 통과 — 앵커 7: {sorted(pass7) or '없음'} · "
          f"앵커 11: {sorted(pass11) or '없음'} · "
          f"{'같다' if same else '다르다 ← 뒤집힘'}")
    if not same:
        flipped.append((axis, sorted(pass7), sorted(pass11)))


def run_coin(flipped):
    stock_eq = _stock_fixed(A11)
    print("\n" + "#" * 70)
    print("# 코인 열 축 — bt_coin_param_resweep.py (2026-09-22)")
    print("#" * 70)
    print("현물 위상 7 =", [R.PH_SPOT[a] for a in A7], "→ 11 =",
          [R.PH_SPOT[a] for a in A11])
    print("선물 위상 7 =", [R.PH_FUT[a] for a in A7], "→ 11 =",
          [R.PH_FUT[a] for a in A11])

    def make(stress):
        def build(name, a):
            k = R._key(name, a, stress)
            sp, fu = R._load("sp", k), R._load("fu", k)
            if sp is None or fu is None:
                return None
            return R.build_alloc(stock_eq[a], sp, fu)
        return build

    builders = {False: make(False), True: make(True)}
    for axis, arms in R.AXES.items():
        compare_axis(axis, [a[0] for a in arms], builders, flipped)


def run_stock(flipped):
    import bt_stock_param_resweep as S
    print("\n" + "#" * 70)
    print("# 주식 아홉 축 — bt_stock_param_resweep.py (2026-09-25)")
    print("#" * 70)
    print("주식 앵커는 스태거 23일 위상 그대로 0~6 → 0~10")

    coin = {}
    for a in A11:
        for stress in (False, True):
            suf = "__s5" if stress else ""
            sp = _read(f"{S.COIN_EQ}/sp__현행__a{a}{suf}.csv")
            fu = _read(f"{S.COIN_EQ}/fu__현행__a{a}{suf}.csv")
            if sp is None or fu is None:
                raise SystemExit(
                    f"코인 현행 곡선이 없다 (앵커 {a}, 비용 {'5배' if stress else '기본'}) — "
                    f"{S.COIN_EQ} 에 앵커 7~10 을 채워라")
            coin[(a, stress)] = (sp, fu)

    def make(stress):
        def build(name, a):
            eq = S._load_st(S._key(name, a, stress))
            if eq is None:
                return None
            sp, fu = coin[(a, stress)]
            return R.build_alloc(eq, sp, fu)
        return build

    builders = {False: make(False), True: make(True)}
    for axis, arms in S.AXES.items():
        compare_axis(axis, [a[0] for a in arms], builders, flipped)


def _coin_builders():
    stock_eq = _stock_fixed(A11)

    def make(stress):
        def build(name, a):
            k = R._key(name, a, stress)
            sp, fu = R._load("sp", k), R._load("fu", k)
            if sp is None or fu is None:
                return None
            return R.build_alloc(stock_eq[a], sp, fu)
        return build

    return R.AXES, {False: make(False), True: make(True)}


def _stock_builders():
    import bt_stock_param_resweep as S
    coin = {}
    for a in A11:
        for stress in (False, True):
            suf = "__s5" if stress else ""
            coin[(a, stress)] = (_read(f"{S.COIN_EQ}/sp__현행__a{a}{suf}.csv"),
                                 _read(f"{S.COIN_EQ}/fu__현행__a{a}{suf}.csv"))

    def make(stress):
        def build(name, a):
            eq = S._load_st(S._key(name, a, stress))
            sp, fu = coin[(a, stress)]
            if eq is None or sp is None or fu is None:
                return None
            return R.build_alloc(eq, sp, fu)
        return build

    return S.AXES, {False: make(False), True: make(True)}


def stability() -> None:
    """7곳이라는 답이 "어느 일곱을 골랐나" 에 얼마나 달려 있나.

    뒤집힘 개수만으로는 7 과 11 중 어느 쪽이 나은지 안 갈린다 — 뒤집혔다는 것은
    둘이 다르다는 뜻이지 어느 쪽이 맞다는 뜻이 아니다. 그래서 앵커 위상을 성가신
    변수로 보고, 열한 자리에서 일곱을 고르는 330가지를 전부 매겨 통과 집합이
    고르는 방식에 따라 갈리는지 센다. 갈리면 일곱은 위상 운에 기대고 있는 것이고,
    안 갈리면 넷을 더 재는 값이 없다.
    """
    from itertools import combinations
    sizes = [5, 7, 9]
    subs = {k: list(combinations(range(11), k)) for k in sizes}
    subsets = subs[7]
    print("기간 %s ~ %s · 열한 자리에서 일곱을 고르는 %d가지"
          % (R.START, R.END, len(subsets)))
    size_rows = []
    rows = []
    for side, (axes, builders) in (("코인", _coin_builders()),
                                   ("주식", _stock_builders())):
        for axis, arms in axes.items():
            names = [a[0] for a in arms]
            base = rank_by_anchor(names, builders[False], A11)
            stress = rank_by_anchor(names, builders[True], A11)
            full = gate_pass(base, stress)
            counts: dict[frozenset, int] = {}
            for sub in subsets:
                s = gate_pass(base, stress, list(sub))
                counts[s] = counts.get(s, 0) + 1
            rows.append((side, axis, full, counts))
            for k in sizes:
                agree = sum(1 for sub in subs[k]
                            if gate_pass(base, stress, list(sub)) == full)
                size_rows.append((k, agree / len(subs[k])))

    print(f"\n{'축':<22}{'서로다른답':>10}{'최빈답비율':>11}"
          f"{'11곳답과같음':>13}  11곳 통과집합")
    split = 0
    for side, axis, full, counts in rows:
        top = max(counts.values())
        same = counts.get(full, 0)
        if len(counts) > 1:
            split += 1
        print(f"{side + ' ' + axis:<22}{len(counts):>10}"
              f"{top / len(subsets):>11.1%}{same / len(subsets):>13.1%}"
              f"  {sorted(full) or '없음'}")
        if len(counts) > 1:
            for s, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"{'':<22}  {c:>4}가지 ({c / len(subsets):>5.1%}) "
                      f"{sorted(s) or '없음'}")
    print("\n일곱을 어떻게 고르냐에 따라 답이 갈린 축: %d/%d" % (split, len(rows)))

    print("\n앵커를 몇 곳 보느냐에 따른 흔들림 (열한 곳 답과 일치하는 비율)")
    print(f"{'고르는 수':<10}{'축 평균':>9}{'답이 갈린 축':>14}")
    for k in sizes:
        vals = [v for kk, v in size_rows if kk == k]
        print(f"{k:<10}{float(np.mean(vals)):>9.1%}"
              f"{sum(1 for v in vals if v < 1.0):>11}/{len(vals)}")
    print("주의: 열한 곳 답 자체도 위상 열하나의 표본이다. 이 표가 보이는 것은 "
          "고르는 수가 줄수록 답이 흔들린다는 방향이지 열한 곳이 참이라는 게 아니다.")


def detail(side: str, axis: str, names: list[str]) -> None:
    """뒤집힌 축 하나를 등수가 아니라 실제 수치로 본다.

    순위합은 미세차를 등수로 벌리므로(2026-09-25 현금버퍼 자리에서 확인), 뒤집힌
    팔은 칼마·수익률·최대낙폭·샤프를 앵커별로 직접 보고 판단한다.
    """
    if side == "stock":
        import bt_stock_param_resweep as S
        coin = {}
        for a in A11:
            for stress in (False, True):
                suf = "__s5" if stress else ""
                coin[(a, stress)] = (_read(f"{S.COIN_EQ}/sp__현행__a{a}{suf}.csv"),
                                     _read(f"{S.COIN_EQ}/fu__현행__a{a}{suf}.csv"))

        def build(name, a, stress):
            eq = S._load_st(S._key(name, a, stress))
            sp, fu = coin[(a, stress)]
            if eq is None or sp is None or fu is None:
                return None
            return R.build_alloc(eq, sp, fu)
    else:
        stock_eq = _stock_fixed(A11)

        def build(name, a, stress):
            k = R._key(name, a, stress)
            sp, fu = R._load("sp", k), R._load("fu", k)
            if sp is None or fu is None:
                return None
            return R.build_alloc(stock_eq[a], sp, fu)

    print(f"\n=== {axis} — 등수 말고 실제 수치 ===")
    for stress in (False, True):
        tag = "비용 5배" if stress else "기본 비용"
        print(f"\n[{tag}] 합성 60/25/15")
        print(f"{'팔':<14}{'앵커':>5}{'칼마':>8}{'연수익%':>9}"
              f"{'최대낙폭%':>10}{'샤프':>7}")
        acc: dict[str, list[dict]] = {n: [] for n in names}
        for name in names:
            for a in A11:
                eq = build(name, a, stress)
                if eq is None:
                    continue
                m = R.metrics(eq)
                acc[name].append(m)
                print(f"{name:<14}{a:>5}{m['Cal']:>8.3f}{m['CAGR']:>9.2f}"
                      f"{m['MDD']:>10.2f}{m['Sharpe']:>7.3f}")
        for label, idx in (("앵커 0~6 평균", A7), ("앵커 0~10 평균", A11)):
            print(f"  -- {label} --")
            for name in names:
                ms = [acc[name][a] for a in idx if a < len(acc[name])]
                if not ms:
                    continue
                print(f"{name:<14}{'':>5}"
                      f"{np.mean([m['Cal'] for m in ms]):>8.3f}"
                      f"{np.mean([m['CAGR'] for m in ms]):>9.2f}"
                      f"{np.mean([m['MDD'] for m in ms]):>10.2f}"
                      f"{np.mean([m['Sharpe'] for m in ms]):>7.3f}")
        base = names[0]
        for name in names[1:]:
            w = sum(1 for a in A11
                    if a < len(acc[name]) and a < len(acc[base])
                    and acc[name][a]['Cal'] > acc[base][a]['Cal'])
            print(f"  칼마로 {name} 이 {base} 를 이긴 앵커: {w}/11")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "detail":
        # detail <coin|stock> <축> <팔...>  (팔을 안 주면 그 축 전부)
        side = sys.argv[2] if len(sys.argv) > 2 else "stock"
        axis = sys.argv[3] if len(sys.argv) > 3 else "단기 모멘텀"
        names = sys.argv[4:]
        if not names:
            if side == "stock":
                import bt_stock_param_resweep as S
                names = [a[0] for a in S.AXES[axis]]
            else:
                names = [a[0] for a in R.AXES[axis]]
            names = [R.CUR] + [n for n in names if n != R.CUR]
        detail(side, axis, names)
        return
    if which == "stability":
        stability()
        return
    print("기간 %s ~ %s · 앵커 7곳 대 11곳 · 창 %s × 보폭 %s"
          % (R.START, R.END, R.WIN_SIZES, R.STRIDES))
    flipped: list = []
    if which in ("all", "coin"):
        run_coin(flipped)
    if which in ("all", "stock"):
        run_stock(flipped)

    print("\n" + "=" * 70)
    if flipped:
        print("뒤집힌 축 %d개:" % len(flipped))
        for axis, a, b in flipped:
            print(f"  {axis}: 앵커 7 {a or '없음'} → 앵커 11 {b or '없음'}")
    else:
        print("뒤집힌 축 0개 — 앵커를 11곳으로 늘려도 통과 집합이 그대로다.")


if __name__ == "__main__":
    main()
