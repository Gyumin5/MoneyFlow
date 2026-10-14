"""선물 리필 규칙 셋을 한 판에 놓고 잰다 — 라이브 현행 · 백테스트 정본 · 넓은풀 시총순 (2026-10-15).

왜 다시 재나. ai-debate run-20261014T224537Z(1라운드 합의)가 시총순 전환을 보류로 닫으면서
순서를 바꿨다. 지금 라이브와 백테스트의 리필 규칙이 애초에 다른데(라이브 = 넓은 풀·모멘텀순·
비트코인 제외, 백테스트 = 그날 fresh 선정 최대 3종·알파벳순), 2026-10-01 측정에서 백테스트
쪽이 라이브를 꽤 크게 이겼다. 그러니 "시총순으로 바꿀까" 보다 "라이브를 백테스트 쪽으로
맞출까" 가 먼저이고, 넓은풀 시총순은 그 정본 대비 우위가 확인될 때 다시 판정하라는 것이다.

토론이 같이 못박은 제약 하나 — 비교 대상 수가 다른 순위값을 직접 비교하지 않는다. 그래서
2026-10-01 의 두 팔 숫자와 2026-10-15 의 두 팔 숫자를 이어 붙이지 않고, 세 규칙을 같은 판에
넣어 처음부터 다시 센다. 팔 셋 × 출처 둘 × 앵커 일곱 × 비용 둘 = 84런.

읽는 법 — 수준이 아니라 차이를 읽는다. 고정 출처 팔은 날짜와 무관하게 마지막 달 목록을 써서
과거 구간에 생존편향이 들어간다. 라이브가 코인게코 상위 40 을 매일 그대로 읽는 것을 흉내 낸
민감도 시험이지 성과 추정이 아니다. 출처끼리 수준을 나란히 놓지 않고, 출처 안에서 잰 팔 사이
차이만 본다.

판정 (결과 보기 전에 못박는다).
  후보는 둘이다 — A = 백테스트 정본(좁은 풀·알파벳순), B = 넓은풀 시총순(비트코인 제외).
  기준선은 라이브 현행(넓은 풀·모멘텀순·비트코인 제외)이다.
  한 후보가 "통과" 하려면 넷을 다 채워야 한다.
    ⑴ 두 비용 조건 모두에서 순위 평균이 라이브 현행보다 낮다.
    ⑵ 두 비용 조건 모두에서 이긴 앵커가 5/7 이상이다.
    ⑶ ⑴⑵ 가 두 출처 모두에서 성립한다.
    ⑷ 합성 최대낙폭이 라이브 현행보다 0.5%포인트 넘게 깊어지지 않는다 — 네 칸(출처 둘 ×
       비용 둘) 모두에서. 이 한도는 이 프로젝트가 써 온 값이고 이번에 새로 만든 게 아니다.
  결과 갈래.
    ① 한 후보만 통과 → 그 후보를 전환 제안으로 올린다. 승인 전 배포 금지는 그대로다.
    ② 둘 다 통과 → 두 출처 네 칸에서 순위가 더 낮은 쪽을 올린다. 칸마다 갈리면 운영자 판단.
    ③ 아무도 통과 못 함 → 라이브 현행을 유지하고 이 축을 닫는다.
  전기간 칼마·샤프·연복리는 같이 찍되 판정에 쓰지 않는다. 토론이 지적한 대로 결과를 보고
  새 최적화 목표로 올리지 않기 위해서다. 낙폭만 ⑷ 로 판정에 들어간다.

선행조건. 리필 토글은 정본 엔진에 없다(규칙상 되돌려 둔다). 두 패치를 먼저 얹어야 한다.
  cd /home/gmoh/mon/251229
  git apply --3way strategies/cap_defend/research/bt_fut_refill_variants.engine.patch
  git apply --3way strategies/cap_defend/research/bt_fut_universe_source.engine.patch
안 얹고 돌리면 세 팔이 같은 곡선을 내는데 그건 "차이 없음" 과 눈으로 구분이 안 된다
(2026-10-15 에 실제로 904초를 그렇게 버렸다 — history #52). 그래서 재기 전에 한 벌로
확인하고 안 갈리면 멈춘다. 끝나면 git checkout 으로 엔진을 되돌린다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_fut_refill_three_policies.py
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

import bt_coin_param_resweep as R  # noqa: E402
import bt_fut_refill_variants as V  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261015-threepol"
EQDIR = os.path.join(OUT, "eq")

LIVE = "라이브 현행(넓은풀 모멘텀순)"
BTCUR = "백테스트 정본(좁은풀 알파벳순)"
MCAP = "넓은풀 시총순"

# (팔 이름, 리필 환경변수 덮어쓰기). 빈 dict 는 ENV_DEFAULT = 백테스트 정본이다.
ARMS: list[tuple[str, dict]] = [
    (LIVE, dict(REFILL_ORDER="mom", REFILL_POOL="wide", REFILL_EXCL_BTC="1")),
    (BTCUR, dict()),
    (MCAP, dict(REFILL_ORDER="mcap", REFILL_POOL="wide", REFILL_EXCL_BTC="1")),
]
CANDIDATES = [BTCUR, MCAP]

# (출처 이름, UNIVERSE_FIXED 값)
SOURCES: list[tuple[str, str]] = [
    ("시점별(백테스트 현행)", "0"),
    ("고정(라이브 코인게코 대역 근사)", "1"),
]

MDD_LIMIT = 0.5  # %포인트. 이보다 깊어지면 ⑷ 탈락.


def _key(arm: str, src: str, anchor: int, stress: bool) -> str:
    safe = f"{arm}__{src}".replace("/", "_").replace(" ", "_")
    safe = safe.replace("(", "").replace(")", "")
    return f"{safe}__a{anchor}{'__s5' if stress else ''}"


def _save(key: str, eq: pd.Series) -> None:
    eq.to_csv(os.path.join(EQDIR, f"fu__{key}.csv"), header=["eq"])


def _load(key: str):
    p = os.path.join(EQDIR, f"fu__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def run_arm(env: dict, fixed: str, anchor: int, stress: bool) -> pd.Series:
    os.environ["UNIVERSE_FIXED"] = fixed
    try:
        eq = V.run_fut_arm(env, anchor, stress)
    finally:
        os.environ["UNIVERSE_FIXED"] = "0"
    return eq


def report(stress: bool, stock_eq, spot_eq) -> dict:
    """출처마다 따로 세 팔 순위합을 낸다. 출처끼리는 비교하지 않는다."""
    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== 세 규칙 한 판 · {tag} · 앵커 {len(R.ANCHORS)}곳 · 합성 60/25/15 ===")
    print("순위 평균은 낮을수록 우세(3팔 비교). 차·낙폭차는 라이브 현행 대비.")
    out: dict = {}
    for sname, _fixed in SOURCES:
        print(f"\n[{sname}]")
        print(f"{'규칙':<26}{'순위평균':>9}{'차':>9}{'이긴앵커':>9}"
              f"{'Cal':>7}{'Sharpe':>8}{'CAGR':>8}{'MDD':>9}{'낙폭차':>9}")
        ranks: dict[str, list[float]] = {a: [] for a, _ in ARMS}
        mets: dict[str, list[dict]] = {a: [] for a, _ in ARMS}
        for a in R.ANCHORS:
            comp = {}
            for an, _env in ARMS:
                e = _load(_key(an, sname, a, stress))
                if e is None:
                    comp = {}
                    break
                comp[an] = R.build_alloc(stock_eq[a], spot_eq[(a, stress)], e)
            if not comp:
                continue
            sums, _w, n = R.window_rs(comp)
            for an in comp:
                ranks[an].append(sums[an] / max(n, 1))
                mets[an].append(R.metrics(comp[an]))
        if not ranks[LIVE]:
            print("  (곡선 없음)")
            continue
        base_r = float(np.mean(ranks[LIVE]))
        base_mdd = float(np.mean([x["MDD"] for x in mets[LIVE]]))
        cell: dict[str, tuple] = {}
        for an, _env in ARMS:
            r = float(np.mean(ranks[an]))
            mdd = float(np.mean([x["MDD"] for x in mets[an]]))
            wins = sum(1 for x, y in zip(ranks[an], ranks[LIVE]) if x < y)
            # MDD 는 음수(%)다. 더 깊다 = 더 작다. 악화폭은 양수로 적는다.
            worse = base_mdd - mdd
            print(f"{an:<26}{r:>9.3f}{r - base_r:>+9.3f}{wins:>6}/{len(ranks[an])}"
                  f"{np.mean([x['Cal'] for x in mets[an]]):>7.2f}"
                  f"{np.mean([x['Sharpe'] for x in mets[an]]):>8.2f}"
                  f"{np.mean([x['CAGR'] for x in mets[an]]):>7.1f}%"
                  f"{mdd:>8.2f}%{worse:>+9.2f}p")
            cell[an] = (r, r - base_r, wins, len(ranks[an]), mdd, worse)
        out[sname] = cell
    return out


def verdict(res: dict) -> None:
    print("\n=== 판정 (사전등록 기준) ===")
    passed = []
    for cand in CANDIDATES:
        print(f"\n[{cand}]")
        ok = True
        for stress in (False, True):
            tag = "비용 5배" if stress else "기본 비용"
            for sname, _f in SOURCES:
                c = res.get(stress, {}).get(sname, {}).get(cand)
                if c is None:
                    print(f"  {tag} · {sname}: 곡선이 모자라 판정 불가")
                    ok = False
                    continue
                _r, d, w, n, _mdd, worse = c
                g1 = d < 0
                g2 = w >= 5
                g4 = worse <= MDD_LIMIT
                if not (g1 and g2 and g4):
                    ok = False
                print(f"  {tag} · {sname}: 차 {d:+.3f}{'' if g1 else ' (탈락 ⑴)'} · "
                      f"이긴앵커 {w}/{n}{'' if g2 else ' (탈락 ⑵)'} · "
                      f"낙폭 {worse:+.2f}p{'' if g4 else ' (탈락 ⑷)'}")
        print(f"  → {'통과' if ok else '탈락'}")
        if ok:
            passed.append(cand)

    print()
    if not passed:
        print("③ 아무도 통과 못 함 — 라이브 현행을 유지하고 이 축을 닫는다.")
    elif len(passed) == 1:
        print(f"① {passed[0]} 만 통과 — 전환 제안으로 올린다. 승인 전 배포 금지.")
    else:
        split = False
        for stress in (False, True):
            for sname, _f in SOURCES:
                a = res[stress][sname][passed[0]][0]
                b = res[stress][sname][passed[1]][0]
                if (a < b) != (res[False][SOURCES[0][0]][passed[0]][0]
                               < res[False][SOURCES[0][0]][passed[1]][0]):
                    split = True
        if split:
            print("② 둘 다 통과하는데 칸마다 순위가 갈린다 — 운영자 판단으로 올린다.")
        else:
            best = min(passed, key=lambda c: res[False][SOURCES[0][0]][c][0])
            print(f"② 둘 다 통과 — 네 칸에서 일관되게 앞선 {best} 를 올린다. 승인 전 배포 금지.")


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()
    V.set_env({})
    os.environ["UNIVERSE_FIXED"] = "0"

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    print("주식·현물 슬리브 준비 중...", flush=True)
    stock_eq, spot_eq = {}, {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if os.path.exists(p):
            stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = R.run_stock(a, False)
            stock_eq[a].to_csv(p, header=["eq"])
        for s in (False, True):
            spot_eq[(a, s)] = V.ensure_spot(a, s)
    print(f"  완료 ({time.time()-t0:.0f}초)", flush=True)

    # ─ 사전 점검: 세 팔이 실제로 다른 곡선을 내나 ─
    # 빈 측정과 "차이 없음" 은 눈으로 구분이 안 된다(history #52). 두 패치가 다 얹혀야
    # 세 팔이 갈린다 — variants 패치가 리필 정렬·풀을, universe 패치가 목록 출처를 연다.
    print("사전 점검: 세 팔이 다른 곡선을 내는지 한 벌로 확인...", flush=True)
    probe = {}
    for an, env in ARMS:
        probe[an] = run_arm(dict(env), SOURCES[0][1], R.ANCHORS[0], False)
    stuck = []
    for i in range(len(ARMS)):
        for j in range(i + 1, len(ARMS)):
            x, y = probe[ARMS[i][0]], probe[ARMS[j][0]]
            k = x.index.intersection(y.index)
            d = float(np.max(np.abs(np.asarray(x[k]) / np.asarray(y[k]) - 1.0)))
            print(f"  {ARMS[i][0]} 대 {ARMS[j][0]}: 최대 상대괴리 {d:.3e}", flush=True)
            if d < 1e-9:
                stuck.append((ARMS[i][0], ARMS[j][0]))
    if stuck:
        print("  같은 곡선인 쌍이 있다 — 리필 토글이 엔진에 없다는 뜻이다.\n"
              "  cd /home/gmoh/mon/251229 && git apply --3way "
              "strategies/cap_defend/research/bt_fut_refill_variants.engine.patch\n"
              "  를 먼저 얹어라. 재지 않고 멈춘다.")
        return

    # 출처 토글이 실제로 먹는지도 같이 본다. universe 패치가 없으면 고정 출처가 시점별과
    # 같은 곡선을 낸다 — 이것도 "민감도 없음" 과 구분이 안 된다.
    pit = probe[MCAP]
    fix = run_arm(dict(ARMS[2][1]), SOURCES[1][1], R.ANCHORS[0], False)
    k = pit.index.intersection(fix.index)
    d = float(np.max(np.abs(np.asarray(pit[k]) / np.asarray(fix[k]) - 1.0)))
    print(f"  시총순의 시점별 대 고정: 최대 상대괴리 {d:.3e}", flush=True)
    if d < 1e-9:
        print("  출처 토글이 안 먹는다 — bt_fut_universe_source.engine.patch 를 얹어라."
              " 재지 않고 멈춘다.")
        return

    todo = [(an, env, sn, fx, a, s)
            for an, env in ARMS for sn, fx in SOURCES
            for a in R.ANCHORS for s in (False, True)]
    print(f"\n선물 계산 조합 {len(todo)}건 (이미 있는 것은 건너뜀)", flush=True)
    done = 0
    for an, env, sname, fixed, a, s in todo:
        k2 = _key(an, sname, a, s)
        if _load(k2) is None:
            _save(k2, run_arm(env, fixed, a, s))
        done += 1
        if done % 5 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  전부 완료 ({time.time()-t0:.0f}초)", flush=True)

    res = {}
    for stress in (False, True):
        res[stress] = report(stress, stock_eq, spot_eq)
    verdict(res)
    print(f"\n총 {time.time()-t0:.0f}초. 곡선 {EQDIR}")


if __name__ == "__main__":
    main()
