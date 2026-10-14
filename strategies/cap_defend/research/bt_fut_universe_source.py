"""선물 리필 시총순의 우위가 유니버스 '출처' 를 바꿔도 남나 (2026-10-15).

태스크 3 의 남은 선행조건 하나다. 2026-10-01 에 "넓은풀 시총순 BTC제외" 가 실제 라이브
규칙(넓은풀 모멘텀순 BTC제외)을 두 비용 조건 모두에서 이겼고 못박은 기준 넷을 다 통과했다.
순위 교란(30일 지연·인접쌍 교환)에도 견뎠다. 그런데 교란은 같은 목록 안에서 순서를 흔든
것이고, 안 잰 것은 목록 자체가 어디서 오느냐다.

왜 중요한가. 시총순은 정렬 축이 유니버스에서 바로 온다 — 모멘텀순은 가격에서 오므로
유니버스가 바뀌어도 순서 자체는 안 바뀐다. 라이브는 코인게코 상위 40 을 매일 읽고(= 오늘의
한 벌), 백테스트는 data/historical_universe.json 의 시점별 목록을 쓴다. 2026-06-06 패리티에서
이 차이만으로 선정이 5.69%(114/2005일) 갈렸다.

재는 법. 팔 둘(라이브 현행 = 넓은풀 모멘텀순 BTC제외, 후보 = 넓은풀 시총순 BTC제외)을
출처 둘(시점별 · 고정) 아래에서 각각 돌린다. 네 벌이다.

읽는 법 — 수준이 아니라 차이를 읽는다. 고정 출처는 과거 구간에 생존편향이 들어가므로
절대 성과를 시점별과 나란히 놓으면 안 된다. 보는 것은 출처마다 따로 잰
'시총순 − 모멘텀순' 하나이고, 그 차이가 출처를 바꿔도 같은 부호로 남는지만 본다.

판정(결과 보기 전에 못박는다).
  ① 고정 출처에서도 시총순이 모멘텀순을 두 비용 조건 모두에서 이긴다 → 선행조건 통과.
  ② 한 조건이라도 부호가 뒤집히면 → 시총순의 우위는 시점별 목록 모양에 얹힌 것으로 보고
     태스크 3 을 기각한다. 라이브는 그 목록을 안 쓰기 때문이다.
  ③ 부호는 같은데 이긴 앵커가 5/7 아래로 떨어지면 → 통과도 기각도 아니고 운영자 판단으로
     올린다. 출처 민감도가 있다는 사실만 보고한다.

엔진은 환경변수로만 바뀐다 — UNIVERSE_FIXED=1 이면 get_mcap 이 날짜와 무관하게 마지막 달
목록을 돌려준다. 측정이 끝나면 정본에서 되돌리고 patch 만 남긴다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_fut_universe_source.py
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

OUT = "/home/gmoh/mon/251229/state/20261015-univsrc"
EQDIR = os.path.join(OUT, "eq")

MOM = "모멘텀순(라이브 현행)"
MCAP = "시총순(후보)"

# (팔 이름, 리필 환경변수)
ARMS: list[tuple[str, dict]] = [
    (MOM, dict(REFILL_ORDER="mom", REFILL_POOL="wide", REFILL_EXCL_BTC="1")),
    (MCAP, dict(REFILL_ORDER="mcap", REFILL_POOL="wide", REFILL_EXCL_BTC="1")),
]

# (출처 이름, UNIVERSE_FIXED 값)
SOURCES: list[tuple[str, str]] = [
    ("시점별(백테스트 현행)", "0"),
    ("고정(라이브 코인게코 대역 근사)", "1"),
]


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
    """출처마다 따로 쌍대 순위합을 낸다. 출처끼리는 비교하지 않는다."""
    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== 출처별 쌍대 비교 · {tag} · 앵커 {len(R.ANCHORS)}곳 · 합성 60/25/15 ===")
    print("낮은 쪽이 우세. 차 = 시총순 − 모멘텀순 (음수면 시총순 우세).")
    print(f"{'출처':<26}{'시총순':>9}{'모멘텀순':>10}{'차':>9}{'이긴앵커':>9}"
          f"{'시총Cal':>9}{'모멘Cal':>9}{'시총MDD':>10}{'모멘MDD':>10}")
    out = {}
    for sname, fixed in SOURCES:
        ra, rb, ma, mb = [], [], [], []
        for a in R.ANCHORS:
            ea = _load(_key(MCAP, sname, a, stress))
            eb = _load(_key(MOM, sname, a, stress))
            if ea is None or eb is None:
                continue
            ca = R.build_alloc(stock_eq[a], spot_eq[(a, stress)], ea)
            cb = R.build_alloc(stock_eq[a], spot_eq[(a, stress)], eb)
            sums, _w, n = R.window_rs({MCAP: ca, MOM: cb})
            ra.append(sums[MCAP] / max(n, 1))
            rb.append(sums[MOM] / max(n, 1))
            ma.append(R.metrics(ca))
            mb.append(R.metrics(cb))
        if not ra:
            print(f"{sname:<26}  (곡선 없음)")
            continue
        wins = sum(1 for x, y in zip(ra, rb) if x < y)
        diff = float(np.mean(ra)) - float(np.mean(rb))
        out[sname] = (diff, wins, len(ra))
        print(f"{sname:<26}{float(np.mean(ra)):>9.3f}{float(np.mean(rb)):>10.3f}"
              f"{diff:>+9.3f}{wins:>6}/{len(ra)}"
              f"{np.mean([x['Cal'] for x in ma]):>9.2f}"
              f"{np.mean([x['Cal'] for x in mb]):>9.2f}"
              f"{np.mean([x['MDD'] for x in ma]):>9.2f}%"
              f"{np.mean([x['MDD'] for x in mb]):>9.2f}%")
    return out


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

    print("주식·현물 슬리브(고정 팔) 준비 중...", flush=True)
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

    # ─ 사전 점검: 두 팔이 실제로 다른 곡선을 내나 ─
    # 2026-10-15 첫 실행이 56벌을 다 돌고 '차 +0.000' 을 냈는데, 결과가 아니라 결함이었다.
    # 리필 정렬 토글(REFILL_ORDER 등)은 정본 엔진에 없고 research/bt_fut_refill_variants
    # .engine.patch 를 얹어야 생긴다. 패치 없이 돌리면 환경변수가 아무것도 안 해서 두 팔이
    # 같은 곡선을 내고, 그게 "차이 없음" 으로 보인다. 빈 결과와 같은 결과는 구분이 안 되므로
    # 재기 전에 한 벌로 확인하고 다르지 않으면 멈춘다.
    print("사전 점검: 두 팔이 다른 곡선을 내는지 한 벌로 확인...", flush=True)
    _a = run_arm(dict(ARMS[0][1]), SOURCES[0][1], R.ANCHORS[0], False)
    _b = run_arm(dict(ARMS[1][1]), SOURCES[0][1], R.ANCHORS[0], False)
    _j = _a.index.intersection(_b.index)
    _d = float(np.max(np.abs(np.asarray(_a[_j]) / np.asarray(_b[_j]) - 1.0)))
    print(f"  두 팔 최대 상대괴리 {_d:.3e}", flush=True)
    if _d < 1e-9:
        print("  두 팔이 같은 곡선이다 — 리필 정렬 토글이 엔진에 없다는 뜻이다.\n"
              "  cd /home/gmoh/mon/251229 && git apply --3way "
              "strategies/cap_defend/research/bt_fut_refill_variants.engine.patch\n"
              "  를 먼저 얹어라. 재지 않고 멈춘다.")
        return

    todo = [(an, env, sn, fx, a, s)
            for an, env in ARMS for sn, fx in SOURCES
            for a in R.ANCHORS for s in (False, True)]
    shard = os.environ.get("SHARD", "")
    if shard:
        si, sn_ = (int(x) for x in shard.split("/"))
        todo = [t for i, t in enumerate(todo) if i % sn_ == si]
        print(f"  이 몫 {si}/{sn_} = {len(todo)}건", flush=True)
    print(f"\n선물 계산 조합 {len(todo)}건 (이미 있는 것은 건너뜀)", flush=True)
    done = 0
    for an, env, sname, fixed, a, s in todo:
        k = _key(an, sname, a, s)
        if _load(k) is None:
            _save(k, run_arm(env, fixed, a, s))
        done += 1
        if done % 5 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  전부 완료 ({time.time()-t0:.0f}초)", flush=True)

    if shard:
        print("쪼갠 몫이라 보고는 건너뛴다 — 전부 모인 뒤에 보고만 따로 돌려라.")
        return

    res = {}
    for stress in (False, True):
        res[stress] = report(stress, stock_eq, spot_eq)

    print("\n=== 판정 ===")
    pit, fix = SOURCES[0][0], SOURCES[1][0]
    ok = True
    flipped = False
    for stress in (False, True):
        tag = "비용 5배" if stress else "기본 비용"
        r = res.get(stress, {})
        if pit not in r or fix not in r:
            print(f"  {tag}: 곡선이 모자라 판정 불가")
            ok = False
            continue
        dp, wp, np_ = r[pit]
        df, wf, nf = r[fix]
        same = (dp < 0) == (df < 0)
        enough = wf >= 5
        print(f"  {tag}: 시점별 차 {dp:+.3f}({wp}/{np_}) · 고정 차 {df:+.3f}({wf}/{nf}) · "
              f"부호 {'같다' if same else '뒤집힌다'} · "
              f"고정 이긴앵커 {'5/7 이상' if enough else '5/7 미만'}")
        if not same or df >= 0:
            flipped = True
        if not enough:
            ok = False
    if flipped:
        print("  ② 기각 — 고정 출처에서 부호가 뒤집힌다. 라이브는 시점별 목록을 안 쓴다.")
    elif ok:
        print("  ① 통과 — 출처를 바꿔도 시총순 우위가 두 비용 조건 모두에서 남는다.")
    else:
        print("  ③ 운영자 판단 — 부호는 남지만 고정 출처에서 이긴 앵커가 5/7 아래로 떨어진다.")
    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
