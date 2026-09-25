"""배수는 천천히 올리고 한 번에 내린다 — 비대칭 래칫 (2026-09-25).

운영자 물음: "배수를 올릴 땐 딜레이를 두고 내릴 때는 한 번에 내리는 방식은?
연속 몇 일일 때만 한 배씩 올릴 수 있도록 하고 내릴 때는 한방인 거야."

현행은 양방향 즉시다. 목표 배수(= min(BTC캡, 7일선 이격 K2))가 어제 기준으로 바뀌면
그날 그대로 따라간다. 여기서 재는 것은 올리는 쪽만 늦추는 규칙이다.

  적용 배수 상태를 코인별로 들고 간다.
  목표 < 상태 → 확인 없이 그날 목표까지 내린다(한방).
  목표 > 상태 → 그 조건이 연속 N일 유지될 때만 올린다. 한 칸 제한이 있으면 +1 칸씩.
  올린 직후 카운터는 0 이다 — 다음 칸은 또 N일을 채워야 한다.

이미 재고 기각한 것과 다른 축이다(그래서 돈다):
  · 전일 급등 억제(2026-09-24) — 하루짜리 사건을 조건으로 쓴다. 이건 사건이 아니라
    목표가 오르려는 모든 자리에 걸린다.
  · 연속 양봉 하향(2026-09-24) — 내리는 쪽 조건이고, 여기는 올리는 쪽 속도다.
  · 매일 L 재조정(2026-08-21) — 이미 잡은 포지션의 명목을 언제 맞추느냐는 적용 시점 축이다.
    이건 목표 배수 자체를 어떻게 정하느냐다.
  · 다중선 배수(2026-09-24) — 배수를 정하는 신호를 갈아끼운다. 여기는 신호를 그대로 두고
    따라가는 속도만 비대칭으로 만든다.

축 하나만 바꾼다. 선정·헬스·카나리·스냅·드리프트 0.015·체결밴드 5%·비용 전부 고정.
정본 엔진(backtest_futures_v25.py)은 건드리지 않는다 — leverage 인자로만 주입한다.

판정 기준 (결과 보기 전 고정):
  1. 결정층은 합성 60/25/15 윈도우 순위합이고, 앵커 7곳에서 각각 내어 평균·중앙·최악을
     같이 적는다. 슬리브만 좋고 합성이 반대면 채택하지 않는다.
  2. N 스윕에 플래토가 있어야 한다. 끝값이 최고면 꺾이는 자리가 구간 밖이라는 뜻이라
     채택 근거로 쓰지 않고 구간을 넓힌다.
  3. 비용 5배에서 순서가 뒤집히면 현행 유지로 판정한다.
  4. 규칙이 실제로 배수를 바꾼 비율을 먼저 본다. 분모는 전체 코인-일이 아니라 이 규칙이
     작동할 수 있는 날 — 현행 목표가 직전보다 올라가려는 코인-일이다. 0 이면 그 팔은
     현행과 같고, 동률을 개선으로 읽지 않는다.
  5. 대조군 둘을 같이 돌린다. 비대칭이 값인지 딴 것이 값인지 가른다.
     · 대칭 지연 N: 양방향 모두 연속 N일 확인. 둔해지는 건 같고 비대칭만 없다.
     · 평균맞춤 축소: 현행 배수에 고정 배율을 곱해 평균 배수를 래칫 팔과 맞춘다.
       래칫은 올리는 쪽만 늦추므로 평균 배수가 내려간다 — 그 노출 감소만으로 설명되는지 본다.
     래칫이 현행만 이기고 이 둘을 못 이기면 "비대칭이 통한다"가 아니다.
  6. 위 다섯을 통과해도 채택 전에 평가창을 사건 유무로 쪼갠다(bt_invvol_windows.py 방식).
     이득이 최대낙폭 한 구간을 품은 창에만 있으면 기각한다.

중간 결과는 state/20260925-ratchet/eq/*.csv 에 남고 재시작하면 이어 간다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_lev_ratchet.py
     ARMS=quick 로 현행 + 대표 두 팔만 돌린다(smoke).
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

import bt_fut_spike_hold as H  # noqa: E402  — 하니스 재사용(앵커·합성·순위합·슬리브 실행)

OUT = "/home/gmoh/mon/251229/state/20260925-ratchet"
EQDIR = os.path.join(OUT, "eq")

CUR = "현행"
L_MIN, L_MAX = 2.0, 4.0
START, END = H.START, H.END
ANCHORS = H.ANCHORS


# ────────────────────────── 레버리지 시그널 ──────────────────────────

def build_ratchet(base_k2: dict, n_days: int, step_limit: bool,
                  symmetric: bool = False) -> dict:
    """올리는 쪽만 연속 N일 확인. 내리는 쪽은 즉시(symmetric=True 면 양쪽 다 확인).

    base_k2 는 이미 shift(1) 된 목표 배수다(오늘 배수 = 어제까지의 신호).
    상태는 코인별로 시간축을 따라 이어지고 보유 여부와 무관하다 — 라이브도 코인별
    상태 한 칸으로 같게 만들 수 있다.
    """
    out = {}
    for coin, lev in base_k2.items():
        v = lev.values.astype(float)
        adj = np.empty_like(v)
        cur = float(v[0]) if len(v) and not np.isnan(v[0]) else L_MIN
        up = dn = 0
        for i in range(len(v)):
            t = v[i]
            if np.isnan(t):
                adj[i] = cur
                continue
            if t > cur:
                dn = 0
                up += 1
                if up >= n_days:
                    cur = min(cur + 1.0, t) if step_limit else t
                    up = 0
            elif t < cur:
                up = 0
                if symmetric:
                    dn += 1
                    if dn >= n_days:
                        cur = max(cur - 1.0, t) if step_limit else t
                        dn = 0
                else:
                    cur = t
                    dn = 0
            else:
                up = dn = 0
            adj[i] = cur
        out[coin] = pd.Series(adj, index=lev.index)
    return out


def build_scaled(base_k2: dict, scale: float) -> dict:
    """현행 배수에 고정 배율. 평균 배수만 낮추고 반응 속도는 현행 그대로."""
    return {c: s * scale for c, s in base_k2.items()}


def _flat(d: dict, coins=None) -> np.ndarray:
    vs = []
    for coin, s in d.items():
        if coins is not None and coin not in coins:
            continue
        vs.append(s.loc[START:END].values.astype(float))
    if not vs:
        return np.array([])
    v = np.concatenate(vs)
    return v[~np.isnan(v)]


def mean_lev(d: dict) -> float:
    v = _flat(d)
    return float(v.mean()) if len(v) else float('nan')


def rise_days(base_k2: dict) -> tuple[int, int]:
    """규칙이 작동할 수 있는 날 — 현행 목표가 직전 목표보다 올라가는 코인-일."""
    tot = rise = 0
    for s in base_k2.values():
        v = s.loc[START:END].values.astype(float)
        ok = ~np.isnan(v)
        v = v[ok]
        if len(v) < 2:
            continue
        tot += len(v) - 1
        rise += int((np.diff(v) > 0).sum())
    return rise, tot


def lev_profile(base: dict, arm: dict, rise_n: int) -> dict:
    """바뀐 비율(전체 코인-일 기준과 작동 가능한 날 기준 둘 다), 평균 배수, 칸 분포.

    래칫은 한 번 뒤처지면 여러 날 뒤처진 채로 있으므로 바뀐 날이 작동 가능한 날보다
    많을 수 있다(비율 100% 초과). 그래서 두 분모를 같이 적는다.
    """
    b, a = [], []
    for coin in arm:
        if coin not in base:
            continue
        idx = base[coin].index.intersection(arm[coin].index)
        b.append(base[coin].loc[idx].loc[START:END].values.astype(float))
        a.append(arm[coin].loc[idx].loc[START:END].values.astype(float))
    if not b:
        return {}
    bv, av = np.concatenate(b), np.concatenate(a)
    ok = ~(np.isnan(bv) | np.isnan(av))
    bv, av = bv[ok], av[ok]
    diff = int((np.abs(av - bv) > 1e-9).sum())
    out: dict = {f"L{int(L)}": float((np.abs(av - L) < 1e-9).mean()) for L in (2, 3, 4)}
    out['mean'] = float(av.mean())
    out['drop'] = float(np.mean(bv - av))
    out['ch_all'] = diff / len(bv) if len(bv) else 0.0
    out['ch_rise'] = diff / rise_n if rise_n else 0.0
    return out


# ────────────────────────── 팔 정의 ──────────────────────────

def arm_specs(k2: dict) -> list[tuple[str, dict]]:
    """(이름, 배수 시그널). 평균맞춤 배율은 래칫 팔의 평균에서 역산한다."""
    mode = os.environ.get('ARMS', 'full')
    base_mean = mean_lev(k2)

    ratchet: list[tuple[str, dict]] = []
    for n in (1, 2, 3, 5, 10):
        ratchet.append((f"한칸 N{n}", build_ratchet(k2, n, True)))
    for n in (2, 3, 5):
        ratchet.append((f"목표 N{n}", build_ratchet(k2, n, False)))

    if mode == 'quick':
        return [(CUR, k2), ratchet[1], ratchet[2]]

    specs: list[tuple[str, dict]] = [(CUR, k2)] + ratchet
    for n in (2, 3):
        specs.append((f"대칭 N{n}", build_ratchet(k2, n, True, symmetric=True)))
    for ref in ("한칸 N2", "한칸 N3"):
        d = dict(ratchet)[ref]
        sc = mean_lev(d) / base_mean if base_mean else 1.0
        specs.append((f"평균맞춤 {ref}", build_scaled(k2, sc)))
    if mode == 'round1':
        return specs

    # 2차 자리 — 1차에서 끝값(N1 또는 N10)이 최고면 여기를 넓힌다.
    for n in (15, 20):
        specs.append((f"한칸 N{n}", build_ratchet(k2, n, True)))
    return specs


# ────────────────────────── 저장·실행 ──────────────────────────

def _key(name: str, anchor: int, stress: bool) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace(".", "p")
    return f"{safe}__a{anchor}{'__s5' if stress else ''}"


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(os.path.join(EQDIR, f"{kind}__{key}.csv"), header=["eq"])


def _load(kind: str, key: str) -> pd.Series | None:
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def report(names: list[str], stress: bool, stock_eq: dict) -> None:
    per_rank: dict[str, list[float]] = {n: [] for n in names}
    per_cal: dict[str, list[float]] = {n: [] for n in names}
    per_mdd: dict[str, list[float]] = {n: [] for n in names}
    for a in ANCHORS:
        allocs = {}
        eq_sp = _load("sp", _key(CUR, a, stress))
        if eq_sp is None:
            continue
        for name in names:
            eq_fu = _load("fu", _key(name, a, stress))
            if eq_fu is None:
                continue
            alloc = H.build_alloc(stock_eq[a], eq_sp, eq_fu)
            allocs[name] = alloc
            mm = H.metrics(alloc)
            per_cal[name].append(mm["Cal"])
            per_mdd[name].append(mm["MDD"])
        if len(allocs) == len(names):
            sums, _wins, n = H.window_rs(allocs)
            for name in names:
                per_rank[name].append(sums[name] / max(n, 1))

    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== 합성 60/25/15 · {tag} · 앵커 {len(ANCHORS)}곳 ===")
    print(f"  {'팔':<16s}{'평균순위':>9s}{'중앙':>8s}{'최악':>8s}{'현행대비':>10s}"
          f"{'칼마평균':>10s}{'낙폭평균':>10s}{'승앵커':>8s}")
    base = per_rank.get(CUR, [])
    for name in sorted(names, key=lambda x: float(np.mean(per_rank[x])) if per_rank[x] else 9):
        r = per_rank[name]
        if not r:
            print(f"  {name:<16s}  (자료 없음)")
            continue
        wins = sum(1 for i in range(len(r)) if base and r[i] < base[i])
        gap = float(np.mean(r)) - (float(np.mean(base)) if base else 0.0)
        print(f"  {name:<16s}{float(np.mean(r)):>9.3f}{float(np.median(r)):>8.3f}"
              f"{float(np.max(r)):>8.3f}{gap:>+10.3f}"
              f"{float(np.mean(per_cal[name])):>10.2f}"
              f"{float(np.mean(per_mdd[name])):>10.2f}{wins:>6d}/{len(r)}")


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    H._bars, H._funding = load_data("D")
    k2 = build_K2_signal(H._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    H._k2 = k2
    print(f"  심볼 {len(H._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    specs = arm_specs(k2)
    levs = dict(specs)
    names = [n for n, _ in specs]
    print(f"\n기간 {START} ~ {END} | 축: 배수 상향만 연속 N일 확인 하나")
    print(f"팔 {len(specs)}개 · 앵커 {len(ANCHORS)}곳 · 비용 2조건 "
          f"= 선물 {len(specs)*len(ANCHORS)*2}런", flush=True)

    rise, tot = rise_days(k2)
    print(f"\n[규칙이 작동할 수 있는 날] 목표 배수가 올라가려는 코인-일 "
          f"{rise:,}일 / 전체 {tot:,}일 ({rise/tot:.2%})", flush=True)
    print(f"  현행 평균 배수 {mean_lev(k2):.3f}")
    print("\n[규칙이 실제로 배수를 바꾼 비율]")
    print(f"  {'팔':<16s}{'전체대비':>9s}{'작동가능대비':>13s}{'평균감소':>9s}"
          f"{'평균배수':>9s}{'2배':>8s}{'3배':>8s}{'4배':>8s}")
    for name in names:
        p = lev_profile(k2, levs[name], rise)
        print(f"  {name:<16s}{p.get('ch_all', 0):>9.2%}{p.get('ch_rise', 0):>13.2%}"
              f"{p.get('drop', 0):>9.3f}{p.get('mean', 0):>9.3f}"
              f"{p.get('L2', 0):>8.1%}{p.get('L3', 0):>8.1%}{p.get('L4', 0):>8.1%}",
              flush=True)

    print("\n주식 슬리브(합성용 고정 팔) 앵커별 계산 중...", flush=True)
    stock_eq = {}
    for a in ANCHORS:
        k = _key("주식", a, False)
        eq = _load("st", k)
        if eq is None:
            eq = H.run_stock(a)
            _save("st", k, eq)
        stock_eq[a] = eq
    print(f"  완료 ({time.time()-t0:.0f}초)", flush=True)

    total = len(specs) * len(ANCHORS) * 2
    done = 0
    for stress in (False, True):
        for a in ANCHORS:
            kc = _key(CUR, a, stress)
            if _load("sp", kc) is None:
                _save("sp", kc, H.run_spot(H.PH_SPOT[a], stress))
            for name in names:
                k = _key(name, a, stress)
                done += 1
                if _load("fu", k) is not None:
                    continue
                t1 = time.time()
                _save("fu", k, H.run_fut(levs[name], H.PH_FUT[a], stress))
                print(f"  [{done}/{total}] {name} a{a}"
                      f"{' s5' if stress else ''} {time.time()-t1:.0f}초", flush=True)

    print("\n[슬리브 단독 — 앵커 0, 기본 비용]")
    print(f"  {'팔':<16s}{'칼마':>7s}{'연수익':>9s}{'낙폭':>9s}{'최종':>9s}")
    for name in names:
        eq = _load("fu", _key(name, 0, False))
        if eq is None:
            continue
        mm = H.metrics(eq)
        print(f"  {name:<16s}{mm['Cal']:>7.2f}{mm['CAGR']:>+8.1f}%"
              f"{mm['MDD']:>+8.1f}%{eq.iloc[-1]/eq.iloc[0]:>8.1f}x")

    for stress in (False, True):
        report(names, stress, stock_eq)

    print(f"\n소요 {time.time()-t0:.0f}초")


if __name__ == '__main__':
    main()
