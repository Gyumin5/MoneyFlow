"""C7 — 2024-01-11 미국 현물 비트코인 상장지수펀드 승인 전후의 구조 변화 측정 (2026-09-22).

계기: 2026-09-22 논문 조사에서 추린 백테스트 후보 세 건 중 첫째다. 여러 논문이
2024년 1월 미국 현물 비트코인 상장지수펀드 승인을 코인 시장의 구조 분기점으로 다룬다.
우리 세 슬리브가 실제로 그 영향을 받았는지를 먼저 재고, 받았을 때만 다음 후보로 간다.

전략 코드는 한 줄도 바꾸지 않는다. 이건 진단이지 변형 실험이 아니다.

무엇을 재나 — 둘이다.

A. 원자료 상관. 전략을 거치지 않은 일간 수익률로 90일 롤링 상관을 낸다.
   비트코인 대 미국 주가지수, 알트 바스켓 대 미국 주가지수, 비트코인 대 알트 바스켓,
   그리고 알트 코인들끼리의 평균 쌍상관. 마지막 값이 우리에게 가장 직접적이다 —
   현물·선물 슬리브가 코인 3종 분산에 방어를 걸고 있으니, 알트끼리 같이 움직이면
   그 분산이 얇아진다.

B. 합성 포트폴리오 꼬리 동조화. 세 슬리브 자산곡선을 실제 채택 설정으로 뽑아
   60/25/15 로 합성하고, 합성이 최악인 날들에 세 슬리브가 동시에 마이너스인 비율과
   슬리브 쌍상관을 분기점 전후로 비교한다.

가짜 발견을 거르는 장치: 분기점을 2024-01-11 이 아닌 다른 날짜로 바꿔 같은 계산을
반복한다(위약 분할). 2024-01-11 의 변화폭이 임의 날짜들의 변화폭 분포 안에 들어가면
그건 상장지수펀드 때문이 아니라 표본을 자르면 늘 생기는 흔들림이다.

종료 조건 (결과 보기 전 고정):
  1. 위약 분할 대비 2024-01-11 이 특별하지 않으면 여기서 끝낸다. 후속 없음.
  2. 구조 변화가 있어도 우리 방어 규칙에 닿는 함의가 없으면 기록만 남기고 끝낸다.
  3. 어느 쪽이든 전략 변경 제안은 이 스크립트의 결과만으로 하지 않는다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_c7_etf_regime.py
출력은 state/20260922-c7-etf-regime/ 로 리다이렉트한다. 원시 로그는 리포트가 아니다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAP))
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

DATA = os.path.join(ROOT, "data")
SPLIT = pd.Timestamp("2024-01-11")          # 미국 현물 비트코인 상장지수펀드 거래 개시일
PLACEBO = [pd.Timestamp(d) for d in
           ["2021-07-11", "2022-01-11", "2022-07-11", "2023-01-11", "2023-07-11",
            "2024-07-11", "2025-01-11", "2025-07-11"]]
ROLL = 90                                    # 롤링 상관 창(일)
TAIL_Q = 0.05                                # 합성 최악 5% 를 꼬리로 본다

# 알트 바스켓 = 분기점 앞뒤로 계속 상위권이었고 2020년 이전 가격이 있는 것들.
# 코인을 사후에 고르면 생존편향이 들어가므로 우리 유니버스 상단에서 기계적으로 뽑는다.
ALT_MIN_ROWS = 1800
COVER_END = pd.Timestamp("2026-06-01")       # 이 날짜 뒤까지 있는 계열만 쓴다


def load_close(ticker: str) -> pd.Series | None:
    path = os.path.join(DATA, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Adj_Close" not in df.columns:
        return None
    s = pd.Series(df["Adj_Close"].values,
                  index=pd.to_datetime(df["Date"])).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < ALT_MIN_ROWS:
        return None
    # 분기점 뒤가 끊긴 계열은 쓰지 않는다. 섞으면 전 종목 공통일이 그 코인이
    # 사라진 날까지로 잘려서 이후 구간이 통째로 짧아진다(MATIC 이 그랬다).
    if s.index[0] > pd.Timestamp("2020-09-01") or s.index[-1] < COVER_END:
        return None
    return s


def alt_basket() -> list[str]:
    """유니버스 상단에서 기계적으로 뽑는다 — 사후 선택 금지."""
    with open(os.path.join(DATA, "historical_universe.json")) as f:
        months = json.load(f)
    key = sorted(k for k in months if k <= SPLIT.strftime("%Y-%m-%d"))[-1]
    stable = {"USDT", "USDC", "DAI", "TUSD", "BUSD", "FDUSD", "PYUSD", "USDD",
              "RLUSD", "USDG", "EURC", "XAUT", "PAXG", "LEO", "OKB", "BGB",
              "KCS", "CRO", "MNT", "BNB", "WBTC", "STETH", "WSTETH"}
    out = []
    for t in months[key]:
        sym = t.replace("-USD", "")
        if sym in stable or sym in ("BTC", "ETH"):
            continue
        if load_close(t) is not None:
            out.append(t)
        if len(out) >= 8:
            break
    return out


def avg_pairwise_corr(rets: pd.DataFrame, window: int) -> pd.Series:
    """각 시점의 롤링 평균 쌍상관 (대각 제외)."""
    cols = list(rets.columns)
    acc = None
    cnt = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = rets[cols[i]].rolling(window).corr(rets[cols[j]])
            acc = c if acc is None else acc + c
            cnt += 1
    return acc / cnt if cnt else pd.Series(dtype=float)


def split_stats(s: pd.Series, split: pd.Timestamp) -> dict:
    pre = s[s.index < split].dropna()
    post = s[s.index >= split].dropna()
    if len(pre) < 60 or len(post) < 60:
        return {}
    return dict(n_pre=len(pre), n_post=len(post),
                pre_mean=float(pre.mean()), post_mean=float(post.mean()),
                pre_med=float(pre.median()), post_med=float(post.median()),
                pre_q90=float(pre.quantile(0.90)), post_q90=float(post.quantile(0.90)),
                delta=float(post.mean() - pre.mean()))


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ───────────────────────────── A. 원자료 상관 ─────────────────────────────

def part_a() -> dict:
    section("A. 원자료 상관 — 전략을 거치지 않은 일간 수익률")

    alts = alt_basket()
    print(f"알트 바스켓({len(alts)}종, 분기점 직전 시가총액 순서): {', '.join(alts)}")

    series = {"BTC": load_close("BTC-USD"), "ETH": load_close("ETH-USD"),
              "SPY": load_close("SPY")}
    for t in alts:
        series[t] = load_close(t)
    series = {k: v for k, v in series.items() if v is not None}

    px = pd.DataFrame(series).sort_index()
    px = px[px.index >= "2019-01-01"]
    # 주가지수는 주말이 없다. 코인만 있는 날은 상관 계산에서 빼야 짝이 맞는다.
    px = px.dropna(subset=["SPY"])
    px = px.ffill().dropna()
    rets = np.log(px / px.shift(1)).dropna()
    print(f"공통 거래일 {len(rets)}일: {rets.index[0].date()} ~ {rets.index[-1].date()}")

    alt_cols = [c for c in rets.columns if c not in ("SPY",)]
    alt_only = [c for c in alt_cols if c not in ("BTC", "ETH")]

    curves = {
        "BTC~SPY": rets["BTC"].rolling(ROLL).corr(rets["SPY"]),
        "ETH~SPY": rets["ETH"].rolling(ROLL).corr(rets["SPY"]),
        "BTC~알트평균": avg_pairwise_corr(rets[["BTC"] + alt_only], ROLL)
        if alt_only else pd.Series(dtype=float),
        "알트끼리평균": avg_pairwise_corr(rets[alt_only], ROLL)
        if len(alt_only) > 1 else pd.Series(dtype=float),
        "코인전체평균": avg_pairwise_corr(rets[alt_cols], ROLL),
    }

    print(f"\n{ROLL}일 롤링 상관, 분기점 {SPLIT.date()} 전후")
    print(f"{'축':<16}{'이전평균':>10}{'이후평균':>10}{'차이':>9}"
          f"{'이전중앙':>10}{'이후중앙':>10}{'이전일수':>9}{'이후일수':>9}")
    stats = {}
    for name, cur in curves.items():
        st = split_stats(cur, SPLIT)
        if not st:
            continue
        stats[name] = st
        print(f"{name:<16}{st['pre_mean']:>10.3f}{st['post_mean']:>10.3f}"
              f"{st['delta']:>+9.3f}{st['pre_med']:>10.3f}{st['post_med']:>10.3f}"
              f"{st['n_pre']:>9}{st['n_post']:>9}")

    print(f"\n위약 분할 — 같은 계산을 다른 날짜로. 2024-01-11 의 차이가 이 분포 안이면 무의미하다.")
    print(f"{'축':<16}{'실제':>9} | " + " ".join(f"{d.strftime('%y-%m'):>7}" for d in PLACEBO))
    placebo = {}
    for name, cur in curves.items():
        if name not in stats:
            continue
        deltas = []
        for d in PLACEBO:
            st = split_stats(cur, d)
            deltas.append(st.get("delta", np.nan))
        placebo[name] = deltas
        real = stats[name]["delta"]
        absd = [abs(x) for x in deltas if not np.isnan(x)]
        rank = sum(1 for x in absd if x >= abs(real)) + 1
        print(f"{name:<16}{real:>+9.3f} | " +
              " ".join(f"{x:>+7.3f}" if not np.isnan(x) else f"{'-':>7}" for x in deltas) +
              f"   |변화| 순위 {rank}/{len(absd)+1}")

    return dict(alts=alts, stats=stats, placebo=placebo,
                curves={k: v.dropna() for k, v in curves.items() if len(v.dropna())})


# ──────────────────── B. 합성 포트폴리오 꼬리 동조화 ────────────────────

def part_b(end_date: str) -> dict:
    section("B. 합성 60/25/15 꼬리 동조화 — 실제 채택 설정의 슬리브 자산곡선")

    import bt_fut_rebal_band as base
    base.END = end_date
    print(f"기간 {base.START} ~ {base.END}")

    print("  선물 슬리브 계산 중...", flush=True)
    m_fu = base.run_fut(0.05)
    eq_fu = m_fu["_equity"]
    print("  현물 슬리브 계산 중...", flush=True)
    eq_sp = base.run_spot_live()
    print("  주식 슬리브 계산 중...", flush=True)
    m_st = base.run_stock_live()
    eq_st = m_st["_equity"] if isinstance(m_st, dict) else m_st

    for name, eq in (("주식", eq_st), ("코인현물", eq_sp), ("코인선물", eq_fu)):
        mt = base.metrics(eq)
        print(f"  {name:<6} {eq.index[0].date()}~{eq.index[-1].date()} "
              f"CAGR {mt['CAGR']:>7.1f}%  MDD {mt['MDD']:>7.1f}%  "
              f"Sharpe {mt['Sharpe']:>5.2f}  Calmar {mt['Cal']:>5.2f}")

    comp = base.build_alloc(eq_st, eq_sp, eq_fu)
    mt = base.metrics(comp)
    print(f"  {'합성':<6} {comp.index[0].date()}~{comp.index[-1].date()} "
          f"CAGR {mt['CAGR']:>7.1f}%  MDD {mt['MDD']:>7.1f}%  "
          f"Sharpe {mt['Sharpe']:>5.2f}  Calmar {mt['Cal']:>5.2f}")

    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    r = pd.DataFrame({
        "주식": eq_st.loc[common].pct_change(),
        "코인현물": eq_sp.loc[common].pct_change(),
        "코인선물": eq_fu.loc[common].pct_change(),
    }).dropna()
    rc = comp.pct_change().reindex(r.index).dropna()
    r = r.loc[rc.index]

    def window_report(label: str, rr: pd.DataFrame, cc: pd.Series) -> dict:
        thr = cc.quantile(TAIL_Q)
        tail = rr.loc[cc[cc <= thr].index]
        all_neg = float((tail < 0).all(axis=1).mean())
        two_neg = float(((tail < 0).sum(axis=1) >= 2).mean())
        full_corr = rr.corr()
        tail_corr = tail.corr()
        out = dict(n=len(rr), n_tail=len(tail), thr=float(thr),
                   all_neg=all_neg, two_neg=two_neg,
                   worst_day=float(cc.min()),
                   mdd=float(((1 + cc).cumprod() / (1 + cc).cumprod().cummax() - 1).min()))
        for a, b in (("주식", "코인현물"), ("주식", "코인선물"), ("코인현물", "코인선물")):
            out[f"전체 {a}~{b}"] = float(full_corr.loc[a, b])
            out[f"꼬리 {a}~{b}"] = float(tail_corr.loc[a, b])
        print(f"\n[{label}] 거래일 {out['n']}일, 꼬리 {out['n_tail']}일 "
              f"(합성 일수익 {out['thr']*100:.2f}% 이하)")
        print(f"  꼬리에서 세 슬리브 동시 마이너스 {all_neg*100:.1f}% · "
              f"둘 이상 마이너스 {two_neg*100:.1f}%")
        print(f"  최악 하루 {out['worst_day']*100:.2f}% · 구간 최대낙폭 {out['mdd']*100:.2f}%")
        for a, b in (("주식", "코인현물"), ("주식", "코인선물"), ("코인현물", "코인선물")):
            print(f"  {a}~{b}: 전체 {out[f'전체 {a}~{b}']:+.3f} · "
                  f"꼬리 {out[f'꼬리 {a}~{b}']:+.3f}")
        return out

    res = {"전체": window_report("전체 구간", r, rc)}
    pre_idx = rc.index[rc.index < SPLIT]
    post_idx = rc.index[rc.index >= SPLIT]
    if len(pre_idx) > 120 and len(post_idx) > 120:
        res["이전"] = window_report(f"{SPLIT.date()} 이전", r.loc[pre_idx], rc.loc[pre_idx])
        res["이후"] = window_report(f"{SPLIT.date()} 이후", r.loc[post_idx], rc.loc[post_idx])

        print("\n  분기점 전후 차이 (이후 빼기 이전)")
        for k in ("all_neg", "two_neg"):
            lbl = "세 슬리브 동시 마이너스" if k == "all_neg" else "둘 이상 마이너스"
            print(f"    꼬리 {lbl}: {res['이전'][k]*100:>5.1f}% -> "
                  f"{res['이후'][k]*100:>5.1f}%  ({(res['이후'][k]-res['이전'][k])*100:+.1f}%p)")
        for a, b in (("주식", "코인현물"), ("주식", "코인선물"), ("코인현물", "코인선물")):
            for scope in ("전체", "꼬리"):
                key = f"{scope} {a}~{b}"
                print(f"    {key}: {res['이전'][key]:+.3f} -> {res['이후'][key]:+.3f}"
                      f"  ({res['이후'][key]-res['이전'][key]:+.3f})")

        print("\n  위약 분할 — 같은 비교를 다른 날짜로")
        print(f"    {'분할':<12}{'동시마이너스 변화':>18}{'현물~선물 꼬리상관 변화':>26}")
        for d in [SPLIT] + PLACEBO:
            pi = rc.index[rc.index < d]
            qi = rc.index[rc.index >= d]
            if len(pi) < 120 or len(qi) < 120:
                continue
            a1 = window_quiet(r.loc[pi], rc.loc[pi])
            a2 = window_quiet(r.loc[qi], rc.loc[qi])
            mark = " <- 실제" if d == SPLIT else ""
            print(f"    {str(d.date()):<12}{(a2['all_neg']-a1['all_neg'])*100:>+17.1f}%"
                  f"{a2['sf']-a1['sf']:>+26.3f}{mark}")

    res["_eq"] = dict(주식=eq_st, 코인현물=eq_sp, 코인선물=eq_fu)
    return res


def window_quiet(rr: pd.DataFrame, cc: pd.Series) -> dict:
    thr = cc.quantile(TAIL_Q)
    tail = rr.loc[cc[cc <= thr].index]
    return dict(all_neg=float((tail < 0).all(axis=1).mean()),
                sf=float(tail.corr().loc["코인현물", "코인선물"]))


# ─────────── C. 상관 게이트를 실제 전략에 붙였을 때 ───────────

GATE_Q = [0.75, 0.90]          # 자기 과거 분포의 상위 25% / 상위 10%
GATE_F = [0.50, 0.75]          # 게이트가 켜진 날 코인 두 팔의 노출 배수
GATE_MIN_OBS = 504             # 분위수를 말하려면 최소 2년치 과거가 있어야 한다
TX_SPOT, TX_FUT = 0.004, 0.0006


def gate_series(corr: pd.Series, q: float) -> pd.Series:
    """자기 과거만으로 만든 분위수 문턱을 넘었나. 미래를 보지 않는다.

    전체 구간 분위수를 쓰면 그 문턱 자체가 미래 정보다 — 2021년의 판단에
    2025년 분포가 들어간다. 그래서 확장창 분위수를 쓰고, 그날 신호는
    다음 거래일부터 적용한다(전일 종가 판단·당일 체결 규약과 같다).
    """
    c = corr.dropna()
    thr = c.expanding(min_periods=GATE_MIN_OBS).quantile(q)
    return (c > thr).fillna(False).shift(1).fillna(False)


def part_c(eqs: dict, corr: pd.Series) -> None:
    section("C. 상관 게이트를 현재 전략에 붙인다 — 합성층 덮개")

    import bt_fut_rebal_band as base

    eq_st, eq_sp, eq_fu = eqs["주식"], eqs["코인현물"], eqs["코인선물"]
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    r_st = eq_st.loc[common].pct_change().fillna(0)
    r_sp = eq_sp.loc[common].pct_change().fillna(0)
    r_fu = eq_fu.loc[common].pct_change().fillna(0)

    print("규칙: 알트끼리 평균 롤링상관이 자기 과거 분위수를 넘은 다음 날부터,")
    print("코인 현물 25% + 코인 선물 15% 의 노출을 배수만큼 줄이고 나머지는 현금.")
    print("주식 60% 는 건드리지 않는다. 전략 엔진은 한 줄도 안 바꾼다.")
    print(f"분위수는 확장창(최소 {GATE_MIN_OBS}일) — 미래 분포를 쓰지 않는다.")
    print(f"덮개 회전비용: 현물 {TX_SPOT*100:.1f}% · 선물 {TX_FUT*100:.2f}% 편도")

    def arm(f_series: pd.Series, tx_mult: float = 1.0) -> pd.Series:
        f = f_series.reindex(common).fillna(1.0)
        ret = 0.60 * r_st + 0.25 * f * r_sp + 0.15 * f * r_fu
        turn = f.diff().abs().fillna(0.0)
        cost = turn * (0.25 * TX_SPOT + 0.15 * TX_FUT) * tx_mult
        return (1 + ret - cost).cumprod()

    arms: dict[str, pd.Series] = {}
    fmap: dict[str, pd.Series] = {}
    ones = pd.Series(1.0, index=common)
    arms["현행(게이트 없음)"] = arm(ones)
    fmap["현행(게이트 없음)"] = ones
    for q in GATE_Q:
        on = gate_series(corr, q).reindex(common).fillna(False)
        for fmul in GATE_F:
            name = f"상위{int((1-q)*100)}% 넘으면 ×{fmul:.2f}"
            fs = pd.Series(np.where(on.values, fmul, 1.0), index=common)
            arms[name] = arm(fs)
            fmap[name] = fs

    print(f"\n게이트가 켜져 있던 날 비율과 전환 횟수 (거래일 {len(common)}일)")
    print(f"{'팔':<26}{'켜진날':>8}{'전환':>7}{'덮개회전':>10}")
    for name, fs in fmap.items():
        on_pct = float((fs < 1.0).mean()) * 100
        flips = int((fs.diff().abs() > 1e-9).sum())
        turn = float(fs.diff().abs().sum())
        print(f"{name:<26}{on_pct:>7.1f}%{flips:>7}{turn:>10.2f}")

    print(f"\n합성 60/25/15 — 전체 구간 {common[0].date()} ~ {common[-1].date()}")
    print(f"{'팔':<26}{'CAGR':>9}{'MDD':>9}{'Sharpe':>8}{'Calmar':>8}")
    for name, eq in arms.items():
        mt = base.metrics(eq)
        print(f"{name:<26}{mt['CAGR']:>8.1f}%{mt['MDD']:>8.1f}%"
              f"{mt['Sharpe']:>8.2f}{mt['Cal']:>8.2f}")

    sums, wins, n = base.window_rs(arms)
    print(f"\n윈도우 순위합 — 창 {base.WIN_SIZES} × 보폭 {base.STRIDES}, n={n}")
    print(f"{'팔':<26}{'평균순위':>10}{'1위횟수':>9}")
    for k, v in sorted(sums.items(), key=lambda x: x[1] / max(n, 1)):
        print(f"{k:<26}{v/max(n,1):>10.3f}{wins[k]:>9}")

    arms5 = {k: arm(fmap[k], tx_mult=5.0) for k in fmap}
    s5, w5, n5 = base.window_rs(arms5)
    print("\n덮개 회전비용 5배")
    print(f"{'팔':<26}{'평균순위':>10}{'Calmar':>9}{'CAGR':>9}")
    for k, v in sorted(s5.items(), key=lambda x: x[1] / max(n5, 1)):
        mt = base.metrics(arms5[k])
        print(f"{k:<26}{v/max(n5,1):>10.3f}{mt['Cal']:>9.2f}{mt['CAGR']:>8.1f}%")

    print("\n개선이 회전 감소로 설명되는지 — 노출을 줄인 만큼의 단순 기대손실 대조")
    base_eq = arms["현행(게이트 없음)"]
    bm = base.metrics(base_eq)
    for name, eq in arms.items():
        if name == "현행(게이트 없음)":
            continue
        mt = base.metrics(eq)
        fs = fmap[name]
        avg_exp = float(fs.mean())
        print(f"  {name:<26} 평균 코인노출 {avg_exp*40:.1f}% (현행 40.0%) · "
              f"CAGR {mt['CAGR']-bm['CAGR']:+.1f}%p · MDD {mt['MDD']-bm['MDD']:+.1f}%p · "
              f"Calmar {mt['Cal']-bm['Cal']:+.2f}")

    print("\n판정은 보고서에서 한다. 합성 순위합에서 이기지 못하면 채택하지 않는다.")


def main() -> None:
    end_date = os.environ.get("C7_END", "2026-09-18")
    a = part_a()
    b = part_b(end_date)
    corr = a.get("curves", {}).get("알트끼리평균")
    if corr is not None and len(corr) and "_eq" in b:
        part_c(b["_eq"], corr)

    section("판정")
    print("아래는 관측이다. 채택·기각 판정은 보고서에서 한다.")
    if a.get("stats"):
        key = "알트끼리평균" if "알트끼리평균" in a["stats"] else next(iter(a["stats"]))
        st = a["stats"][key]
        print(f"- 원자료: {key} 롤링 상관이 {st['pre_mean']:.3f} -> {st['post_mean']:.3f} "
              f"({st['delta']:+.3f})")
    if "이전" in b:
        print(f"- 합성 꼬리 동시 마이너스: {b['이전']['all_neg']*100:.1f}% -> "
              f"{b['이후']['all_neg']*100:.1f}%")
        print(f"- 합성 최대낙폭: {b['이전']['mdd']*100:.2f}% -> {b['이후']['mdd']*100:.2f}%")


if __name__ == "__main__":
    main()
