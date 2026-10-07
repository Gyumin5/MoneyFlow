"""선정만 비교하는 검사 — 손익을 안 센다 (2026-10-07, 채택 명세 5항).

토론이 확정한 절차의 5항이다: "손익을 계산하지 않는 선정 검사를 먼저 수행한다. 모든 재선정
시점에 전체 풀과 원시 월별 시총 목록의 앞 40개를 먼저 자른 풀을 비교한다. 40개 절단은
심볼 매핑·데이터 필터·헬스 이전에 적용하며 탈락분을 41위 이하로 보충하지 않는다. 흡수 전·후
집합, 현금, 경계 교란 가능 여부를 저장한다. 동일한 선정·비중 경로를 만드는 팔은 하나만
재생하고 별칭으로 보고한다. 중복 판단에 성과는 사용하지 않는다."

내는 것 넷.
  ⑴ 팔끼리 같은 경로인가 — 매일의 (흡수 후 종목 집합, 비중) 이 전부 같으면 같은 팔이다.
     자산곡선을 안 보고 가른다. 중복이면 둘 중 하나만 보고하고 나머지는 별칭으로 적는다.
  ⑵ 전체 풀 대 원시 시총 앞 40종 풀 — 흡수 전·후 집합 불일치율, 목표 비중 차이
     0.5×Σ|w_전체 − w_40|, 현금 차이, 그리고 전체 풀에서 원시 40위 밖 코인에 간 비중.
     게이트 ③ 의 저비용 보고분이다.
  ⑶ 경계 교란(3위↔4위 교환)이 가능한 날의 비율. 4위가 없으면 교란이 아무것도 안 바꾸므로
     그 날은 안정성의 근거가 못 된다 — 게이트 ④ 가 요구하는 집계다.
  ⑷ 흡수 전·후 종목 수 분포와 현금 비중. 게이트 ⑧ 보고분이다.

안 재는 것: 스냅 평균 후 목표 비중. 그건 스냅 일정까지 복제해야 하는데, 복제본이 엔진과
어긋나면 조용히 틀린 수가 나온다. 그 값은 실제 백테스트 산출물에서 뽑는다.

이 검사는 한 스냅샷 기준이다 — 매 신호일을 재선정 시점으로 보고 그 날의 선정 결과를 비교한다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 probe_rank_select_compare.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import backtest_futures_v25 as bf  # noqa: E402
import rank_axes_select as ra  # noqa: E402

START = "2020-10-01"
END = "2026-09-18"

MOM_LONG = 127
VOL_BARS = 90
VOL_TH = 0.05
SMA = 42
UNIVERSE_SIZE = 3
CAP_W = 1.0 / 3.0
POOL_N = 40

ARMS_SPOT = ["b0", "smom", "slong", "seff", "cmix", "ualt"]
ARMS_FUT = ARMS_SPOT + ["sdvol"]
LABEL = {"b0": "B0 시총", "smom": "S단기", "slong": "S장기", "seff": "S효율",
         "cmix": "C시총효율", "ualt": "U교대3", "sdvol": "S거래대금"}


def healthy_at(bars, mcap_order, sig_date, mom_short):
    """엔진 _compute_weights 의 헬스 루프 그대로. 시총 내림차순을 유지한다."""
    out = []
    min_bars = max(mom_short, MOM_LONG, VOL_BARS, SMA)
    for coin in mcap_order:
        df = bars.get(coin)
        if df is None:
            continue
        ci = df.index.get_indexer([sig_date], method="ffill")[0]
        if ci < 0 or ci < min_bars:
            continue
        c = df["Close"].values[:ci + 1]
        if (bf.calc_mom(c, mom_short) > 0 and bf.calc_mom(c, MOM_LONG) > 0
                and bf.calc_vol_daily(c, 1, lookback_bars=VOL_BARS) <= VOL_TH):
            out.append((coin, df, ci, c))
    return out


def absorb(picks, info, mom_short):
    """엔진 greedy 흡수 그대로."""
    picks = list(picks)
    if len(picks) > 1:
        for i in range(len(picks) - 1, 0, -1):
            # 튜플은 (coin, df, ci, close) — 종가는 3번이다.
            ma = bf.calc_mom(info[picks[i - 1]][3], mom_short)
            mb = bf.calc_mom(info[picks[i]][3], mom_short)
            if ma >= mb:
                picks.pop(i)
    return picks


def weights_of(picks):
    if not picks:
        return {"CASH": 1.0}
    w = min(1.0 / len(picks), CAP_W)
    out = {c: w for c in picks}
    t = sum(out.values())
    if t < 0.999:
        out["CASH"] = 1.0 - t
    return out


def inputs_for(hs):
    """하루치 선택 입력. 팔마다 다시 계산하면 변동성 호출이 팔 수만큼 곱해진다."""
    closes = {x[0]: x[3] for x in hs}
    vols = {x[0]: bf.calc_vol_daily(x[3], 1, lookback_bars=VOL_BARS) for x in hs}
    dvols = {}
    for coin, df, ci, _c in hs:
        lo = max(0, ci + 1 - ra.DVOL_BARS)
        v = df["Volume"].values[lo:ci + 1]
        p = df["Close"].values[lo:ci + 1]
        dvols[coin] = float(np.mean(v * p)) if len(v) else 0.0
    return [x[0] for x in hs], closes, vols, dvols


def select_for(mode, inp, mom_short, swap=False):
    """rank_axes_select 를 엔진과 같은 인자로 부른다."""
    healthy, closes, vols, dvols = inp
    return ra.select(mode, healthy, UNIVERSE_SIZE, closes=closes, vols=vols,
                     dvols=dvols, mom_short_bars=mom_short, mom_long_bars=MOM_LONG,
                     calc_mom=bf.calc_mom, swap=swap)


def wdiff(wa, wb):
    keys = set(wa) | set(wb)
    return 0.5 * sum(abs(wa.get(k, 0.0) - wb.get(k, 0.0)) for k in keys)


def main() -> None:
    t0 = time.time()
    from unified_backtest import load_data
    bars, _f = load_data("D")
    btc = bars.get("BTC")
    dates = [d for d in btc.index if START <= str(d)[:10] <= END]
    print(f"일봉 {len(bars)}종 · 신호일 {len(dates)}일 ({time.time() - t0:.0f}s)", flush=True)

    for label, mom_short, arms in (("선물(단기 18봉)", 18, ARMS_FUT),
                                   ("현물(단기 20봉)", 20, ARMS_SPOT)):
        # 팔별 집계
        pre_n = {a: [] for a in arms}
        post_n = {a: [] for a in arms}
        cash = {a: [] for a in arms}
        swap_ok = {a: 0 for a in arms}
        pre_mis = {a: 0 for a in arms}     # 전체 풀 대 40종 풀, 흡수 전 집합 불일치
        post_mis = {a: 0 for a in arms}    # 흡수 후 집합 불일치
        wgap = {a: [] for a in arms}       # 0.5 Σ|w_전체 - w_40|
        cgap = {a: [] for a in arms}       # 현금 차이
        tail_w = {a: [] for a in arms}     # 전체 풀에서 원시 40위 밖에 간 비중
        sig = {a: [] for a in arms}        # 중복 판정용 경로 지문
        n_eval = 0

        for i, d in enumerate(dates):
            mo_full = bf.get_mcap(d)
            mo_40 = mo_full[:POOL_N]
            top40 = set(mo_40)
            hs_full = healthy_at(bars, mo_full, d, mom_short)
            hs_40 = healthy_at(bars, mo_40, d, mom_short)
            if not hs_full and not hs_40:
                for a in arms:
                    sig[a].append("CASH")
                continue
            n_eval += 1
            info_f = {x[0]: x for x in hs_full}
            info_4 = {x[0]: x for x in hs_40}
            inp_f = inputs_for(hs_full)
            inp_4 = inputs_for(hs_40)
            for a in arms:
                pf = select_for(a, inp_f, mom_short) if hs_full else []
                p4 = select_for(a, inp_4, mom_short) if hs_40 else []
                af = absorb(pf, info_f, mom_short)
                a4 = absorb(p4, info_4, mom_short)
                wf, w4 = weights_of(af), weights_of(a4)
                pre_n[a].append(len(pf))
                post_n[a].append(len(af))
                cash[a].append(wf.get("CASH", 0.0))
                if set(pf) != set(p4):
                    pre_mis[a] += 1
                if set(af) != set(a4):
                    post_mis[a] += 1
                wgap[a].append(wdiff(wf, w4))
                cgap[a].append(abs(wf.get("CASH", 0.0) - w4.get("CASH", 0.0)))
                tail_w[a].append(sum(v for k, v in wf.items()
                                     if k != "CASH" and k not in top40))
                # 경계 교란 가능 여부 — swap=True 호출이 실제로 자리를 바꿨는지
                ra.reset_stats()
                select_for(a, inp_f, mom_short, swap=True)
                if ra.get_stats()["swap_applied"]:
                    swap_ok[a] += 1
                sig[a].append("|".join(sorted(af)) + "@" + f"{wf.get('CASH', 0.0):.4f}")
            if (i + 1) % 400 == 0:
                print(f"  ... {i + 1}/{len(dates)}일 ({time.time() - t0:.0f}s)", flush=True)

        print(f"\n=== {label} === 선정이 있는 날 {n_eval}/{len(dates)}")
        print("\n[팔 중복] 같은 경로를 만드는 팔 쌍 (매일 흡수 후 집합·현금이 전부 같음)")
        dup = []
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                if sig[a] == sig[b]:
                    dup.append((a, b))
        print("  없음" if not dup else
              "\n".join(f"  {LABEL[a]} ≡ {LABEL[b]}" for a, b in dup))

        print(f"\n{'팔':<10} {'흡수전 평균':>10} {'흡수후 평균':>10} {'현금 평균':>9}"
              f" {'경계교란 가능':>12}")
        for a in arms:
            print(f"{LABEL[a]:<10} {np.mean(pre_n[a]):>10.2f} {np.mean(post_n[a]):>10.2f}"
                  f" {np.mean(cash[a]) * 100:>8.1f}%"
                  f" {swap_ok[a] / n_eval * 100:>11.1f}%")

        print(f"\n[풀 폭] 전체 풀 대 원시 시총 앞 {POOL_N}종")
        print(f"{'팔':<10} {'흡수전 불일치':>12} {'흡수후 불일치':>12} {'비중차 평균':>11}"
              f" {'현금차 평균':>11} {'40위밖 비중':>11}")
        for a in arms:
            print(f"{LABEL[a]:<10} {pre_mis[a] / n_eval * 100:>11.1f}%"
                  f" {post_mis[a] / n_eval * 100:>11.1f}%"
                  f" {np.mean(wgap[a]) * 100:>10.2f}%"
                  f" {np.mean(cgap[a]) * 100:>10.2f}%"
                  f" {np.mean(tail_w[a]) * 100:>10.2f}%")

    print(f"\n걸린 시간 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
