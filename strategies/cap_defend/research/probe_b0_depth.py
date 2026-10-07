"""B0 가 실제로 몇 위까지 내려가 뽑는지, 그 위에서 빠진 코인이 몇이나 되는지 센다.

선행 검증의 마지막 칸이다. 유니버스 감사가 원시 시총 앞 40칸의 11.9%(스테이블·금 제외)가
자료에 없다고 알려줬는데, 그 수치만으로는 측정이 깨졌는지 알 수 없다. B0 가 앞 몇 위
안에서만 뽑으면 깊은 자리의 결측은 선정을 못 바꾼다. 그래서 두 가지를 센다.

  ⑴ B0 의 흡수 후 종목이 원시(필터 전) 시총 목록에서 몇 위였나 — 그 날의 최대 순위.
  ⑵ 그 최대 순위보다 앞에서 자료가 없어 건너뛴 코인 수 — 스테이블·금·포장토큰을 뺀 수.
     이 수가 0 이면 그 날의 선정은 결측과 무관하다. 1 이상이면 그 코인이 헬스를 통과했을
     경우 뽑힌 셋 중 하나를 밀어냈을 수 있다 — 상한이지 실제 변경 건수가 아니다.

손익을 안 센다. 선정만 본다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 probe_b0_depth.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAP))
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import backtest_futures_v25 as bf  # noqa: E402
from probe_rank_select_compare import START, END, healthy_at, absorb  # noqa: E402

RAW = os.path.join(ROOT, "data", "historical_universe.json")
OUTDIR = os.path.join(ROOT, "state", "20261007-b0robust")

# 선정 대상이 될 수 없는 것들 — 스테이블코인, 금 연동, 포장·스테이킹 파생.
# 이것들이 빠진 자리는 전략이 애초에 못 고르는 자리라 결측으로 세지 않는다.
NON_SELECTABLE = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "PAX", "UST", "USTC",
    "FDUSD", "PYUSD", "USDD", "FRAX", "LUSD", "GUSD", "EURS", "EURT",
    "USDE", "SUSDE", "USD0", "USDS", "USDF", "RLUSD", "USDY", "BUIDL",
    "USD1", "USDG", "USDX", "USDB", "USDL", "AUSD", "EURC", "XSGD",
    "PAXG", "XAUT", "XAUM",
    "WBTC", "CBBTC", "SOLVBTC", "LBTC", "BTCB", "HBTC", "RENBTC",
    "STETH", "WSTETH", "WETH", "WEETH", "RETH", "CBETH", "METH", "EZETH",
    "WBETH", "RSETH", "SWETH", "FRXETH", "SFRXETH", "BSC-USD", "WBT",
    "JITOSOL", "MSOL", "JUPSOL", "BNSOL", "WSOL",
    "STBTC", "SBTC", "ETHX", "ANKRETH", "OSETH", "WSTUSDT",
}


# 거래소 자체 토큰 — 업비트 원화장·바이낸스 무기한 어느 쪽에도 대체로 없다.
# 빠진 것 중 이 몫을 따로 적어서, 실제로 살 수 있었을 코인의 결측을 가린다.
EXCHANGE = {"LEO", "CRO", "MNT", "OKB", "BGB", "KCS", "NEXO", "WBT", "HT",
            "FTT", "GT", "BNB", "CET", "MX", "BMX", "ZRO"}


def load_raw():
    d = json.load(open(RAW))
    keys = sorted(d.keys())
    return d, keys


def raw_rank_map(d, keys, date):
    mk = date.strftime("%Y-%m") + "-01"
    use = None
    for k in keys:
        if k <= mk:
            use = k
        else:
            break
    if use is None:
        use = keys[0]
    lst = [t.replace("-USD", "") for t in d[use]]
    return {s: i + 1 for i, s in enumerate(lst)}, lst


def main() -> None:
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    from unified_backtest import load_data
    bars, _f = load_data("D")
    btc = bars.get("BTC")
    dates = [d for d in btc.index if START <= str(d)[:10] <= END]
    draw, dkeys = load_raw()
    print(f"일봉 {len(bars)}종 · 신호일 {len(dates)}일 ({time.time() - t0:.0f}s)", flush=True)

    report = {}
    for label, mom_short, key in (("선물(단기 18봉)", 18, "fut"),
                                  ("현물(단기 20봉)", 20, "spot")):
        depth = []          # 그 날 뽑힌 종목의 원시 최대 순위
        gaps = []           # 그 최대 순위 위에서 빠진 선정가능 코인 수
        ex_gaps = []        # 그중 거래소 토큰까지 뺀 수
        gap_names = Counter()
        n_pick0 = 0
        for d in dates:
            mo = bf.get_mcap(d)
            hs = healthy_at(bars, mo, d, mom_short)
            if not hs:
                n_pick0 += 1
                continue
            info = {x[0]: x for x in hs}
            picks = absorb([x[0] for x in hs][:3], info, mom_short)
            if not picks:
                n_pick0 += 1
                continue
            rmap, rlist = raw_rank_map(draw, dkeys, d)
            ranks = [rmap.get(c, 999) for c in picks]
            dmax = max(ranks)
            depth.append(dmax)
            # 그 깊이 위에서 자료에 없어 못 본 선정가능 코인
            miss = [s for s in rlist[:dmax]
                    if s not in NON_SELECTABLE and bars.get(s) is None]
            gaps.append(len(miss))
            ex_gaps.append(sum(1 for s in miss if s not in EXCHANGE))
            for s in miss:
                gap_names[s] += 1

        n = len(depth)
        sd = sorted(depth)
        q = lambda p: sd[min(n - 1, int(p * n))]  # noqa: E731
        zero = sum(1 for g in gaps if g == 0)
        report[key] = dict(
            days=n, no_pick=n_pick0,
            depth_med=q(0.5), depth_p90=q(0.9), depth_p99=q(0.99), depth_max=sd[-1],
            depth_le10=sum(1 for x in depth if x <= 10) / n,
            depth_le20=sum(1 for x in depth if x <= 20) / n,
            gap_zero_frac=zero / n,
            gap_mean=sum(gaps) / n, gap_max=max(gaps),
            exgap_zero_frac=sum(1 for g in ex_gaps if g == 0) / n,
            exgap_mean=sum(ex_gaps) / n, exgap_max=max(ex_gaps),
            gap_top=gap_names.most_common(12),
        )
        r = report[key]
        print(f"\n== {label} ==")
        print(f"  선정일 {n}일 · 전부현금 {n_pick0}일")
        print(f"  뽑힌 종목의 원시 최대 순위 — 중앙 {r['depth_med']} · 90분위 {r['depth_p90']}"
              f" · 99분위 {r['depth_p99']} · 최대 {r['depth_max']}")
        print(f"  10위 이내로만 뽑은 날 {r['depth_le10']*100:.1f}% · 20위 이내 {r['depth_le20']*100:.1f}%")
        print(f"  그 깊이 위에서 빠진 선정가능 코인 — 0인 날 {r['gap_zero_frac']*100:.1f}%"
              f" · 평균 {r['gap_mean']:.2f}종 · 최대 {r['gap_max']}종")
        print(f"  거래소 토큰까지 빼면 — 0인 날 {r['exgap_zero_frac']*100:.1f}%"
              f" · 평균 {r['exgap_mean']:.2f}종 · 최대 {r['exgap_max']}종")
        print(f"  자주 빠진 것: {r['gap_top']}")

    with open(os.path.join(OUTDIR, "b0_depth.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n저장 {OUTDIR}/b0_depth.json · 총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
