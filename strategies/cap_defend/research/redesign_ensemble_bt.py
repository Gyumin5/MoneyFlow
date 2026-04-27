"""Step 6 — Ensemble k=2/3 실제 백테스트 + tail co-crash 지표.

입력:
  redesign_rank_{asset}.csv  (Step 5 랭킹 후 Top N)
  redesign_ensemble_candidates_{asset}.csv (analyze 에서 bucket 조합)
  redesign_top500_{asset}_k1.csv (cfg 참조)

로직:
  1. 각 combo 멤버 cfg 복원 (동일 snap + iv 버킷 확인)
  2. 각 멤버 daily equity/returns 생성 (BT 재실행 or cache)
  3. Pearson corr, bad_day_overlap (worst 5% day 공통 손실), joint_2022_loss, joint_max_DD
  4. EW 앙상블 백테스트 Cal/CAGR/MDD/Sh 계산
  5. k=1 대비 개선 체크 (med_Cal / p25_Cal / MDD 중 2개 이상)

출력: redesign_ensemble_bt_{asset}.csv
  members, k, Cal, CAGR, MDD, Sh, p25_Cal, corr_max, bad_day_overlap,
  joint_2022_loss, joint_max_DD, improve_count, status

병렬 24 worker. resume 지원 (members key).
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from redesign_common import parse_cfg, run_bt, status_resume_keys, TX_BY_ASSET

_DATA = None


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def member_return_series(asset, cfg, phase=0):
    """멤버 single bt 를 돌려 daily returns Series 반환.
    TODO: unified_backtest 에 equity 반환 옵션 추가 필요. 현재는 placeholder.
    """
    # 실제 구현은 unified_backtest.run 에서 equity DataFrame 반환하도록 확장.
    # 지금은 skeleton — metrics 만 반환.
    return None


def tail_metrics(returns_list):
    """멤버별 returns Series 들로 tail co-crash 지표 계산.
    placeholder: None 반환.
    """
    return {"corr_max": None, "bad_day_overlap": None,
            "joint_2022_loss": None, "joint_max_DD": None}


def ensemble_fut_sae(traces, leverage=3.0, tx_cost=0.0004):
    """fut L3 전용: futures_ensemble_engine.SingleAccountEngine 으로 진짜 single-account 시뮬.
    traces: list of [{date, target}] (멤버별 D 또는 4h trace)
    1h bars 기반 SAE 실행 — 정확한 leverage/margin/funding/liquidation 회계.
    """
    import pandas as pd
    import sys, os
    HERE = os.path.dirname(os.path.abspath(__file__))
    CAP = os.path.dirname(HERE)
    if CAP not in sys.path:
        sys.path.insert(0, CAP)
    from futures_ensemble_engine import SingleAccountEngine
    if not traces:
        return None
    if not hasattr(ensemble_fut_sae, "_bars_1h"):
        from unified_backtest import load_data
        bars_1h, funding_1h = load_data("1h")
        ensemble_fut_sae._bars_1h = bars_1h
        ensemble_fut_sae._funding_1h = funding_1h
    bars_1h = ensemble_fut_sae._bars_1h
    funding_1h = ensemble_fut_sae._funding_1h

    # 멤버별 (date, target) sorted events
    member_events = []
    for trace in traces:
        rows = [(pd.Timestamp(t.get("date") or t.get("Date")), dict(t.get("target") or {}))
                for t in trace]
        rows.sort(key=lambda x: x[0])
        # dedup (last per date)
        dedup = {}
        for d, tgt in rows:
            dedup[d] = tgt
        member_events.append(sorted(dedup.items()))
    if any(not e for e in member_events):
        return None

    # 1h grid 위에 멤버 target ffill + EW merge
    ts_1h = next(iter(bars_1h.values())).index
    # 시작/종료 시점은 모든 멤버 trace 의 max-start ~ min-end 교집합 (ts_1h 안에서만)
    starts = [e[0][0] for e in member_events]
    ends = [e[-1][0] for e in member_events]
    t0 = max(starts)
    t1 = min(ends)
    ts_1h = ts_1h[(ts_1h >= t0) & (ts_1h <= t1)]
    if len(ts_1h) < 24:
        return None
    k = len(member_events)
    indices = [0] * k
    current = [{} for _ in range(k)]
    target_series = []
    for ts in ts_1h:
        for mi in range(k):
            ev = member_events[mi]
            while indices[mi] < len(ev) and ev[indices[mi]][0] <= ts:
                current[mi] = ev[indices[mi]][1]
                indices[mi] += 1
        merged = {}
        for tgt in current:
            for asset_name, w in tgt.items():
                key = str(asset_name).upper()
                if key == "CASH":
                    continue
                merged[key] = merged.get(key, 0.0) + float(w) / k
        cash_w = max(0.0, 1.0 - sum(merged.values()))
        if cash_w > 1e-9:
            merged["CASH"] = cash_w
        target_series.append((ts, merged))

    sae = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=float(leverage), tx_cost=float(tx_cost),
        stop_kind="none", stop_pct=0.0,
        leverage_mode="fixed",
    )
    metrics = sae.run(target_series)
    if not metrics:
        return None
    return {
        "Cal": float(metrics.get("Cal", 0) or 0),
        "CAGR": float(metrics.get("CAGR", 0) or 0),
        "MDD": float(metrics.get("MDD", 0) or 0),
        "Sh": float(metrics.get("Sharpe", 0) or 0),
        "equity": metrics.get("_equity"),
    }


def ensemble_from_traces(traces, asset, leverage=1.0, tx_cost=None):
    """True single-account ensemble: 멤버 traces (target weights) 를 매 rebal date 마다
    EW merge → 단일 portfolio 시뮬. live engine 동일 방식.

    traces: list of [{Date, target}] (멤버별)
    asset: 'fut'/'spot'/'stock' (tx_cost 결정용)
    """
    import pandas as pd
    from redesign_common import TX_BY_ASSET
    if tx_cost is None:
        tx_cost = TX_BY_ASSET.get(asset, 0.0004)
    if not traces:
        return None
    # 1) 멤버 별 daily target weights DataFrame 생성 (target 변경 시점만 row, ffill 로 채움)
    target_dfs = []
    for trace in traces:
        rows = []
        for t in trace:
            # 4h trace 의 시각 component drop → date 단위 (cross-iv 호환)
            d = pd.Timestamp(t.get("date") or t.get("Date")).normalize()
            tgt = t.get("target") or {}
            row = {"date": d}
            for k, v in tgt.items():
                kk = str(k).upper()
                if kk == "CASH":
                    continue
                row[kk] = float(v)
            rows.append(row)
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("date").fillna(0.0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        target_dfs.append(df)
    # 2) union dates 위에서 ffill 후 EW merge
    all_dates = sorted(set().union(*[df.index for df in target_dfs]))
    aligned = [df.reindex(all_dates).ffill().fillna(0.0) for df in target_dfs]
    all_assets = sorted(set().union(*[set(df.columns) for df in aligned]))
    if not all_assets:
        return None
    k = len(traces)
    merged = pd.DataFrame(0.0, index=all_dates, columns=all_assets)
    for df in aligned:
        df3 = df.reindex(columns=all_assets, fill_value=0.0)
        merged += df3 / k
    # 비-cash 합 cap (1.0 초과면 정규화)
    row_sum = merged.sum(axis=1)
    over = row_sum > 1.0 + 1e-9
    if over.any():
        merged.loc[over] = merged.loc[over].div(row_sum[over], axis=0)
    # 3) 단순 portfolio 시뮬 (close-to-close, leverage 1=현물, 3=fut)
    # 가격: 각 자산의 close. bars 는 _DATA 에 dict {asset: df}.
    # cross-iv ensemble 가능 (snap_days_eq 일치 시). portfolio sim 은 D 기준 사용.
    if asset == "stock":
        # stock 은 stock_engine 의 가격 사용 (SPY/QQQ/...). _g_prices 는 _init 으로 주입됨.
        try:
            from redesign_stock_adapter import _init_once
            _init_once()
            import stock_engine as tsi
            bars = {ticker: pd.DataFrame({"Close": ser}) for ticker, ser in tsi._g_prices.items()}
        except Exception as e:
            return None
    else:
        bars = _DATA.get("D", ({}, None))[0]
    # 자산 close 추출 (대문자 일치)
    px = pd.DataFrame()
    for a in all_assets:
        # bars 키 매칭: 대소문자 둘 다
        if a in bars:
            df = bars[a]
        elif a.lower() in bars:
            df = bars[a.lower()]
        else:
            continue
        col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        if col is not None:
            px[a] = df[col]
    if px.empty:
        return None
    px = px.sort_index()
    common = px.index.intersection(merged.index)
    if len(common) < 2:
        return None
    px = px.loc[common]
    tgt = merged.reindex(common, method="ffill").fillna(0.0)
    pv = 10000.0
    qty = {a: 0.0 for a in all_assets}
    cash = pv
    pv_hist = []
    prev = {a: 0.0 for a in all_assets}
    for d in common:
        row_px = px.loc[d]
        # mark-to-market (pre-trade)
        eq = sum(qty[a] * row_px[a] for a in all_assets if not pd.isna(row_px.get(a)))
        pv = cash + eq * leverage
        if pv <= 0:
            pv_hist.append({"Date": d, "PV": 0.0})
            continue
        cur = {a: float(tgt.loc[d].get(a, 0.0)) for a in all_assets}
        if any(abs(cur[a] - prev[a]) > 1e-6 for a in all_assets):
            new_qty = {}
            for a in all_assets:
                w = cur[a]
                p = row_px.get(a)
                if pd.isna(p) or p <= 0:
                    new_qty[a] = qty[a]
                    continue
                target_qty = (pv * w) / p
                turnover = abs(target_qty - qty[a])
                tx = turnover * p * tx_cost
                cash -= (target_qty - qty[a]) * p + tx
                new_qty[a] = target_qty
            qty = new_qty
            prev = dict(cur)
            # post-trade re-mark
            eq = sum(qty[a] * row_px[a] for a in all_assets if not pd.isna(row_px.get(a)))
            pv = cash + eq * leverage
        pv_hist.append({"Date": d, "PV": pv})
    pv_df = pd.DataFrame(pv_hist).set_index("Date")
    if pv_df.empty or len(pv_df) < 2:
        return None
    eq = pv_df["PV"]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[0] <= 0:
        return None
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * (252 ** 0.5)) if dr.std() > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    cal = float(cagr / abs(mdd)) if mdd != 0 else 0.0
    return {"Cal": cal, "CAGR": float(cagr), "MDD": mdd, "Sh": sh, "equity": eq}


def ensemble_from_equity(equities):
    """[Legacy] 멤버별 equity Series → EW 앙상블 equity + 지표 (simplified, 진단용).
    각 멤버 equity 를 1.0 기준 정규화 후 EW 평균. 진짜 single-account 아님.
    """
    import pandas as pd
    s = pd.concat(equities, axis=1)
    s = s.dropna()
    if len(s) == 0:
        return None
    normalized = s.div(s.iloc[0])
    ew = normalized.mean(axis=1)
    dr = ew.pct_change().dropna()
    yrs = (ew.index[-1] - ew.index[0]).days / 365.25 if len(ew) > 1 else 0
    if yrs <= 0 or ew.iloc[-1] <= 0:
        return None
    cagr = ew.iloc[-1] ** (1 / yrs) - 1
    mdd = (ew / ew.cummax() - 1).min()
    sh = dr.mean() / dr.std() * (365.25 ** 0.5) if dr.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return {
        "Cal": float(cal), "CAGR": float(cagr),
        "MDD": float(mdd), "Sh": float(sh),
        "equity": ew,
    }


def compute_tail_metrics(equity_list):
    """멤버 equity Series 리스트 → daily resample → corr/bad_day/joint metrics.
    4h equity 는 daily last 로 resample 후 pct_change 적용 (scale 일치).
    """
    import pandas as pd
    if not equity_list:
        return {}
    daily = []
    for eq in equity_list:
        try:
            d = eq.resample("D").last().dropna()
            daily.append(d.pct_change().dropna())
        except Exception:
            daily.append(eq.pct_change().dropna())
    df = pd.concat(daily, axis=1).dropna()
    if len(df) < 20:
        return {}
    corr_max = float(df.corr().values[np.triu_indices(df.shape[1], k=1)].max()) if df.shape[1] >= 2 else 0.0
    # Bad day overlap: worst 5% days, fraction where all members negative
    n_bad = max(int(len(df) * 0.05), 1)
    worst_rows = df.index[df.sum(axis=1).rank() <= n_bad]
    overlap = (df.loc[worst_rows] < 0).all(axis=1).sum() / max(len(worst_rows), 1)
    # 2022 joint loss
    y2022 = df[df.index.year == 2022]
    joint_2022 = None
    if len(y2022):
        joint_2022 = float((y2022.mean(axis=1)).sum())
    # Joint max DD
    eq = (1 + df.mean(axis=1)).cumprod()
    joint_max_dd = float((eq / eq.cummax() - 1).min())
    return {
        "corr_max": corr_max,
        "bad_day_overlap": float(overlap),
        "joint_2022_loss": joint_2022,
        "joint_max_DD": joint_max_dd,
    }


def run_one(task):
    members = task["members"]
    asset = task["asset"]
    k = task["k"]
    cfgs = task["cfgs"]
    try:
        member_results = []
        for cfg in cfgs:
            r = run_bt(asset, cfg, bars_funding=_DATA, phase_offset=0,
                      with_equity=True, with_trace=True)
            if r["status"] != "ok":
                return {"members": "|".join(members), "asset": asset, "k": k,
                        "status": "error", "error": f"member BT fail: {r.get('error')}"}
            member_results.append(r)
        equities = [r["_equity"] for r in member_results if "_equity" in r]
        traces = [r["_trace"] for r in member_results if "_trace" in r]
        if len(equities) != len(cfgs) or len(traces) != len(cfgs):
            return {"members": "|".join(members), "asset": asset, "k": k,
                    "status": "error", "error": "equity/trace missing"}
        # fut L3 는 SAE-based true single-account (1h bars, margin/liquidation/funding 정확)
        # spot/stock 은 L1 라 trace-based portfolio sim
        if asset == "fut":
            lev = float(cfgs[0].get("lev", 3.0))
            ens = ensemble_fut_sae(traces, leverage=lev, tx_cost=0.0004)
        else:
            ens = ensemble_from_traces(traces, asset, leverage=1.0)
        if ens is None:
            return {"members": "|".join(members), "asset": asset, "k": k,
                    "status": "error", "error": "ensemble aggregation failed"}
        tail = compute_tail_metrics(equities)
        # Gate 적용 (plan v3 Step 6)
        #  - corr_max < 0.7
        #  - bad_day_overlap < 0.8 (worst 5% day 에서 90%+ 동시 하락 방지)
        #  - joint_2022_loss ≥ -0.20 (코인) / ≥ -0.10 (주식)
        #  - joint_max_DD ≥ -0.50 (극단 tail drawdown 거부)
        #  - k=1 대비 개선 (median Cal / MDD 중 2개 이상) — BT 후 별도 check
        corr_max = tail.get("corr_max")
        bad_overlap = tail.get("bad_day_overlap")
        joint_2022 = tail.get("joint_2022_loss")
        joint_dd = tail.get("joint_max_DD")
        # 모든 ensemble-special gate 제거 (2026-04-25): 앙상블이라고 특별 조건 안 검.
        # corr_max / bad_day_overlap / joint_2022 / joint_max_DD 는 진단 컬럼으로만 유지.
        # k=1 멤버와 동일 metric (Cal/MDD/Sh/yearly...) 으로 평가.
        _ = corr_max  # noqa: F841
        _ = bad_overlap  # noqa: F841
        _ = joint_2022  # noqa: F841
        _ = joint_dd  # noqa: F841
        passes = []
        reasons = []
        gates_pass = True  # 항상 통과 — 평가는 metric 기반 ranking 으로
        # k=1 대비 개선 체크: member 개별 metrics 과 앙상블 비교
        # plan: med_Cal / p25_Cal (=MDD 대용) / MDD 중 2개 이상 strict improvement
        improve_count = 0
        improve_detail = []
        if member_results:
            best_k1_cal = max(m["Cal"] for m in member_results)
            best_k1_mdd = max(m["MDD"] for m in member_results)  # MDD 는 음수 → max = 덜 나쁜 것
            best_k1_sh = max(m["Sh"] for m in member_results)
            if ens["Cal"] > best_k1_cal:
                improve_count += 1
                improve_detail.append(f"Cal+{(ens['Cal']-best_k1_cal):.2f}")
            if ens["MDD"] > best_k1_mdd:
                improve_count += 1
                improve_detail.append(f"MDD+{(ens['MDD']-best_k1_mdd):.2%}")
            if ens["Sh"] > best_k1_sh:
                improve_count += 1
                improve_detail.append(f"Sh+{(ens['Sh']-best_k1_sh):.2f}")
        improve_pass = improve_count >= 2
        out = {
            "members": "|".join(members), "asset": asset, "k": k,
            "status": "ok",
            "Cal": ens["Cal"], "CAGR": ens["CAGR"],
            "MDD": ens["MDD"], "Sh": ens["Sh"],
            "gates_pass": gates_pass,
            "gate_fails": ",".join(reasons) if reasons else "",
            "improve_count": improve_count,
            "improve_pass": improve_pass,
            "improve_detail": "|".join(improve_detail) if improve_detail else "",
            **tail,
        }
        return out
    except Exception as e:
        return {"members": "|".join(members), "asset": asset, "k": k,
                "status": "error", "error": str(e)[:200]}


def build_tasks(asset, resume):
    cand_csv = os.path.join(HERE, f"redesign_ensemble_candidates_{asset}.csv")
    if not os.path.exists(cand_csv) or os.path.getsize(cand_csv) <= 1:
        return []
    try:
        cands = pd.read_csv(cand_csv)
    except pd.errors.EmptyDataError:
        return []
    if cands.empty or "members" not in cands.columns:
        return []
    top_csv = os.path.join(HERE, f"redesign_top500_{asset}_k1.csv")
    top = pd.read_csv(top_csv).set_index("tag")
    out_csv = os.path.join(HERE, f"redesign_ensemble_bt_{asset}.csv")
    done = status_resume_keys(out_csv, ["members"])
    tasks = []
    for _, row in cands.iterrows():
        members = str(row["members"]).split("|")
        if (tuple(["|".join(members)]),) and ("|".join(members),) in done:
            continue
        try:
            cfgs = [parse_cfg(asset, top.loc[m]) for m in members]
        except KeyError:
            continue
        if any(c is None for c in cfgs):
            continue
        tasks.append({"members": members, "asset": asset,
                      "k": int(row["k"]), "cfgs": cfgs})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--flush", type=int, default=100)
    args = ap.parse_args()

    out_csv = os.path.join(HERE, f"redesign_ensemble_bt_{args.asset}.csv")
    tasks = build_tasks(args.asset, resume=True)
    print(f"[{args.asset}] {len(tasks)} ensemble tasks", flush=True)
    if not tasks:
        return

    t0 = time.time()
    header_written = os.path.exists(out_csv)
    buf = []

    def flush_buf():
        nonlocal header_written, buf
        if not buf:
            return
        pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not header_written, index=False)
        header_written = True
        buf = []

    with Pool(args.workers, initializer=preload) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1), 1):
            buf.append(res)
            if len(buf) >= args.flush:
                flush_buf()
            if i % 100 == 0:
                rate = i / max(time.time()-t0, 1e-6)
                eta = (len(tasks)-i) / max(rate, 1e-6)
                print(f"[{args.asset} {i}/{len(tasks)}] {rate:.2f}/s ETA {eta/60:.1f}m", flush=True)
    flush_buf()
    print(f"[{args.asset} done] {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
