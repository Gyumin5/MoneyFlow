"""Shared helpers for redesign pipeline.

함수:
  - parse_cfg(asset, row): CSV row → cfg dict
  - is_prime_x3_snap(snap): snap 값이 prime×3 (nice number 아님) 인가
  - run_bt(asset, cfg, phase_offset, snap_override, start, end, tx_mult, exec_delay):
      통합 백테스트 wrapper. status/metrics 반환
  - status_resume_keys(path, key_cols): 출력 csv 에서 status==ok 인 rows 의 key 집합
"""
from __future__ import annotations
import os
import sys
import math
from typing import Dict, Any, Optional, Tuple, Iterable

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
CAP = os.path.dirname(HERE)
REPO = os.path.dirname(CAP)
sys.path.insert(0, REPO)
sys.path.insert(0, CAP)

ANCHOR = "2020-10-01"
END = "2026-04-13"
ANCHOR_STOCK = "2017-04-01"
END_STOCK = "2025-12-31"

TX_BY_ASSET = {"fut": 0.0004, "spot": 0.004, "stock": 0.0025}

# Nice numbers (snap 에 흔히 쓰이는 달력 친화적 값). soft penalty 대상.
NICE_SNAPS = {7, 14, 21, 28, 30, 60, 90, 120, 150, 180, 210, 270, 360, 540, 720,
              6, 12, 18, 24, 48, 72, 96, 144, 192, 240, 288, 336, 432, 480}


def is_prime_x3_snap(snap: int) -> bool:
    """3 × p (p 소수 ≥ 7) 형태. NICE_SNAPS 제외.
    - 3×5=15 는 5 가 소수지만 nice (3-5 월의 약수 흔함) → 제외
    - 3×7=21 부터 prime×3 로 인정
    """
    if snap in NICE_SNAPS or snap <= 18:
        return False
    if snap % 3 != 0:
        return False
    q = snap // 3
    if q < 7:
        return False
    for i in range(2, int(math.isqrt(q)) + 1):
        if q % i == 0:
            return False
    return True


def generate_prime_x3_list(max_snap: int = 500) -> list:
    """is_prime_x3_snap() 기준으로 최대값까지 prime×3 리스트 생성.
    snap_nudge.PRIME_X3 와 단일 소스로 쓰기 위함.
    """
    return [n for n in range(21, max_snap + 1, 3) if is_prime_x3_snap(n)]


def parse_cfg(asset: str, row: pd.Series) -> Optional[Dict[str, Any]]:
    """CSV row → cfg dict. 컬럼 우선 (tag 파싱 fallback).
    필수 컬럼:
      fut/spot: sma, ms, ml, vmode, vthr, snap, iv, (lev for fut)
      stock: snap_days, canary_sma, canary_hyst, canary_type, select, def_mom_period, health
    """
    if asset == "stock":
        required = ["snap_days", "canary_sma", "canary_hyst", "canary_type",
                    "select", "def_mom_period", "health"]
        if any(c not in row or pd.isna(row[c]) for c in required):
            return None
        cfg = {
            "iv": "D",
            "snap": int(row["snap_days"]),
            "canary_sma": int(row["canary_sma"]),
            "canary_hyst": float(row["canary_hyst"]),
            "canary_type": str(row["canary_type"]),
            "select": str(row["select"]),
            "def_mom": int(row["def_mom_period"]),
            "health": str(row["health"]),
        }
        # 신규 grid v3 옵션 (있으면 추가, 없으면 adapter 기본 사용)
        if "sharpe_lookback" in row and pd.notna(row["sharpe_lookback"]):
            cfg["sharpe_lookback"] = int(row["sharpe_lookback"])
        if "mom_style" in row and pd.notna(row["mom_style"]):
            cfg["mom_style"] = str(row["mom_style"])
        if "n_pick" in row and pd.notna(row["n_pick"]):
            cfg["n_pick"] = int(row["n_pick"])
        return cfg
    # fut/spot
    required = ["sma", "ms", "ml", "vthr", "snap"]
    if any(c not in row or pd.isna(row[c]) for c in required):
        return None
    iv = row["iv"] if "iv" in row and pd.notna(row["iv"]) else (
        "D" if "_1D_" in str(row.get("tag", "")) else (
            "4h" if "_4h_" in str(row.get("tag", "")) else None))
    vmode = row["vmode"] if "vmode" in row and pd.notna(row["vmode"]) else (
        "daily" if "_d" in str(row.get("tag", "")) else "bar")
    cfg = {
        "iv": iv,
        "sma": int(row["sma"]),
        "ms": int(row["ms"]),
        "ml": int(row["ml"]),
        "vmode": str(vmode),
        "vthr": float(row["vthr"]),
        "snap": int(row["snap"]),
        "lev": float(row.get("lev", 1.0)) if asset == "fut" else 1.0,
    }
    if cfg["iv"] is None:
        return None
    return cfg


def anchor_end_for(asset: str, year: Optional[int] = None) -> Tuple[str, str]:
    if year is not None:
        return f"{year}-01-01", f"{year}-12-31"
    if asset == "stock":
        return ANCHOR_STOCK, END_STOCK
    return ANCHOR, END


def run_bt(
    asset: str,
    cfg: Dict[str, Any],
    *,
    bars_funding: Optional[Dict[str, Any]] = None,
    phase_offset: int = 0,
    snap_override: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tx_mult: float = 1.0,
    exec_delay_bars: int = 0,
    drop_top_contributor: bool = False,
    with_equity: bool = False,
    with_trace: bool = False,
) -> Dict[str, Any]:
    """통합 BT wrapper. 반환 {status, Cal, CAGR, MDD, Sh, rebal?, error?}."""
    tx = TX_BY_ASSET[asset] * tx_mult
    _start, _end = anchor_end_for(asset, None)
    if start:
        _start = start
    if end:
        _end = end
    snap = snap_override if snap_override is not None else cfg["snap"]

    try:
        if asset == "stock":
            from redesign_stock_adapter import run_stock_from_cfg
            year = None
            if start and end and start[:4] == end[:4] and start.endswith("01-01") and end.endswith("12-31"):
                year = int(start[:4])
            if drop_top_contributor:
                # 2-pass: baseline trace → 기간가중 누적 weight top asset → exclude 재실행
                # (Codex r10 fix: 리밸 이벤트 합이 아닌 weight × duration 으로 측정)
                r0 = run_stock_from_cfg(cfg, phase_offset=phase_offset,
                                          tx_cost=tx, start=start or ANCHOR_STOCK,
                                          end=end or END_STOCK, year_only=year,
                                          execution_delay_bars=int(exec_delay_bars or 0),
                                          with_trace=True)
                if r0.get("status") != "ok":
                    return r0
                trace = r0.get("_trace") or []
                cum = {}
                for i, row in enumerate(trace):
                    tgt = row.get("target") or {}
                    # duration = 이 리밸부터 다음 리밸까지 일수 (마지막은 1일로)
                    if i + 1 < len(trace):
                        dur = max(1, (trace[i + 1]["Date"] - row["Date"]).days)
                    else:
                        dur = 1
                    for k, v in tgt.items():
                        if k.lower() == "cash":
                            continue
                        cum[k] = cum.get(k, 0.0) + float(v) * dur
                if not cum:
                    return {"status": "error", "error": "drop_top: no non-cash"}
                top_asset = max(cum, key=cum.get)
                r = run_stock_from_cfg(cfg, phase_offset=phase_offset,
                                         tx_cost=tx, start=start or ANCHOR_STOCK,
                                         end=end or END_STOCK, year_only=year,
                                         execution_delay_bars=int(exec_delay_bars or 0),
                                         exclude_assets=frozenset({top_asset}))
                if r.get("status") == "ok":
                    r["_dropped_asset"] = top_asset
            else:
                r = run_stock_from_cfg(cfg, phase_offset=phase_offset,
                                           tx_cost=tx, start=start or ANCHOR_STOCK,
                                           end=end or END_STOCK, year_only=year,
                                           execution_delay_bars=int(exec_delay_bars or 0),
                                           with_trace=with_trace)
            if not with_equity and "_equity" in r:
                r = {k: v for k, v in r.items() if k != "_equity"}
            if not with_trace and "_trace" in r:
                r = {k: v for k, v in r.items() if k != "_trace"}
            return r
        from unified_backtest import run as bt_run
        if bars_funding is None:
            from unified_backtest import load_data
            bars_funding = {cfg["iv"]: load_data(cfg["iv"])}
        bars, funding = bars_funding[cfg["iv"]]
        lev = 1.0 if asset == "spot" else cfg["lev"]

        common_kwargs = dict(
            interval=cfg["iv"], asset_type=asset,
            leverage=lev, universe_size=3, cap=1 / 3, tx_cost=tx,
            sma_bars=cfg["sma"], mom_short_bars=cfg["ms"], mom_long_bars=cfg["ml"],
            vol_mode=cfg["vmode"], vol_threshold=cfg["vthr"],
            snap_interval_bars=snap, n_snapshots=3,
            phase_offset_bars=phase_offset,
            canary_hyst=0.015, health_mode="mom2vol",
            stop_kind="none", stop_pct=0.0,
            drift_threshold=0.10, post_flip_delay=5,
            dd_lookback=60, dd_threshold=-0.25,
            bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
            start_date=_start, end_date=_end,
            execution_delay_bars=int(exec_delay_bars or 0),
        )

        # with_trace 지원: 비-drop_top, 비-stock 경로에서 trace 추출
        trace_buf = [] if (with_trace and not drop_top_contributor) else None
        if trace_buf is not None:
            common_kwargs["_trace"] = trace_buf

        if drop_top_contributor:
            # 2-pass: 1) baseline trace 수집 → 누적 weight × bars 1위 자산 식별
            #         2) 해당 자산 exclude 로 재실행
            trace = []
            m_base = bt_run(bars, funding, _trace=trace, **common_kwargs)
            top_asset = None
            if trace:
                # 자산별 누적 weight (= 보유 비중 × bars 수, 비현금)
                cum = {}
                for row in trace:
                    tgt = row.get("target", {}) or {}
                    for k, v in tgt.items():
                        if k.upper() == "CASH":
                            continue
                        cum[k] = cum.get(k, 0.0) + float(v)
                if cum:
                    top_asset = max(cum, key=cum.get)
            if top_asset is None:
                return {"status": "error",
                        "error": "drop_top: no non-cash holdings in trace"}
            common_kwargs["exclude_assets"] = frozenset({top_asset})
            m = bt_run(bars, funding, **common_kwargs)
            m["_dropped_asset"] = top_asset
        else:
            m = bt_run(bars, funding, **common_kwargs)
        out = {
            "status": "ok",
            "Sh": float(m.get("Sharpe", 0) or 0),
            "Cal": float(m.get("Cal") or 0),
            "CAGR": float(m.get("CAGR", 0) or 0),
            "MDD": float(m.get("MDD", 0) or 0),
            "rebal": int(m.get("Rebal", 0) or 0),
        }
        if with_equity and "_equity" in m:
            out["_equity"] = m["_equity"]
        if "_dropped_asset" in m:
            out["_dropped_asset"] = m["_dropped_asset"]
        if with_trace and trace_buf is not None:
            out["_trace"] = trace_buf
        return out
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def status_resume_keys(path: str, key_cols: Iterable[str]) -> set:
    """출력 csv 에서 status==ok 인 rows 의 key 집합 반환. error row 는 retry.
    - status 없는 legacy CSV: error 컬럼 비어있는 row 만 done (codex review fix)
    - empty/parse error CSV: 빈 set 반환 (gemini review fix)
    - critical metric NaN row 제외 (Cal/MDD)
    """
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return set()
    if df.empty:
        return set()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    elif "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    if "Cal" in df.columns:
        df = df.dropna(subset=["Cal"])
    if "MDD" in df.columns:
        df = df.dropna(subset=["MDD"])
    key_cols = list(key_cols)
    if any(col not in df.columns for col in key_cols):
        return set()
    return set(tuple(row[col] for col in key_cols) for _, row in df[key_cols].iterrows())
