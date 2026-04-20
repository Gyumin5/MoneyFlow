#!/usr/bin/env python3
"""Parallel validator framework — 24 worker joblib + checkpoint + resume.

사용:
    from parallel_validator import run_grid_validator
    run_grid_validator(
        name="pullback",
        engine_fn=run_pullback,
        configs=[...],
        funding_loader=None,  # or load_funding fn
    )

체크포인트:
  out/<name>_grid_partial.csv 에 한 config씩 append.
  재실행 시 이미 done인 config_hash는 skip → resume.
"""
from __future__ import annotations
import os, sys, json, hashlib
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from common_next import (load_all, aggregate_ew_portfolio, simple_metrics,
                          TRAIN_END, HOLDOUT_START, FULL_END)

OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

N_JOBS = 24


def config_hash(cfg: dict) -> str:
    """configs dict → 결정적 hash."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _run_one_coin(coin_symbol: str, engine_fn, cfg: dict, funding_loader=None):
    """단일 코인 실행. engine_fn + cfg + optional funding."""
    from c_engine_v5 import load_coin
    df = load_coin(coin_symbol + "USDT")
    if df is None:
        return coin_symbol, None
    kwargs = dict(cfg)
    if funding_loader is not None:
        kwargs["funding"] = funding_loader(coin_symbol)
    eq, evs = engine_fn(df, **kwargs)
    return coin_symbol, eq


def _validate_one_config(args):
    cfg, name, avail, funding_loader_name = args
    from joblib import Parallel, delayed
    # dynamic import
    import importlib
    if name == "pullback":
        from engine_pullback import run_pullback as engine_fn
        fl = None
    elif name == "vbo":
        from engine_vbo import run_vbo as engine_fn
        fl = None
    elif name == "short_mom":
        from engine_short_mom import run_short_mom as engine_fn, load_funding
        fl = load_funding
    elif name == "breakdown_short":
        from engine_breakdown_short import run_breakdown_short as engine_fn
        from engine_short_mom import load_funding
        fl = load_funding
    elif name == "range_mr":
        from engine_range_mr import run_range_mr as engine_fn
        fl = None
    else:
        raise ValueError(f"Unknown strategy: {name}")

    # 코인별 순차 실행 (config 내부는 싱글 코어)
    eq_map = {}
    for c in avail:
        _, eq = _run_one_coin(c, engine_fn, cfg, fl)
        if eq is not None:
            eq_map[c] = eq

    port_eq = aggregate_ew_portfolio(eq_map)
    if len(port_eq) == 0:
        return {**cfg, "cfg_hash": config_hash(cfg), "n_coins": 0}

    # split metrics
    import pandas as _pd
    idx = _pd.to_datetime(port_eq.index)
    s = port_eq.copy(); s.index = idx
    def slc(start, end):
        sub = s[(s.index >= start) & (s.index <= end)]
        if len(sub) < 30 or sub.iloc[0] <= 0:
            return {"CAGR": 0, "MDD": 0, "Cal": 0}
        return simple_metrics(sub / sub.iloc[0])
    m_full = slc(port_eq.index[0], FULL_END)
    m_train = slc(port_eq.index[0], TRAIN_END)
    m_hout = slc(HOLDOUT_START, FULL_END)

    return {
        **cfg, "cfg_hash": config_hash(cfg),
        "n_coins": len(eq_map),
        "full_Cal": m_full.get("Cal", 0),
        "full_CAGR": m_full.get("CAGR", 0),
        "full_MDD": m_full.get("MDD", 0),
        "train_Cal": m_train.get("Cal", 0),
        "train_CAGR": m_train.get("CAGR", 0),
        "hout_Cal": m_hout.get("Cal", 0),
        "hout_CAGR": m_hout.get("CAGR", 0),
        "hout_MDD": m_hout.get("MDD", 0),
    }


def run_grid_validator(name: str, configs: list, checkpoint_every: int = 5):
    """Grid 실행. N_JOBS 병렬, 매 N개마다 checkpoint 저장, 이미 done은 skip."""
    partial_path = os.path.join(OUT_DIR, f"{name}_grid_partial.csv")
    done_hashes = set()
    if os.path.exists(partial_path):
        try:
            prev = pd.read_csv(partial_path)
            if "cfg_hash" in prev.columns:
                done_hashes = set(prev["cfg_hash"].tolist())
            print(f"[resume] {len(done_hashes)} configs already done (from {partial_path})")
        except Exception as e:
            print(f"[resume] 기존 partial 읽기 실패: {e}")

    avail, _, _, _ = load_all()

    todo = [c for c in configs if config_hash(c) not in done_hashes]
    print(f"Total configs: {len(configs)}, TODO: {len(todo)}, avail coins: {len(avail)}")

    # 병렬: 각 worker가 1 config × N 코인 순차
    args_list = [(cfg, name, avail, None) for cfg in todo]

    # 배치 처리로 checkpoint (step = batch_size로 overlap 제거)
    batch_size = max(1, checkpoint_every * N_JOBS)
    for batch_start in range(0, len(args_list), batch_size):
        batch = args_list[batch_start:batch_start + batch_size]
        if not batch:
            continue
        results = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(_validate_one_config)(a) for a in batch)
        # append to CSV
        new_df = pd.DataFrame(results)
        if os.path.exists(partial_path):
            new_df.to_csv(partial_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(partial_path, index=False)
        print(f"[checkpoint] {batch_start + len(batch)}/{len(args_list)} configs done")

    # 최종 정리
    final = pd.read_csv(partial_path)
    final_path = os.path.join(OUT_DIR, f"{name}_grid.csv")
    final.to_csv(final_path, index=False)

    print(f"\n=== {name} Top 15 by train_Cal ===")
    print(final.sort_values("train_Cal", ascending=False).head(15).to_string(index=False))
    print(f"\n=== {name} Top 10 by hout_Cal ===")
    print(final.sort_values("hout_Cal", ascending=False).head(10).to_string(index=False))
    print(f"\n저장: {final_path}")

    return final
