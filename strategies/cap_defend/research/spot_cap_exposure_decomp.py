"""상한 0.5 가 무엇을 바꾸나 — 선정 수별 노출 분해.

토론(run-20261102T022434Z) action_step 4: "상한 변경은 실제 선정 수 g에 대해 목표
노출 g×min(1/g, cap)을 바꾸므로, 상한 0.5의 직접 효과는 선정 수가 둘 이하일 때
발생한다. 이는 정당한 노출 정책일 수 있지만 선정 개선과 구별해야 한다."

엔진에서 확인한 것(`unified_backtest._compute_weights` 끝):
    w = min(1.0 / len(picks), cap)
그래서 선정 수 g 일 때 투자 비중은 g × min(1/g, cap) 이고,
  g=3: min(1/3, 0.5) = 1/3 → 100%. 상한 1/3 과 글자 그대로 같다.
  g=2: 1/3 이면 67%, 0.5 면 100%.
  g=1: 1/3 이면 33%, 0.5 면 50%.
즉 상한 0.5 는 종목을 더 잘 고르는 게 아니라 "고른 게 적을 때 현금을 덜 든다" 는
노출 정책이다. 선정 수가 자주 셋 미만인 이유는 greedy 흡수다 — 앞 코인의 단기
모멘텀이 뒤 코인 이상이면 뒤를 떨어뜨린다.

여기서 재는 것 둘.
  ⑴ 스냅샷 합성 목표의 투자 비중 분포. 현금을 얼마나, 얼마나 자주 드나.
  ⑵ 두 상한의 투자 비중 차이가 실제로 어느 구간에서 나나.
합성 목표는 스냅샷 일곱의 평균이라 선정 수가 정수로 안 떨어진다. 그래서 선정 수를
직접 세는 대신 투자 비중(1 - 현금)으로 본다 — 상한이 바꾸는 것이 바로 그 값이다.

실행: python3 spot_cap_exposure_decomp.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import unified_backtest as ub  # noqa: E402

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-cap-exposure')
START = '2017-01-01'
END = '2026-11-01'
PHASES = [0, 4, 8, 12, 16, 20, 24]

BASE = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=7, universe_size=3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
    snap_interval_bars=217,
)


def exposure_profile(trace):
    """봉마다 투자 비중(1 - 현금)을 모은다. 카나리 OFF(전액 현금) 봉은 따로 센다."""
    inv = []
    for t in trace:
        tgt = t.get('target') or {}
        cash = float(tgt.get('CASH', tgt.get('Cash', 0.0)))
        inv.append(1.0 - cash)
    v = np.asarray(inv, dtype=float)
    on = v[v > 1e-9]          # 들어가 있는 봉만
    return dict(
        bars=int(len(v)),
        in_bars=int(len(on)),
        cash_only_pct=float(100.0 * (len(v) - len(on)) / len(v)) if len(v) else 0.0,
        mean_inv=float(np.mean(v)) if len(v) else 0.0,
        mean_inv_when_in=float(np.mean(on)) if len(on) else 0.0,
        # 들어가 있는 봉 중 투자 비중이 거의 100% 인 비율 = 선정 수가 상한을 안 물린 자리
        full_pct=float(100.0 * np.mean(on > 0.999)) if len(on) else 0.0,
        # 67% 근처 = 선정 둘 × 상한 1/3
        two_third_pct=float(100.0 * np.mean((on > 0.66) & (on < 0.68))) if len(on) else 0.0,
        p10=float(np.percentile(on, 10)) if len(on) else 0.0,
        p50=float(np.percentile(on, 50)) if len(on) else 0.0,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    rows = []
    for cname, cval in (('c33', 1 / 3), ('c50', 0.50)):
        for ph in PHASES:
            tr = []
            kw = dict(BASE, cap=cval, phase_offset_bars=ph)
            ub.run(bars, fund, start_date=START, end_date=END, _trace=tr, **kw)
            p = exposure_profile(tr)
            p.update(cap=cname, phase=ph)
            rows.append(p)
            print(f"  {cname} 위상 {ph}: 평균투자 {p['mean_inv']:.4f} "
                  f"(들어간 봉만 {p['mean_inv_when_in']:.4f}) "
                  f"전액현금 {p['cash_only_pct']:.1f}% 100%투자 {p['full_pct']:.1f}%",
                  flush=True)
    with open(os.path.join(OUT, 'rows.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary(rows)
    return 0


def summary(rows=None):
    if rows is None:
        rows = [json.loads(l) for l in
                open(os.path.join(OUT, 'rows.jsonl'), encoding='utf-8')]
    print("\n상한별 노출 (앵커 7곳 평균)")
    print("  상한   평균투자  들어간봉만  전액현금   100%투자   67%대   중앙값")
    keys = ('mean_inv', 'mean_inv_when_in', 'cash_only_pct', 'full_pct',
            'two_third_pct', 'p50')
    agg = {}
    for c in ('c33', 'c50'):
        g = [r for r in rows if r['cap'] == c]
        if not g:
            continue
        a = {k: float(np.mean([r[k] for r in g])) for k in keys}
        agg[c] = a
        print(f"  {c}  {a['mean_inv']:9.4f} {a['mean_inv_when_in']:10.4f} "
              f"{a['cash_only_pct']:9.1f}% {a['full_pct']:9.1f}% "
              f"{a['two_third_pct']:7.1f}% {a['p50']:8.4f}")
    if len(agg) == 2:
        d = agg['c50']['mean_inv'] - agg['c33']['mean_inv']
        print(f"\n  평균 투자 비중 차이 {d:+.4f} "
              f"({100*d/max(agg['c33']['mean_inv'], 1e-9):+.1f}%)")
        print("  읽는 법. 이 차이가 상한 0.5 의 직접 효과 전부다. 선정 수가 셋이면 두")
        print("  상한이 같은 비중을 내므로, 차이는 선정이 둘 이하인 봉에서만 생긴다.")
        print("  성과 개선이 이 노출 증가로 설명되면 '종목을 더 잘 고른다' 가 아니라")
        print("  '현금을 덜 든다' 는 정책이다 — 둘은 위험 성격이 다르다.")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
        raise SystemExit(0)
    raise SystemExit(main())
