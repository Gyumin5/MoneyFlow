"""선물에서 종목 수와 상한이 무엇을 바꾸나 — 현물과 같은 2x2 분해.

현물에서 갈린 것(history #118): 종목 2개는 선정 효율 개선이고(노출당 수익 +15%),
상한 50% 는 노출 정책이다(노출당 수익 제자리). 운영자 지시로 선물도 같은 방법으로
잰다.

선물이 현물과 다른 자리 셋.
  ⑴ 레버리지가 이미 노출 축이다. V25 는 코인마다 L = min(BTC_cap, K2) 로 2~4배를
     오간다. 상한을 올리는 것과 레버리지를 올리는 것이 같은 일을 하므로, 이 측정에서는
     레버리지를 V25 라이브 그대로 고정하고 종목 수와 상한만 움직인다. 레버리지 축은
     따로 묶어 본다 — 한 격자에 섞으면 어느 쪽이 효과를 냈는지 안 갈린다.
  ⑵ 앵커가 다르다. 선물은 스냅샷 5개 · 간격 95봉 · 스태거 19 라 위상을 0~18 에서
     일곱 곳 고르게 잡는다(현물은 스태거 31 이라 0~24 였다).
  ⑶ 거래비용이 0.06% 다(현물 0.4%).

레버리지는 build_K2_signal 로 만들어 dict 로 넘긴다. 이 helper 를 안 쓰고 숫자만
넘기면 고정 배수가 되어 V25 가 아니다 — 2026-11-02 에 한 칸을 그렇게 잘못 재서
낙폭 -69.45% 를 얻었다.

재는 것은 현물판과 같다. 투자 비중(1 - 현금) 분포, 네 구성의 성과, 노출당 수익
(연복리/평균투자). 노출만 올려서 번 수익은 분자와 분모가 같이 커져 마지막 값이
안 오른다.

실행: python3 fut_cap_exposure_decomp.py
요약만: python3 fut_cap_exposure_decomp.py --summary
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

import backtest_futures_v25 as f25  # noqa: E402

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-fut-cap-exposure')
START = '2017-01-01'
END = '2026-11-01'
# 선물 스태거 19 안쪽을 일곱 곳 고르게.
PHASES = [0, 3, 6, 9, 12, 15, 18]

# V25 라이브 스펙. CLAUDE.md "선물 전략 규칙 (V25)" 과 같은 값이다.
BASE = dict(
    interval='D',
    sma_days=42, mom_short_days=18, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=5, snap_interval_bars=95, drift_threshold=0.015,
    post_flip_delay=5, health_mode='mom2vol', vol_mode='daily',
    tx_cost=0.0006, maint_rate=0.004,
)

# 현물과 같은 2x2 + 종목 수 축이 봉우리인지 보는 세 칸.
ARMS = [
    ('n3-c33', 3, 1 / 3),   # 현행 V25
    ('n3-c50', 3, 0.50),
    ('n2-c33', 2, 1 / 3),
    ('n2-c50', 2, 0.50),
    ('n1-c50', 1, 0.50),
    ('n4-c50', 4, 0.50),
    ('n5-c50', 5, 0.50),
]


def exposure_profile(trace):
    """봉마다 투자 비중(1 - 현금)을 모은다. 카나리 OFF(전액 현금) 봉은 따로 센다.

    선물이라도 target 은 자본 대비 비중이다 — 레버리지는 그 비중에 곱해져 명목이
    되지 평가액 비중 자체를 바꾸지 않는다. 그래서 현물과 같은 읽기가 선다.
    """
    inv = []
    for t in trace:
        tgt = t.get('target') or {}
        cash = float(tgt.get('CASH', tgt.get('Cash', 0.0)))
        inv.append(1.0 - cash)
    v = np.asarray(inv, dtype=float)
    on = v[v > 1e-9]
    return dict(
        bars=int(len(v)),
        in_bars=int(len(on)),
        cash_only_pct=float(100.0 * (len(v) - len(on)) / len(v)) if len(v) else 0.0,
        mean_inv=float(np.mean(v)) if len(v) else 0.0,
        mean_inv_when_in=float(np.mean(on)) if len(on) else 0.0,
        full_pct=float(100.0 * np.mean(on > 0.999)) if len(on) else 0.0,
        two_third_pct=float(100.0 * np.mean((on > 0.66) & (on < 0.68))) if len(on) else 0.0,
        p10=float(np.percentile(on, 10)) if len(on) else 0.0,
        p50=float(np.percentile(on, 50)) if len(on) else 0.0,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    from long_history import load_extended
    bars, fund = load_extended(verbose=False)
    # V25 동적 per-coin L. 숫자를 넘기면 고정 배수가 되어 라이브가 아니다.
    lev = f25.build_K2_signal(bars)

    rows = []
    for arm, nsel, cval in ARMS:
        for ph in PHASES:
            tr = []
            kw = dict(BASE, universe_size=nsel, cap=cval, phase_offset_bars=ph)
            r = f25.run(bars, fund, leverage=lev, start_date=START, end_date=END,
                        _trace=tr, **kw)
            p = exposure_profile(tr)
            p.update(arm=arm, nsel=nsel, phase=ph,
                     cal=float(r.get('Cal', 0)), cagr=float(r.get('CAGR', 0)),
                     mdd=float(r.get('MDD', 0)), sharpe=float(r.get('Sharpe', 0)),
                     liq=int(r.get('Liq', 0)))
            rows.append(p)
            print(f"  {arm} 위상 {ph}: 평균투자 {p['mean_inv']:.4f} "
                  f"전액현금 {p['cash_only_pct']:.1f}% 칼마 {p['cal']:.3f} "
                  f"연복리 {p['cagr']:.4f} 낙폭 {p['mdd']:.4f} 청산 {p['liq']}",
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
    keys = ('mean_inv', 'mean_inv_when_in', 'cash_only_pct', 'full_pct',
            'p50', 'cal', 'cagr', 'mdd', 'sharpe', 'liq')
    agg = {}
    for arm, _n, _c in ARMS:
        g = [r for r in rows if r.get('arm') == arm]
        if g:
            agg[arm] = {k: float(np.mean([r.get(k, 0) for r in g])) for k in keys}

    print("\n노출 (앵커 7곳 평균)")
    print("  구성      평균투자  들어간봉만  전액현금  100%투자   중앙값")
    for arm in agg:
        a = agg[arm]
        print(f"  {arm}  {a['mean_inv']:9.4f} {a['mean_inv_when_in']:10.4f} "
              f"{a['cash_only_pct']:8.1f}% {a['full_pct']:8.1f}% {a['p50']:8.4f}")

    print("\n성과 (앵커 7곳 평균)")
    print("  구성         칼마     연복리      낙폭      샤프   노출당수익   청산")
    for arm in agg:
        a = agg[arm]
        eff = a['cagr'] / max(a['mean_inv'], 1e-9)
        print(f"  {arm}  {a['cal']:9.3f} {a['cagr']:9.4f} {a['mdd']:9.4f} "
              f"{a['sharpe']:9.3f} {eff:9.3f} {a['liq']:7.1f}")

    need = {'n3-c33', 'n3-c50', 'n2-c33', 'n2-c50'}
    if not need <= set(agg):
        return
    inv = {a: agg[a]['mean_inv'] for a in need}
    cal = {a: agg[a]['cal'] for a in need}
    cagr = {a: agg[a]['cagr'] for a in need}

    print("\n분해 — 전체 차이를 두 축으로 가른다 (n3-c33 → n2-c50)")
    print(f"  전체        평균투자 {inv['n2-c50']-inv['n3-c33']:+.4f}  "
          f"칼마 {cal['n2-c50']-cal['n3-c33']:+.3f}  "
          f"연복리 {cagr['n2-c50']-cagr['n3-c33']:+.4f}")
    for lab, a, b in (("상한만(n3)", 'n3-c33', 'n3-c50'),
                      ("종목수만(c33)", 'n3-c33', 'n2-c33'),
                      ("상한만(n2)", 'n2-c33', 'n2-c50'),
                      ("종목수만(c50)", 'n3-c50', 'n2-c50')):
        print(f"  {lab:<13} 평균투자 {inv[b]-inv[a]:+.4f}  "
              f"칼마 {cal[b]-cal[a]:+.3f}  연복리 {cagr[b]-cagr[a]:+.4f}")

    eff = {a: cagr[a] / max(inv[a], 1e-9) for a in need}
    print(f"\n  노출당 수익  상한만(n3) {eff['n3-c50']-eff['n3-c33']:+.3f} · "
          f"종목수만(c33) {eff['n2-c33']-eff['n3-c33']:+.3f}")
    print("  오르면 같은 노출로 더 번 것이고, 제자리면 현금을 덜 든 것뿐이다.")
    print("  청산 칸도 같이 본다 — 선물은 노출을 올리면 청산이 따라 늘 수 있고,")
    print("  그건 칼마가 못 담는 위험이다.")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
        raise SystemExit(0)
    raise SystemExit(main())
