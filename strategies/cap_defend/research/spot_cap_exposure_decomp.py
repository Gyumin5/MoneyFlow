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

1차 판정이 고른 구성은 종목 2개 · 상한 50%(n2-c50)이고 현행은 종목 3개 · 상한
1/3(n3-c33)이다. 두 축이 같이 갈렸으므로 어느 쪽이 우세를 만들었는지가 안 보인다.
그래서 2×2 요인으로 넷을 다 돌린다 — n3-c33 · n3-c50 · n2-c33 · n2-c50.
  · 상한 효과 = 같은 종목 수에서 c33 → c50 의 차이
  · 종목 수 효과 = 같은 상한에서 n3 → n2 의 차이
  · 둘이 더해서 전체 차이를 못 메우면 그만큼이 상호작용이다
종목 수를 줄이면 그 자체로 노출이 바뀐다 — 선정이 둘로 제한되면 상한 1/3 에서
최대 67% 라 늘 현금을 3분의 1 든다. 상한 50% 가 그 자리를 100% 로 되돌린다.
즉 n2-c50 은 "종목 둘에 몰아서 전액 투자" 이고, 그게 선정 개선인지 노출 증가인지를
가르는 것이 이 측정의 목적이다.

여기서 재는 것 셋.
  ⑴ 스냅샷 합성 목표의 투자 비중 분포. 현금을 얼마나, 얼마나 자주 드나.
  ⑵ 네 구성의 투자 비중 차이가 실제로 어느 구간에서 나나.
  ⑶ 같은 칸의 성과(칼마·연복리·낙폭·샤프). 노출 차이와 성과 차이를 붙여 읽는다.
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

# 2×2 요인. 이름은 교차 격자와 같은 형식이라 그쪽 표와 바로 맞춰 읽는다.
ARMS = [
    ('n3-c33', 3, 1 / 3),   # 현행
    ('n3-c50', 3, 0.50),    # 상한만 바꿈
    ('n2-c33', 2, 1 / 3),   # 종목 수만 바꿈
    ('n2-c50', 2, 0.50),    # 1차 판정 통과 구성
]


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
    for arm, nsel, cval in ARMS:
        for ph in PHASES:
            tr = []
            kw = dict(BASE, universe_size=nsel, cap=cval, phase_offset_bars=ph)
            r = ub.run(bars, fund, start_date=START, end_date=END, _trace=tr, **kw)
            p = exposure_profile(tr)
            p.update(arm=arm, nsel=nsel, phase=ph,
                     cal=float(r.get('Cal', 0)), cagr=float(r.get('CAGR', 0)),
                     mdd=float(r.get('MDD', 0)), sharpe=float(r.get('Sharpe', 0)))
            rows.append(p)
            print(f"  {arm} 위상 {ph}: 평균투자 {p['mean_inv']:.4f} "
                  f"(들어간 봉만 {p['mean_inv_when_in']:.4f}) "
                  f"전액현금 {p['cash_only_pct']:.1f}% 100%투자 {p['full_pct']:.1f}% "
                  f"칼마 {p['cal']:.3f}", flush=True)
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
            'two_third_pct', 'p50', 'cal', 'cagr', 'mdd', 'sharpe')
    agg = {}
    for arm, _n, _c in ARMS:
        g = [r for r in rows if r.get('arm') == arm]
        if g:
            agg[arm] = {k: float(np.mean([r.get(k, 0) for r in g])) for k in keys}

    print("\n노출 (앵커 7곳 평균)")
    print("  구성      평균투자  들어간봉만  전액현금  100%투자   67%대   중앙값")
    for arm in agg:
        a = agg[arm]
        print(f"  {arm}  {a['mean_inv']:9.4f} {a['mean_inv_when_in']:10.4f} "
              f"{a['cash_only_pct']:8.1f}% {a['full_pct']:8.1f}% "
              f"{a['two_third_pct']:7.1f}% {a['p50']:8.4f}")

    print("\n성과 (앵커 7곳 평균)")
    print("  구성         칼마     연복리      낙폭      샤프   노출당수익")
    for arm in agg:
        a = agg[arm]
        # 노출당 수익 = 연복리 / 평균투자. 노출만 올려서 번 수익은 이 값이 안 오른다.
        # 두 구성의 노출이 다를 때 "더 잘 골랐나" 를 가르는 자리다.
        eff = a['cagr'] / max(a['mean_inv'], 1e-9)
        print(f"  {arm}  {a['cal']:9.3f} {a['cagr']:9.4f} {a['mdd']:9.4f} "
              f"{a['sharpe']:9.3f} {eff:9.3f}")

    need = {'n3-c33', 'n3-c50', 'n2-c33', 'n2-c50'}
    if not need <= set(agg):
        return
    inv = {a: agg[a]['mean_inv'] for a in need}
    cal = {a: agg[a]['cal'] for a in need}
    cagr = {a: agg[a]['cagr'] for a in need}

    print("\n분해 — 전체 차이를 두 축으로 가른다 (n3-c33 → n2-c50)")
    tot_i = inv['n2-c50'] - inv['n3-c33']
    tot_c = cal['n2-c50'] - cal['n3-c33']
    tot_g = cagr['n2-c50'] - cagr['n3-c33']
    print(f"  전체        평균투자 {tot_i:+.4f}  칼마 {tot_c:+.3f}  연복리 {tot_g:+.4f}")
    for lab, a, b in (("상한만(n3)", 'n3-c33', 'n3-c50'),
                      ("종목수만(c33)", 'n3-c33', 'n2-c33'),
                      ("상한만(n2)", 'n2-c33', 'n2-c50'),
                      ("종목수만(c50)", 'n3-c50', 'n2-c50')):
        print(f"  {lab:<13} 평균투자 {inv[b]-inv[a]:+.4f}  "
              f"칼마 {cal[b]-cal[a]:+.3f}  연복리 {cagr[b]-cagr[a]:+.4f}")
    add = (cal['n3-c50'] - cal['n3-c33']) + (cal['n2-c33'] - cal['n3-c33'])
    print(f"  두 축 단순합 칼마 {add:+.3f} · 전체 {tot_c:+.3f} → "
          f"상호작용 {tot_c - add:+.3f}")

    eff = {a: cagr[a] / max(inv[a], 1e-9) for a in need}
    print(f"\n  노출당 수익  상한만(n3) {eff['n3-c50']-eff['n3-c33']:+.3f} · "
          f"종목수만(c33) {eff['n2-c33']-eff['n3-c33']:+.3f}")
    print("  이 줄이 두 축의 성격을 가른다. 노출을 올려서 번 수익은 분자와 분모가")
    print("  같이 커지므로 이 값이 안 오른다. 오르면 같은 노출로 더 번 것이다.")

    print("\n  읽는 법. 상한 0.5 는 선정 수가 셋이면 1/3 과 같은 비중을 내므로, 그")
    print("  차이는 선정이 둘 이하인 봉에서만 생긴다. 종목 수를 둘로 줄이면 그 봉이")
    print("  훨씬 잦아져 상한의 효과가 커진다 — 두 축이 곱해지는 자리다.")
    print("  성과 개선이 평균투자 증가로 설명되면 '종목을 더 잘 고른다' 가 아니라")
    print("  '현금을 덜 든다' 는 정책이다. 둘은 위험 성격이 다르다 — 노출을 올려 얻은")
    print("  수익은 하락장에서 같은 배수로 돌려준다.")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
        raise SystemExit(0)
    raise SystemExit(main())
