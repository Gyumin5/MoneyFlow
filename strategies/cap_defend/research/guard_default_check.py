"""엔진 기본값에서 전략 가드가 꺼졌는지 확인한다 — 세 갈래를 한 번에 본다.

왜 재나. 2026-11-02 에 `unified_backtest.run()` 의 가드 기본값을 껐다(토론
run-20261102T071404Z). 라이브 현물 V24·선물 V25 에 전략 가드가 없는데 엔진만
켜져 있어서, 인자를 안 넘긴 호출부가 아무 경고 없이 라이브와 다른 전략을 쟀다.
바꾼 것이 실제로 듣는지, 그리고 켜는 길이 막히지 않았는지를 같은 자료로 본다.

재는 것 셋.
  ⑴ 기본 호출 == 명시적 무가드 호출. 자산곡선이 한 점도 안 달라야 한다.
  ⑵ 명시적 가드 켬 호출은 그 둘과 달라야 한다. 안 다르면 끄기만 되고 켜기가
     죽은 것이다 — 가드 비교 실험이 통째로 무의미해진다.
  ⑶ 채택 호출부(trade/v24_shadow_today.py)와 같은 인자 묶음이 ⑴ 과 같아야 한다.
     그 파일은 가드 인자를 안 넘기므로 이번 변경으로 라이브 정합이 된다.

실행: python3 guard_default_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import unified_backtest as ub  # noqa: E402

# 채택 현물 V24 (trade/v24_shadow_today.py 45~51행과 같은 값, 가드 인자는 없다)
ADOPTED = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=7,
    universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
    snap_interval_bars=217,
)

# 가드가 실제로 발동하는 구간을 품어야 한다 — 2021-05 와 2022-05 의 급락이 들어간다.
START, END = '2020-10-01', '2023-01-01'

NOGUARD = dict(dd_lookback=0, bl_drop=0, crash_threshold=-10)
GUARD_ON = dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
                crash_threshold=-0.10)


def curve(bars, funding, extra):
    """자산곡선과 지표 넷. run() 은 `_equity` 로 곡선을 돌려준다 — `_trace` 에는
    날짜·목표·스냅샷만 있고 평가액이 없다."""
    res = ub.run(bars, funding, start_date=START, end_date=END,
                 **dict(ADOPTED, **extra))
    eq = np.asarray(res.get('_equity'), dtype=float)
    keep = {k: res.get(k) for k in ('CAGR', 'MDD', 'Cal', 'Sharpe', 'Trades', 'Rebal')}
    return eq, keep


def main():
    print(f"자료 로드 ({START} ~ {END})", flush=True)
    bars, funding = ub.load_data('D')
    print(f"  심볼 {len(bars)}개\n", flush=True)

    base, m_b = curve(bars, funding, {})
    expl, m_e = curve(bars, funding, NOGUARD)
    on, m_o = curve(bars, funding, GUARD_ON)
    print(f"봉 수 — 기본 {len(base)} · 명시 무가드 {len(expl)} · 가드 켬 {len(on)}")
    if len(base) == 0:
        print("[실패] 자산곡선이 비었다 — 비교할 수 없다.")
        return 1

    ok = True

    # ⑴ 기본 == 명시 무가드
    if len(base) != len(expl):
        print(f"[실패] ⑴ 봉 수가 다르다: {len(base)} 대 {len(expl)}")
        ok = False
    else:
        d = float(np.max(np.abs(base - expl))) if len(base) else 0.0
        rel = d / float(np.max(np.abs(base))) if len(base) and np.max(np.abs(base)) else 0.0
        print(f"⑴ 기본 대 명시 무가드 — 최대 절대차 {d:.6e} (상대 {rel:.3e})")
        if rel > 1e-12:
            print("   [실패] 기본값이 아직 가드를 켜고 있다.")
            ok = False
        else:
            print("   [통과] 같다 — 기본값이 꺼졌다.")

    # ⑵ 가드 켬은 달라야 한다
    if len(base) == len(on):
        d2 = float(np.max(np.abs(base - on)))
        rel2 = d2 / float(np.max(np.abs(base))) if np.max(np.abs(base)) else 0.0
        print(f"⑵ 기본 대 가드 켬 — 최대 절대차 {d2:.6e} (상대 {rel2:.3e})")
        if rel2 < 1e-9:
            print("   [실패] 가드를 켜도 안 달라진다 — 켜는 길이 죽었다.")
            ok = False
        else:
            print("   [통과] 다르다 — 명시로 켜면 가드가 산다.")
    else:
        print(f"⑵ 봉 수가 달라 비교 생략: {len(base)} 대 {len(on)}")

    # ⑶ 채택 호출부 인자 묶음 == ⑴
    #    ADOPTED 자체가 그 묶음이라 base 가 곧 채택 호출부의 결과다.
    print("⑶ 채택 호출부(trade/v24_shadow_today.py)는 가드 인자를 안 넘긴다"
          " → 위 ⑴ 의 '기본' 쪽이 그 결과다.")

    print("\n지표 넷 (같은 자료·같은 구간)")
    for name, m in (('기본', m_b), ('명시 무가드', m_e), ('가드 켬', m_o)):
        print(f"   {name:<12} 칼마 {m['Cal']:.3f} · 연복리 {m['CAGR']:.4f}"
              f" · 낙폭 {m['MDD']:.4f} · 샤프 {m['Sharpe']:.3f}"
              f" · 매매 {m['Trades']} · 리밸 {m['Rebal']}")

    print("\n판정:", "통과" if ok else "실패")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
