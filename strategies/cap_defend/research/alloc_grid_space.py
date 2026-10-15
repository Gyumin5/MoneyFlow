"""토론이 동결한 후보 문법을 실제 후보 표로 펼친다 — 성과는 일절 안 본다.

토론(run-20261015T005408Z) action_steps 2~7 이 명목 <금액>개를 적고 "이 수는 고유
전략 수가 아니다. 최종 고유 수와 제외 목록은 실행 담당자가 생성 결과로 확정한다" 로
넘겼다. 여기서 확정한다.

합치는 근거는 둘뿐이고 둘 다 성과를 보기 전에 정해진 것이다.
  ① 전이 신호 E 중 발화일 집합이 글자 그대로 같은 것 (alloc_trigger_space 가 이미 합침)
  ② 한 번도 발화하지 않는 E — 그 E 를 쓰는 후보는 결합에 따라 대조군 중 하나와 같아진다.
     버리되 무발화 진단으로 번호를 남긴다 (action_steps 7).
기간·방향·범위 축은 서로 다른 전략이라 합치지 않는다. W 11 과 R 21 은 이미 중복 없는
열거다 (0 하나 + 양수 5종 × 방향 2 [× 시작점 2]).

제외하지 않는 것: 혼합 방향, T1 제거, 긴 대기. "이미 알려진 결과" 로 미리 빼지 않는다.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alloc_ledger_sim import SPAN_MONTH  # noqa: E402

N_COMBO = 6
BANDS_N = 3
TSCOPE_N = 2
SPANS_POS = (10, 21, 42, 63, SPAN_MONTH)     # 양수 기간 다섯 (영업일 넷 + 달력 한 달)

# 대조군 — 번호는 음수로 둬서 후보 번호와 안 섞이게 한다
CONTROLS = (
    ("현행", dict(combo=0, e_idx=-1)),                 # C 그대로
    ("송금 없음", dict(combo=3, e_idx=-1)),            # E 단독 + E 없음 = 영영 발화 없음
    ("T1만", dict(combo=1, e_idx=-1)),
    ("U만", dict(combo=2, e_idx=-1)),
    ("월 고정일", dict(combo=3, e_idx=-1, fixed='M')),
    ("분기 고정일", dict(combo=3, e_idx=-1, fixed='Q')),
)


def w_specs():
    """대기 W = 0 하나 + 양수 5종 × (전이 경로만 / 모든 경로) = 11."""
    out = [(0, 1)]
    for k in SPANS_POS:
        out.append((k, 0))
        out.append((k, 1))
    return out


def r_specs():
    """휴지 R = 0 하나 + 양수 5종 × 방향 2 × 시작점 2 = 21."""
    out = [(0, 1, 0)]
    for k in SPANS_POS:
        for allp in (0, 1):
            for done in (0, 1):
                out.append((k, allp, done))
    return out


def block_specs():
    """전이를 '금지 신호' 로 쓰는 갈래 = 양수 휴지 5 × (U만 / C전체) 2 × 송금범위 2."""
    out = []
    for k in SPANS_POS:
        for allp in (0, 1):          # 0 = U 만 차단, 1 = C 전체 차단
            for ts in range(TSCOPE_N):
                out.append((k, allp, ts))
    return out


def build(e_live: int, e_dead: int = 0):
    """후보 표를 만든다. e_live = 한 번이라도 발화하는 고유 E 개수."""
    W, Rr, BL = w_specs(), r_specs(), block_specs()
    cols = {k: [] for k in ('e_idx', 'combo', 'w_key', 'w_all', 'r_key',
                            'r_all', 'r_from_done', 'band_i', 'tscope',
                            'block_mode')}

    def push(**kw):
        for k in cols:
            cols[k].append(kw.get(k, 0))

    for e in range(e_live):
        for c in range(N_COMBO):
            for (wk, wa) in W:
                for (rk, ra, rd) in Rr:
                    for b in range(BANDS_N):
                        for ts in range(TSCOPE_N):
                            push(e_idx=e, combo=c, w_key=wk, w_all=wa,
                                 r_key=rk, r_all=ra, r_from_done=rd,
                                 band_i=b, tscope=ts)
        for (rk, ra, ts) in BL:
            push(e_idx=e, combo=0, w_key=0, w_all=1, r_key=rk, r_all=ra,
                 r_from_done=0, band_i=0, tscope=ts, block_mode=1)

    spec = {k: np.asarray(v, dtype=np.int64) for k, v in cols.items()}
    counts = dict(
        고유E=e_live, 무발화E=e_dead,
        발화후보=e_live * N_COMBO * len(W) * len(Rr) * BANDS_N * TSCOPE_N,
        금지후보=e_live * len(BL),
        대조=len(CONTROLS),
    )
    counts['합계'] = counts['발화후보'] + counts['금지후보'] + counts['대조']
    return spec, counts


def main() -> int:
    # 고유 E 개수는 장부 축에서 정해진다(alloc_ledger_sim 이 센다). 인자로 받는다.
    e_live = int(sys.argv[1]) if len(sys.argv) > 1 else 85
    e_dead = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    spec, counts = build(e_live, e_dead)
    print(f"전이 신호 E 발화 {e_live}종 (무발화 {e_dead}종 제외)")
    print(f"  대기 W {len(w_specs())}종 · 휴지 R {len(r_specs())}종"
          f" · 결합 {N_COMBO}종 · 최소편차 {BANDS_N}종 · 송금범위 {TSCOPE_N}종")
    print(f"  금지 신호 갈래 {len(block_specs())}종/E")
    for k, v in counts.items():
        print(f"  {k:<8} {v:>12,}")
    print(f"  실제 생성된 행 {len(spec['e_idx']):,}")
    assert len(spec['e_idx']) == counts['발화후보'] + counts['금지후보']
    # 같은 축 조합이 두 번 들어가지 않았는지 (문법 자체의 중복 점검)
    key = np.stack([spec[k] for k in sorted(spec)], axis=1)
    uniq = np.unique(key, axis=0)
    print(f"  행 중복 {len(key) - len(uniq):,}건")
    return 0


if __name__ == '__main__':
    sys.exit(main())
