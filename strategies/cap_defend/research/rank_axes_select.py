"""선정 순위 축 실험의 선택 함수 (2026-10-07, ai-debate run-20261007T013223Z 명세).

정본 엔진은 환경변수 SELECT_RANK 가 비어 있으면 이 모듈을 아예 import 하지 않는다.
실험이 끝나면 엔진 쪽 호출부를 되돌리고 이 파일과 .engine.patch 만 남긴다.

공통 명세 (결과를 보기 전에 고정했다):
  - s = sig_date = t-1. 모든 입력은 s 완료봉까지만 쓴다.
  - H_s = 정본이 만든 헬스 통과 집합. 이 모듈은 H_s 를 만들지 않는다 — 받아서 줄만 세운다.
    들어오는 순서가 곧 원시 시총 순위다(엔진이 시총 목록 순으로 append 한다).
  - 모멘텀·변동성은 엔진 함수를 그대로 받아 쓴다. 다시 구현하지 않는다.
  - 동률은 원시 시총 순위, 그 다음 심볼 사전순으로 푼다.
  - 뽑은 뒤에는 반드시 시총순으로 다시 세워서 돌려준다. 이게 설계의 핵심이다 —
    새 순위를 그대로 greedy 흡수에 넣으면 앞뒤 비교 조건(ma >= mb)이 항상 참이 되어
    1종으로 무너진다(실측: 해당일 1,245일 전부 100%, 현금 2/3 고정).

팔:
  b0      시총 내림차순 앞 3종 (현행 기준선 — 엔진 원래 경로와 같은 결과여야 한다)
  smom    단기 모멘텀(선물 18봉 / 현물 20봉) 큰 순
  slong   127봉 모멘텀 큰 순
  seff    127봉 모멘텀 / 90봉 일간 변동성 큰 순
  cmix    시총 순위 + 효율 순위 작은 순
  ualt    시총 목록과 효율 목록을 번갈아 읽어 중복 빼고 앞 3종 (합집합)
  sdvol   종가×거래량 30봉 평균 큰 순 (선물 전용)
  placebo 심볼 고정 해시 기반 무작위 우선순위 (위약 대조)

교란 (채택 검증용):
  lag   점수만 s-L 봉까지로 계산한다. H_s 와 시총순은 그대로 둔다.
  swap  흡수 전 최종 선정 순서의 3위와 4위를 바꾼 뒤 앞 3종을 고른다.
        4위가 없으면 교란 불가로 기록한다(_last_swap_applied=False).
"""
from __future__ import annotations

import hashlib

import numpy as np

EFF_FLOOR = -1e18          # 변동성이 0이거나 점수가 비유한일 때 최하위로 보낸다
DVOL_BARS = 30

# 마지막 호출에서 교란이 실제로 적용됐는지. 교란이 아무것도 안 바꿨으면
# 안정성의 근거로 쓰지 않는다는 채택 기준 ④ 때문에 센다.
_stats = {"calls": 0, "swap_applied": 0, "swap_impossible": 0}


def reset_stats() -> None:
    for k in _stats:
        _stats[k] = 0


def get_stats() -> dict:
    return dict(_stats)


def _placebo_key(coin: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{coin}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def select(mode, healthy, universe_size, *, closes, vols, dvols,
           mom_short_bars, mom_long_bars, calc_mom,
           lag=0, swap=False, seed=0):
    """H_s 에서 universe_size 종을 고르고 시총순으로 다시 세워 돌려준다.

    healthy  : 원시 시총 내림차순 코인 리스트 (엔진이 만든 그대로)
    closes   : {coin: np.ndarray}  s 완료봉까지의 종가
    vols     : {coin: float}       정본 90봉 일간 변동성 (lag 적용분을 엔진이 넣어 준다)
    dvols    : {coin: float}       종가×거래량 30봉 평균. 선물에서만 채워진다.
    calc_mom : 정본 모멘텀 함수
    """
    _stats["calls"] += 1
    if not healthy:
        return []
    mcap_rank = {c: i for i, c in enumerate(healthy)}

    def _tie(coin):
        return (mcap_rank[coin], coin)

    def _ordered_by(score):
        # 점수 내림차순, 동률은 시총 순위 → 심볼.
        return sorted(healthy, key=lambda c: (-score[c],) + _tie(c))

    def _cut(c):
        """lag 를 적용한 종가 배열. 부족하면 원본을 그대로 쓴다."""
        a = closes[c]
        if lag and len(a) > lag:
            return a[:-lag]
        return a

    if mode == "b0":
        order = list(healthy)
    elif mode in ("smom", "slong"):
        n = mom_short_bars if mode == "smom" else mom_long_bars
        score = {c: float(calc_mom(_cut(c), n)) for c in healthy}
        order = _ordered_by(score)
    elif mode == "seff":
        score = {}
        for c in healthy:
            m = float(calc_mom(_cut(c), mom_long_bars))
            v = vols.get(c, 0.0)
            score[c] = m / v if (v and np.isfinite(v) and v > 0) else EFF_FLOOR
            if not np.isfinite(score[c]):
                score[c] = EFF_FLOOR
        order = _ordered_by(score)
    elif mode == "cmix":
        eff = {}
        for c in healthy:
            m = float(calc_mom(_cut(c), mom_long_bars))
            v = vols.get(c, 0.0)
            eff[c] = m / v if (v and np.isfinite(v) and v > 0) else EFF_FLOOR
            if not np.isfinite(eff[c]):
                eff[c] = EFF_FLOOR
        eff_order = _ordered_by(eff)
        r_eff = {c: i for i, c in enumerate(eff_order)}
        # 순위합 오름차순. 동률은 시총 순위 → 심볼.
        order = sorted(healthy, key=lambda c: (mcap_rank[c] + r_eff[c],) + _tie(c))
    elif mode == "ualt":
        eff = {}
        for c in healthy:
            m = float(calc_mom(_cut(c), mom_long_bars))
            v = vols.get(c, 0.0)
            eff[c] = m / v if (v and np.isfinite(v) and v > 0) else EFF_FLOOR
            if not np.isfinite(eff[c]):
                eff[c] = EFF_FLOOR
        a = list(healthy)[:universe_size]            # 시총 상위 목록
        b = _ordered_by(eff)[:universe_size]          # 효율 상위 목록
        order, seen = [], set()
        for i in range(universe_size):
            for lst in (a, b):
                if i < len(lst) and lst[i] not in seen:
                    seen.add(lst[i])
                    order.append(lst[i])
        # 합집합이 모자라면 남은 건강 코인을 시총순으로 뒤에 붙인다 — 교란(3위↔4위)에
        # 쓸 자리를 남기기 위해서다. 앞 universe_size 종만 실제로 뽑힌다.
        order += [c for c in healthy if c not in seen]
    elif mode == "sdvol":
        order = _ordered_by({c: dvols.get(c, 0.0) for c in healthy})
    elif mode == "placebo":
        order = _ordered_by({c: _placebo_key(c, seed) for c in healthy})
    else:
        raise ValueError(f"모르는 팔: {mode}")

    if swap:
        if len(order) > universe_size:
            i = universe_size - 1
            order = list(order)
            order[i], order[i + 1] = order[i + 1], order[i]
            _stats["swap_applied"] += 1
        else:
            _stats["swap_impossible"] += 1

    picks = order[:universe_size]
    # 핵심 — 흡수 전에 시총순으로 되돌린다.
    return sorted(picks, key=lambda c: mcap_rank[c])
