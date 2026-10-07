"""현행 선정 규칙의 우연 여부를 재는 실험의 선택 함수 (2026-10-07,
ai-debate 깊은 회차 run-20261007T032923Z 명세).

직전 실험(rank_axes_select.py)은 "다른 순위축이 더 나은가" 를 물었다. 이번은 반대다 —
현행(헬스 통과 집합을 원시 시총 내림차순으로 훑어 앞 3종)이 무작위 선정보다 나은가,
그리고 몇몇 코인을 빼면 그 우위가 남는가를 묻는다.

팔:
  b0        현행. 들어온 순서(=원시 시총 내림차순) 그대로 앞 3종. 피시험자다.
  pfix:N    고정 해시 무작위 순위. 시드 N 으로 심볼마다 한 번 정해진 우선순위를 끝까지 쓴다.
            "특정 코인을 계속 좋아하는" 임의 선호가 현행만큼 버는지 본다.
  pref:N    갱신 해시 무작위 순위. 시드 N 과 스냅 간격 구간 번호로 우선순위를 다시 뽑는다.
            "시점마다 아무거나 고르는" 선정이 현행만큼 버는지 본다.
  top1      시총 1위 한 종만. 상한 1/3 은 그대로라 나머지는 현금이다.
  fullpool  헬스 통과 전부. 각 종목 min(1/3, 1/n), 나머지 현금. 흡수를 생략한다
            (엔진 쪽 SELECT_NO_ABSORB=1 과 짝이다).

고정된 것: 헬스 게이트, 카나리, 스냅 수·간격, 드리프트 문턱, 리필 규칙, 상한, 체결, 비용.
b0·pfix·pref 는 뽑은 뒤 시총순으로 다시 세워 흡수에 넣는다 — 모멘텀 내림차순을 그대로
넣으면 흡수 조건이 항상 참이라 1종으로 무너진다(직전 실험 실측).
top1 은 1종이라 흡수가 무의미하고, fullpool 은 흡수를 끈다.

갱신형의 한계 하나를 명시한다. 명세는 "해당 스냅의 유효기간 동안 순위를 유지" 라고 했는데,
엔진이 선택 함수에 어느 스냅을 채우는지 안 알려준다. 그래서 구간 번호를 신호일의 봉 번호를
스냅 간격으로 나눈 몫으로 잡았다. 한 스냅의 수명이 두 구간에 걸치면 그 스냅의 리필은 새
순위를 쓴다. 검사 의도(시점마다 임의 선정)는 그대로 성립한다.
"""
from __future__ import annotations

import hashlib

MODES = ("b0", "pfix", "pref", "top1", "fullpool")


def _key(token: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{token}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def parse_mode(raw: str):
    """'pfix:7' → ('pfix', 7). 시드가 없으면 0."""
    if ":" in raw:
        name, s = raw.split(":", 1)
        return name, int(s)
    return raw, 0


def select(raw_mode, healthy, universe_size, *, epoch=0):
    """헬스 통과 집합에서 종목을 골라 시총순으로 돌려준다.

    healthy : 원시 시총 내림차순 코인 리스트 (엔진이 만든 그대로)
    epoch   : 갱신형 무작위 순위의 구간 번호. 다른 팔은 안 쓴다.
    """
    if not healthy:
        return []
    mode, seed = parse_mode(raw_mode)
    mcap_rank = {c: i for i, c in enumerate(healthy)}

    if mode == "b0":
        order = list(healthy)
    elif mode == "top1":
        return list(healthy[:1])
    elif mode == "fullpool":
        return list(healthy)
    elif mode in ("pfix", "pref"):
        tok = (lambda c: c) if mode == "pfix" else (lambda c: f"{c}#{epoch}")
        score = {c: _key(tok(c), seed) for c in healthy}
        # 점수 내림차순, 동률은 시총 순위 → 심볼.
        order = sorted(healthy, key=lambda c: (-score[c], mcap_rank[c], c))
    else:
        raise ValueError(f"모르는 팔: {raw_mode}")

    picks = order[:universe_size]
    return sorted(picks, key=lambda c: mcap_rank[c])
