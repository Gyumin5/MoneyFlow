"""V23 통일 텔레그램 보고 포맷 helper.

3개 자산 (코인 spot, 선물 fut, 주식) + Daily Report 가 공통 skeleton 으로 출력.
구조: [자산] EMOJI 자산이름 V23 (시각) → 🎯 목표 → 📦 보유 → 🔄 주문 → 🦅 카나리 → 📊 상태
"""
from typing import Dict, Iterable, Optional


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def build_header(asset_label: str, emoji: str, name: str, ts_str: str) -> str:
    return f"[{asset_label}] {emoji} {name} V23 ({ts_str})"


def build_target_section(target: Dict[str, float]) -> str:
    lines = ["🎯 목표"]
    items = [(k, v) for k, v in target.items() if not k.startswith('_')]
    for k, v in sorted(items, key=lambda kv: -kv[1]):
        if v < 1e-4 and k.lower() != 'cash':
            continue
        lines.append(f"  {k}: {fmt_pct(v)}")
    return "\n".join(lines)


def build_holdings_section(holdings: Iterable[Dict]) -> str:
    """holdings: list[{ticker, value_str, weight (0~1), pnl?}]."""
    holdings = list(holdings)
    if not holdings:
        return "📦 보유: 없음"
    lines = ["📦 보유"]
    for h in holdings:
        line = f"  {h['ticker']}: {h['value_str']} ({fmt_pct(h['weight'])}"
        if 'pnl' in h and h['pnl'] is not None:
            line += f", PnL {h['pnl']:+.2f}"
        line += ")"
        lines.append(line)
    return "\n".join(lines)


def build_orders_section(orders_text: str) -> str:
    return f"🔄 주문: {orders_text}"


def build_canary_section(canary_lines: Iterable[str]) -> str:
    """canary_lines: per-멤버 한 줄 (예: 'D_SMA42: ON 🟢 BTC $75,750 vs SMA42 $72,065 ratio 1.0511')."""
    canary_lines = [l for l in canary_lines if l]
    if not canary_lines:
        return ""
    return "🦅 카나리\n" + "\n".join(f"  {l}" for l in canary_lines)


def build_status_section(status: Dict[str, str]) -> str:
    lines = ["📊 상태"]
    for k, v in status.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def build_report(
    asset_label: str,
    emoji: str,
    name: str,
    ts_str: str,
    target: Dict[str, float],
    holdings: Iterable[Dict],
    orders_text: str,
    canary_lines: Iterable[str],
    status: Dict[str, str],
    extra: Optional[str] = None,
) -> str:
    parts = [build_header(asset_label, emoji, name, ts_str), ""]
    parts.append(build_target_section(target))
    parts.append("")
    parts.append(build_holdings_section(holdings))
    parts.append("")
    parts.append(build_orders_section(orders_text))
    canary = build_canary_section(canary_lines)
    if canary:
        parts.append("")
        parts.append(canary)
    parts.append("")
    parts.append(build_status_section(status))
    if extra:
        parts.append("")
        parts.append(extra)
    return "\n".join(parts)
