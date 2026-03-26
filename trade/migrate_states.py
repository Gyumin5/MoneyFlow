#!/usr/bin/env python3
"""기존 state 파일을 새 스키마로 마이그레이션.

사용법:
  python3 migrate_states.py                     # dry-run (파일 안 씀)
  python3 migrate_states.py --apply             # 실제 적용
  python3 migrate_states.py --input-dir /tmp    # 다른 디렉토리의 파일 변환

기존 파일:
  signal_state.json     → signal_state.json (새 스키마)
  trade_state.json      → coin_trade_state.json
  kis_trade_state.json  → kis_trade_state.json (새 스키마)

기존 파일은 *.backup으로 보존.
"""

import json, os, sys, argparse
from datetime import datetime


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path, data, dry_run=True):
    if dry_run:
        print(f"  [DRY-RUN] Would write {path}")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
        print("  ...")
        return
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    print(f"  [WRITTEN] {path}")


def migrate_signal_state(old, dir_path):
    """기존 signal_state → 새 스키마."""
    # 주식 신호
    offense_picks = old.get('execution_plan', {}).get('stock', {}).get('ideal_picks', old.get('stock_holdings', []))
    n = len(offense_picks) if offense_picks else 1
    cash = 0.02
    w = (1.0 - cash) / max(n, 1)
    offense_weights = {t: round(w, 4) for t in offense_picks}
    offense_weights['Cash'] = cash

    # 방어 종목 — 기존 signal_state에 없으므로 기본값 사용
    defense_picks = ["IEF", "GLD", "PDBC"]
    dw = (1.0 - cash) / len(defense_picks)
    defense_weights = {t: round(dw, 4) for t in defense_picks}
    defense_weights['Cash'] = cash

    new = {
        "stock": {
            "offense_picks": offense_picks,
            "offense_weights": offense_weights,
            "defense_picks": defense_picks,
            "defense_weights": defense_weights,
            "risk_on": old.get('risk_on', True),
            "vt_prev_close": old.get('vt_prev_close', 0),
            "vt_sma10": 0,  # recommend가 다음 실행 시 채움
        },
        "coin": {
            "picks": old.get('execution_plan', {}).get('coin', {}).get('ideal_picks', []),
            "weights": old.get('execution_plan', {}).get('coin', {}).get('ideal_weights', {}),
            "risk_on": old.get('coin_risk_on', False),
            "guard_refs": {},  # recommend가 다음 실행 시 채움
        },
        "meta": {
            "signal_date": datetime.now().strftime('%Y-%m-%d'),
            "updated_at": old.get('updated', old.get('execution_plan', {}).get('updated_at', '')),
        },
    }
    return new


def migrate_coin_trade_state(old):
    """기존 trade_state → coin_trade_state 새 스키마."""
    old_tranches = old.get('tranches', {})
    new_tranches = {}
    for day, tr in old_tranches.items():
        new_tranches[day] = {
            "picks": tr.get('picks', []),
            "weights": tr.get('weights', {}),
            "anchor_month": tr.get('last_anchor_month', tr.get('anchor_month', '')),
        }

    new = {
        "tranches": new_tranches,
        "rebalancing_needed": False,  # 전환 직후 안정 상태
        "guard_state": {
            "crash_active": False,
            "crash_date": None,
            "crash_cooldown_until": None,
            "exclusions": {},
        },
        "prev_risk_on": old.get('coin_risk_on', True),
        "flip_date": None,
        "pfd_done": True,
        "last_action": "migration",
        "last_trade_date": datetime.now().strftime('%Y-%m-%d'),
    }
    return new


def migrate_kis_trade_state(old):
    """기존 kis_trade_state → 새 스키마."""
    old_tranches = old.get('tranches', {})
    new_tranches = {}
    for day, tr in old_tranches.items():
        new_tranches[day] = {
            "picks": tr.get('picks', []),
            "weights": tr.get('weights', {}),
            "anchor_month": tr.get('anchor_month', ''),
        }

    new = {
        "tranches": new_tranches,
        "rebalancing_needed": old.get('rebalancing_needed', False),
        "guard_state": {
            "crash_active": False,
            "crash_date": None,
            "crash_cooldown_until": None,
        },
        "prev_risk_on": True,  # 기존에 없던 필드, 기본값
        "last_action": old.get('last_action', 'migration'),
        "last_trade_date": old.get('last_date', datetime.now().strftime('%Y-%m-%d')),
    }
    return new


def main():
    parser = argparse.ArgumentParser(description='State file migration')
    parser.add_argument('--apply', action='store_true', help='Actually write files (default: dry-run)')
    parser.add_argument('--input-dir', default='.', help='Directory with existing state files')
    args = parser.parse_args()

    dry_run = not args.apply
    d = args.input_dir

    if dry_run:
        print("=" * 60)
        print("DRY-RUN MODE — 파일을 쓰지 않습니다. --apply로 실제 적용.")
        print("=" * 60)

    # 1. signal_state
    sig_path = os.path.join(d, 'signal_state.json')
    if os.path.exists(sig_path):
        print(f"\n[1/3] signal_state.json 마이그레이션")
        old_sig = load_json(sig_path)
        new_sig = migrate_signal_state(old_sig, d)
        if not dry_run:
            os.rename(sig_path, sig_path + '.backup')
        save_json(sig_path, new_sig, dry_run)
    else:
        print(f"  [SKIP] {sig_path} 없음")

    # 2. trade_state → coin_trade_state
    ts_path = os.path.join(d, 'trade_state.json')
    coin_path = os.path.join(d, 'coin_trade_state.json')
    if os.path.exists(ts_path):
        print(f"\n[2/3] trade_state.json → coin_trade_state.json")
        old_ts = load_json(ts_path)
        new_coin = migrate_coin_trade_state(old_ts)
        if not dry_run:
            os.rename(ts_path, ts_path + '.backup')
        save_json(coin_path, new_coin, dry_run)
    else:
        print(f"  [SKIP] {ts_path} 없음")

    # 3. kis_trade_state
    kis_path = os.path.join(d, 'kis_trade_state.json')
    if os.path.exists(kis_path):
        print(f"\n[3/3] kis_trade_state.json 마이그레이션")
        old_kis = load_json(kis_path)
        new_kis = migrate_kis_trade_state(old_kis)
        if not dry_run:
            os.rename(kis_path, kis_path + '.backup')
        save_json(kis_path, new_kis, dry_run)
    else:
        print(f"  [SKIP] {kis_path} 없음")

    print(f"\n{'적용 완료' if not dry_run else 'DRY-RUN 완료'}.")
    if dry_run:
        print("실제 적용하려면: python3 migrate_states.py --apply")


if __name__ == '__main__':
    main()
