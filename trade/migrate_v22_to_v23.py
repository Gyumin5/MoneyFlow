#!/usr/bin/env python3
"""V22 → V23 상태파일 마이그레이션.

대상 파일
- trade_state.json (코인 spot, V22 1D+4h → V23 1D 단일 sn=217 n=7 drift=0.10)
- binance_state.json (선물, V22 1D+4h → V23 1D 단일 sn=57 n=3 drift=0.05)
- kis_trade_state.json (주식, V22 sd=125 → V23 sd=69 stagger=23)

절차 (각 state 파일별)
1. 백업: {file}.v22.bak.{YYYYMMDDHHMMSS}
2. members 단일화 (H4_SMA240 / 4h_SMA240 멤버 제거)
3. snapshots 길이 변경: spot 3 → 7 (D_SMA42), fut 3 유지, stock 3 유지
   - 새 슬롯은 last_combined 또는 빈 {'CASH': 1.0} 으로 초기화
4. bar_counter 0 리셋 (n_snap 변경 시 stagger 재정렬)
5. last_bar_ts 클리어 (다음 새 봉 처리)
6. schema_version = 'V23' 추가
7. last_rebal_reason = 'v23_migration'

Usage
  python3 migrate_v22_to_v23.py --dry-run     # 결과만 출력
  python3 migrate_v22_to_v23.py --apply       # 실제 백업 + 변경
  python3 migrate_v22_to_v23.py --rollback {YYYYMMDDHHMMSS}  # 백업 복원
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, Any, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

V23_COIN_SPOT_TARGET = {
    'snap_interval_bars': 217,
    'n_snapshots': 7,
    'drift_threshold': 0.10,
    'remove_members': ['H4_SMA240'],
    'keep_member': 'D_SMA42',
}

V23_COIN_FUT_TARGET = {
    'snap_interval_bars': 57,
    'n_snapshots': 3,
    'drift_threshold': 0.05,
    'remove_members': ['4h_SMA240'],
    'keep_member': 'D_SMA42',
}

V23_STOCK_TARGET = {
    'snap_period_days': 69,
    'snap_stagger_days': 23,
    'n_snaps': 3,
}


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def backup_file(path: str, ts: str) -> str:
    if not os.path.exists(path):
        return ''
    bak = f'{path}.v22.bak.{ts}'
    shutil.copy2(path, bak)
    return bak


def migrate_coin_spot(state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    """trade_state.json (V22 → V23 spot)."""
    members = state.get('members', {})
    keep = target['keep_member']
    n_snap_new = target['n_snapshots']

    # keep 외 모든 멤버 제거 (H4_SMA240 외 V21 잔여 D_SMA* 도 정리)
    for name in list(members.keys()):
        if name != keep:
            print(f'  - removing member: {name}')
            members.pop(name, None)

    # keep 멤버의 snapshots/bar_counter/last_bar_ts 갱신
    keep_state = members.get(keep, {})
    snapshots = keep_state.get('snapshots', [])
    last_combined = keep_state.get('last_combined', {})

    # snapshots 길이 조정
    if len(snapshots) < n_snap_new:
        # 부족한 슬롯은 last_combined (또는 CASH) 으로 채움
        fill = dict(last_combined) if last_combined else {'CASH': 1.0}
        for _ in range(n_snap_new - len(snapshots)):
            snapshots.append(dict(fill))
        print(f'  - snapshots {len(snapshots)-(n_snap_new-len(snapshots))} → {n_snap_new} ({n_snap_new-(n_snap_new-len(snapshots))}개 추가, fill={list(fill.keys())[:3]})')
    elif len(snapshots) > n_snap_new:
        snapshots = snapshots[:n_snap_new]
        print(f'  - snapshots truncated to {n_snap_new}')

    keep_state['snapshots'] = snapshots
    keep_state['bar_counter'] = 0  # 재정렬 위해 리셋
    keep_state['last_bar_ts'] = None  # 다음 새 봉 강제 처리
    members[keep] = keep_state

    state['members'] = members
    state['schema_version'] = 'V23'
    state['last_rebal_reason'] = 'v23_migration'
    state['rebalancing_needed'] = True  # 첫 실행 정합 위해 강제 리밸
    print(f'  ✓ keep={keep} n_snap={n_snap_new} bar_counter=0 last_bar_ts=None rebalancing_needed=True')
    return state


def migrate_coin_fut(state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    """binance_state.json (V22 → V23 fut)."""
    strategies = state.get('strategies', {})
    keep = target['keep_member']
    n_snap_new = target['n_snapshots']

    for name in list(strategies.keys()):
        if name != keep:
            print(f'  - removing strategy: {name}')
            strategies.pop(name, None)

    keep_state = strategies.get(keep, {})
    snapshots = keep_state.get('snapshots', [])
    last_combined = keep_state.get('last_combined', {})

    if len(snapshots) < n_snap_new:
        fill = dict(last_combined) if last_combined else {'CASH': 1.0}
        for _ in range(n_snap_new - len(snapshots)):
            snapshots.append(dict(fill))
        print(f'  - snapshots → {n_snap_new}')
    elif len(snapshots) > n_snap_new:
        snapshots = snapshots[:n_snap_new]
        print(f'  - snapshots truncated to {n_snap_new}')

    keep_state['snapshots'] = snapshots
    keep_state['bar_counter'] = 0
    keep_state['last_bar_ts'] = None
    strategies[keep] = keep_state

    state['strategies'] = strategies
    state['schema_version'] = 'V23'
    state['last_rebal_reason'] = 'v23_migration'
    state['rebalancing_needed'] = True
    print(f'  ✓ keep={keep} n_snap={n_snap_new} bar_counter=0 last_bar_ts=None rebalancing_needed=True')
    return state


def migrate_stock(state: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    """kis_trade_state.json (V22 → V23 stock).

    snap-based 3-tranche 는 유지 (n=3). snap_period 만 126→69, stagger 42→23.
    snapshots 의 last_rebal_date 은 stagger 변경 반영해 재계산.
    """
    snapshots = state.get('snapshots', {})
    today = datetime.now().date()

    # snap_id 0/1/2 의 last_rebal_date 을 staggered 23일 단위로 재할당
    # (전 트랜치 즉시 rebal 트리거되도록 SNAP_PERIOD 이상 경과한 날짜로)
    for snap_id in range(target['n_snaps']):
        key = str(snap_id)
        if key in snapshots:
            # 기존 picks/weights 보존, last_rebal_date 만 재정렬
            pass
        # last_rebal_date 강제 재계산 (즉시 rebal 트리거)
        days_back = target['snap_period_days'] + snap_id * target['snap_stagger_days']
        from datetime import timedelta
        new_date = (today - timedelta(days=days_back)).isoformat()
        snapshots.setdefault(key, {})['last_rebal_date'] = new_date
        print(f'  - snap{snap_id}: last_rebal_date = {new_date} (강제 재정렬)')

    state['snapshots'] = snapshots
    state['schema_version'] = 'V23'
    state['last_rebal_reason'] = 'v23_migration'
    print(f'  ✓ stock V23: snap_period={target["snap_period_days"]} stagger={target["snap_stagger_days"]} n={target["n_snaps"]}')
    return state


def find_state_files() -> Dict[str, str]:
    """state 파일 탐색. 서버 (/home/ubuntu/) + 로컬 cache 모두 확인."""
    candidates = {
        'coin_spot': [
            os.path.join(SCRIPT_DIR, 'trade_state.json'),
            '/home/ubuntu/trade_state.json',
            os.path.expanduser('~/trade_state.json'),
        ],
        'coin_fut': [
            os.path.join(SCRIPT_DIR, 'binance_state.json'),
            '/home/ubuntu/binance_state.json',
            os.path.expanduser('~/binance_state.json'),
        ],
        'stock': [
            os.path.join(SCRIPT_DIR, 'kis_trade_state.json'),
            '/home/ubuntu/kis_trade_state.json',
            os.path.expanduser('~/kis_trade_state.json'),
        ],
    }
    found = {}
    for kind, paths in candidates.items():
        for p in paths:
            if os.path.exists(p):
                found[kind] = p
                break
    return found


def rollback(ts: str) -> None:
    """{ts} 시점의 백업 복원."""
    files = find_state_files()
    for kind, path in files.items():
        bak = f'{path}.v22.bak.{ts}'
        if os.path.exists(bak):
            shutil.copy2(bak, path)
            print(f'✓ {kind}: {bak} → {path} 복원')
        else:
            print(f'✗ {kind}: {bak} 백업 없음')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--rollback')
    args = p.parse_args()

    if args.rollback:
        rollback(args.rollback)
        return

    if not args.dry_run and not args.apply:
        print('Use --dry-run or --apply')
        sys.exit(1)

    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    files = find_state_files()
    if not files:
        print('No state files found. 종료.')
        sys.exit(1)

    print(f'== V22 → V23 마이그레이션 ({"dry-run" if args.dry_run else "APPLY"}) ts={ts} ==')
    for kind, path in files.items():
        print(f'\n[{kind}] {path}')
        state = load_json(path)
        if not state:
            print('  (empty)')
            continue

        if args.apply:
            bak = backup_file(path, ts)
            print(f'  백업: {bak}')

        if kind == 'coin_spot':
            new_state = migrate_coin_spot(state, V23_COIN_SPOT_TARGET)
        elif kind == 'coin_fut':
            new_state = migrate_coin_fut(state, V23_COIN_FUT_TARGET)
        elif kind == 'stock':
            new_state = migrate_stock(state, V23_STOCK_TARGET)
        else:
            print(f'  unknown kind: {kind}')
            continue

        if args.apply:
            save_json_atomic(path, new_state)
            print(f'  ✓ 저장 완료')
        else:
            print(f'  (dry-run: 저장 안함)')

    print(f'\n== 완료 ==')
    if args.apply:
        print(f'rollback: python3 {os.path.basename(__file__)} --rollback {ts}')


if __name__ == '__main__':
    main()
