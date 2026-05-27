#!/usr/bin/env python3
"""V23 → V24 상태파일 마이그레이션.

V24 변경점 (vs V23):
- 코인 spot + fut sleeve 에 refill v2 도입 (drift fire 시 mom2 음수 슬롯 교체)
- 주식 sleeve 변경 없음 (refill 미적용)
- 파라미터 구조 동일 (snap_interval, n_snap, drift_threshold 그대로)
- schema_version: 'V23' → 'V24'

이 마이그레이션은 단순 schema_version 라벨 변경 + 백업.
snapshots 구조 호환 (refill 은 런타임에 snapshots 수정).

Usage
  python3 migrate_v23_to_v24.py --dry-run
  python3 migrate_v23_to_v24.py --apply
  python3 migrate_v23_to_v24.py --rollback {YYYYMMDDHHMMSS}
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

TARGETS = [
    {'file': 'trade_state.json',   'sleeve': 'coin_spot'},
    {'file': 'binance_state.json', 'sleeve': 'coin_fut'},
    {'file': 'kis_trade_state.json', 'sleeve': 'stock'},
]


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  load 실패: {e}")
        return {}


def save_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def migrate_one(state: Dict[str, Any], sleeve: str) -> Dict[str, Any]:
    """단일 파일 마이그레이션. 단순 schema_version 라벨 + 마커만."""
    out = dict(state)
    prev_v = out.get('schema_version', 'unknown')
    out['schema_version'] = 'V24'
    out['v24_migrated_at'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    out['last_rebal_reason'] = 'v24_migration'
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--rollback', help='YYYYMMDDHHMMSS 백업 stamp')
    p.add_argument('--state-dir', default=os.path.expanduser('~'),
                   help='state 파일 디렉토리 (서버: ~/ , 로컬: project root 등)')
    args = p.parse_args()

    if not (args.dry_run or args.apply or args.rollback):
        p.print_help()
        sys.exit(1)

    state_dir = args.state_dir
    stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')

    if args.rollback:
        for t in TARGETS:
            bak = os.path.join(state_dir, f"{t['file']}.v23.bak.{args.rollback}")
            tgt = os.path.join(state_dir, t['file'])
            if os.path.exists(bak):
                shutil.copy2(bak, tgt)
                print(f"✅ rollback {t['file']} <- {bak}")
            else:
                print(f"⚠️  백업 없음: {bak}")
        return

    for t in TARGETS:
        fp = os.path.join(state_dir, t['file'])
        if not os.path.exists(fp):
            print(f"⏭  {t['file']} 없음, skip")
            continue
        state = load_json(fp)
        prev_v = state.get('schema_version', 'unknown')
        new_state = migrate_one(state, t['sleeve'])

        print(f"\n=== {t['file']} ({t['sleeve']}) ===")
        print(f"  schema_version: {prev_v} → {new_state['schema_version']}")
        print(f"  v24_migrated_at: {new_state['v24_migrated_at']}")
        print(f"  last_rebal_reason: {new_state['last_rebal_reason']}")

        if args.apply:
            bak = f"{fp}.v23.bak.{stamp}"
            shutil.copy2(fp, bak)
            save_json_atomic(fp, new_state)
            print(f"  ✅ 적용 + 백업 → {bak}")
        else:
            print(f"  🔍 dry-run (적용 안 함)")

    if args.apply:
        print(f"\n완료. rollback 명령: python3 migrate_v23_to_v24.py --rollback {stamp}")
    else:
        print(f"\ndry-run 완료. 적용하려면 --apply")


if __name__ == '__main__':
    main()
