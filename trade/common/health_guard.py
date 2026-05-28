"""공통 cron 안전장치 — fsync abort log, health.json, lock, abort_streak.

cycle 6 ai-debate (2026-05-28) 합의 — executor_coin / executor_stock / auto_trade_binance V25 공통.
각 executor 는 자기 이름으로 인스턴스화: HealthGuard(name='coin') / HealthGuard(name='stock').

사용 패턴:
    hg = HealthGuard(name='coin')
    if hg.is_locked():
        return  # cron skip
    try:
        ... 매매 ...
        hg.record_success()
    except UnknownExecutionError as e:
        hg.lock(f'UNKNOWN_EXECUTION: {e}')
        raise
    except Exception as e:
        hg.record_abort(str(e))
        raise
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Optional


ABORT_STREAK_LOCK_THRESHOLD = 3


class UnknownExecutionError(Exception):
    """주문 송신 후 응답 불명 — 중복 주문 위험. 즉시 lock 필요."""
    pass


class HealthGuard:
    def __init__(self, name: str, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.expanduser('~')
        self.name = name
        self.health_file = os.path.join(base_dir, f'.{name}_health.json')
        self.lock_file = os.path.join(base_dir, f'.{name}_lock')
        self.abort_log = os.path.join(base_dir, f'{name}_abort.log')

    def _read(self) -> dict:
        try:
            with open(self.health_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _write(self, data: dict):
        try:
            tmp = self.health_file + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                try: os.fsync(f.fileno())
                except Exception: pass
            os.replace(tmp, self.health_file)
        except Exception:
            pass

    def is_locked(self) -> Optional[str]:
        try:
            with open(self.lock_file) as f:
                return f.read().strip() or 'locked'
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def lock(self, reason: str):
        try:
            with open(self.lock_file, 'w') as f:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{ts} KST\n{reason}\n")
                f.flush()
                try: os.fsync(f.fileno())
                except Exception: pass
            self.persist_log(f"LOCKED: {reason}")
        except Exception:
            pass

    def unlock(self):
        try: os.remove(self.lock_file)
        except FileNotFoundError: pass
        except Exception: pass

    def persist_log(self, msg: str):
        """ABORT/LOCK 시 디스크 영구 기록 (Telegram 의존 제거)."""
        try:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.abort_log, 'a') as f:
                f.write(f"[{ts} KST] {msg}\n")
                f.flush()
                try: os.fsync(f.fileno())
                except Exception: pass
        except Exception:
            pass

    def record_success(self):
        h = self._read()
        h['last_success_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        h['abort_streak'] = 0
        h['last_abort_reason'] = ''
        self._write(h)

    def record_abort(self, reason: str) -> int:
        """ABORT 기록 + streak 증가. 3 연속 시 자동 lock. Returns: streak."""
        h = self._read()
        streak = int(h.get('abort_streak', 0)) + 1
        h['abort_streak'] = streak
        h['last_abort_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        h['last_abort_reason'] = (reason or '')[:500]
        self._write(h)
        self.persist_log(f"ABORT (streak={streak}): {reason}")
        if streak >= ABORT_STREAK_LOCK_THRESHOLD:
            self.lock(f"abort_streak={streak} reached threshold")
        return streak
