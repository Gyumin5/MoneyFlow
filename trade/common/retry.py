"""재시도 유틸리티."""
import time
import logging
from typing import Callable, Sequence

log = logging.getLogger(__name__)


def retry_call(fn: Callable, *,
               attempts: int = 3,
               delays: Sequence[float] = (1.0, 2.0, 5.0),
               is_retryable: Callable[[Exception], bool] = None,
               on_retry: Callable[[int, Exception], None] = None):
    """재시도 래퍼. is_retryable이 None이면 모든 예외에 재시도."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i == attempts - 1:
                raise
            if is_retryable and not is_retryable(e):
                raise
            delay = delays[min(i, len(delays) - 1)]
            if on_retry:
                on_retry(i + 1, e)
            else:
                log.warning(f'retry {i+1}/{attempts}: {e} (wait {delay}s)')
            time.sleep(delay)
    raise last_exc
