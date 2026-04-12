"""로깅 유틸리티."""
import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


def setup_file_logger(name: str, log_file: str, console: bool = False) -> logging.Logger:
    """TimedRotatingFileHandler 기반 로거 생성."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = TimedRotatingFileHandler(
            log_file, when='midnight', backupCount=14, encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        if console and sys.stderr.isatty():
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(sh)
    return logger


def make_log_fn(logger: logging.Logger, run_id_ref: list):
    """run_id가 포함된 log() 함수를 반환. run_id_ref는 [run_id] 형태의 mutable ref."""
    def log(msg: str):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rid = run_id_ref[0][:8] if run_id_ref[0] else '--------'
        logger.info(f'[{ts}] [{rid}] {msg}')
    return log
