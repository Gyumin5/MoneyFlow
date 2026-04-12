"""설정/환경변수 유틸리티."""
import os
import sys


def get_config_or_env(name: str, default: str = '', script_dir: str = None) -> str:
    """config.py에서 먼저 찾고 없으면 환경변수에서 조회."""
    try:
        if script_dir and script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        import config
        val = getattr(config, name, None)
        if val is not None:
            return val
    except ImportError:
        pass
    return os.environ.get(name, default)
