"""JSON 상태 파일 입출력 (원자적 저장)."""
import json
import os
from typing import Any


def load_json(path: str, default: Any = None) -> dict:
    """JSON 파일 로드. 파일 없거나 파싱 실패 시 default 반환."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def save_json(path: str, data: dict, default=None):
    """원자적 JSON 저장 (tmp + os.replace)."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default)
    os.replace(tmp, path)
