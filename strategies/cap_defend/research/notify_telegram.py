#!/usr/bin/env python3
"""텔레그램 알림 헬퍼."""
import json
import os
import sys
try:
    import requests
except Exception:
    requests = None

def _creds():
    """TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID를 ~/.config/telegram_bot.json 또는 환경변수에서 로드."""
    cfg = os.path.expanduser("~/.config/telegram_bot.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg) as f:
                d = json.load(f)
            return d.get("TELEGRAM_BOT_TOKEN", ""), d.get("TELEGRAM_CHAT_ID", "")
        except Exception:
            pass
    return os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")

def send(text):
    if requests is None:
        return False
    tok, chat = _creds()
    if not tok:
        print("no telegram token", file=sys.stderr)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                         json={"chat_id": chat, "text": text}, timeout=10)
        return r.ok
    except Exception as e:
        print(f"telegram fail: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    msg = sys.stdin.read() if not sys.argv[1:] else " ".join(sys.argv[1:])
    sys.exit(0 if send(msg) else 1)
