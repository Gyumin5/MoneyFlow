"""텔레그램 알림 유틸리티."""
import logging
import requests

logger = logging.getLogger(__name__)


def send_telegram(token: str, chat_id: str, msg: str, prefix: str = '', timeout: int = 5):
    """텔레그램 메시지 전송. prefix가 있으면 앞에 붙임."""
    if not token or not chat_id:
        return
    text = f'[{prefix}] {msg}' if prefix else msg
    try:
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        requests.post(url, data={'chat_id': chat_id, 'text': text}, timeout=timeout)
    except Exception as e:
        logger.warning('텔레그램 전송 실패: %s', e)


class BufferedNotifier:
    """이벤트를 모아서 한 번에 전송."""

    def __init__(self, token: str, chat_id: str, prefix: str = ''):
        self.token = token
        self.chat_id = chat_id
        self.prefix = prefix
        self.events = []

    def add(self, msg: str):
        self.events.append(msg)

    def flush(self):
        if not self.events:
            return
        text = '\n'.join(self.events)
        send_telegram(self.token, self.chat_id, text, prefix=self.prefix)
        self.events.clear()

    def send_now(self, msg: str):
        send_telegram(self.token, self.chat_id, msg, prefix=self.prefix)
