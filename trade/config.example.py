"""실거래용 로컬 설정 예시.

실제 운영 시:
- 이 파일을 `trade/config.py`로 복사해서 사용
- 실제 비밀값은 코드 레포가 아니라 private 설정 레포(`moneyflow-config`)에서 관리
"""

# ─── Binance USDT-M Futures ─────────────────────────────────────
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# ─── Telegram ───────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# ─── Upbit (선택) ───────────────────────────────────────────────
UPBIT_ACCESS_KEY = ""
UPBIT_SECRET_KEY = ""

# ─── 공통 실행 옵션 ─────────────────────────────────────────────
TURNOVER_THRESHOLD = 0.02
RETRY_COUNT = 3
RETRY_DELAY = 1
