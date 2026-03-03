# config/settings.py — 통합 설정 파일
# 이 파일을 settings.py로 복사한 후 실제 키를 채워넣으세요.
# cp config/settings.example.py config/settings.py

# ==========================================
# 1. 빗썸 API (Connect API 1.0)
# ==========================================
BITHUMB_API_KEY = "YOUR_BITHUMB_API_KEY"
BITHUMB_SECRET_KEY = "YOUR_BITHUMB_SECRET_KEY"

# ==========================================
# 2. 업비트 API Keys
# ==========================================
UPBIT_ACCESS_KEY = "YOUR_UPBIT_ACCESS_KEY"
UPBIT_SECRET_KEY = "YOUR_UPBIT_SECRET_KEY"

# ==========================================
# 3. 공통 설정
# ==========================================
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# 거래 설정
DRY_RUN = False
MAX_BUDGET = None
TURNOVER_THRESHOLD = 0.30
EXECUTION_TIME = "09:00"
RETRY_COUNT = 3
RETRY_DELAY = 5

# ==========================================
# 4. 거래 비밀번호
# ==========================================
TRADE_PASSWORD = "YOUR_TRADE_PASSWORD"
