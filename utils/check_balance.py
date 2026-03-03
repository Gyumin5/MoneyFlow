import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pybithumb
from config.bithumb import BITHUMB_API_KEY, BITHUMB_SECRET_KEY

b = pybithumb.Bithumb(BITHUMB_API_KEY, BITHUMB_SECRET_KEY)

print("=== 현재 잔고 확인 ===")
for ticker in ['BTC', 'PAXG', 'BCH', 'BNB', 'WLFI', 'SKY']:
    try:
        balance = b.get_balance(ticker)
        # balance: (total_coin, start_coin, total_krw, start_krw)
        print(f"{ticker}: {balance[0]} (KRW: {balance[2]})")
    except:
        print(f"{ticker}: 조회 실패")

print("\n=== 최근 주문 내역 (User Transactions) ===")
# This might not be easy to fetch via pybithumb simple interface without parsing
# Let's try a small buy order of a minimal amount or just check the return type behavior
print("API Key check:", BITHUMB_API_KEY[:4])
