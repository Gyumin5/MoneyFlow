#!/bin/bash
# 첫 주식 자동매매 테스트 — 장 열릴 때까지 대기 후 실행
# nohup bash kis_first_trade.sh >> kis_first_trade.log 2>&1 &

cd /home/ubuntu

echo "$(date) 장 개시 대기 중 (23:35 KST)..."

# 23:35까지 대기
TARGET="2335"
while true; do
    NOW=$(date +%H%M)
    if [ "$NOW" -ge "$TARGET" ] && [ "$NOW" -lt "2359" ]; then
        break
    fi
    # 이미 자정 넘었으면 (00:00~05:59) 바로 실행
    if [ "$NOW" -ge "0000" ] && [ "$NOW" -lt "0600" ]; then
        break
    fi
    sleep 60
done

echo "$(date) === 1차 매매 시도 ==="
python3 auto_trade_kis.py --trade 2>&1

# 5분 대기 후 미체결 확인
sleep 300
echo "$(date) === 미체결 확인 1 ==="
python3 auto_trade_kis.py --monitor 2>&1

# 30분 대기 후 재확인
sleep 1800
echo "$(date) === 미체결 확인 2 ==="
python3 auto_trade_kis.py --monitor 2>&1

# 잔고 확인
echo "$(date) === 최종 잔고 ==="
python3 auto_trade_kis.py --balance 2>&1

echo "$(date) === 완료 ==="
