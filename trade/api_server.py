"""
Trade API Server - Flask based
Provides web API to trigger auto_trade scripts + stock holdings management
"""
from flask import Flask, jsonify, request
import subprocess
import threading
import os
import json
from datetime import datetime

app = Flask(__name__)

# 4자리 PIN (서버 환경변수 TRADE_PIN 필수 설정)
TRADE_PIN = os.environ.get('TRADE_PIN', '')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

running_tasks = {}
HOLDINGS_FILE = '/home/ubuntu/my_stock_holdings.json'

def run_trade_async(exchange: str, force: bool = False, trade: bool = True, target_amount: int = 0):
    task_id = f"{exchange}_{int(os.times().elapsed)}"
    running_tasks[task_id] = {"status": "running", "output": ""}
    try:
        cmd = ["python3", f"/home/ubuntu/auto_trade_{exchange}.py"]
        if trade: cmd.append("--trade")
        if force: cmd.append("--force")
        if target_amount > 0: cmd.extend(["--amount", str(target_amount)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd="/home/ubuntu")
        running_tasks[task_id] = {"status": "completed", "output": result.stdout + result.stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        running_tasks[task_id] = {"status": "timeout", "output": "Script timed out after 5 minutes"}
    except Exception as e:
        running_tasks[task_id] = {"status": "error", "output": str(e)}
    return task_id

@app.route('/api/trade/upbit', methods=['POST'])
def trade_upbit():
    # 암호 검증
    data = request.get_json(silent=True) or {}
    if str(data.get('password', '')) != TRADE_PIN:
        return jsonify({"error": "잘못된 비밀번호"}), 403

    # 중복 실행 방지 (flock 기반)
    lock_file = '/tmp/auto_trade_upbit.lock'
    try:
        import fcntl
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        return jsonify({"error": "이미 매매 실행 중 (cron 또는 다른 요청)"}), 409

    for tid, task in running_tasks.items():
        if "upbit" in tid and task.get("status") == "running":
            lock_fd.close()
            return jsonify({"error": "Upbit trade is already running", "task_id": tid}), 409

    target_amount = int(data.get('target_amount', 0))
    thread = threading.Thread(target=run_trade_async, args=("upbit", True, True, target_amount))
    thread.start()
    msg = f"Upbit force trade started (Target: {target_amount} KRW)" if target_amount > 0 else "Upbit force trade started (Full Equity)"
    return jsonify({"message": msg, "status": "running"})

# --- Stock Holdings API ---
@app.route('/api/holdings', methods=['GET'])
def get_holdings():
    try:
        with open(HOLDINGS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"tickers": [], "updated": ""})

@app.route('/api/holdings', methods=['POST'])
def set_holdings():
    data = request.get_json(silent=True) or {}
    tickers_str = data.get('tickers', '').strip().upper()
    if not tickers_str:
        holdings = {"tickers": [], "updated": datetime.now().strftime('%Y-%m-%d %H:%M')}
    else:
        tickers = [t.strip() for t in tickers_str.split() if t.strip()]
        holdings = {"tickers": tickers, "updated": datetime.now().strftime('%Y-%m-%d %H:%M')}
    with open(HOLDINGS_FILE, 'w') as f:
        json.dump(holdings, f, indent=2)
    return jsonify({"message": f"Saved {len(holdings['tickers'])} tickers", "holdings": holdings})

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(running_tasks)

TRADE_STATE_FILE = '/home/ubuntu/trade_state.json'

@app.route('/api/cash_buffer', methods=['POST'])
def update_cash_buffer():
    data = request.get_json() or {}
    if str(data.get('password', '')) != TRADE_PIN:
        return jsonify({"error": "잘못된 비밀번호"}), 403
    new_buffer = data.get('cash_buffer')
    if new_buffer is None or not isinstance(new_buffer, (int, float)):
        return jsonify({"error": "cash_buffer 값 필요 (0.02~0.80)"}), 400
    if not (0.01 <= new_buffer <= 0.95):
        return jsonify({"error": "범위: 0.01~0.95"}), 400

    # Read existing state, update buffer
    state = {}
    try:
        with open(TRADE_STATE_FILE, 'r') as f:
            state = json.load(f)
    except Exception:
        pass
    state['cash_buffer'] = round(new_buffer, 2)
    state['buffer_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    try:
        tmp = TRADE_STATE_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, TRADE_STATE_FILE)
    except Exception as e:
        return jsonify({"error": f"저장 실패: {e}"}), 500

    invest_pct = round((1 - new_buffer) * 100)
    return jsonify({"message": f"Cash buffer {new_buffer:.0%} (투자 {invest_pct}%) 설정 완료"})

@app.route('/api/cash_buffer', methods=['GET'])
def get_cash_buffer():
    try:
        with open(TRADE_STATE_FILE, 'r') as f:
            state = json.load(f)
        return jsonify({"cash_buffer": state.get('cash_buffer', 0.02)})
    except Exception:
        return jsonify({"cash_buffer": 0.02})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
