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
    for tid, task in running_tasks.items():
        if "upbit" in tid and task.get("status") == "running":
            return jsonify({"error": "Upbit trade is already running", "task_id": tid}), 409
    data = request.get_json(silent=True) or {}
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
