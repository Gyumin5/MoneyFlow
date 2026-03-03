"""
Trade API Server - Flask based
Provides web API to trigger auto_trade scripts
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from flask import Flask, jsonify, request
import subprocess
import threading

app = Flask(__name__)

# CORS 허용 (브라우저에서 다른 포트 접근 허용)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# 실행 상태 추적
running_tasks = {}

def run_trade_async(exchange: str, force: bool = False, trade: bool = True, target_amount: int = 0):
    """비동기로 트레이딩 스크립트 실행"""
    task_id = f"{exchange}_{int(os.times().elapsed)}"
    running_tasks[task_id] = {"status": "running", "output": ""}
    
    try:
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        cmd = ["python3", os.path.join(project_root, "trade", f"auto_trade_{exchange}.py")]
        if trade:
            cmd.append("--trade")
        if force:
            cmd.append("--force")
        if target_amount > 0:
            cmd.extend(["--amount", str(target_amount)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=project_root)
        running_tasks[task_id] = {
            "status": "completed",
            "output": result.stdout + result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        running_tasks[task_id] = {"status": "timeout", "output": "Script timed out after 5 minutes"}
    except Exception as e:
        running_tasks[task_id] = {"status": "error", "output": str(e)}
    
    return task_id

@app.route('/api/trade/upbit', methods=['POST'])
def trade_upbit():
    """업비트 Force Trade 실행 (--trade --force --amount)"""
    # 이미 실행 중인지 확인
    for tid, task in running_tasks.items():
        if "upbit" in tid and task.get("status") == "running":
            return jsonify({"error": "Upbit trade is already running", "task_id": tid}), 409
    
    data = request.get_json(silent=True) or {}
    target_amount = int(data.get('target_amount', 0))
    
    # 비동기 실행
    thread = threading.Thread(target=run_trade_async, args=("upbit", True, True, target_amount))
    thread.start()
    
    msg = f"Upbit force trade started (Target: {target_amount} KRW)" if target_amount > 0 else "Upbit force trade started (Full Equity)"
    return jsonify({"message": msg, "status": "running"})

@app.route('/api/trade/bithumb', methods=['POST'])
def trade_bithumb():
    """빗썸 Force Trade 실행 (--trade --force --amount)"""
    # 이미 실행 중인지 확인
    for tid, task in running_tasks.items():
        if "bithumb" in tid and task.get("status") == "running":
            return jsonify({"error": "Bithumb trade is already running", "task_id": tid}), 409
    
    data = request.get_json(silent=True) or {}
    target_amount = int(data.get('target_amount', 0))
    
    # 비동기 실행
    thread = threading.Thread(target=run_trade_async, args=("bithumb", True, True, target_amount))
    thread.start()
    
    msg = f"Bithumb force trade started (Target: {target_amount} KRW)" if target_amount > 0 else "Bithumb force trade started (Full Equity)"
    return jsonify({"message": msg, "status": "running"})

@app.route('/api/status', methods=['GET'])
def get_status():
    """실행 상태 확인"""
    return jsonify(running_tasks)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # 포트 5000에서 실행 (8080은 정적 파일 서버용)
    app.run(host='0.0.0.0', port=5000, debug=False)
