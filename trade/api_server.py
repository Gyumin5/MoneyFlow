"""
Trade API Server - Flask based
Provides web API to trigger auto_trade scripts + stock holdings management + asset dashboard
"""
from flask import Flask, jsonify, request
import subprocess
import threading
import os
import json
import sqlite3
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
        # run_trade.sh 경유: flock 보호 일관 적용
        cmd = [f"/home/ubuntu/run_trade.sh", exchange]
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

    # 중복 실행 방지: run_trade.sh의 flock이 담당
    # API 측에서는 running_tasks로 중복 요청만 차단
    for tid, task in running_tasks.items():
        if "upbit" in tid and task.get("status") == "running":
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
    state['buffer_changed'] = True  # 다음 trade에서 트리거로 인식
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

# ─── Asset Dashboard ────────────────────────────────────────────
ASSETS_DB = '/home/ubuntu/assets.db'

def init_assets_db():
    """SQLite 초기화."""
    conn = sqlite3.connect(ASSETS_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT NOT NULL UNIQUE,
        stock_krw REAL DEFAULT 0,
        coin_krw REAL DEFAULT 0,
        cash_krw REAL DEFAULT 0,
        total_krw REAL DEFAULT 0,
        income_krw REAL DEFAULT 0,
        expense_krw REAL DEFAULT 0,
        memo TEXT DEFAULT '',
        accounts_json TEXT DEFAULT '{}',
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

init_assets_db()

@app.route('/api/assets/snapshots', methods=['GET'])
def get_snapshots():
    """전체 히스토리 조회."""
    conn = sqlite3.connect(ASSETS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM snapshots ORDER BY month").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/assets/snapshots', methods=['POST'])
def save_snapshot():
    """월별 스냅샷 저장 (upsert)."""
    data = request.get_json() or {}
    month = data.get('month')  # '2026-03'
    if not month:
        return jsonify({"error": "month 필요 (예: 2026-03)"}), 400

    stock = float(data.get('stock_krw', 0))
    coin = float(data.get('coin_krw', 0))
    cash = float(data.get('cash_krw', 0))
    total = stock + coin + cash
    income = float(data.get('income_krw', 0))
    expense = float(data.get('expense_krw', 0))
    memo = data.get('memo', '')
    accounts = json.dumps(data.get('accounts', {}), ensure_ascii=False)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    conn = sqlite3.connect(ASSETS_DB)
    conn.execute("""INSERT INTO snapshots (month, stock_krw, coin_krw, cash_krw, total_krw,
                    income_krw, expense_krw, memo, accounts_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(month) DO UPDATE SET
                    stock_krw=?, coin_krw=?, cash_krw=?, total_krw=?,
                    income_krw=?, expense_krw=?, memo=?, accounts_json=?, created_at=?""",
                 (month, stock, coin, cash, total, income, expense, memo, accounts, now,
                  stock, coin, cash, total, income, expense, memo, accounts, now))
    conn.commit()
    conn.close()
    return jsonify({"message": f"{month} 저장 완료", "total_krw": total})

@app.route('/api/assets/coin_balance', methods=['GET'])
def get_coin_balance():
    """업비트 코인 잔고 자동 조회."""
    try:
        import pyupbit
        from config import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
        upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        balances = upbit.get_balances()
        total_krw = 0
        krw_balance = 0
        holdings = []
        for b in balances:
            if not isinstance(b, dict): continue
            currency = b.get('currency', '')
            bal = float(b.get('balance', 0)) + float(b.get('locked', 0))
            if currency == 'KRW':
                krw_balance = bal
                continue
            if bal <= 0: continue
            price = pyupbit.get_current_price(f"KRW-{currency}") or 0
            val = bal * price
            if val >= 1000:
                holdings.append({'ticker': currency, 'qty': bal, 'price': price, 'value': val})
                total_krw += val
        total_krw += krw_balance
        return jsonify({
            "total_krw": total_krw,
            "krw_balance": krw_balance,
            "holdings": holdings,
            "updated": datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/assets/rebalance', methods=['POST'])
def calc_rebalance():
    """리밸런싱 배분 계산."""
    data = request.get_json() or {}
    stock = float(data.get('stock_krw', 0))
    coin = float(data.get('coin_krw', 0))
    cash = float(data.get('cash_krw', 0))
    additional = float(data.get('additional_krw', 0))

    total = stock + coin + cash + additional
    if total <= 0:
        return jsonify({"error": "총자산이 0"}), 400

    # 목표 비중: 주식 58.8%, 코인 39.2%, 현금 2%
    target_stock = total * 0.588
    target_coin = total * 0.392
    target_cash = total * 0.02

    diff_stock = target_stock - stock
    diff_coin = target_coin - coin
    diff_cash = target_cash - cash

    return jsonify({
        "total": total,
        "current": {"stock": stock, "coin": coin, "cash": cash},
        "current_pct": {
            "stock": stock / total * 100,
            "coin": coin / total * 100,
            "cash": cash / total * 100
        },
        "target": {"stock": target_stock, "coin": target_coin, "cash": target_cash},
        "target_pct": {"stock": 58.8, "coin": 39.2, "cash": 2.0},
        "diff": {"stock": diff_stock, "coin": diff_coin, "cash": diff_cash},
        "action": {
            "stock": f"+{diff_stock:,.0f}원 매수" if diff_stock > 0 else f"{diff_stock:,.0f}원 매도/출금",
            "coin": f"+{diff_coin:,.0f}원 입금" if diff_coin > 0 else f"{diff_coin:,.0f}원 출금",
            "cash": f"+{diff_cash:,.0f}원 확보" if diff_cash > 0 else f"{diff_cash:,.0f}원 투자로",
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
