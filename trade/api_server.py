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
APP_HOME = os.environ.get('MONEYFLOW_APP_HOME', os.getcwd())

# 인증 토큰 (서버 환경변수 TRADE_PIN — 길고 랜덤한 값 권장)
TRADE_PIN = os.environ.get('TRADE_PIN', '')

# CORS: 같은 서버에서만 허용 (포트 8080 = serve.py)
ALLOWED_ORIGINS = [o for o in os.environ.get('ALLOWED_ORIGINS', '').split(',') if o]  # 서버에서 환경변수로 설정

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin', '')
    if not ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = '*'
    elif origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

def require_auth():
    """쓰기 API 인증 체크. PIN이 없거나 불일치하면 403."""
    data = request.get_json(silent=True) or {}
    pwd = data.get('password', request.args.get('password', ''))
    if not TRADE_PIN or str(pwd) != TRADE_PIN:
        return False
    return True

running_tasks = {}
HOLDINGS_FILE = os.environ.get('HOLDINGS_FILE', os.path.join(APP_HOME, 'my_stock_holdings.json'))

def run_trade_async(exchange: str, force: bool = False, trade: bool = True, target_amount: int = 0):
    task_id = f"{exchange}_{int(os.times().elapsed)}"
    running_tasks[task_id] = {"status": "running", "output": ""}
    try:
        # run_trade.sh 경유: flock 보호 일관 적용
        cmd = [os.environ.get('RUN_TRADE_SCRIPT', os.path.join(APP_HOME, 'run_trade.sh')), exchange]
        if trade: cmd.append("--trade")
        if force: cmd.append("--force")
        if target_amount > 0: cmd.extend(["--amount", str(target_amount)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=APP_HOME)
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
    if not require_auth():
        return jsonify({"error": "인증 필요"}), 403
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

TRADE_STATE_FILE = os.environ.get('COIN_TRADE_STATE_FILE', os.path.join(APP_HOME, 'trade_state.json'))

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
    state['buffer_pct'] = state['cash_buffer']
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
        return jsonify({"cash_buffer": state.get('cash_buffer', state.get('buffer_pct', 0.02))})
    except Exception:
        return jsonify({"cash_buffer": 0.02})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# ─── Asset Dashboard ────────────────────────────────────────────
ASSETS_DB = os.environ.get('ASSETS_DB', os.path.join(APP_HOME, 'assets.db'))

def init_assets_db():
    """SQLite 초기화 (v3: futures_krw 추가)."""
    conn = sqlite3.connect(ASSETS_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date TEXT UNIQUE NOT NULL,
        stock_krw REAL DEFAULT 0,
        coin_krw REAL DEFAULT 0,
        futures_krw REAL DEFAULT 0,
        cash_krw REAL DEFAULT 0,
        total_krw REAL DEFAULT 0,
        fx_rate REAL DEFAULT 0,
        usd_cash REAL DEFAULT 0,
        memo TEXT DEFAULT '',
        accounts_json TEXT DEFAULT '{}',
        created_at TEXT
    )""")
    # 기존 DB 마이그레이션
    try:
        conn.execute("ALTER TABLE snapshots ADD COLUMN futures_krw REAL DEFAULT 0")
    except Exception:
        pass
    conn.commit()
    conn.close()

init_assets_db()

@app.route('/api/assets/snapshots', methods=['GET'])
def get_snapshots():
    """전체 히스토리 조회."""
    conn = sqlite3.connect(ASSETS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM snapshots ORDER BY snapshot_date").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/assets/snapshots', methods=['POST'])
def save_snapshot():
    """스냅샷 저장 (upsert by date)."""
    if not require_auth():
        return jsonify({"error": "인증 필요"}), 403
    data = request.get_json() or {}
    date = data.get('month', data.get('snapshot_date'))  # 호환: month 또는 snapshot_date
    if not date:
        return jsonify({"error": "snapshot_date 필요 (예: 2026-03-26)"}), 400

    stock = float(data.get('stock_krw', 0))
    coin = float(data.get('coin_krw', 0))
    futures = float(data.get('futures_krw', 0))
    cash = float(data.get('cash_krw', 0))
    total = stock + coin + futures + cash
    fx_rate = float(data.get('fx_rate', 0))
    usd_cash = float(data.get('usd_cash', 0))
    memo = data.get('memo', '')
    accounts = json.dumps(data.get('accounts', {}), ensure_ascii=False)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    conn = sqlite3.connect(ASSETS_DB)
    conn.execute("""INSERT INTO snapshots (snapshot_date, stock_krw, coin_krw, futures_krw, cash_krw, total_krw,
                    fx_rate, usd_cash, memo, accounts_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_date) DO UPDATE SET
                    stock_krw=?, coin_krw=?, futures_krw=?, cash_krw=?, total_krw=?,
                    fx_rate=?, usd_cash=?, memo=?, accounts_json=?, created_at=?""",
                 (date, stock, coin, futures, cash, total, fx_rate, usd_cash, memo, accounts, now,
                  stock, coin, futures, cash, total, fx_rate, usd_cash, memo, accounts, now))
    conn.commit()
    conn.close()
    return jsonify({"message": f"{date} 저장 완료", "total_krw": total})

def _get_coin_balance_data() -> dict:
    """업비트 코인 잔고 자동 조회."""
    import pyupbit
    from config import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
    upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
    balances = upbit.get_balances()
    total_krw = 0.0
    krw_balance = 0.0
    holdings = []
    for b in balances:
        if not isinstance(b, dict):
            continue
        currency = b.get('currency', '')
        bal = float(b.get('balance', 0)) + float(b.get('locked', 0))
        if currency == 'KRW':
            krw_balance = bal
            continue
        if bal <= 0:
            continue
        try:
            price = pyupbit.get_current_price(f"KRW-{currency}") or 0
        except Exception:
            price = 0
        val = bal * price
        if val >= 1000:
            holdings.append({
                'ticker': currency,
                'qty': bal,
                'price': price,
                'value': val,
                'weight_value_krw': val,
            })
            total_krw += val
    total_krw += krw_balance
    weights = {}
    if total_krw > 0:
        for h in holdings:
            weights[h['ticker']] = float(h.get('weight_value_krw', 0.0)) / total_krw
        weights['현금'] = krw_balance / total_krw
    return {
        "total_krw": total_krw,
        "krw_balance": krw_balance,
        "holdings": holdings,
        "weights": weights,
        "updated": datetime.now().strftime('%Y-%m-%d %H:%M')
    }


@app.route('/api/assets/coin_balance', methods=['GET'])
def get_coin_balance():
    try:
        return jsonify(_get_coin_balance_data())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _get_usdkrw_rate() -> float:
    """KIS 기준 USD/KRW 환율."""
    rate = 1500.0
    try:
        from auto_trade_kis import _get, KIS_ACCOUNT, KIS_ACCOUNT_PROD
        data = _get("/uapi/overseas-stock/v1/trading/foreign-margin", "TTTC2101R", {
            "CANO": KIS_ACCOUNT, "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        }, retries=1)
        for item in data.get('output', []):
            if isinstance(item, dict) and item.get('natn_name') == '미국' and item.get('crcy_cd') == 'USD':
                rate = float(item.get('bass_exrt', 1500))
                break
    except Exception:
        pass
    return rate


def _get_upbit_usdt_krw_rate() -> float:
    """업비트 KRW-USDT 가격 기반 환율. 김프 반영 목적."""
    rate = 0.0
    try:
        import pyupbit
        price = pyupbit.get_current_price("KRW-USDT")
        if price and float(price) > 0:
            rate = float(price)
    except Exception:
        pass
    return rate


def _get_stock_balance_data() -> dict:
    """한투 해외주식 잔고 자동 조회."""
    from auto_trade_kis import get_balance, get_buying_power_usd

    holdings_raw, _ = get_balance()
    stock_eval = sum(h['eval_amt'] for h in holdings_raw)
    buying_power = get_buying_power_usd()
    rate = _get_usdkrw_rate()
    total_usd = stock_eval + buying_power
    total_krw = total_usd * rate
    holdings = []
    for h in holdings_raw:
        holdings.append({
            **h,
            "price_krw": float(h.get("current_price", 0.0)) * rate,
            "value_krw": float(h.get("eval_amt", 0.0)) * rate,
            "weight_value_krw": float(h.get("eval_amt", 0.0)) * rate,
        })
    weights = {}
    if total_krw > 0:
        for h in holdings:
            weights[h['ticker']] = float(h.get('weight_value_krw', 0.0)) / total_krw
        weights['현금'] = (buying_power * rate) / total_krw
    return {
        "total_krw": total_krw,
        "total_usd": total_usd,
        "stock_eval_usd": stock_eval,
        "buying_power_usd": buying_power,
        "cash_usd": buying_power,
        "cash_krw": buying_power * rate,
        "exchange_rate": rate,
        "holdings": holdings,
        "weights": weights,
        "updated": datetime.now().strftime('%Y-%m-%d %H:%M')
    }


@app.route('/api/assets/stock_balance', methods=['GET'])
def get_stock_balance():
    try:
        return jsonify(_get_stock_balance_data())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_binance_balance_data(exchange_rate: float | None = None) -> dict:
    """바이낸스 USDT-M 선물 잔고/포지션 조회."""
    from binance.client import Client
    import config

    rate = exchange_rate or _get_upbit_usdt_krw_rate() or _get_usdkrw_rate()
    api_key = getattr(config, 'BINANCE_API_KEY', '')
    api_secret = getattr(config, 'BINANCE_API_SECRET', '')
    client = Client(api_key, api_secret)

    account = client.futures_account()
    fut_total_usdt = float(account.get('totalMarginBalance') or account.get('totalWalletBalance') or 0.0)
    fut_available_usdt = float(account.get('availableBalance') or 0.0)
    unrealized_usdt = float(account.get('totalUnrealizedProfit') or 0.0)

    # Spot 잔고 합산 (USDT는 현금, 그 외 토큰은 holdings에 추가)
    spot_usdt = 0.0
    spot_other_value_usdt = 0.0
    spot_holdings = []
    try:
        spot_acc = client.get_account()
        spot_prices = None
        for b in spot_acc.get('balances', []):
            asset = b.get('asset', '')
            qty = float(b.get('free', 0)) + float(b.get('locked', 0))
            if qty <= 0:
                continue
            if asset == 'USDT':
                spot_usdt += qty
                continue
            # 다른 토큰: USDT 페어 시세로 평가, 없으면 0 처리
            if spot_prices is None:
                try:
                    spot_prices = {p['symbol']: float(p['price']) for p in client.get_all_tickers()}
                except Exception:
                    spot_prices = {}
            sym = asset + 'USDT'
            price = spot_prices.get(sym, 0.0)
            value = qty * price
            if value < 1.0:  # 1 USDT 미만 dust는 무시
                continue
            spot_other_value_usdt += value
            spot_holdings.append({
                "ticker": asset,
                "symbol": sym,
                "qty": qty,
                "entry_price": 0.0,
                "price": price,
                "price_krw": price * rate,
                "value_usdt": value,
                "value_krw": value * rate,
                "weight_value_krw": value * rate,
                "pnl_usdt": 0.0,
                "pnl_krw": 0.0,
                "account": "spot",
            })
    except Exception:
        pass

    total_usdt = fut_total_usdt + spot_usdt + spot_other_value_usdt
    available_usdt = fut_available_usdt + spot_usdt

    holdings = []
    for p in client.futures_position_information():
        qty = float(p.get('positionAmt') or 0.0)
        if abs(qty) <= 1e-12:
            continue
        symbol = p.get('symbol', '')
        mark = float(p.get('markPrice') or 0.0)
        entry = float(p.get('entryPrice') or 0.0)
        notional = abs(float(p.get('notional') or (qty * mark)))
        pnl = float(p.get('unRealizedProfit') or 0.0)
        margin_usdt = float(
            p.get('positionInitialMargin')
            or p.get('initialMargin')
            or p.get('isolatedMargin')
            or 0.0
        )
        lev = float(p.get('leverage') or 0.0)
        if margin_usdt > 0:
            real_notional = margin_usdt
            if lev <= 0:
                lev = notional / margin_usdt if margin_usdt > 0 else 1.0
        elif lev > 0:
            real_notional = notional / lev
        else:
            real_notional = notional
            lev = 1.0
        holdings.append({
            "ticker": symbol.replace("USDT", ""),
            "symbol": symbol,
            "qty": qty,
            "entry_price": entry,
            "price": mark,
            "price_krw": mark * rate,
            "value_usdt": real_notional,
            "value_krw": real_notional * rate,
            "weight_value_krw": (margin_usdt or real_notional) * rate,
            "leverage": lev,
            "notional_usdt": notional,
            "pnl_usdt": pnl,
            "pnl_krw": pnl * rate,
        })
    weights = {}
    try:
        with open(os.environ.get('BINANCE_STATE_FILE', os.path.join(APP_HOME, 'binance_state.json')), 'r') as f:
            state = json.load(f)
        last_target = state.get('last_target') or {}
        total_target = sum(float(v) for v in last_target.values() if isinstance(v, (int, float)))
        if total_target > 0:
            for k, v in last_target.items():
                if isinstance(v, (int, float)):
                    weights['현금' if str(k).upper() == 'CASH' else str(k)] = float(v) / total_target
    except Exception:
        pass
    holdings.extend(spot_holdings)
    if not weights and total_usdt > 0:
        total_krw = total_usdt * rate
        for h in holdings:
            weights[h['ticker']] = float(h.get('weight_value_krw', 0.0)) / total_krw
        weights['현금'] = (available_usdt * rate) / total_krw
    return {
        "total_krw": total_usdt * rate,
        "total_usdt": total_usdt,
        "cash_usdt": available_usdt,
        "cash_krw": available_usdt * rate,
        "spot_usdt": spot_usdt,
        "spot_other_value_usdt": spot_other_value_usdt,
        "futures_total_usdt": fut_total_usdt,
        "futures_cash_usdt": fut_available_usdt,
        "unrealized_usdt": unrealized_usdt,
        "exchange_rate": rate,
        "holdings": holdings,
        "weights": weights,
        "updated": datetime.now().strftime('%Y-%m-%d %H:%M')
    }


@app.route('/api/assets/binance_balance', methods=['GET'])
def get_binance_balance():
    try:
        return jsonify(_get_binance_balance_data())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/assets/live_overview', methods=['GET'])
def get_live_overview():
    """한 번에 한투/업비트/바이낸스 실계좌 현황 조회."""
    result = {"updated": datetime.now().strftime('%Y-%m-%d %H:%M'), "accounts": {}}
    total_krw = 0.0

    try:
        stock = _get_stock_balance_data()
        result["accounts"]["stock_kis"] = stock
        total_krw += float(stock.get("total_krw", 0.0))
    except Exception as e:
        result["accounts"]["stock_kis"] = {"error": str(e)}

    try:
        coin_data = _get_coin_balance_data()
        result["accounts"]["coin_upbit"] = coin_data
        total_krw += float(coin_data.get("total_krw", 0.0))
    except Exception as e:
        result["accounts"]["coin_upbit"] = {"error": str(e)}

    try:
        rate = _get_upbit_usdt_krw_rate()
        fut = _get_binance_balance_data(rate)
        result["accounts"]["coin_binance"] = fut
        total_krw += float(fut.get("total_krw", 0.0))
    except Exception as e:
        result["accounts"]["coin_binance"] = {"error": str(e)}

    result["total_krw"] = total_krw
    return jsonify(result)


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
