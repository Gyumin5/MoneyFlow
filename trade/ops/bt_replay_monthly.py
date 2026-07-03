#!/usr/bin/env python3
"""V24 월 1회 BT replay — 라이브 파라미터로 BT 재실행, 기록된 baseline 대비 drift 확인.

목적: 라이브 코드의 잠재적 변경/데이터 변경이 BT 결과를 흔드는지 monthly 감시.
실행 결과는 CSV 누적 + 텔레그램 알림 (드리프트 또는 회귀 발견 시).

사용
- cron 월 1회 (예: 매월 1일 09:30 KST)
- baseline: 2026-05-14 펀딩 fix 후 V24 정식 Cal 4.05 (CAGR +256%, MDD -63.1%)
"""
import os, sys, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
TRADE_DIR = os.path.abspath(os.path.join(HERE, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(TRADE_DIR, '..'))
STRAT_DIR = os.path.join(PROJECT_ROOT, 'strategies', 'cap_defend')
sys.path.insert(0, HERE)
sys.path.insert(0, TRADE_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, STRAT_DIR)

OUT_DIR = os.path.join(HERE, 'bt_replay_log')
os.makedirs(OUT_DIR, exist_ok=True)
LOG_CSV = os.path.join(OUT_DIR, 'bt_replay_history.csv')

# 2026-05-14 펀딩 fix 후 V24 정식 (sma=42 ms=18 ml=127 sn=95 n=5 drift=0.03, tx=0.0006, 2020-10-01~2026-05-13)
BASELINE = dict(Cal=4.05, CAGR=2.56, MDD=-0.631, Sharpe=None)
# 허용 drift: Cal ±15%, CAGR ±15%, MDD ±5pp
TOL = dict(Cal=0.15, CAGR=0.15, MDD=0.05)

V24_FUT_CFG = dict(
    interval='D', leverage=3.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
    universe_size=3, selection='greedy', cap=1/3,
    tx_cost=0.0006, maint_rate=0.004,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, health_mode='mom2vol',
    start_date='2020-10-01', end_date='2026-05-13',
)


def metrics(eq):
    import numpy as np
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return dict(Cal=float(cal), CAGR=float(cagr), MDD=float(mdd), Sharpe=float(sh))


def run_bt():
    from backtest_futures_full import load_data, run
    bars, funding = load_data('D')
    m = run(bars, funding, **V24_FUT_CFG)
    return metrics(m['_equity'])


def check_drift(cur):
    msgs = []
    for k, tol in TOL.items():
        b = BASELINE.get(k)
        c = cur.get(k)
        if b is None or c is None:
            continue
        if k == 'MDD':
            if abs(c - b) > tol:
                msgs.append(f"{k}: {c:+.1%} vs baseline {b:+.1%} (Δ{(c-b)*100:+.1f}pp 허용 ±{tol*100:.0f}pp)")
        else:
            ratio = abs(c - b) / abs(b) if b != 0 else 0
            if ratio > tol:
                msgs.append(f"{k}: {c:.2f} vs baseline {b:.2f} (Δ{ratio*100:+.0f}% 허용 ±{tol*100:.0f}%)")
    return msgs


def append_log(cur):
    new = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, 'a') as f:
        if new:
            f.write("date,Cal,CAGR,MDD,Sharpe\n")
        f.write(f"{dt.date.today()},{cur['Cal']:.3f},{cur['CAGR']:.4f},{cur['MDD']:.4f},{cur['Sharpe']:.3f}\n")


def telegram_alert(msg):
    try:
        from common.notify import send_telegram
        import importlib.util
        cfg_spec = importlib.util.spec_from_file_location(
            'config', os.path.join(TRADE_DIR, 'config.py'))
        if cfg_spec is None or cfg_spec.loader is None:
            return
        cfg = importlib.util.module_from_spec(cfg_spec)
        cfg_spec.loader.exec_module(cfg)
        token = getattr(cfg, 'TELEGRAM_BOT_TOKEN', None)
        chat = getattr(cfg, 'TELEGRAM_CHAT_ID', None)
        if token and chat:
            send_telegram(token, chat, msg, prefix='경고', timeout=10)
    except Exception as e:
        print(f"(텔레그램 알림 실패: {e})")


def main():
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M}] V24 fut BT replay 시작")
    cur = run_bt()
    print(f"  현재: Cal={cur['Cal']:.2f} CAGR={cur['CAGR']:+.1%} MDD={cur['MDD']:+.1%} Sharpe={cur['Sharpe']:.2f}")
    print(f"  baseline: Cal={BASELINE['Cal']} CAGR={BASELINE['CAGR']:+.1%} MDD={BASELINE['MDD']:+.1%}")
    append_log(cur)
    drift = check_drift(cur)
    if drift:
        msg = "⚠️ V24 fut BT replay drift 발견\n" + "\n".join(drift)
        print(msg)
        telegram_alert(msg)
        sys.exit(1)
    print("✅ V24 fut BT replay OK — baseline tolerance 내")
    sys.exit(0)


if __name__ == '__main__':
    main()
