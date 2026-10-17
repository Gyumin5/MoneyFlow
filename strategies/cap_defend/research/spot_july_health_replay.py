"""라이브 현물 유니버스 16종에 현물 헬스 필터를 걸어 본다 — 07-14 플립 봉과 그 뒤 며칠."""
import os
import numpy as np, pandas as pd

ROOT = "/home/gmoh/mon/251229"
UNIV = ["BTC","ETH","XRP","SOL","TRX","DOGE","XLM","LINK","ADA","BCH",
        "SUI","HBAR","AVAX","NEAR","SHIB","UNI"]
VOL_TH, VOL_LB = 0.05, 90

def daily(sym):
    p = os.path.join(ROOT, f"data/futures/{sym}USDT_1h.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    s = df["Close"] if "Close" in df.columns else df.iloc[:, -1]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.resample("1D").last().dropna()

series = {s: daily(s) for s in UNIV}
missing = [s for s, v in series.items() if v is None]
if missing:
    print("자료 없음:", missing)

for bar in ["2026-07-14","2026-07-15","2026-07-20","2026-07-25","2026-07-31","2026-08-01"]:
    ok20, ok18 = [], []
    for s, v in series.items():
        if v is None:
            continue
        x = v[v.index <= pd.Timestamp(bar, tz="UTC")]
        if len(x) < 140 or str(x.index[-1].date()) != bar:
            continue
        c = x.values.astype(float)
        m20 = c[-1]/c[-21]-1; m18 = c[-1]/c[-19]-1; m127 = c[-1]/c[-128]-1
        vol = float(np.std(np.diff(np.log(c[-VOL_LB-1:])), ddof=1))
        if m20 > 0 and m127 > 0 and vol <= VOL_TH: ok20.append(s)
        if m18 > 0 and m127 > 0 and vol <= VOL_TH: ok18.append(s)
    print(f"{bar} 완료봉 — 현물조건(20일) 통과 {len(ok20)}종 {ok20} | 선물조건(18일) {len(ok18)}종 {ok18}")

print()
print("TRX·UNI 상세")
for s in ["TRX","UNI"]:
    v = series[s]
    for bar in ["2026-07-14","2026-07-15","2026-08-01"]:
        x = v[v.index <= pd.Timestamp(bar, tz="UTC")]
        c = x.values.astype(float)
        m20 = c[-1]/c[-21]-1; m18 = c[-1]/c[-19]-1; m127 = c[-1]/c[-128]-1
        vol = float(np.std(np.diff(np.log(c[-VOL_LB-1:])), ddof=1))
        print(f"  {s} {bar}: m20={m20*100:+6.2f}% m18={m18*100:+6.2f}% m127={m127*100:+7.2f}% vol90={vol*100:5.2f}%")
