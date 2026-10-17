# 현물 라이브 카나리(바이낸스 spot BTCUSDT)와 BT 카나리(perp)를 같은 날짜축에서 비교한다.
import io, json, os, urllib.request
import numpy as np, pandas as pd

def binance_spot_daily(sym="BTCUSDT", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit={limit}"
    d = json.loads(urllib.request.urlopen(url, timeout=30).read())
    idx = pd.to_datetime([r[0] for r in d], unit="ms", utc=True)
    return pd.Series([float(r[4]) for r in d], index=idx, name="spot")

ROOT = "/home/gmoh/mon/251229"
perp = pd.read_csv(os.path.join(ROOT, "data/futures/BTCUSDT_1h.csv"),
                   index_col=0, parse_dates=True)
pc = perp["Close"] if "Close" in perp.columns else perp.iloc[:, -1]
pc.index = pd.to_datetime(pc.index, utc=True)
perp_d = pc.resample("1D").last().dropna()
perp_d.name = "perp"

sp = binance_spot_daily()
df = pd.concat([sp, perp_d], axis=1).dropna()

def run(series, hyst=0.015, p=42, start_on=False):
    sma = series.rolling(p).mean()
    on, out = start_on, {}
    for d, c in series.items():
        s = sma.get(d, np.nan)
        if not np.isfinite(s):
            continue
        on = (not (c < s * (1 - hyst))) if on else (c > s * (1 + hyst))
        out[d] = (on, c / s)
    return out

a = run(df["spot"]); b = run(df["perp"])
print("날짜(UTC 완료봉)  spot종가  spot비율 spotON | perp비율 perpON")
flips = []
prev = None
for d in df.loc["2026-05-01":"2026-08-10"].index:
    if d not in a or d not in b:
        continue
    so, sr = a[d]; po, pr = b[d]
    if prev is not None and (so, po) != prev:
        flips.append((d.date(), so, po))
    prev = (so, po)
    if "2026-07-05" <= str(d.date()) <= "2026-08-05":
        print(f"{d.date()}  {df['spot'][d]:10.2f}  {sr:.4f} {'ON ' if so else 'OFF'} | {pr:.4f} {'ON ' if po else 'OFF'}"
              + ("   <<< 다름" if so != po else ""))
print()
print("전환 (spot, perp) 5~8월:")
for d, so, po in flips:
    print(f"  {d}  spot={'ON' if so else 'OFF'}  perp={'ON' if po else 'OFF'}")
