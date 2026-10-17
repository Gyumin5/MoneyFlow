"""2026-07-14 완료봉에서 현물 헬스 필터(20/127, vol90<=5%)를 통과하는 코인을 센다.
같은 날 선물 파라미터(18/127)와 비교한다."""
import glob, os
import numpy as np, pandas as pd

ROOT = "/home/gmoh/mon/251229"
BAR = "2026-07-14"
VOL_TH, VOL_LB = 0.05, 90

def daily(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df["Close"] if "Close" in df.columns else df.iloc[:, -1]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.resample("1D").last().dropna()

def mom(c, n):
    return c[-1] / c[-1 - n] - 1.0

def vol(c, n):
    r = np.diff(np.log(c[-n - 1:]))
    return float(np.std(r, ddof=1))

rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "data/futures/*_1h.csv"))):
    sym = os.path.basename(p).replace("_1h.csv", "").replace("USDT", "")
    try:
        s = daily(p)
    except Exception:
        continue
    s = s[s.index <= pd.Timestamp(BAR, tz="UTC")]
    if len(s) < 140:
        continue
    if str(s.index[-1].date()) != BAR:
        continue
    c = s.values.astype(float)
    m20, m18, m127, v = mom(c, 20), mom(c, 18), mom(c, 127), vol(c, VOL_LB)
    rows.append((sym, m20, m18, m127, v,
                 m20 > 0 and m127 > 0 and v <= VOL_TH,
                 m18 > 0 and m127 > 0 and v <= VOL_TH))

rows.sort(key=lambda r: -r[1])
sp = [r for r in rows if r[5]]
fu = [r for r in rows if r[6]]
print(f"{BAR} 완료봉 — 조사 대상 {len(rows)}종")
print(f"현물 조건(단기20일) 통과: {len(sp)}종 {[r[0] for r in sp]}")
print(f"선물 조건(단기18일) 통과: {len(fu)}종 {[r[0] for r in fu]}")
print()
print("단기 모멘텀 상위 12 (sym  m20  m18  m127  vol90  현물OK 선물OK)")
for r in rows[:12]:
    print(f"  {r[0]:<6} {r[1]*100:+7.2f}% {r[2]*100:+7.2f}% {r[3]*100:+8.2f}% {r[4]*100:6.2f}%   "
          f"{'O' if r[5] else 'X'}      {'O' if r[6] else 'X'}")
