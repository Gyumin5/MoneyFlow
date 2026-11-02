#!/usr/bin/env python3
"""비트파이넥스 공개 통로에서 코인 일봉을 받아온다 — 2017년까지 구간을 늘리기 위한 연구용 수집기.

운영자 지시(2026-11-02 텔레그램 9520·9521): "알트가 없어도 있는 코인만이라도 2017년 수준까지
늘릴 방법을 찾아봐" / "그때 그것만 있었다는 거니 그 전략이 그때도 그렇게 작동했을 거란 기대를
할 수 있잖아".

왜 비트파이넥스인가. 키 없이 일봉 이력을 주고 시작이 제일 이르다(실측 2026-11-02):
  · 야후       비트코인·라이트코인만 2014-09, 나머지는 전부 2017-11-09
  · 크립토컴페어 이제 키를 요구한다(401)
  · 바이낸스 현물 최초가 2017-08-17 이라 2017 상반기가 통째로 없다
  · 비트파이넥스 비트코인·라이트코인 2015-01, 이더리움 2016-03, 이더리움클래식 2016-07,
               지캐시 2016-10, 모네로 2016-11 → 2017-01 에 여섯 종목이 선다

상장폐지된 심볼도 이력은 그대로 준다. 그래서 지금 상장 목록(44개)만 보면 2017년 자료를
놓친다 — 당시 티커를 따로 적어 두고 하나씩 두드린다.

속도 제한이 빡빡하다(공개 통로 분당 30회 안팎). 간격을 두고 429 면 물러섰다 다시 친다.
한 번에 다 받지 말고 이어서 받을 수 있게 종목마다 파일을 따로 쓴다 — 중간에 끊겨도
이미 받은 것은 건너뛴다.

쓰는 법:
  python3 fetch_bitfinex_daily.py --probe          # 어느 종목이 언제부터 있나만 본다
  python3 fetch_bitfinex_daily.py --fetch          # 실제로 받는다 (이어받기)
  python3 fetch_bitfinex_daily.py --fetch --only BTC,ETH

산출물은 연구용이다. data/ 의 실매매·채택 백테스트 입력은 안 건드린다.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', '..', '..', 'data_research', 'bitfinex_daily')
PROBE_PATH = os.path.join(OUT_DIR, '_probe.json')

API = 'https://api-pub.bitfinex.com/v2'
HDR = {'User-Agent': 'Mozilla/5.0 (research)'}

# 2017~2020 에 비트파이넥스에 있던 티커. 지금 상장돼 있는지와 무관하게 두드린다.
# (비트파이넥스는 세 글자 티커를 쓴다 — 대시=DSH, 아이오타=IOT, 퀀텀=QTM, 마나=MNA)
CANDIDATES = """
BTC LTC ETH ETC XRP XMR ZEC DSH IOT EOS OMG NEO BCH BAB BSV BTG QTM AVT EDO DAT QSH YYW RRT
SNT GNT FUN ZRX TNB SPK TRX MNA ATM RCN RLC AID SNG REP ELF IOS AIO REQ BFT ODE ANT DGB BCI
STJ XLM XVG MIT PAI DRN UTK VEE ESS IQX DTX SEE ORS GOT WAX LYM UOS BOX MLN NEC UDC TKN LRC
AGI NCA CND DAI XTZ ZIL KNC LINK MKR BAT XAUT USTF0 ALG CHZ ADA DOT FIL SOL AVAX MATIC DOGE
SHIB UNI AAVE YFI COMP SUSHI CRV LEO
""".split()

START_MS = int(datetime(2014, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _get(url, tries=6):
    """공개 통로를 두드린다. 429 면 물러섰다 다시 친다."""
    delay = 3.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return None


def probe(gap):
    """각 후보가 언제부터 있는지 한 줄씩 본다."""
    os.makedirs(OUT_DIR, exist_ok=True)
    found = {}
    if os.path.exists(PROBE_PATH):
        found = json.load(open(PROBE_PATH, encoding='utf-8'))
    todo = [s for s in CANDIDATES if s not in found]
    print(f"두드릴 것 {len(todo)}개 (이미 본 것 {len(found)}개)", flush=True)
    for i, sym in enumerate(todo):
        url = f"{API}/candles/trade:1D:t{sym}USD/hist?limit=1&sort=1&start={START_MS}"
        try:
            d = _get(url)
        except Exception as e:
            print(f"  {sym}: 실패 {str(e)[:50]}", flush=True)
            found[sym] = None
            continue
        if d:
            first = datetime.fromtimestamp(d[0][0] / 1000, timezone.utc).date().isoformat()
            found[sym] = first
            print(f"  [{i+1}/{len(todo)}] {sym}: {first}", flush=True)
        else:
            found[sym] = None
            print(f"  [{i+1}/{len(todo)}] {sym}: 없음", flush=True)
        json.dump(found, open(PROBE_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        time.sleep(gap)

    live = sorted((v, k) for k, v in found.items() if v)
    print(f"\n응답한 종목 {len(live)}개")
    for cut in ('2017-01-01', '2017-07-01', '2018-01-01', '2019-01-01', '2020-10-01'):
        print(f"  {cut} 시점 가용: {sum(1 for d, _ in live if d < cut)}개")
    print("\n2018-01-01 이전 시작분:")
    for d, s in live:
        if d < '2018-01-01':
            print(f"  {d}  {s}")
    return 0


def fetch_one(sym, gap):
    """한 종목의 일봉 전체를 받는다. 비트파이넥스는 호출당 10,000행까지 준다."""
    path = os.path.join(OUT_DIR, f"{sym}USD.csv")
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return 'skip', 0
    rows = []
    start = START_MS
    while True:
        url = (f"{API}/candles/trade:1D:t{sym}USD/hist"
               f"?limit=10000&sort=1&start={start}")
        d = _get(url)
        if not d:
            break
        rows.extend(d)
        if len(d) < 10000:
            break
        start = d[-1][0] + 86400000
        time.sleep(gap)
    if not rows:
        return 'empty', 0
    # [ts, open, close, high, low, volume] — 종가가 셋째다. 순서를 틀리면 조용히 다른 전략이 된다.
    seen = set()
    out = []
    for ts, o, c, h, lo, v in rows:
        if ts in seen:
            continue
        seen.add(ts)
        day = datetime.fromtimestamp(ts / 1000, timezone.utc).date().isoformat()
        if float(c) <= 0:
            continue
        out.append((day, float(o), float(h), float(lo), float(c), float(v)))
    out.sort()
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Date,Open,High,Low,Close,Volume\n')
        for r in out:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
    return 'ok', len(out)


def fetch(only, gap):
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PROBE_PATH):
        print("먼저 --probe 를 돌려라 — 어느 종목이 있는지부터 안다")
        return 2
    found = json.load(open(PROBE_PATH, encoding='utf-8'))
    syms = [k for k, v in found.items() if v]
    if only:
        want = {x.strip().upper() for x in only.split(',')}
        syms = [s for s in syms if s in want]
    syms.sort(key=lambda s: found[s])
    print(f"받을 종목 {len(syms)}개", flush=True)
    for i, sym in enumerate(syms):
        try:
            status, n = fetch_one(sym, gap)
        except Exception as e:
            print(f"  [{i+1}/{len(syms)}] {sym}: 실패 {str(e)[:60]}", flush=True)
            continue
        print(f"  [{i+1}/{len(syms)}] {sym}: {status} {n}행 (시작 {found[sym]})", flush=True)
        if status != 'skip':
            time.sleep(gap)
    print(f"\n끝. 산출물 {os.path.abspath(OUT_DIR)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--gap', type=float, default=2.5, help="호출 사이 간격(초)")
    a = ap.parse_args()
    if a.probe:
        return probe(a.gap)
    if a.fetch:
        return fetch(a.only, a.gap)
    ap.print_help()
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
