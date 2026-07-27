"""vol_cap plateau 비단조(7% 골짜기) 원인 규명 — 아웃라이어 코인 탐지.

메인 헬스 vol_threshold 를 0.05~0.12 로 스윕(라이브 SPOT_KW 기준), 각 vthr 의
일별 held picks 를 trace 로 수집. 0.05 대비 새로 편입되는 코인과 그 코인의
보유기간 수익 기여를 계산해, 7% 에서 성능을 끌어내리는 아웃라이어를 특정.

read-only 분석. 라이브 코드 무수정.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import unified_backtest as ub

SPOT_KW = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, canary_hyst=0.015, n_snapshots=7,
    universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
    snap_interval_bars=217,
)
START, END = '2020-10-01', '2026-05-13'
VTHRS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]


def daily_target_from_trace(trace, all_dates):
    """trace(list of {date,target,rebal}) → DataFrame[date x coin] 일별 비중(ffill)."""
    rows = {}
    for t in trace:
        tgt = t.get('target') or {}
        rows[pd.Timestamp(t['date'])] = {k: v for k, v in tgt.items() if str(k).upper() != 'CASH'}
    df = pd.DataFrame(rows).T.reindex(all_dates).ffill().fillna(0.0)
    return df


def main():
    t0 = time.time()
    bars, funding = ub.load_data('D')
    # 코인 일별 수익률
    closes = {c: df['Close'] for c, df in bars.items() if c != 'BTC'}

    results = {}
    targets = {}
    for vthr in VTHRS:
        tr = []
        kw = dict(SPOT_KW); kw.update(start_date=START, end_date=END, vol_threshold=vthr, _trace=tr)
        m = ub.run(bars, funding, **kw)
        eq = m['_equity'].dropna()
        results[vthr] = (m, eq)
        tw = daily_target_from_trace(tr, eq.index)
        targets[vthr] = tw
        held = [c for c in tw.columns if tw[c].sum() > 0]
        print(f"vthr={vthr:.2f}: Cal={m['Cal']:.3f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} "
              f"Trades={m['Trades']} | 편입된 코인수={len(held)}")

    base = 0.05
    tw_base = targets[base]
    base_coins = set(c for c in tw_base.columns if tw_base[c].sum() > 0)

    print("\n=== 0.05 대비 각 vthr 에서 새로 편입되는 코인 (보유일수 / 보유기간 코인수익) ===")
    for vthr in VTHRS:
        if vthr == base:
            continue
        tw = targets[vthr]
        new_coins = [c for c in tw.columns if tw[c].sum() > 0 and c not in base_coins]
        info = []
        for c in new_coins:
            held_days = int((tw[c] > 0).sum())
            cser = closes.get(c)
            # 보유구간 코인 자체 수익 (held 마스크 구간의 누적 일수익)
            ret_contrib = np.nan
            if cser is not None:
                cr = cser.reindex(tw.index).pct_change().fillna(0.0)
                # 그날 비중 * 그날수익 합 (근사 기여도)
                ret_contrib = float((tw[c].shift(1).fillna(0.0) * cr).sum())
            info.append((c, held_days, ret_contrib))
        info.sort(key=lambda x: (x[2] if not np.isnan(x[2]) else 0))
        tag = " ← 골짜기" if abs(vthr - 0.07) < 1e-9 else ""
        print(f"\n vthr={vthr:.2f}{tag}: 신규편입 {len(new_coins)}개")
        for c, hd, rc in info:
            print(f"    {c:6s} 보유{hd:>4d}일  비중가중수익기여={rc:+.3f}")

    # 0.07 vs 0.05 equity 최대 이탈 구간
    _, eq5 = results[0.05]; _, eq7 = results[0.07]
    idx = eq5.index.intersection(eq7.index)
    rel = (eq7.reindex(idx) / eq7.reindex(idx).iloc[0]) / (eq5.reindex(idx) / eq5.reindex(idx).iloc[0])
    worst_date = rel.idxmin()
    print(f"\n=== 0.07이 0.05 대비 가장 뒤진 시점: {worst_date.date()} (상대비 {rel.min():.3f}) ===")
    win = pd.date_range(worst_date - pd.Timedelta(days=45), worst_date)
    tw7 = targets[0.07]
    held_then = {}
    for d in win:
        if d in tw7.index:
            for c in tw7.columns:
                if tw7.loc[d, c] > 0:
                    held_then[c] = held_then.get(c, 0) + 1
    print("  그 직전 45일간 0.07이 보유한 코인 (보유일수):")
    for c, n in sorted(held_then.items(), key=lambda x: -x[1]):
        extra = " [0.05엔 없음]" if c not in base_coins else ""
        print(f"    {c:6s} {n}일{extra}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
