"""alloc_spot_n2 결과 조회 — 판정 규칙은 측정 전에 고정한 것 그대로 쓴다.

기준선은 60/20/20 n3 (현행 라이브). 보는 것 셋.
  1. 여섯 칸 전부 — 앵커 평균 지표와 기준선 대비 같은 창 짝 우세 비율
  2. 종목수 효과 — 같은 배분 안에서 n2 가 n3 를 이기나
  3. 판정 — 수익·칼마 두 기준 모두 기준선을 이기는 칸이 있나
"""
import glob
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'state/20261103-spot-n2'
files = sorted(glob.glob(f'{D}/n2_a*.npz'))
if not files:
    print(f'{D} 에 결과가 없다')
    raise SystemExit(1)

z0 = np.load(files[0], allow_pickle=True)
labels = [str(x) for x in z0['label']]
K = len(labels)

cal = np.mean([np.load(p)['cal'] for p in files], axis=0)
cagr = np.mean([np.load(p)['cagr'] for p in files], axis=0)
mdd = np.mean([np.load(p)['mdd'] for p in files], axis=0)
shp = np.mean([np.load(p)['sharpe'] for p in files], axis=0)
fpy = np.mean([np.load(p)['fires_per_year'] for p in files], axis=0)

# 창별 등수는 앵커를 가로로 잇는다 — 창 하나가 한 표다.
rc = np.concatenate([np.load(p)['rank_cal'] for p in files], axis=0)
rg = np.concatenate([np.load(p)['rank_cg'] for p in files], axis=0)
W = rc.shape[0]

base = labels.index('60/20/20 n3')


def row(i):
    dg, dc = rg[:, i] - rg[:, base], rc[:, i] - rc[:, base]
    mark = ' (기준선)' if i == base else ''
    return (f'  {labels[i]:<14} 칼마 {cal[i]:5.2f} · 연복리 {cagr[i]:6.1f}% · '
            f'낙폭 {mdd[i]:6.1f}% · 샤프 {shp[i]:4.2f} · 송금 {fpy[i]:5.2f}/년 · '
            f'우세창 수익 {100 * (dg < 0).mean():5.1f}% 칼마 {100 * (dc < 0).mean():5.1f}%'
            f'{mark}')


print(f'칸 {K}개 · 앵커 {len(files)}곳 · 창 {W}개')
print(f'기간 {z0["d0"]}~{z0["d1"]} ({int(z0["ndays"])}일) · 송금 지연 {int(z0["delay"])}영업일')
print('\n' + '=' * 108)
print('1. 여섯 칸 — 앵커 평균과 기준선 대비 같은 창 짝 우세 비율')
print('=' * 108)
for i in range(K):
    print(row(i))

print('\n' + '=' * 108)
print('2. 종목수 효과 — 같은 배분 안에서 둘이 셋을 이기나 (창 짝 비교)')
print('=' * 108)
for mix in ['60/20/20', '60/25/15', '60/30/10']:
    i2, i3 = labels.index(f'{mix} n2'), labels.index(f'{mix} n3')
    dg, dc = rg[:, i2] - rg[:, i3], rc[:, i2] - rc[:, i3]
    print(f'  {mix}: 둘이 셋보다 나은 창 — 수익 {100 * (dg < 0).mean():5.1f}% · '
          f'칼마 {100 * (dc < 0).mean():5.1f}% · '
          f'연복리 차 {cagr[i2] - cagr[i3]:+6.1f}%p · 칼마 차 {cal[i2] - cal[i3]:+5.2f}')

print('\n' + '=' * 108)
print('3. 판정 — 두 기준 모두 기준선보다 나은 칸')
print('=' * 108)
win = []
for i in range(K):
    if i == base:
        continue
    dg, dc = rg[:, i] - rg[:, base], rc[:, i] - rc[:, base]
    if (dg < 0).mean() > 0.5 and (dc < 0).mean() > 0.5:
        win.append(i)
if win:
    for i in win:
        print(row(i))
    n2w = [i for i in win if labels[i].endswith('n2')]
    print(f'\n  후보 {len(win)}개 중 종목 둘은 {len(n2w)}개.')
else:
    print('  없다 — 종목 둘의 효율 이득은 전체 합성에서 살아나지 않는다.')
    print('  미리 정한 규칙대로 여기서 종결한다. 격자로 넓히지 않는다.')
