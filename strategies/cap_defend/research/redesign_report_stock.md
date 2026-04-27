# STOCK 최종 후보 리포트

총 rank 후보: 6 / ensemble: 35 / stress: 30

## Primary 3 (stress 전 시나리오 pass, 6개 중)
### stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq
- rank_sum: 35.0 | Cal: 0.938 | CAGR: 12.780% | Cal×CAGR: 0.120 | MDD: -13.62% | Sh: 0.00
- phase_med: 0.938
- yearly_med: 1.931
- worst_year: -0.303
- snap_nudge_CV: 0.064
- stress:
  - tx_1.5x: Cal 0.869 (-7.3%)
  - tx_2.0x: Cal 0.739 (-21.2%)
  - delay_1bar: Cal 0.824 (-12.2%)
  - drop_top: Cal 0.617 (-34.3%)

### stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault
- rank_sum: 35.5 | Cal: 1.029 | CAGR: 13.380% | Cal×CAGR: 0.138 | MDD: -13.01% | Sh: 0.00
- phase_med: 0.972
- yearly_med: 1.931
- worst_year: -0.454
- snap_nudge_CV: 0.081
- stress:
  - tx_1.5x: Cal 0.937 (-9.0%)
  - tx_2.0x: Cal 0.833 (-19.0%)
  - delay_1bar: Cal 0.842 (-18.1%)
  - drop_top: Cal 0.681 (-33.8%)

### stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault
- rank_sum: 37.5 | Cal: 0.938 | CAGR: 12.780% | Cal×CAGR: 0.120 | MDD: -13.62% | Sh: 0.00
- phase_med: 0.938
- yearly_med: 1.931
- worst_year: -0.303
- snap_nudge_CV: 0.064
- stress:
  - drop_top: Cal 0.617 (-34.3%)
  - tx_1.5x: Cal 0.869 (-7.3%)
  - tx_2.0x: Cal 0.739 (-21.2%)
  - delay_1bar: Cal 0.824 (-12.2%)

## Backup 5~9
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual | rank_sum 38.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual | rank_sum 48.0 | Cal 1.00 | CAGR 13.38% | Cal×CAGR 0.13
- stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual | rank_sum 56.5 | Cal 0.97 | CAGR 13.28% | Cal×CAGR 0.13

## 다중 metric 정렬 — top 10 (단독 멤버 k=1)

### rank_sum 낮은순
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq | rank_sum 35.0 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault | rank_sum 35.5 | Cal 1.03 | CAGR 13.38% | Cal×CAGR 0.14 | MDD -13.0%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault | rank_sum 37.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual | rank_sum 38.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual | rank_sum 48.0 | Cal 1.00 | CAGR 13.38% | Cal×CAGR 0.13 | MDD -13.4%
- stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual | rank_sum 56.5 | Cal 0.97 | CAGR 13.28% | Cal×CAGR 0.13 | MDD -13.6%

### Cal 높은순
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault | rank_sum 35.5 | Cal 1.03 | CAGR 13.38% | Cal×CAGR 0.14 | MDD -13.0%
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual | rank_sum 48.0 | Cal 1.00 | CAGR 13.38% | Cal×CAGR 0.13 | MDD -13.4%
- stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual | rank_sum 56.5 | Cal 0.97 | CAGR 13.28% | Cal×CAGR 0.13 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq | rank_sum 35.0 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault | rank_sum 37.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual | rank_sum 38.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%

### CAGR 높은순
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault | rank_sum 35.5 | Cal 1.03 | CAGR 13.38% | Cal×CAGR 0.14 | MDD -13.0%
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual | rank_sum 48.0 | Cal 1.00 | CAGR 13.38% | Cal×CAGR 0.13 | MDD -13.4%
- stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual | rank_sum 56.5 | Cal 0.97 | CAGR 13.28% | Cal×CAGR 0.13 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq | rank_sum 35.0 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault | rank_sum 37.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual | rank_sum 38.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%

### Cal×CAGR 높은순
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault | rank_sum 35.5 | Cal 1.03 | CAGR 13.38% | Cal×CAGR 0.14 | MDD -13.0%
- stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual | rank_sum 48.0 | Cal 1.00 | CAGR 13.38% | Cal×CAGR 0.13 | MDD -13.4%
- stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual | rank_sum 56.5 | Cal 0.97 | CAGR 13.28% | Cal×CAGR 0.13 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq | rank_sum 35.0 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault | rank_sum 37.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%
- stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual | rank_sum 38.5 | Cal 0.94 | CAGR 12.78% | Cal×CAGR 0.12 | MDD -13.6%

## Ensemble 후보 (k=2/3) — 다중 metric 정렬

### Ensemble Cal 높은순 — top 10
- k=2: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.61% | Cal×CAGR 0.156 | MDD -11.91% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=2: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.61% | Cal×CAGR 0.156 | MDD -11.91% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=2: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.61% | Cal×CAGR 0.156 | MDD -11.91% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=2: stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.66% | Cal×CAGR 0.156 | MDD -11.95% | Sh 5.63 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.141 | CAGR 13.59% | Cal×CAGR 0.155 | MDD -11.91% | Sh 5.59 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.141 | CAGR 13.59% | Cal×CAGR 0.155 | MDD -11.91% | Sh 5.59 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.141 | CAGR 13.59% | Cal×CAGR 0.155 | MDD -11.91% | Sh 5.59 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00

### Ensemble CAGR 높은순 — top 10
- k=2: stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.125 | CAGR 13.68% | Cal×CAGR 0.154 | MDD -12.17% | Sh 5.62 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.123 | CAGR 13.67% | Cal×CAGR 0.153 | MDD -12.17% | Sh 5.62 | improve 3/3 corr 1.00 bad 1.00
- k=2: stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.122 | CAGR 13.67% | Cal×CAGR 0.153 | MDD -12.18% | Sh 5.62 | improve 3/3 corr 1.00 bad 1.00
- k=2: stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.66% | Cal×CAGR 0.156 | MDD -11.95% | Sh 5.63 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.120 | CAGR 13.62% | Cal×CAGR 0.153 | MDD -12.16% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.120 | CAGR 13.62% | Cal×CAGR 0.153 | MDD -12.16% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.120 | CAGR 13.62% | Cal×CAGR 0.153 | MDD -12.16% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00

### Ensemble Cal×CAGR 높은순 — top 10
- k=2: stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.66% | Cal×CAGR 0.156 | MDD -11.95% | Sh 5.63 | improve 3/3 corr 1.00 bad 1.00
- k=2: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.61% | Cal×CAGR 0.156 | MDD -11.91% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=2: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.61% | Cal×CAGR 0.156 | MDD -11.91% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=2: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.143 | CAGR 13.61% | Cal×CAGR 0.156 | MDD -11.91% | Sh 5.60 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_dm63_sma200_sh252_mstdual → Cal 1.139 | CAGR 13.62% | Cal×CAGR 0.155 | MDD -11.95% | Sh 5.61 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.141 | CAGR 13.59% | Cal×CAGR 0.155 | MDD -11.91% | Sh 5.59 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_msteq|stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.141 | CAGR 13.59% | Cal×CAGR 0.155 | MDD -11.91% | Sh 5.59 | improve 3/3 corr 1.00 bad 1.00
- k=3: stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdefault|stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual|stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual → Cal 1.141 | CAGR 13.59% | Cal×CAGR 0.155 | MDD -11.91% | Sh 5.59 | improve 3/3 corr 1.00 bad 1.00