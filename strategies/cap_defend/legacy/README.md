# Legacy Strategy Files

현재 전략 상위 디렉터리를 단순화하기 위해 옮겨 둔 레거시 코인/통합 백테스트 파일들이다.

포함:
- `backtest_official.py`
- `coin_engine.py`
- `coin_helpers.py`
- `coin_dd_exit.py`
- `coin_backtest_howto.md`
- `daily_history.py`
- `run_portfolio_backtest.py`

의도:
- 현재 운영 기준 진입점은 상위 디렉터리의 `run_current_*_backtest.py` 3개만 남긴다.
- V18 계열 코인 백테스트와 과거 통합 포트폴리오/히스토리 도구는 여기서만 유지한다.

주의:
- `research/` 아래 과거 실험 스크립트 중 일부는 이 레거시 파일들을 기준으로 작성됐다.
- 레거시 재실행이 필요하면 import 경로를 이 디렉터리 기준으로 맞춰 사용해야 한다.
