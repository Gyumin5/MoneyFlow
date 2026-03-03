# Security Setup Guide

이 프로젝트를 실행하려면 아래 설정 파일들을 먼저 생성해야 합니다.

## 1. 설정 파일 생성

```bash
cp config/settings.example.py config/settings.py
cp config/upbit.example.py config/upbit.py
cp config/bithumb.example.py config/bithumb.py
```

## 2. API 키 발급 및 입력

### 업비트 (Upbit)

1. https://upbit.com → 로그인 → Open API 관리
2. API 키 발급 (자산조회 + 주문 권한)
3. `config/settings.py`와 `config/upbit.py`에 `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY` 입력

### 빗썸 (Bithumb)

1. https://www.bithumb.com → 로그인 → API 관리
2. Connect API 키 발급
3. `config/settings.py`와 `config/bithumb.py`에 `BITHUMB_API_KEY`, `BITHUMB_SECRET_KEY` 입력

### 거래 비밀번호

- `config/settings.py`에서 `TRADE_PASSWORD`를 원하는 비밀번호로 설정
- 웹 리포트에서 Force Trade 실행 시 사용됩니다

## 3. 보안 주의사항

- `config/*.py` (example 제외) 파일은 `.gitignore`에 의해 Git 추적에서 제외됩니다
- **절대 API 키가 포함된 파일을 커밋하지 마세요**
- API 키 유출 시 즉시 거래소에서 키를 재발급하세요
