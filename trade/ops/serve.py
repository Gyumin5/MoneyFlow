#!/usr/bin/env python3
"""Secure static file server — whitelist HTML files only."""

import os
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Only these files are accessible
#
# 2026-11-03: '/v22_alloc_report.html' 를 뺐다. 2026-04-28 자 V22 세대 자산배분 보고서인데
# 집 저장소에 원본도 생성 스크립트도 없어서 서버 파일이 유일본이고 고칠 길이 없었다.
# 내용이 현행(60/20/20, 트리거 25pp·20%)과 달라 공개를 멈췄다. 서버 파일은 지우지 않았고
# 사본을 /home/gmoh/backups/251229-reports/2026-11-03/ 에 해시와 같이 보존했다.
# 되돌리려면 이 줄을 다시 넣고 배포 후 정적 서버만 재시작한다 — 실행 중인 목록은 자동으로
# 갱신되지 않는다. 토론 run-20261103T060344Z 판단 2.
ALLOWED_FILES = {
    '/portfolio_result_gmoh.html',
    '/portfolio_result.html',
    '/strategy.html',
    '/asset_dashboard.html',
    '/strategy_guide.html',
}


class SecureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ALLOWED_FILES:
            self.send_error(404)
            return

        filepath = os.path.join(BASE_DIR, self.path.lstrip('/'))
        if not os.path.isfile(filepath):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Robots-Tag', 'noindex, nofollow')
        self.end_headers()
        with open(filepath, 'rb') as f:
            self.wfile.write(f.read())

    def do_HEAD(self):
        self.send_error(404)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), SecureHandler)
    print(f"Serving on port {PORT}")
    print(f"Allowed: {ALLOWED_FILES}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
