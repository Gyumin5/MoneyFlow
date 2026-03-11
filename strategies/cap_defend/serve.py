#!/usr/bin/env python3
"""Secure static file server — whitelist HTML files only."""

import os
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Only these files are accessible
ALLOWED_FILES = {
    '/portfolio_result_gmoh.html',
    '/portfolio_result.html',
    '/strategy.html',
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
