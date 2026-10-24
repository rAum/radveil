#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Minimal on-the-fly defacing service (standard library only).

    python examples/service_http.py --port 8080 --workers 8
    curl --data-binary @t1.nii.gz http://localhost:8080/deface -o t1_defaced.nii.gz

POST a NIfTI volume (.nii or .nii.gz bytes) to /deface; the response is the
anonymized volume as .nii.gz. Query parameters: mode (blur|adaptive|cut|
block), region (face|ears|face+ears). A volume without a validated face is
refused with HTTP 422 and a JSON reason. Nothing touches the disk. The
pipeline report is returned in the X-RadVeil-Report header (JSON).

Requests run concurrently in threads: the Rust core releases the GIL, so one
process uses all cores. At most --workers volumes are processed at once
(each needs ~4-5x its voxel data in memory); further requests wait. Put it
behind your usual reverse proxy for TLS and authentication.
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import radveil

MAX_UPLOAD = 1 << 30  # 1 GiB
SLOTS = threading.BoundedSemaphore(os.cpu_count() or 1)


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, body, ctype="application/json", headers=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg):
        self._reply(code, json.dumps({"error": msg}).encode())

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._reply(200, json.dumps({"status": "ok", "version": radveil.__version__}).encode())
        else:
            self._error(404, "not found")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/deface":
            return self._error(404, "not found")
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_UPLOAD:
            return self._error(413, "body must be a NIfTI volume up to 1 GiB")
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        body = self.rfile.read(n)
        with SLOTS:
            try:
                vol = radveil.Volume.from_bytes(body)
            except Exception as e:  # not a readable NIfTI volume
                return self._error(400, f"cannot read NIfTI: {e}")
            try:
                report = radveil.deface_volume(
                    vol, mode=q.get("mode", "blur"), region=q.get("region", "face")
                )
            except ValueError as e:  # bad parameters
                return self._error(400, str(e))
            except RuntimeError as e:  # no validated face: refuse, never guess
                return self._error(422, str(e))
            out = vol.to_bytes()
        summary = {
            k: report.get(k)
            for k in ("views_detected", "verify_rounds", "residual_detections",
                      "escalated_to_block", "voxels_changed", "eye_dist_mm")
        }
        self._reply(200, out, "application/gzip",
                    [("X-RadVeil-Report", json.dumps(summary))])

    def log_message(self, fmt, *args):  # one line per request, no payloads
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="volumes processed concurrently (default: CPU count)")
    a = ap.parse_args()
    global SLOTS
    SLOTS = threading.BoundedSemaphore(max(1, a.workers))
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"RadVeil {radveil.__version__} serving on http://{a.host}:{a.port}/deface", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
