"""
run.py — Production entrypoint for Railway (and any OCI-compliant container host).

Why this file exists instead of a bare `uvicorn` CMD:
  1. $PORT         — Railway injects a random port at runtime.  Shell-form CMD
                     handles this, but h11_max_incomplete_event_size is a Python-
                     level kwarg that has no CLI equivalent, so we need Python.
  2. proxy-headers — Without this, slowapi sees Railway's internal proxy IP for
                     every request and all users share one rate-limit bucket.
  3. h11 event size — Without this, h11 rejects HTTP requests whose header block
                     is larger than its default ~16 KB limit before FastAPI even
                     sees them, meaning large CSV uploads fail at the protocol layer
                     with an opaque error instead of a clean 413 from server.py.
"""

import os
import uvicorn

PORT = int(os.environ.get("PORT", 8000))

uvicorn.run(
    "server:app",
    host="0.0.0.0",
    port=PORT,
    proxy_headers=True,
    forwarded_allow_ips="*",        # trust Railway's load-balancer
    h11_max_incomplete_event_size=1024 * 1024 * 1024,  # 1 GB — matches MAX_UPLOAD_BYTES in server.py
    reload=False,
)
