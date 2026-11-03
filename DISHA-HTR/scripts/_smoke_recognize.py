"""One-shot smoke test for /api/recognize (local testing server)."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "demo" / "approved" / "demo_06_cu01_hello.png"
URL = "http://127.0.0.1:8766/api/recognize"


def main() -> int:
    if not IMG.is_file():
        print(f"missing image: {IMG}", file=sys.stderr)
        return 2
    boundary = "----dishaBound"
    raw = IMG.read_bytes()
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="force_single_line"\r\n\r\n'
        "true\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{IMG.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL elapsed={time.time() - t0:.2f}s: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - t0
    text = (payload.get("text") or "").strip()
    print(f"OK elapsed={elapsed:.2f}s text={text!r}")
    print(f"inference_time_s={payload.get('inference_time_s')}")
    return 0 if text else 2


if __name__ == "__main__":
    raise SystemExit(main())
