"""
Download IAM Handwriting Database artifacts for DISHA-HTR Phase 1.

Official IAM images/metadata require FKI registration + credentials.
This script NEVER bypasses authentication.

Public Aachen split lists (OpenSLR SLR56) can be downloaded without IAM login.

Env (for official IAM only):
  IAM_USERNAME / IAM_PASSWORD   — FKI account from registration
  or pass --username / --password (prefer env; avoid shell history)

Resumable: skips files that already exist with matching size when Content-Length
is available; uses .partial + rename. Does not delete existing raw files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha_htr.logging_config import setup_logging  # noqa: E402

# Official FKI endpoints (registration required)
FKI_BASE = "https://fki.tic.heia-fr.ch/DBs/iamDB"
OFFICIAL_ARTIFACTS = {
    "lines.tgz": f"{FKI_BASE}/data/lines/lines.tgz",
    "ascii.tgz": f"{FKI_BASE}/data/ascii/ascii.tgz",
    "largeWriterIndependentTextLineRecognitionTask.zip": (
        f"{FKI_BASE}/tasks/largeWriterIndependentTextLineRecognitionTask.zip"
    ),
}

# Public split lists (no IAM credentials)
AACHEN_SPLITS_URL = "https://www.openslr.org/resources/56/splits.zip"
AACHEN_FILENAME = "splits.zip"


def _requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "requests is required (already listed in this project's venv). "
            "Do not pip-install without approval if missing."
        ) from exc
    return requests


def download_file(
    url: str,
    dest: Path,
    *,
    username: str | None = None,
    password: str | None = None,
    timeout: int = 120,
) -> tuple[str, str]:
    """
    Resumable-ish download: if dest exists and remote size matches, skip.
    Writes to dest.with_suffix(dest.suffix + '.partial') then renames.

    Returns (status, detail) where status in {ok, skipped, blocked, error}.
    """
    requests = _requests()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    auth = (username, password) if username and password else None

    try:
        head = requests.head(url, auth=auth, allow_redirects=True, timeout=timeout)
    except requests.RequestException as exc:
        return "error", f"HEAD failed: {exc}"

    if head.status_code in (401, 403):
        return (
            "blocked",
            f"HTTP {head.status_code} for {url} — register at FKI and pass credentials",
        )
    if head.status_code == 404:
        return "blocked", f"HTTP 404 for {url}"

    remote_size = head.headers.get("Content-Length")
    if dest.is_file():
        if remote_size and dest.stat().st_size == int(remote_size):
            return "skipped", f"already present ({dest.stat().st_size} bytes)"
        if not remote_size and dest.stat().st_size > 0:
            return "skipped", f"already present ({dest.stat().st_size} bytes; no Content-Length)"

    partial = dest.with_name(dest.name + ".partial")
    headers = {}
    mode = "wb"
    existing = 0
    if partial.is_file() and partial.stat().st_size > 0:
        existing = partial.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    try:
        with requests.get(
            url,
            auth=auth,
            stream=True,
            timeout=timeout,
            headers=headers,
            allow_redirects=True,
        ) as resp:
            if resp.status_code in (401, 403):
                return "blocked", f"HTTP {resp.status_code} during GET {url}"
            if resp.status_code == 416:
                # Partial already complete?
                if remote_size and existing == int(remote_size):
                    partial.replace(dest)
                    return "ok", "completed from partial"
                return "error", "HTTP 416 Invalid Range"
            if resp.status_code not in (200, 206):
                return "error", f"HTTP {resp.status_code} during GET {url}"

            with partial.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        partial.replace(dest)
        return "ok", f"downloaded {dest.stat().st_size} bytes"
    except requests.RequestException as exc:
        return "error", f"GET failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download IAM / Aachen artifacts")
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "datasets" / "iam" / "raw"),
        help="Destination for archives (never deletes existing)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("IAM_USERNAME") or "",
        help="FKI username (or IAM_USERNAME env)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("IAM_PASSWORD") or "",
        help="FKI password (or IAM_PASSWORD env)",
    )
    parser.add_argument(
        "--aachen-only",
        action="store_true",
        help="Only download public Aachen splits.zip (no IAM credentials needed)",
    )
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Only attempt official FKI artifacts (requires credentials)",
    )
    args = parser.parse_args()

    import os as _os

    _os.chdir(ROOT)
    logger = setup_logging(level="INFO", log_dir=ROOT / "logs")
    logger.info("=== DISHA HTR IAM download (Phase 1) ===")

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str, str]] = []

    if not args.official_only:
        status, detail = download_file(
            AACHEN_SPLITS_URL, raw_dir / AACHEN_FILENAME
        )
        results.append((AACHEN_FILENAME, status, detail))
        logger.info("Aachen %s: %s — %s", AACHEN_FILENAME, status, detail)
        print(f"[aachen] {AACHEN_FILENAME}: {status} — {detail}")

    if not args.aachen_only:
        user = (args.username or "").strip()
        password = args.password or ""
        if not user or not password:
            msg = (
                "Official IAM archives require FKI registration. "
                "Set IAM_USERNAME and IAM_PASSWORD (or --username/--password). "
                "Register: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"
            )
            logger.warning(msg)
            print(f"[official] blocked — {msg}")
            for name in OFFICIAL_ARTIFACTS:
                results.append((name, "blocked", "missing credentials"))
        else:
            for name, url in OFFICIAL_ARTIFACTS.items():
                status, detail = download_file(
                    url, raw_dir / name, username=user, password=password
                )
                results.append((name, status, detail))
                logger.info("Official %s: %s — %s", name, status, detail)
                print(f"[official] {name}: {status} — {detail}")

    blocked = [r for r in results if r[1] == "blocked"]
    errors = [r for r in results if r[1] == "error"]
    print("\nSummary:")
    for name, status, detail in results:
        print(f"  {status:8}  {name}  ({detail})")

    if errors:
        return 1
    if blocked and args.official_only:
        return 2
    # Partial success (e.g. Aachen ok, official blocked) is acceptable for Phase 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
