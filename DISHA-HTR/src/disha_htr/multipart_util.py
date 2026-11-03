"""Minimal multipart/form-data parser (stdlib only; no cgi — removed in 3.13+)."""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from typing import Any


@dataclass
class FormFile:
    filename: str
    content_type: str
    data: bytes


def parse_multipart(content_type: str, body: bytes) -> dict[str, Any]:
    """
    Parse multipart body into {field_name: str | FormFile}.

    Raises ValueError on invalid content type / parse errors.
    """
    if "multipart/form-data" not in (content_type or ""):
        raise ValueError("expected multipart/form-data")
    # email parser expects a full message with headers
    raw = b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
    msg = BytesParser(policy=default).parsebytes(raw)
    if not msg.is_multipart():
        raise ValueError("not a multipart message")

    out: dict[str, Any] = {}
    for part in msg.iter_parts():
        disp = part.get_content_disposition()
        if disp != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = b""
        if isinstance(payload, str):
            payload = payload.encode("utf-8", errors="replace")
        if filename:
            out[name] = FormFile(
                filename=filename,
                content_type=part.get_content_type() or "application/octet-stream",
                data=bytes(payload),
            )
        else:
            out[name] = bytes(payload).decode("utf-8", errors="replace")
    return out
