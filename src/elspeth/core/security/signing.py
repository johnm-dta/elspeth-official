"""Simple signing helpers using HMAC digests."""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Literal

Algorithm = Literal["hmac-sha256", "hmac-sha512"]


def _normalize_key(key: str | bytes) -> bytes:
    if isinstance(key, bytes):
        return key
    return key.encode("utf-8")


def _resolve_digest(algorithm: Algorithm):
    if algorithm == "hmac-sha256":
        return hashlib.sha256
    if algorithm == "hmac-sha512":
        return hashlib.sha512
    raise ValueError(f"Unsupported algorithm '{algorithm}'")


def generate_signature(data: bytes, key: str | bytes, algorithm: Algorithm = "hmac-sha256") -> str:
    digest = _resolve_digest(algorithm)
    signer = hmac.new(_normalize_key(key), data, digest)
    return base64.b64encode(signer.digest()).decode("ascii")


def verify_signature(data: bytes, signature: str, key: str | bytes, algorithm: Algorithm = "hmac-sha256") -> bool:
    expected = generate_signature(data, key, algorithm)
    return hmac.compare_digest(expected, signature)
