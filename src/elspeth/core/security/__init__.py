"""Security utilities (signing, classification, etc.)."""

from .signing import generate_signature, verify_signature

SECURITY_LEVELS = [
    "unofficial",
    "official",
    "official-sensitive",
    "secret",
    "top-secret",
]


def normalize_security_level(level: str | None) -> str:
    if level is None or not str(level).strip():
        return SECURITY_LEVELS[0]
    normalized = str(level).strip().lower()
    if normalized not in SECURITY_LEVELS:
        raise ValueError(f"Unknown security level '{level}'")
    return normalized


def is_security_level_allowed(data_level: str | None, clearance_level: str | None) -> bool:
    normalized_data = normalize_security_level(data_level)
    normalized_clearance = normalize_security_level(clearance_level)
    data_idx = SECURITY_LEVELS.index(normalized_data)
    clearance_idx = SECURITY_LEVELS.index(normalized_clearance)
    return clearance_idx >= data_idx


def resolve_security_level(*levels: str | None) -> str:
    normalized = [normalize_security_level(level) for level in levels if level is not None]
    if not normalized:
        return SECURITY_LEVELS[0]
    return max(normalized, key=lambda lvl: SECURITY_LEVELS.index(lvl))


__all__ = [
    "SECURITY_LEVELS",
    "generate_signature",
    "is_security_level_allowed",
    "normalize_security_level",
    "resolve_security_level",
    "verify_signature",
]
