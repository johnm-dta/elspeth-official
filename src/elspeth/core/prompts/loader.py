"""Helpers for loading prompt templates from disk."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .engine import PromptEngine

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .template import PromptTemplate


def load_template(
    path: str | Path,
    *,
    engine: PromptEngine | None = None,
    name: str | None = None,
    defaults: Mapping[str, object] | None = None,
) -> PromptTemplate:
    engine = engine or PromptEngine()
    text = Path(path).read_text(encoding="utf-8")
    return engine.compile(text, name=name or Path(path).stem, defaults=defaults)


def load_template_pair(
    system_path: str | Path,
    user_path: str | Path,
    *,
    engine: PromptEngine | None = None,
    defaults: Mapping[str, object] | None = None,
) -> tuple[PromptTemplate, PromptTemplate]:
    engine = engine or PromptEngine()
    system = load_template(system_path, engine=engine, name="system_prompt", defaults=defaults)
    user = load_template(user_path, engine=engine, name="user_prompt", defaults=defaults)
    return system, user
