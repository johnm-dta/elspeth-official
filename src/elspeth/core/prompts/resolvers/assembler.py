"""Pack assembler for converting fetched files into prompt pack configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import yaml

from .protocols import FetchedFile

logger = logging.getLogger(__name__)


@dataclass
class DefaultPackAssembler:
    """Default assembler that converts fetched files into prompt pack config.

    Recognized files and their mappings:
    - config.yaml/yml -> Base configuration (merged first)
    - system_prompt.md or system.md -> prompts.system (overrides config.yaml)
    - user_prompt.md or user.md -> prompts.user (overrides config.yaml)
    - criteria.yaml/yml -> criteria list
    - defaults.yaml/yml -> prompt_defaults

    Priority: Markdown files (.md) override values in config.yaml for prompts.
    This allows separating long prompts into dedicated files while keeping
    other configuration in config.yaml.
    """

    def assemble(
        self,
        files: list[FetchedFile],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble fetched files into a prompt pack configuration.

        Args:
            files: List of fetched files from the remote pack
            metadata: Optional metadata about the pack source

        Returns:
            Prompt pack configuration dictionary compatible with
            ConfigurationMerger (precedence=2)
        """
        config: dict[str, Any] = {}
        prompts: dict[str, str] = {}

        # Build a map of filename -> file for easy lookup
        file_map = {self._get_filename(f.path): f for f in files}

        # 1. Parse config.yaml/yml as base configuration
        for config_name in ("config.yaml", "config.yml"):
            if config_name in file_map:
                parsed = yaml.safe_load(file_map[config_name].content)
                if parsed:
                    config = dict(parsed)
                    # Extract prompts from config if present
                    if "prompts" in config:
                        prompts = dict(config.get("prompts", {}))
                logger.debug("Loaded base config from %s", config_name)
                break

        # 2. Override prompts with dedicated .md files
        # Support both naming conventions: system_prompt.md and system.md
        for system_name in ("system_prompt.md", "system.md"):
            if system_name in file_map:
                prompts["system"] = file_map[system_name].content
                logger.debug("Loaded system prompt from %s", system_name)
                break

        for user_name in ("user_prompt.md", "user.md"):
            if user_name in file_map:
                prompts["user"] = file_map[user_name].content
                logger.debug("Loaded user prompt from %s", user_name)
                break

        # Update config with assembled prompts
        if prompts:
            config["prompts"] = prompts

        # 3. Load criteria from dedicated file
        for criteria_name in ("criteria.yaml", "criteria.yml"):
            if criteria_name in file_map:
                parsed = yaml.safe_load(file_map[criteria_name].content)
                if parsed:
                    # criteria can be a list or a single dict
                    if isinstance(parsed, list):
                        config["criteria"] = parsed
                    else:
                        config["criteria"] = [parsed]
                    logger.debug("Loaded criteria from %s", criteria_name)
                break

        # 4. Load prompt defaults from dedicated file
        for defaults_name in ("defaults.yaml", "defaults.yml"):
            if defaults_name in file_map:
                parsed = yaml.safe_load(file_map[defaults_name].content)
                if parsed and isinstance(parsed, dict):
                    config["prompt_defaults"] = parsed
                    logger.debug("Loaded prompt_defaults from %s", defaults_name)
                break

        # 5. Add metadata for debugging/tracing
        if metadata:
            config["_remote_pack_metadata"] = metadata

        return config

    @staticmethod
    def _get_filename(path: str) -> str:
        """Extract filename from a path.

        Args:
            path: File path (may contain directories)

        Returns:
            Just the filename portion
        """
        return path.rsplit("/", 1)[-1] if "/" in path else path
