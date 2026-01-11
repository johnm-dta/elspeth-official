"""Remote prompt pack resolvers.

This module provides functionality for loading prompt packs from remote sources
like GitHub and Azure DevOps repositories.

Public API:
    is_remote_pack_url(url) -> bool
        Check if a string is a remote prompt pack URL

    resolve_remote_pack(url, options=None) -> dict[str, Any]
        Resolve a remote URL to a prompt pack configuration

    resolver_registry
        Global registry for resolver implementations

URL Formats:
    GitHub:
        github://owner/repo/path[@ref]

    Azure DevOps:
        azuredevops://org/project/repo/path[@ref]

Examples:
    >>> from elspeth.core.prompts.resolvers import is_remote_pack_url, resolve_remote_pack
    >>>
    >>> # Check if URL is remote
    >>> is_remote_pack_url("github://myorg/prompts/eval")
    True
    >>> is_remote_pack_url("my_local_pack")
    False
    >>>
    >>> # Resolve remote pack (requires GITHUB_TOKEN env var for private repos)
    >>> pack = resolve_remote_pack("github://myorg/prompts/eval@v1.0")
    >>> print(pack["prompts"]["system"])
    ...

Environment Variables:
    GITHUB_TOKEN: GitHub Personal Access Token (for private repos)
    AZURE_DEVOPS_PAT: Azure DevOps Personal Access Token

Recognized Pack Files:
    - config.yaml/yml: Base configuration
    - system_prompt.md: System prompt (overrides config.yaml)
    - user_prompt.md: User prompt (overrides config.yaml)
    - criteria.yaml/yml: Criteria definitions
    - defaults.yaml/yml: Prompt defaults
"""

from __future__ import annotations

from typing import Any

# Import resolvers to trigger registration via decorators
from . import azuredevops as _azuredevops
from . import github as _github
from .assembler import DefaultPackAssembler
from .registry import resolver_registry
from .url_parser import is_remote_pack_url

# Register the default assembler
resolver_registry.register_assembler(DefaultPackAssembler())


def resolve_remote_pack(
    url: str, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve a remote prompt pack URL to configuration.

    Args:
        url: Remote pack URL (e.g., "github://org/repo/path@ref")
        options: Optional resolver configuration options

    Returns:
        Prompt pack configuration dictionary compatible with ConfigurationMerger

    Raises:
        ValueError: If URL format is invalid or no resolver available
        ConnectionError: If remote is unreachable
        PermissionError: If authentication fails
        FileNotFoundError: If pack location doesn't exist

    Example:
        >>> pack = resolve_remote_pack("github://myorg/ai-prompts/sentiment@v1.0")
        >>> print(pack["prompts"]["system"])
        'You are a sentiment analyzer...'
    """
    return resolver_registry.resolve(url, options)


__all__ = [
    "is_remote_pack_url",
    "resolve_remote_pack",
    "resolver_registry",
]
