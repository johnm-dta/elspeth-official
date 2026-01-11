"""URL parsing utilities for remote prompt pack references."""

from __future__ import annotations

import re

from .protocols import ParsedPackURL

# GitHub URL pattern: github://org/repo/path[@ref]
# Examples:
#   github://myorg/prompts/quality-eval
#   github://myorg/prompts/quality-eval@v1.0
#   github://myorg/prompts/quality-eval@main
GITHUB_PATTERN = re.compile(
    r"^github://(?P<org>[^/]+)/(?P<repo>[^/@]+)(?:/(?P<path>[^@]+))?(?:@(?P<ref>.+))?$"
)

# Azure DevOps URL pattern: azuredevops://org/project/repo/path[@ref]
# Examples:
#   azuredevops://myorg/myproject/prompts/sentiment-analysis
#   azuredevops://myorg/myproject/prompts/sentiment-analysis@release
AZUREDEVOPS_PATTERN = re.compile(
    r"^azuredevops://(?P<org>[^/]+)/(?P<project>[^/]+)/(?P<repo>[^/@]+)(?:/(?P<path>[^@]+))?(?:@(?P<ref>.+))?$"
)


def is_remote_pack_url(url: object) -> bool:
    """Check if a string is a remote prompt pack URL.

    Args:
        url: String to check

    Returns:
        True if the string is a recognized remote pack URL scheme

    Examples:
        >>> is_remote_pack_url("github://org/repo/path")
        True
        >>> is_remote_pack_url("azuredevops://org/proj/repo/path")
        True
        >>> is_remote_pack_url("my_local_pack")
        False
    """
    if not isinstance(url, str):
        return False
    return url.startswith(("github://", "azuredevops://"))


def get_scheme(url: str) -> str:
    """Extract the scheme from a prompt pack URL.

    Args:
        url: Remote pack URL

    Returns:
        The URL scheme (e.g., "github", "azuredevops")

    Raises:
        ValueError: If URL scheme is not recognized
    """
    if url.startswith("github://"):
        return "github"
    if url.startswith("azuredevops://"):
        return "azuredevops"
    raise ValueError(f"Unknown URL scheme in: {url}")


def parse_pack_url(url: str) -> ParsedPackURL:
    """Parse a prompt pack URL into components.

    Supported formats:
    - github://org/repo/path[@ref]
    - azuredevops://org/project/repo/path[@ref]

    Args:
        url: Remote pack URL to parse

    Returns:
        ParsedPackURL with extracted components

    Raises:
        ValueError: If URL format is not recognized

    Examples:
        >>> parsed = parse_pack_url("github://myorg/prompts/quality-eval")
        >>> parsed.scheme
        'github'
        >>> parsed.organization
        'myorg'
        >>> parsed.repository
        'prompts'
        >>> parsed.path
        'quality-eval'
        >>> parsed.ref
        'main'

        >>> parsed = parse_pack_url("github://myorg/prompts/quality-eval@v1.0")
        >>> parsed.ref
        'v1.0'

        >>> parsed = parse_pack_url("azuredevops://org/proj/repo/path")
        >>> parsed.scheme
        'azuredevops'
        >>> parsed.project
        'proj'
    """
    # Try GitHub pattern
    if match := GITHUB_PATTERN.match(url):
        return ParsedPackURL(
            scheme="github",
            organization=match.group("org"),
            project=None,
            repository=match.group("repo"),
            path=match.group("path") or "",
            ref=match.group("ref") or "main",
            original_url=url,
        )

    # Try Azure DevOps pattern
    if match := AZUREDEVOPS_PATTERN.match(url):
        return ParsedPackURL(
            scheme="azuredevops",
            organization=match.group("org"),
            project=match.group("project"),
            repository=match.group("repo"),
            path=match.group("path") or "",
            ref=match.group("ref") or "main",
            original_url=url,
        )

    raise ValueError(
        f"Unrecognized prompt pack URL format: {url}. "
        f"Expected github://org/repo/path[@ref] or azuredevops://org/project/repo/path[@ref]"
    )
