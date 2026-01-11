"""Protocols and data classes for remote prompt pack resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ParsedPackURL:
    """Parsed components of a prompt pack URL.

    Attributes:
        scheme: URL scheme (e.g., "github", "azuredevops")
        organization: GitHub org/owner or Azure DevOps organization
        project: Azure DevOps project name (None for GitHub)
        repository: Repository name
        path: Path within the repository to the pack folder
        ref: Branch, tag, or commit reference (default: "main")
        original_url: Original URL string for error messages
    """

    scheme: str
    organization: str
    project: str | None
    repository: str
    path: str
    ref: str
    original_url: str


@dataclass
class FetchedFile:
    """A file fetched from a remote source.

    Attributes:
        path: Relative path within the pack folder
        content: File content as a string
        encoding: Content encoding (default: "utf-8")
    """

    path: str
    content: str
    encoding: str = "utf-8"


@dataclass
class RemotePackManifest:
    """Manifest of files available in a remote prompt pack location.

    Attributes:
        files: List of file paths available in the pack folder
        metadata: Provider-specific metadata about the pack
    """

    files: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PromptPackResolver(Protocol):
    """Protocol for resolving prompt packs from remote sources.

    Implementations handle URL parsing, file discovery, and fetching
    for a specific remote provider (GitHub, Azure DevOps, etc.).
    """

    @property
    def scheme(self) -> str:
        """URL scheme this resolver handles (e.g., 'github', 'azuredevops')."""
        ...

    def can_resolve(self, url: str) -> bool:
        """Check if this resolver can handle the given URL.

        Args:
            url: The URL to check

        Returns:
            True if this resolver can handle the URL
        """
        ...

    def parse_url(self, url: str) -> ParsedPackURL:
        """Parse URL into components.

        Args:
            url: The URL to parse

        Returns:
            Parsed URL components

        Raises:
            ValueError: If URL format is invalid for this resolver
        """
        ...

    def discover_files(self, parsed: ParsedPackURL) -> RemotePackManifest:
        """Discover available files in the remote pack location.

        Args:
            parsed: Parsed URL components

        Returns:
            Manifest of available files

        Raises:
            ConnectionError: If remote is unreachable
            PermissionError: If authentication fails
            FileNotFoundError: If path doesn't exist
        """
        ...

    def fetch_file(self, parsed: ParsedPackURL, file_path: str) -> FetchedFile:
        """Fetch a single file from the remote pack.

        Args:
            parsed: Parsed URL components
            file_path: Relative path to file within pack

        Returns:
            Fetched file with content

        Raises:
            FileNotFoundError: If file doesn't exist
            ConnectionError: If fetch fails
        """
        ...


@runtime_checkable
class PackAssembler(Protocol):
    """Protocol for assembling prompt pack from fetched files."""

    def assemble(
        self,
        files: list[FetchedFile],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble fetched files into a prompt pack configuration.

        Args:
            files: List of fetched files
            metadata: Optional metadata from the manifest

        Returns:
            Prompt pack configuration dictionary compatible with
            ConfigurationMerger (precedence=2)
        """
        ...
