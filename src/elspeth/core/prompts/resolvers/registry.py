"""Registry for prompt pack resolvers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .protocols import (
    FetchedFile,
    PackAssembler,
    ParsedPackURL,
    PromptPackResolver,
    RemotePackManifest,
)
from .url_parser import parse_pack_url

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

# Known pack files to look for when fetching
KNOWN_PACK_FILES = frozenset(
    {
        "config.yaml",
        "config.yml",
        "system_prompt.md",
        "system.md",
        "user_prompt.md",
        "user.md",
        "criteria.yaml",
        "criteria.yml",
        "defaults.yaml",
        "defaults.yml",
    }
)


@dataclass
class ResolverFactory:
    """Factory for creating resolver instances."""

    create: Callable[[dict[str, Any]], PromptPackResolver]
    schema: Mapping[str, Any] | None = None


class PromptPackResolverRegistry:
    """Registry for prompt pack resolvers.

    Manages resolver implementations for different URL schemes (github, azuredevops, etc.)
    and coordinates fetching and assembly of remote prompt packs.

    Example:
        >>> registry = PromptPackResolverRegistry()
        >>> registry.register("github", lambda opts: GitHubResolver(**opts))
        >>> pack = registry.resolve("github://org/repo/prompts")
    """

    def __init__(self) -> None:
        self._resolvers: dict[str, ResolverFactory] = {}
        self._assembler: PackAssembler | None = None

    def register(
        self,
        scheme: str,
        factory: Callable[[dict[str, Any]], PromptPackResolver],
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        """Register a resolver factory for a URL scheme.

        Args:
            scheme: URL scheme to handle (e.g., "github", "azuredevops")
            factory: Callable that creates a resolver instance from options dict
            schema: Optional JSON schema for validating options
        """
        self._resolvers[scheme] = ResolverFactory(create=factory, schema=schema)
        logger.debug("Registered resolver for scheme: %s", scheme)

    def register_assembler(self, assembler: PackAssembler) -> None:
        """Register the pack assembler.

        Args:
            assembler: Assembler instance to use for building packs from files
        """
        self._assembler = assembler

    def has_resolver(self, scheme: str) -> bool:
        """Check if a resolver is registered for the given scheme.

        Args:
            scheme: URL scheme to check

        Returns:
            True if a resolver is registered
        """
        return scheme in self._resolvers

    def get_resolver(
        self, scheme: str, options: dict[str, Any] | None = None
    ) -> PromptPackResolver:
        """Get a resolver instance for the given scheme.

        Args:
            scheme: URL scheme (e.g., "github", "azuredevops")
            options: Optional configuration options for the resolver

        Returns:
            Configured resolver instance

        Raises:
            ValueError: If no resolver is registered for the scheme
        """
        if scheme not in self._resolvers:
            available = ", ".join(sorted(self._resolvers.keys())) or "(none)"
            raise ValueError(
                f"No resolver registered for scheme '{scheme}'. "
                f"Available schemes: {available}"
            )
        return self._resolvers[scheme].create(options or {})

    def resolve(
        self, url: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Resolve a remote prompt pack URL to configuration.

        This is the main entry point for loading remote packs. It:
        1. Parses the URL to determine the scheme
        2. Gets the appropriate resolver
        3. Discovers files in the remote location
        4. Fetches recognized pack files
        5. Assembles them into a configuration dict

        Args:
            url: Remote pack URL (e.g., "github://org/repo/path")
            options: Optional resolver configuration

        Returns:
            Prompt pack configuration dictionary

        Raises:
            ValueError: If URL format is invalid or no resolver available
            ConnectionError: If remote is unreachable
            PermissionError: If authentication fails
            FileNotFoundError: If pack location doesn't exist
            RuntimeError: If no assembler is registered
        """
        parsed = parse_pack_url(url)
        resolver = self.get_resolver(parsed.scheme, options)

        logger.info("Resolving remote prompt pack: %s", url)
        manifest = resolver.discover_files(parsed)
        logger.debug("Discovered %d files in remote pack", len(manifest.files))

        files = self._fetch_pack_files(resolver, parsed, manifest)
        logger.debug("Fetched %d pack files", len(files))

        if self._assembler is None:
            raise RuntimeError(
                "No pack assembler registered. "
                "Ensure the resolvers module is properly initialized."
            )

        return self._assembler.assemble(files, manifest.metadata)

    def _fetch_pack_files(
        self,
        resolver: PromptPackResolver,
        parsed: ParsedPackURL,
        manifest: RemotePackManifest,
    ) -> list[FetchedFile]:
        """Fetch recognized pack files from the manifest.

        Args:
            resolver: Resolver instance to use for fetching
            parsed: Parsed URL components
            manifest: Manifest of available files

        Returns:
            List of fetched files
        """
        files: list[FetchedFile] = []

        for file_path in manifest.files:
            # Extract filename from path
            filename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path

            if filename in KNOWN_PACK_FILES:
                try:
                    fetched = resolver.fetch_file(parsed, file_path)
                    files.append(fetched)
                    logger.debug("Fetched: %s", file_path)
                except FileNotFoundError:
                    # File disappeared between discovery and fetch - skip it
                    logger.warning("File not found during fetch: %s", file_path)
                    continue

        return files


# Global registry instance
resolver_registry = PromptPackResolverRegistry()


def register_resolver(
    scheme: str, schema: Mapping[str, Any] | None = None
) -> Callable[[type], type]:
    """Decorator for registering resolvers with the global registry.

    Usage:
        @register_resolver("github")
        class GitHubResolver:
            ...

    Args:
        scheme: URL scheme to handle
        schema: Optional JSON schema for config validation

    Returns:
        Class decorator
    """

    def decorator(cls: type) -> type:
        resolver_registry.register(
            scheme,
            factory=lambda options: cls(**options),
            schema=schema,
        )
        return cls

    return decorator
