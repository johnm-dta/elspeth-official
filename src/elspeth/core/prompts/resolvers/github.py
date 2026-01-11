"""GitHub resolver for remote prompt packs."""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from .protocols import FetchedFile, ParsedPackURL, RemotePackManifest
from .registry import register_resolver
from .url_parser import parse_pack_url

logger = logging.getLogger(__name__)

GITHUB_RESOLVER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "token_env": {
            "type": "string",
            "description": "Environment variable name for GitHub token",
        },
        "base_url": {
            "type": "string",
            "description": "GitHub API base URL (for Enterprise)",
        },
        "timeout": {
            "type": "number",
            "description": "Request timeout in seconds",
        },
    },
    "additionalProperties": False,
}


@register_resolver("github", schema=GITHUB_RESOLVER_SCHEMA)
@dataclass
class GitHubResolver:
    """Resolve prompt packs from GitHub repositories.

    Uses the GitHub REST API to discover and fetch files from a repository path.

    Attributes:
        token_env: Environment variable containing GitHub token (default: GITHUB_TOKEN)
        base_url: GitHub API base URL (default: https://api.github.com)
        timeout: Request timeout in seconds (default: 30.0)

    Environment Variables:
        GITHUB_TOKEN: Personal access token with repo scope (for private repos)

    Example:
        >>> resolver = GitHubResolver()
        >>> parsed = resolver.parse_url("github://myorg/prompts/quality-eval@main")
        >>> manifest = resolver.discover_files(parsed)
        >>> file = resolver.fetch_file(parsed, "config.yaml")
    """

    token_env: str = "GITHUB_TOKEN"
    base_url: str = "https://api.github.com"
    timeout: float = 30.0
    session: requests.Session | None = None
    _headers_cache: dict[str, str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        self.base_url = self.base_url.rstrip("/")

    @property
    def scheme(self) -> str:
        return "github"

    def can_resolve(self, url: str) -> bool:
        return url.startswith("github://")

    def parse_url(self, url: str) -> ParsedPackURL:
        return parse_pack_url(url)

    def discover_files(self, parsed: ParsedPackURL) -> RemotePackManifest:
        """List files in the GitHub repository path.

        Args:
            parsed: Parsed URL components

        Returns:
            Manifest with list of file paths

        Raises:
            FileNotFoundError: If path doesn't exist
            PermissionError: If authentication fails
            ConnectionError: If request fails
        """
        # Build the contents API URL
        path = parsed.path.strip("/") if parsed.path else ""
        url = f"{self.base_url}/repos/{parsed.organization}/{parsed.repository}/contents/{path}"
        params = {"ref": parsed.ref}

        response = self._request("GET", url, params=params)
        data = response.json()

        # Handle single file vs directory response
        if isinstance(data, dict):
            # Single file - return just that file
            if data.get("type") == "file":
                files = [data["path"]]
            else:
                files = []
        else:
            # Directory listing - filter to files only
            files = [item["path"] for item in data if item.get("type") == "file"]

        return RemotePackManifest(
            files=files,
            metadata={
                "provider": "github",
                "organization": parsed.organization,
                "repository": parsed.repository,
                "ref": parsed.ref,
                "path": parsed.path,
            },
        )

    def fetch_file(self, parsed: ParsedPackURL, file_path: str) -> FetchedFile:
        """Fetch file content from GitHub.

        Args:
            parsed: Parsed URL components
            file_path: Path to the file within the repository

        Returns:
            Fetched file with decoded content

        Raises:
            FileNotFoundError: If file doesn't exist
            ConnectionError: If request fails
        """
        url = f"{self.base_url}/repos/{parsed.organization}/{parsed.repository}/contents/{file_path}"
        params = {"ref": parsed.ref}

        response = self._request("GET", url, params=params, expected_status={200})
        data = response.json()

        # Decode base64 content
        content_b64 = data.get("content", "")
        encoding = data.get("encoding", "base64")

        if encoding == "base64":
            # GitHub returns base64-encoded content with newlines
            content = base64.b64decode(content_b64).decode("utf-8")
        else:
            # Fallback for unexpected encoding
            content = content_b64

        return FetchedFile(
            path=file_path,
            content=content,
            encoding="utf-8",
        )

    def _headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        if self._headers_cache is not None:
            return self._headers_cache

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        token = self._read_token(self.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self._headers_cache = headers
        return headers

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        expected_status: set[int] | None = None,
    ) -> requests.Response:
        """Make an HTTP request with error handling.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            expected_status: Set of acceptable status codes

        Returns:
            Response object

        Raises:
            FileNotFoundError: If resource doesn't exist (404)
            PermissionError: If authentication fails (401, 403)
            ConnectionError: For network or other errors
        """
        expected_status = expected_status or {200}

        try:
            response = self.session.request(  # type: ignore[union-attr]
                method,
                url,
                headers=self._headers(),
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"Failed to connect to GitHub: {exc}") from exc

        # Handle specific error codes
        if response.status_code == 401:
            raise PermissionError(
                f"GitHub authentication failed. "
                f"Ensure {self.token_env} environment variable is set with a valid token."
            )
        if response.status_code == 403:
            raise PermissionError(
                f"GitHub access denied. Check repository permissions. "
                f"Response: {response.text}"
            )
        if response.status_code == 404:
            raise FileNotFoundError(f"Resource not found: {url}")

        if response.status_code not in expected_status:
            raise ConnectionError(
                f"GitHub API error ({response.status_code}): {response.text}"
            )

        return response

    @staticmethod
    def _read_token(env_var: str) -> str | None:
        """Read token from environment variable."""
        token = os.getenv(env_var)
        return token.strip() if token else None
