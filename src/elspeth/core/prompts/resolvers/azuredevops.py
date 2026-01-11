"""Azure DevOps resolver for remote prompt packs."""

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

AZUREDEVOPS_RESOLVER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "token_env": {
            "type": "string",
            "description": "Environment variable name for Azure DevOps PAT",
        },
        "base_url": {
            "type": "string",
            "description": "Azure DevOps API base URL",
        },
        "api_version": {
            "type": "string",
            "description": "API version to use",
        },
        "timeout": {
            "type": "number",
            "description": "Request timeout in seconds",
        },
    },
    "additionalProperties": False,
}


@register_resolver("azuredevops", schema=AZUREDEVOPS_RESOLVER_SCHEMA)
@dataclass
class AzureDevOpsResolver:
    """Resolve prompt packs from Azure DevOps Git repositories.

    Uses the Azure DevOps REST API to discover and fetch files from a repository path.

    Attributes:
        token_env: Environment variable containing PAT (default: AZURE_DEVOPS_PAT)
        base_url: Azure DevOps API base URL (default: https://dev.azure.com)
        api_version: API version (default: 7.1-preview)
        timeout: Request timeout in seconds (default: 30.0)

    Environment Variables:
        AZURE_DEVOPS_PAT: Personal Access Token with Code (Read) permission

    Example:
        >>> resolver = AzureDevOpsResolver()
        >>> parsed = resolver.parse_url("azuredevops://org/project/repo/prompts@main")
        >>> manifest = resolver.discover_files(parsed)
        >>> file = resolver.fetch_file(parsed, "config.yaml")
    """

    token_env: str = "AZURE_DEVOPS_PAT"
    base_url: str = "https://dev.azure.com"
    api_version: str = "7.1-preview"
    timeout: float = 30.0
    session: requests.Session | None = None
    _headers_cache: dict[str, str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        self.base_url = self.base_url.rstrip("/")

    @property
    def scheme(self) -> str:
        return "azuredevops"

    def can_resolve(self, url: str) -> bool:
        return url.startswith("azuredevops://")

    def parse_url(self, url: str) -> ParsedPackURL:
        return parse_pack_url(url)

    def discover_files(self, parsed: ParsedPackURL) -> RemotePackManifest:
        """List files in the Azure DevOps repository path.

        Args:
            parsed: Parsed URL components

        Returns:
            Manifest with list of file paths

        Raises:
            FileNotFoundError: If path doesn't exist
            PermissionError: If authentication fails
            ConnectionError: If request fails
        """
        # Ensure path has leading slash for Azure DevOps API
        path = self._ensure_path(parsed.path) if parsed.path else "/"

        url = (
            f"{self.base_url}/{parsed.organization}/{parsed.project}/_apis/git"
            f"/repositories/{parsed.repository}/items"
        )
        params = {
            "scopePath": path,
            "recursionLevel": "oneLevel",
            "api-version": self.api_version,
            "versionDescriptor.version": parsed.ref,
            "versionDescriptor.versionType": "branch",
        }

        response = self._request("GET", url, params=params)
        data = response.json()

        # Extract file paths (exclude folders)
        files = [
            item["path"].lstrip("/")
            for item in data.get("value", [])
            if not item.get("isFolder", False)
        ]

        return RemotePackManifest(
            files=files,
            metadata={
                "provider": "azuredevops",
                "organization": parsed.organization,
                "project": parsed.project,
                "repository": parsed.repository,
                "ref": parsed.ref,
                "path": parsed.path,
            },
        )

    def fetch_file(self, parsed: ParsedPackURL, file_path: str) -> FetchedFile:
        """Fetch file content from Azure DevOps.

        Args:
            parsed: Parsed URL components
            file_path: Path to the file within the repository

        Returns:
            Fetched file with content

        Raises:
            FileNotFoundError: If file doesn't exist
            ConnectionError: If request fails
        """
        path = self._ensure_path(file_path)

        url = (
            f"{self.base_url}/{parsed.organization}/{parsed.project}/_apis/git"
            f"/repositories/{parsed.repository}/items"
        )
        params = {
            "path": path,
            "api-version": self.api_version,
            "versionDescriptor.version": parsed.ref,
            "versionDescriptor.versionType": "branch",
        }

        # Request raw text content
        headers = {**self._headers(), "Accept": "text/plain"}

        try:
            response = self.session.request(  # type: ignore[union-attr]
                "GET",
                url,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"Failed to connect to Azure DevOps: {exc}") from exc

        if response.status_code == 404:
            raise FileNotFoundError(f"File not found: {path}")
        if response.status_code == 401:
            raise PermissionError(
                f"Azure DevOps authentication failed. "
                f"Ensure {self.token_env} environment variable is set with a valid PAT."
            )
        if response.status_code == 403:
            raise PermissionError(
                f"Azure DevOps access denied. "
                f"Ensure PAT has Code (Read) permission. Response: {response.text}"
            )
        if response.status_code != 200:
            raise ConnectionError(
                f"Azure DevOps API error ({response.status_code}): {response.text}"
            )

        return FetchedFile(
            path=file_path,
            content=response.text,
            encoding="utf-8",
        )

    def _headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        if self._headers_cache is not None:
            return self._headers_cache

        headers = {"Content-Type": "application/json"}

        token = self._read_token(self.token_env)
        if token:
            # Azure DevOps uses Basic auth with empty username and PAT as password
            auth = base64.b64encode(f":{token}".encode()).decode("ascii")
            headers["Authorization"] = f"Basic {auth}"

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
            raise ConnectionError(f"Failed to connect to Azure DevOps: {exc}") from exc

        # Handle specific error codes
        if response.status_code == 401:
            raise PermissionError(
                f"Azure DevOps authentication failed. "
                f"Ensure {self.token_env} environment variable is set with a valid PAT."
            )
        if response.status_code == 403:
            raise PermissionError(
                f"Azure DevOps access denied. "
                f"Ensure PAT has Code (Read) permission. Response: {response.text}"
            )
        if response.status_code == 404:
            raise FileNotFoundError(f"Resource not found: {url}")

        if response.status_code not in expected_status:
            raise ConnectionError(
                f"Azure DevOps API error ({response.status_code}): {response.text}"
            )

        return response

    @staticmethod
    def _ensure_path(path: str) -> str:
        """Ensure path has leading slash for Azure DevOps API."""
        if not path:
            return "/"
        return f"/{path}" if not path.startswith("/") else path

    @staticmethod
    def _read_token(env_var: str) -> str | None:
        """Read token from environment variable."""
        token = os.getenv(env_var)
        return token.strip() if token else None
