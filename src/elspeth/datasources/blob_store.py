"""Helpers for fetching experiment inputs from Azure Blob Storage.

This module focuses on configuration-driven access so we can mirror Azure ML
workspace datastore settings locally.  Credentials default to
``DefaultAzureCredential`` which aligns with managed identity and local
``az login`` flows.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)


class BlobConfigurationError(RuntimeError):
    """Raised when the blob configuration is invalid."""


@dataclass
class BlobConfig:
    """Strongly typed blob configuration extracted from YAML/JSON."""

    connection_name: str
    azureml_datastore_uri: str
    storage_uri: str | None
    account_url: str
    container_name: str
    blob_path: str
    sas_token: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> BlobConfig:
        """Create a ``BlobConfig`` from a decoded mapping."""

        required = ["connection_name", "azureml_datastore_uri"]
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise BlobConfigurationError(f"Missing required blob configuration keys: {', '.join(missing)}")

        storage_uri = data.get("storage_uri")
        account_url = None
        container_name = None
        blob_path = None

        if storage_uri:
            details = _parse_storage_uri(storage_uri)
            account_url = details["account_url"]
            container_name = details["container_name"]
            blob_path = details["blob_path"]
        else:
            account_name = data.get("account_name")
            container_name = data.get("container_name")
            blob_path = data.get("blob_path")

            if not all([account_name, container_name, blob_path]):
                raise BlobConfigurationError("Provide either 'storage_uri' or all of 'account_name', 'container_name', 'blob_path'")

            account_url = data.get(
                "account_url",
                f"https://{account_name}.blob.core.windows.net",
            )

        sas_token = data.get("sas_token")
        if sas_token and sas_token.startswith("?"):
            sas_token = sas_token[1:]

        return cls(
            connection_name=data["connection_name"],
            azureml_datastore_uri=data["azureml_datastore_uri"],
            storage_uri=storage_uri,
            account_url=account_url,
            container_name=container_name,
            blob_path=blob_path,
            sas_token=sas_token,
        )


def _parse_storage_uri(storage_uri: str) -> dict[str, str]:
    """Break a blob URI into account/container/blob components."""

    parsed = urlparse(storage_uri)
    if parsed.scheme not in {"https", "http"}:
        raise BlobConfigurationError(f"Unsupported storage URI scheme: {parsed.scheme or '<missing>'}")

    if not parsed.netloc:
        raise BlobConfigurationError("Storage URI is missing a host component")

    path = parsed.path.lstrip("/")
    if "/" not in path:
        raise BlobConfigurationError("Storage URI must include container and blob path segments")

    container_name, blob_path = path.split("/", 1)
    if not container_name or not blob_path:
        raise BlobConfigurationError("Storage URI is missing container/blob path")

    account_url = f"{parsed.scheme}://{parsed.netloc}"
    return {
        "account_url": account_url,
        "container_name": container_name,
        "blob_path": blob_path,
    }


def load_blob_config(config_path: Path | str, profile: str = "default") -> BlobConfig:
    """Load blob configuration from YAML or JSON."""

    path = Path(config_path)
    if not path.exists():
        raise BlobConfigurationError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise BlobConfigurationError(f"Invalid YAML: {exc}") from exc

    if data is None:
        raise BlobConfigurationError("Configuration file is empty")

    if profile not in data:
        available = ", ".join(sorted(data)) if isinstance(data, dict) else ""
        raise BlobConfigurationError(f"Profile '{profile}' not found in configuration." f" Available: {available}")

    profile_data = data[profile]
    if isinstance(profile_data, str):
        try:
            profile_data = json.loads(profile_data)
        except json.JSONDecodeError as exc:
            raise BlobConfigurationError(f"Profile '{profile}' is a string but not valid JSON") from exc

    if not isinstance(profile_data, dict):
        raise BlobConfigurationError(f"Profile '{profile}' should be a mapping, got {type(profile_data).__name__}")

    config = BlobConfig.from_mapping(profile_data)
    logger.debug(
        "Loaded blob config '%s' targeting %s/%s",
        profile,
        config.container_name,
        config.blob_path,
    )
    return config


class BlobDataLoader:
    """Wrapper around Azure Blob SDK that respects our configuration."""

    def __init__(
        self,
        config: BlobConfig,
        credential: Any | None = None,
        *,
        timeout: int | None = 60,
    ):
        self.config = config
        self.credential = credential
        self.timeout = timeout
        self._blob_client = None

    @property
    def blob_client(self):  # type: ignore[return-any]
        """Instantiate the BlobClient lazily to avoid import costs."""

        if self._blob_client is None:
            try:
                from azure.storage.blob import BlobClient
            except ImportError as exc:  # pragma: no cover - environment specific
                raise BlobConfigurationError("azure-storage-blob is required to access Azure Blob Storage") from exc

            credential = self.credential or self.config.sas_token
            if credential is None:
                try:
                    from azure.identity import DefaultAzureCredential
                except ImportError as exc:
                    raise BlobConfigurationError("azure-identity is required when no credential is provided") from exc
                credential = DefaultAzureCredential()

            self._blob_client = BlobClient(
                account_url=self.config.account_url,
                container_name=self.config.container_name,
                blob_name=self.config.blob_path,
                credential=credential,
                connection_timeout=self.timeout,
                read_timeout=self.timeout,
            )
        return self._blob_client

    def download_to_file(self, destination: Path | str, overwrite: bool = False) -> Path:
        """Download the blob to a local path."""

        path = Path(destination)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading blob %s to %s",
            self.config.blob_path,
            path,
        )
        downloader = self.blob_client.download_blob()
        with path.open("wb") as handle:
            handle.write(downloader.readall())
        return path

    def load_text(self, encoding: str = "utf-8") -> str:
        """Return the blob contents decoded as text."""

        downloader = self.blob_client.download_blob()
        raw = downloader.readall()
        return raw.decode(encoding)

    def load_bytes(self) -> bytes:
        """Return the blob contents as raw bytes."""

        downloader = self.blob_client.download_blob()
        return downloader.readall()

    def load_csv(self, **pandas_kwargs):
        """Load the blob as a Pandas DataFrame."""

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BlobConfigurationError("pandas is required to read blob contents as CSV") from exc

        downloader = self.blob_client.download_blob()
        buffer = io.BytesIO(downloader.readall())
        return pd.read_csv(buffer, **pandas_kwargs)


def load_blob_csv(
    config_path: Path | str = "config/blob_store.yaml",
    *,
    profile: str = "default",
    credential: Any | None = None,
    timeout: int | None = 60,
    pandas_kwargs: dict[str, Any] | None = None,
):
    """Convenience helper that downloads a CSV using the configured blob."""

    config = load_blob_config(config_path, profile=profile)
    loader = BlobDataLoader(config, credential=credential, timeout=timeout)
    kwargs = pandas_kwargs or {}
    return loader.load_csv(**kwargs)


__all__ = [
    "BlobConfig",
    "BlobConfigurationError",
    "BlobDataLoader",
    "load_blob_config",
    "load_blob_csv",
]
