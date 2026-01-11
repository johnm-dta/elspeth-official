"""Plugin wrapping the existing blob loader."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import pandas as pd

from elspeth.core.interfaces import DataSource
from elspeth.core.landscape import get_current_landscape
from elspeth.core.security import normalize_security_level
from elspeth.datasources import load_blob_csv
from elspeth.datasources.blob_store import BlobConfigurationError

logger = logging.getLogger(__name__)


class BlobDataSource(DataSource):
    """Load CSV from Azure Blob Storage.

    Datasources are the starting point of pipelines, so they have no input schema.
    Output is a DataFrame with column names and values.
    """

    # Datasources have no input - they are the origin of data
    input_schema: ClassVar[dict[str, Any]] = {}

    # Output is a DataFrame represented as dict of columns
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "description": "Loaded DataFrame as dict of columns with row data",
    }

    def __init__(
        self,
        *,
        config_path: str,
        profile: str = "default",
        pandas_kwargs: dict[str, Any] | None = None,
        on_error: str = "abort",
        security_level: str | None = None,
        name: str | None = None,
    ):
        self.config_path = config_path
        self.profile = profile
        self.pandas_kwargs = pandas_kwargs or {}
        if on_error != "abort":
            raise ValueError("on_error must be 'abort'")
        self.on_error = on_error
        self.security_level = normalize_security_level(security_level)
        self.name = name or "azure_blob"

    def load(self) -> pd.DataFrame:
        try:
            df = load_blob_csv(
                self.config_path,
                profile=self.profile,
                pandas_kwargs=self.pandas_kwargs,
            )
            df.attrs["security_level"] = self.security_level

            # Save to landscape if active
            landscape = get_current_landscape()
            if landscape:
                save_path = landscape.get_path("inputs", self.name, "source_data.csv")
                df.to_csv(save_path, index=False)
                landscape.register_artifact("inputs", self.name, save_path, {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "profile": self.profile,
                })
                logger.debug("Saved datasource to landscape: %s", save_path)

            return df
        except BlobConfigurationError:
            # Re-raise configuration errors as-is (already have good messages)
            raise
        except FileNotFoundError as exc:
            raise BlobConfigurationError(
                f"Configuration file not found: {self.config_path}"
            ) from exc
        except Exception as exc:
            raise BlobConfigurationError(
                f"Failed to load blob data from config '{self.config_path}' "
                f"profile '{self.profile}': {exc}"
            ) from exc


# --- Plugin Registration ---
from elspeth.core.registry import ON_ERROR_ENUM, registry

AZURE_BLOB_DATASOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "config_path": {"type": "string"},
        "profile": {"type": "string"},
        "pandas_kwargs": {"type": "object"},
        "on_error": ON_ERROR_ENUM,
        "security_level": {"type": "string"},
    },
    "required": ["config_path"],
    "additionalProperties": True,
}

registry.register_datasource("azure_blob", BlobDataSource, AZURE_BLOB_DATASOURCE_SCHEMA)
