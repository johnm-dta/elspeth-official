"""Local CSV datasource for sample suites and offline runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from elspeth.core.interfaces import DataSource
from elspeth.core.landscape import get_current_landscape
from elspeth.core.security import normalize_security_level

logger = logging.getLogger(__name__)


class CSVDataSource(DataSource):
    """Load local CSV files as pandas DataFrames.

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
        path: str | Path,
        dtype: dict[str, Any] | None = None,
        encoding: str = "utf-8",
        on_error: str = "abort",
        security_level: str | None = None,
        name: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.dtype = dtype
        self.encoding = encoding
        if on_error != "abort":
            raise ValueError("on_error must be 'abort'")
        self.on_error = on_error
        self.security_level = normalize_security_level(security_level)
        self.name = name or "csv_local"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV datasource file not found: {self.path}")
        df = pd.read_csv(self.path, dtype=self.dtype, encoding=self.encoding)
        df.attrs["security_level"] = self.security_level

        # Save to landscape if active
        landscape = get_current_landscape()
        if landscape:
            save_path = landscape.get_path("inputs", self.name, "source_data.csv")
            df.to_csv(save_path, index=False)
            landscape.register_artifact("inputs", self.name, save_path, {
                "rows": len(df),
                "columns": list(df.columns),
                "source_path": str(self.path),
            })
            logger.debug("Saved datasource to landscape: %s", save_path)

        return df


__all__ = ["CSVDataSource"]

# --- Plugin Registration ---
# Import at end to avoid circular dependency (registry imports this module)
from elspeth.core.registry import ON_ERROR_ENUM, registry

LOCAL_CSV_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "dtype": {"type": "object"},
        "encoding": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
        "security_level": {"type": "string"},
    },
    "required": ["path"],
    "additionalProperties": True,
}

registry.register_datasource("local_csv", CSVDataSource, LOCAL_CSV_SCHEMA)
