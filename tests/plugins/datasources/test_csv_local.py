"""Tests for the local CSV datasource plugin."""

import pandas as pd

from elspeth.plugins.datasources.csv_local import CSVDataSource


def test_csv_local_loads_and_sets_security_level(tmp_path):
    path = tmp_path / "input.csv"
    df = pd.DataFrame([{"a": 1}, {"a": 2}])
    df.to_csv(path, index=False)

    source = CSVDataSource(path=path, security_level="secret")
    loaded = source.load()

    assert list(loaded["a"]) == [1, 2]
    assert loaded.attrs["security_level"] == "secret"


class TestCSVDataSourceSchemas:
    """Tests for CSVDataSource schema declarations."""

    def test_has_input_schema(self):
        """CSVDataSource should have input_schema class attribute."""
        assert hasattr(CSVDataSource, "input_schema")
        assert isinstance(CSVDataSource.input_schema, dict)

    def test_has_output_schema(self):
        """CSVDataSource should have output_schema class attribute."""
        assert hasattr(CSVDataSource, "output_schema")
        assert isinstance(CSVDataSource.output_schema, dict)

    def test_input_schema_is_empty(self):
        """Datasources have no input - they are the origin of data."""
        assert CSVDataSource.input_schema == {}

    def test_output_schema_describes_dataframe(self):
        """Output schema describes DataFrame output."""
        schema = CSVDataSource.output_schema
        assert schema["type"] == "object"
        assert "description" in schema
