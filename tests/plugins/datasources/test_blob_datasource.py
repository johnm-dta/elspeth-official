"""Tests for Azure Blob Storage datasource plugin."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from elspeth.datasources.blob_store import BlobConfigurationError
from elspeth.plugins.datasources.blob import BlobDataSource


@pytest.fixture
def valid_blob_config(tmp_path: Path) -> Path:
    """Create a valid blob configuration file."""
    config_path = tmp_path / "blob_config.yaml"
    config_path.write_text(
        """
default:
  connection_name: test-connection
  azureml_datastore_uri: azureml://datastores/test
  storage_uri: https://testaccount.blob.core.windows.net/testcontainer/data.csv

production:
  connection_name: prod-connection
  azureml_datastore_uri: azureml://datastores/prod
  storage_uri: https://prodaccount.blob.core.windows.net/prodcontainer/data.csv
"""
    )
    return config_path


@pytest.fixture
def mock_blob_csv():
    """Mock the load_blob_csv function."""
    with patch("elspeth.plugins.datasources.blob.load_blob_csv") as mock:
        yield mock


class TestBlobDataSourceInit:
    """Tests for BlobDataSource initialization."""

    def test_init_with_defaults(self, valid_blob_config: Path):
        """DataSource initializes with default values."""
        source = BlobDataSource(config_path=str(valid_blob_config))

        assert source.config_path == str(valid_blob_config)
        assert source.profile == "default"
        assert source.pandas_kwargs == {}
        assert source.on_error == "abort"
        assert source.name == "azure_blob"

    def test_init_with_custom_options(self, valid_blob_config: Path):
        """DataSource accepts custom options."""
        source = BlobDataSource(
            config_path=str(valid_blob_config),
            profile="production",
            pandas_kwargs={"encoding": "utf-8", "dtype": {"id": str}},
            security_level="official-sensitive",
            name="my_blob_source",
        )

        assert source.profile == "production"
        assert source.pandas_kwargs == {"encoding": "utf-8", "dtype": {"id": str}}
        assert source.security_level == "official-sensitive"
        assert source.name == "my_blob_source"

    def test_init_invalid_on_error(self, valid_blob_config: Path):
        """Rejects invalid on_error value."""
        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            BlobDataSource(
                config_path=str(valid_blob_config),
                on_error="ignore",
            )

    def test_init_normalizes_security_level(self, valid_blob_config: Path):
        """Security level is normalized."""
        source = BlobDataSource(
            config_path=str(valid_blob_config),
            security_level="OFFICIAL",
        )

        assert source.security_level == "official"


class TestBlobDataSourceLoad:
    """Tests for BlobDataSource.load()."""

    def test_load_returns_dataframe(self, valid_blob_config: Path, mock_blob_csv):
        """Load returns a pandas DataFrame."""
        expected_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        mock_blob_csv.return_value = expected_df

        source = BlobDataSource(config_path=str(valid_blob_config))
        result = source.load()

        assert isinstance(result, pd.DataFrame)
        assert list(result["id"]) == [1, 2]
        assert list(result["name"]) == ["Alice", "Bob"]

    def test_load_passes_config_path_and_profile(self, valid_blob_config: Path, mock_blob_csv):
        """Load passes config path and profile to blob loader."""
        mock_blob_csv.return_value = pd.DataFrame()

        source = BlobDataSource(
            config_path=str(valid_blob_config),
            profile="production",
        )
        source.load()

        mock_blob_csv.assert_called_once()
        call_args = mock_blob_csv.call_args
        assert call_args[0][0] == str(valid_blob_config)
        assert call_args[1]["profile"] == "production"

    def test_load_passes_pandas_kwargs(self, valid_blob_config: Path, mock_blob_csv):
        """Load passes pandas kwargs to blob loader."""
        mock_blob_csv.return_value = pd.DataFrame()

        source = BlobDataSource(
            config_path=str(valid_blob_config),
            pandas_kwargs={"encoding": "latin-1", "na_values": ["N/A"]},
        )
        source.load()

        call_args = mock_blob_csv.call_args
        assert call_args[1]["pandas_kwargs"] == {"encoding": "latin-1", "na_values": ["N/A"]}

    def test_load_sets_security_level_attribute(self, valid_blob_config: Path, mock_blob_csv):
        """Security level is set as DataFrame attribute."""
        mock_blob_csv.return_value = pd.DataFrame({"x": [1]})

        source = BlobDataSource(
            config_path=str(valid_blob_config),
            security_level="secret",
        )
        result = source.load()

        assert result.attrs["security_level"] == "secret"

    def test_load_sets_default_security_level_when_not_specified(
        self, valid_blob_config: Path, mock_blob_csv
    ):
        """Security level defaults to 'unofficial' when not specified."""
        mock_blob_csv.return_value = pd.DataFrame({"x": [1]})

        source = BlobDataSource(config_path=str(valid_blob_config))
        result = source.load()

        # normalize_security_level(None) returns 'unofficial'
        assert result.attrs["security_level"] == "unofficial"


class TestConfigFileErrors:
    """Tests for configuration file error handling."""

    def test_load_missing_config_file(self, tmp_path: Path):
        """Missing config file raises BlobConfigurationError."""
        missing_path = tmp_path / "nonexistent.yaml"

        source = BlobDataSource(config_path=str(missing_path))

        with pytest.raises(BlobConfigurationError, match="not found"):
            source.load()

    def test_load_missing_profile(self, valid_blob_config: Path):
        """Missing profile raises BlobConfigurationError."""
        source = BlobDataSource(
            config_path=str(valid_blob_config),
            profile="nonexistent_profile",
        )

        with pytest.raises(BlobConfigurationError, match="not found"):
            source.load()

    def test_load_invalid_yaml(self, tmp_path: Path):
        """Invalid YAML raises BlobConfigurationError."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content: [")

        source = BlobDataSource(config_path=str(config_path))

        with pytest.raises(BlobConfigurationError, match="Invalid YAML"):
            source.load()

    def test_load_empty_config(self, tmp_path: Path):
        """Empty config file raises BlobConfigurationError."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        source = BlobDataSource(config_path=str(config_path))

        with pytest.raises(BlobConfigurationError, match="empty"):
            source.load()


class TestExceptionTranslation:
    """Tests for exception translation with context."""

    def test_load_translates_generic_exception(self, valid_blob_config: Path, mock_blob_csv):
        """Generic exceptions are translated with context."""
        mock_blob_csv.side_effect = RuntimeError("Connection timeout")

        source = BlobDataSource(
            config_path=str(valid_blob_config),
            profile="production",
        )

        with pytest.raises(BlobConfigurationError) as exc_info:
            source.load()

        error_message = str(exc_info.value)
        assert str(valid_blob_config) in error_message
        assert "production" in error_message
        assert "Connection timeout" in error_message

    def test_load_preserves_blob_configuration_error(self, valid_blob_config: Path, mock_blob_csv):
        """BlobConfigurationError is re-raised as-is."""
        original_error = BlobConfigurationError("Specific config error")
        mock_blob_csv.side_effect = original_error

        source = BlobDataSource(config_path=str(valid_blob_config))

        with pytest.raises(BlobConfigurationError) as exc_info:
            source.load()

        assert exc_info.value is original_error

    def test_load_translates_file_not_found(self, tmp_path: Path):
        """FileNotFoundError is translated with config path."""
        missing_path = tmp_path / "missing.yaml"

        source = BlobDataSource(config_path=str(missing_path))

        with pytest.raises(BlobConfigurationError) as exc_info:
            source.load()

        assert str(missing_path) in str(exc_info.value)


class TestLandscapeIntegration:
    """Tests for landscape integration."""

    def test_load_saves_to_landscape_when_active(self, valid_blob_config: Path, mock_blob_csv):
        """Data is saved to landscape when landscape is active."""
        test_df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        mock_blob_csv.return_value = test_df

        with tempfile.TemporaryDirectory() as tmpdir:
            landscape_path = Path(tmpdir) / "landscape"

            # Import here to avoid circular imports
            from elspeth.core.landscape import RunLandscape

            with RunLandscape(base_path=landscape_path, persist=True) as _landscape:
                source = BlobDataSource(
                    config_path=str(valid_blob_config),
                    name="test_source",
                )
                _result = source.load()

                # Verify data was saved
                saved_path = landscape_path / "inputs" / "test_source" / "source_data.csv"
                assert saved_path.exists()

                saved_df = pd.read_csv(saved_path)
                assert len(saved_df) == 3
                assert list(saved_df.columns) == ["id", "value"]

    def test_load_registers_artifact_with_landscape(
        self, valid_blob_config: Path, mock_blob_csv
    ):
        """Artifact is registered with landscape metadata."""
        test_df = pd.DataFrame({"col1": [1], "col2": [2]})
        mock_blob_csv.return_value = test_df

        with tempfile.TemporaryDirectory() as tmpdir:
            landscape_path = Path(tmpdir) / "landscape"

            from elspeth.core.landscape import RunLandscape

            with RunLandscape(base_path=landscape_path, persist=True) as landscape:
                source = BlobDataSource(
                    config_path=str(valid_blob_config),
                    profile="production",
                    name="my_source",
                )
                source.load()

                # Check manifest
                manifest = landscape.get_manifest()
                input_artifacts = [
                    a for a in manifest["artifacts"] if a["category"] == "inputs"
                ]

                assert len(input_artifacts) == 1
                artifact = input_artifacts[0]
                assert artifact["plugin_id"] == "my_source"
                assert artifact["metadata"]["rows"] == 1
                assert artifact["metadata"]["columns"] == ["col1", "col2"]
                assert artifact["metadata"]["profile"] == "production"

    def test_load_works_without_active_landscape(self, valid_blob_config: Path, mock_blob_csv):
        """Load works correctly when no landscape is active."""
        test_df = pd.DataFrame({"x": [1]})
        mock_blob_csv.return_value = test_df

        # Ensure no landscape is active
        from elspeth.core.landscape import get_current_landscape

        assert get_current_landscape() is None

        source = BlobDataSource(config_path=str(valid_blob_config))
        result = source.load()

        assert len(result) == 1


class TestDefaultName:
    """Tests for default name behavior."""

    def test_default_name_is_azure_blob(self, valid_blob_config: Path):
        """Default name is 'azure_blob'."""
        source = BlobDataSource(config_path=str(valid_blob_config))
        assert source.name == "azure_blob"

    def test_custom_name_overrides_default(self, valid_blob_config: Path):
        """Custom name overrides default."""
        source = BlobDataSource(
            config_path=str(valid_blob_config),
            name="my_custom_source",
        )
        assert source.name == "my_custom_source"


class TestBlobDataSourceSchemas:
    """Tests for BlobDataSource schema declarations."""

    def test_has_input_schema(self):
        """BlobDataSource should have input_schema class attribute."""
        assert hasattr(BlobDataSource, "input_schema")
        assert isinstance(BlobDataSource.input_schema, dict)

    def test_has_output_schema(self):
        """BlobDataSource should have output_schema class attribute."""
        assert hasattr(BlobDataSource, "output_schema")
        assert isinstance(BlobDataSource.output_schema, dict)

    def test_input_schema_is_empty(self):
        """Datasources have no input - they are the origin of data."""
        assert BlobDataSource.input_schema == {}

    def test_output_schema_describes_dataframe(self):
        """Output schema describes DataFrame output."""
        schema = BlobDataSource.output_schema
        assert schema["type"] == "object"
        assert "description" in schema
