"""Tests for datasource landscape integration."""

import tempfile
from pathlib import Path

import pandas as pd

from elspeth.core.landscape import RunLandscape


class TestCSVDataSourceLandscape:
    """Test CSV datasource saves to landscape."""

    def test_csv_source_saves_to_landscape(self):
        """CSVDataSource saves loaded data to landscape."""
        from elspeth.plugins.datasources.csv_local import CSVDataSource

        # Create temp CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,name\n1,alice\n2,bob\n")
            csv_path = Path(f.name)

        try:
            with RunLandscape() as landscape:
                source = CSVDataSource(path=str(csv_path), name="test_data")
                _df = source.load()

                # Check landscape has the data
                saved_path = landscape.root / "inputs" / "test_data" / "source_data.csv"
                assert saved_path.exists()

                saved_df = pd.read_csv(saved_path)
                assert len(saved_df) == 2
                assert list(saved_df.columns) == ["id", "name"]
        finally:
            csv_path.unlink()

    def test_csv_source_works_without_landscape(self):
        """CSVDataSource works when no landscape active."""
        from elspeth.plugins.datasources.csv_local import CSVDataSource

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,value\n1,100\n")
            csv_path = Path(f.name)

        try:
            source = CSVDataSource(path=str(csv_path))
            df = source.load()
            assert len(df) == 1
        finally:
            csv_path.unlink()

    def test_csv_source_registers_artifact(self):
        """CSVDataSource registers artifact with metadata."""
        from elspeth.plugins.datasources.csv_local import CSVDataSource

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,name,value\n1,alice,100\n2,bob,200\n3,carol,300\n")
            csv_path = Path(f.name)

        try:
            with RunLandscape() as landscape:
                source = CSVDataSource(path=str(csv_path), name="my_source")
                source.load()

                manifest = landscape.get_manifest()
                artifacts = [a for a in manifest["artifacts"] if a["plugin_id"] == "my_source"]
                assert len(artifacts) == 1
                assert artifacts[0]["category"] == "inputs"
                assert artifacts[0]["metadata"]["rows"] == 3
                assert artifacts[0]["metadata"]["columns"] == ["id", "name", "value"]
        finally:
            csv_path.unlink()
