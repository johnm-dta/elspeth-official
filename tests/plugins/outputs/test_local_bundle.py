"""Tests for LocalBundleSink."""

import json

import pandas as pd

from elspeth.plugins.outputs.local_bundle import LocalBundleSink


def test_local_bundle_writes_manifest_and_results(tmp_path):
    sink = LocalBundleSink(base_path=tmp_path, bundle_name="bundle", timestamped=False, write_json=True, write_csv=True)

    payload = {
        "results": [
            {"row": {"text": "hello"}, "response": {"content": "hi"}},
        ],
        "aggregates": {"sum": 1},
    }
    metadata = {"experiment": "cycle-a", "security_level": "official"}

    sink.write(payload, metadata=metadata)

    bundle_dir = tmp_path / "bundle"
    manifest_path = bundle_dir / "manifest.json"
    results_path = bundle_dir / "results.json"
    csv_path = bundle_dir / "results.csv"

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
    assert manifest["metadata"]["experiment"] == "cycle-a"
    assert manifest["aggregates"]["sum"] == 1

    assert results_path.exists()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["results"][0]["row"]["text"] == "hello"

    assert csv_path.exists()
    csv_df = pd.read_csv(csv_path)
    assert list(csv_df["text"]) == ["hello"]
