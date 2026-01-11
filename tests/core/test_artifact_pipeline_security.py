"""Security enforcement tests for ArtifactPipeline."""

import pytest

from elspeth.core.artifact_pipeline import ArtifactPipeline, SinkBinding
from elspeth.core.interfaces import Artifact


class _ProducerSink:
    def __init__(self):
        self.written = False

    def write(self, payload, metadata=None):
        self.written = True

    def collect_artifacts(self):
        return {"report": Artifact(id="report", type="report")}


class _ConsumerSink:
    def write(self, payload, metadata=None):
        pass


def _binding_for_producer():
    return SinkBinding(
        id="producer",
        plugin="producer",
        sink=_ProducerSink(),
        artifact_config={
            "produces": [
                {"name": "report", "type": "data/report", "security_level": "secret", "alias": "report"},
            ],
        },
        original_index=0,
        security_level="secret",
    )


def _binding_for_consumer(clearance: str):
    return SinkBinding(
        id="consumer",
        plugin="consumer",
        sink=_ConsumerSink(),
        artifact_config={"consumes": [{"token": "@report"}]},
        original_index=1,
        security_level=clearance,
    )


def test_pipeline_blocks_when_consumer_clearance_too_low():
    """Consumer with lower clearance than produced artifact should be blocked."""
    with pytest.raises(PermissionError):
        pipeline = ArtifactPipeline([_binding_for_producer(), _binding_for_consumer(clearance="official")])
        pipeline.execute({"results": []}, metadata={})
