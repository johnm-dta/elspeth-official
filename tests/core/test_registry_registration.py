"""Tests for plugin self-registration API."""

from elspeth.core.registry import PluginRegistry


class DummySink:
    """Test sink class."""
    def __init__(self, path: str):
        self.path = path

    def write(self, results, *, metadata=None):
        pass


class TestPluginRegistration:
    def test_register_sink_adds_to_registry(self):
        reg = PluginRegistry()
        reg.reset()  # Start clean

        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        reg.register_sink("test_sink", DummySink, schema)

        assert "test_sink" in reg._sinks

    def test_create_registered_sink(self):
        reg = PluginRegistry()
        reg.reset()

        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        reg.register_sink("test_sink", DummySink, schema)

        sink = reg.create_sink("test_sink", {"path": "/tmp/test.csv"})
        assert isinstance(sink, DummySink)
        assert sink.path == "/tmp/test.csv"

    def test_reset_clears_all_registrations(self):
        reg = PluginRegistry()
        reg.reset()

        reg.register_sink("test_sink", DummySink, None)
        assert "test_sink" in reg._sinks

        reg.reset()
        assert "test_sink" not in reg._sinks


class DummyDataSource:
    def __init__(self, path: str):
        self.path = path
    def load(self):
        return None


class TestDatasourceRegistration:
    def test_register_datasource(self):
        reg = PluginRegistry()
        reg.reset()

        reg.register_datasource("test_ds", DummyDataSource, None)
        ds = reg.create_datasource("test_ds", {"path": "/data.csv"})
        assert isinstance(ds, DummyDataSource)


class DummyLLM:
    def __init__(self, model: str):
        self.model = model
    def generate(self, *, system_prompt, user_prompt, metadata=None):
        return {"content": "test"}


class TestLLMRegistration:
    def test_register_llm(self):
        reg = PluginRegistry()
        reg.reset()

        reg.register_llm("test_llm", DummyLLM, None)
        llm = reg.create_llm("test_llm", {"model": "gpt-4"})
        assert isinstance(llm, DummyLLM)
