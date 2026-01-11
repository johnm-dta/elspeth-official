"""Tests for default plugin registrations."""

from elspeth.core.sda import plugin_registry


def test_default_transform_plugins_are_registered():
    """Default transform plugins should be auto-registered via import side effects."""
    plugin = plugin_registry.create_transform_plugin({"name": "score_extractor", "options": {"criteria": []}})
    assert getattr(plugin, "name", None) == "score_extractor"
