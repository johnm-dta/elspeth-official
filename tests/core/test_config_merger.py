# tests/core/test_config_merger.py
"""Tests for configuration merging logic."""

from elspeth.core.config_merger import ConfigSource, ConfigurationMerger


def test_merger_exists():
    """Test that ConfigurationMerger can be instantiated."""
    merger = ConfigurationMerger()
    assert merger is not None


def test_merge_single_source():
    """Test merging single configuration source."""
    merger = ConfigurationMerger()
    source = ConfigSource(name="test", data={"key": "value"}, precedence=1)
    result = merger.merge(source)
    assert result == {"key": "value"}


def test_merge_override_strategy():
    """Test that higher precedence overrides lower precedence."""
    merger = ConfigurationMerger()
    source1 = ConfigSource(name="low", data={"key": "low_value"}, precedence=1)
    source2 = ConfigSource(name="high", data={"key": "high_value"}, precedence=2)

    result = merger.merge(source1, source2)
    assert result == {"key": "high_value"}


def test_merge_multiple_keys_override():
    """Test merging multiple keys with override strategy."""
    merger = ConfigurationMerger()
    source1 = ConfigSource(name="base", data={"key1": "value1", "key2": "value2"}, precedence=1)
    source2 = ConfigSource(name="override", data={"key2": "new_value2", "key3": "value3"}, precedence=2)

    result = merger.merge(source1, source2)
    assert result == {"key1": "value1", "key2": "new_value2", "key3": "value3"}


def test_merge_append_strategy():
    """Test that list keys append from all sources."""
    merger = ConfigurationMerger()
    source1 = ConfigSource(name="pack", data={"row_plugins": [{"name": "plugin1"}]}, precedence=1)
    source2 = ConfigSource(name="profile", data={"row_plugins": [{"name": "plugin2"}]}, precedence=2)

    result = merger.merge(source1, source2)
    assert result == {"row_plugins": [{"name": "plugin1"}, {"name": "plugin2"}]}


def test_merge_append_empty_base():
    """Test appending when base doesn't have the key."""
    merger = ConfigurationMerger()
    source1 = ConfigSource(name="base", data={}, precedence=1)
    source2 = ConfigSource(name="override", data={"sinks": [{"plugin": "csv"}]}, precedence=2)

    result = merger.merge(source1, source2)
    assert result == {"sinks": [{"plugin": "csv"}]}


def test_merge_deep_merge_strategy():
    """Test deep merging of nested dictionaries."""
    merger = ConfigurationMerger()
    source1 = ConfigSource(name="pack", data={"llm": {"plugin": "azure_openai", "options": {"temperature": 0.5}}}, precedence=1)
    source2 = ConfigSource(name="profile", data={"llm": {"options": {"temperature": 0.7, "max_tokens": 100}}}, precedence=2)

    result = merger.merge(source1, source2)
    expected = {
        "llm": {
            "plugin": "azure_openai",
            "options": {"temperature": 0.7, "max_tokens": 100},  # Overridden from source2  # Added from source2
        }
    }
    assert result == expected


def test_merge_deep_merge_three_levels():
    """Test deep merging with 3+ levels of nesting."""
    merger = ConfigurationMerger()
    source1 = ConfigSource(name="base", data={"retry": {"max_attempts": 3, "backoff": {"initial": 1.0}}}, precedence=1)
    source2 = ConfigSource(name="override", data={"retry": {"backoff": {"multiplier": 2.0}}}, precedence=2)

    result = merger.merge(source1, source2)
    expected = {"retry": {"max_attempts": 3, "backoff": {"initial": 1.0, "multiplier": 2.0}}}
    assert result == expected


def test_explain_simple_override():
    """Test explain method shows configuration source."""
    merger = ConfigurationMerger()
    source1 = ConfigSource(name="pack", data={"rate_limit": 10}, precedence=1)
    source2 = ConfigSource(name="profile", data={"rate_limit": 50}, precedence=2)

    merged = merger.merge(source1, source2)
    explanation = merger.explain("rate_limit", merged)

    assert "rate_limit = 50" in explanation
    assert "profile" in explanation
    assert "override" in explanation


def test_explain_nested_key():
    """Test explain works with nested keys."""
    merger = ConfigurationMerger()
    source1 = ConfigSource(name="base", data={"llm": {"options": {"temperature": 0.5}}}, precedence=1)
    source2 = ConfigSource(name="override", data={"llm": {"options": {"temperature": 0.7}}}, precedence=2)

    merged = merger.merge(source1, source2)
    explanation = merger.explain("llm.options.temperature", merged)

    assert "0.7" in explanation
    assert "override" in explanation


def test_config_load_uses_merger():
    """Integration test: config.py uses ConfigurationMerger."""
    import tempfile
    from pathlib import Path

    from elspeth.config import load_settings

    # Create temporary config file
    config_yaml = """
default:
  datasource:
    plugin: local_csv
    options:
      path: test.csv
  llm:
    plugin: mock
  prompt_packs:
    test_pack:
      prompts:
        system: "Test system prompt"
      row_plugins:
        - name: plugin1
  prompt_pack: test_pack
  prompts:
    user: "Profile user prompt"
  row_plugins:
    - name: plugin2
  sinks: []
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = f.name

    try:
        settings = load_settings(config_path, profile="default")

        # Verify prompt pack prompts are merged
        assert settings.orchestrator_config.llm_prompt.get("system") == "Test system prompt"
        assert settings.orchestrator_config.llm_prompt.get("user") == "Profile user prompt"

        # Verify plugins are appended (pack plugins + profile plugins)
        plugin_names = [p["name"] for p in settings.orchestrator_config.transform_plugin_defs]
        assert "plugin1" in plugin_names  # From pack
        assert "plugin2" in plugin_names  # From profile
        # Pack plugins should come first
        assert plugin_names.index("plugin1") < plugin_names.index("plugin2")

    finally:
        Path(config_path).unlink()


def test_plugin_accumulation_across_precedence_levels():
    """Integration test: Verify plugins accumulate across all precedence levels.

    This test verifies Critical Issue #1 fix - all plugin key variants
    should use APPEND strategy to accumulate plugins from all sources.
    """
    merger = ConfigurationMerger()

    # Simulate 5-level precedence chain
    system_defaults = ConfigSource(
        name="system_defaults",
        data={
            "row_plugins": [{"name": "system_row_plugin"}],
            "aggregator_plugins": [{"name": "system_agg_plugin"}],
            "baseline_plugins": [{"name": "system_baseline_plugin"}],
            "sinks": [{"plugin": "system_sink"}],
            "llm_middlewares": [{"name": "system_middleware"}],
        },
        precedence=1,
    )

    prompt_pack = ConfigSource(
        name="prompt_pack",
        data={
            "row_plugins": [{"name": "pack_row_plugin"}],
            "aggregator_plugins": [{"name": "pack_agg_plugin"}],
        },
        precedence=2,
    )

    profile = ConfigSource(
        name="profile",
        data={
            "row_plugins": [{"name": "profile_row_plugin"}],
            "sinks": [{"plugin": "profile_sink"}],
        },
        precedence=3,
    )

    suite_defaults = ConfigSource(
        name="suite_defaults",
        data={
            "row_plugins": [{"name": "suite_row_plugin"}],
            "baseline_plugins": [{"name": "suite_baseline_plugin"}],
        },
        precedence=4,
    )

    experiment = ConfigSource(
        name="experiment",
        data={
            "row_plugins": [{"name": "exp_row_plugin"}],
            "llm_middlewares": [{"name": "exp_middleware"}],
        },
        precedence=5,
    )

    # Merge all sources
    result = merger.merge(system_defaults, prompt_pack, profile, suite_defaults, experiment)

    # Verify plugins accumulated in precedence order
    assert result["row_plugins"] == [
        {"name": "system_row_plugin"},
        {"name": "pack_row_plugin"},
        {"name": "profile_row_plugin"},
        {"name": "suite_row_plugin"},
        {"name": "exp_row_plugin"},
    ]

    assert result["aggregator_plugins"] == [
        {"name": "system_agg_plugin"},
        {"name": "pack_agg_plugin"},
    ]

    assert result["baseline_plugins"] == [
        {"name": "system_baseline_plugin"},
        {"name": "suite_baseline_plugin"},
    ]

    assert result["sinks"] == [
        {"plugin": "system_sink"},
        {"plugin": "profile_sink"},
    ]

    assert result["llm_middlewares"] == [
        {"name": "system_middleware"},
        {"name": "exp_middleware"},
    ]


def test_plugin_defs_variant_names_accumulate():
    """Test that normalized plugin def names also accumulate (e.g., row_plugin_defs).

    This test verifies that the *_defs variant names are also registered
    in MERGE_STRATEGIES and accumulate correctly.
    """
    merger = ConfigurationMerger()

    source1 = ConfigSource(
        name="base",
        data={
            "row_plugin_defs": [{"name": "plugin1"}],
            "transform_plugin_defs": [{"name": "transform1"}],
            "aggregator_plugin_defs": [{"name": "agg1"}],
            "aggregation_transform_defs": [{"name": "agg_transform1"}],
            "baseline_plugin_defs": [{"name": "baseline1"}],
            "sink_defs": [{"plugin": "sink1"}],
            "llm_middleware_defs": [{"name": "middleware1"}],
            "early_stop_plugin_defs": [{"name": "early_stop1"}],
            "halt_condition_plugin_defs": [{"name": "halt1"}],
        },
        precedence=1,
    )

    source2 = ConfigSource(
        name="override",
        data={
            "row_plugin_defs": [{"name": "plugin2"}],
            "transform_plugin_defs": [{"name": "transform2"}],
            "aggregator_plugin_defs": [{"name": "agg2"}],
            "aggregation_transform_defs": [{"name": "agg_transform2"}],
            "baseline_plugin_defs": [{"name": "baseline2"}],
            "sink_defs": [{"plugin": "sink2"}],
            "llm_middleware_defs": [{"name": "middleware2"}],
            "early_stop_plugin_defs": [{"name": "early_stop2"}],
            "halt_condition_plugin_defs": [{"name": "halt2"}],
        },
        precedence=2,
    )

    result = merger.merge(source1, source2)

    # Verify all *_defs variants accumulated (not overridden)
    assert result["row_plugin_defs"] == [{"name": "plugin1"}, {"name": "plugin2"}]
    assert result["transform_plugin_defs"] == [{"name": "transform1"}, {"name": "transform2"}]
    assert result["aggregator_plugin_defs"] == [{"name": "agg1"}, {"name": "agg2"}]
    assert result["aggregation_transform_defs"] == [{"name": "agg_transform1"}, {"name": "agg_transform2"}]
    assert result["baseline_plugin_defs"] == [{"name": "baseline1"}, {"name": "baseline2"}]
    assert result["sink_defs"] == [{"plugin": "sink1"}, {"plugin": "sink2"}]
    assert result["llm_middleware_defs"] == [{"name": "middleware1"}, {"name": "middleware2"}]
    assert result["early_stop_plugin_defs"] == [{"name": "early_stop1"}, {"name": "early_stop2"}]
    assert result["halt_condition_plugin_defs"] == [{"name": "halt1"}, {"name": "halt2"}]


def test_suite_defaults_merging_with_prompt_packs():
    """Integration test: Suite defaults merge with prompt pack plugins.

    Verifies that suite defaults (precedence=4) correctly merge with
    prompt pack (precedence=2), accumulating plugins from both.
    """
    merger = ConfigurationMerger()

    prompt_pack = ConfigSource(
        name="prompt_pack",
        data={
            "row_plugins": [{"name": "pack_plugin1"}, {"name": "pack_plugin2"}],
            "aggregator_plugins": [{"name": "pack_agg1"}],
            "prompts": {"system": "Pack system prompt"},
        },
        precedence=2,
    )

    suite_defaults = ConfigSource(
        name="suite_defaults",
        data={
            "row_plugins": [{"name": "suite_plugin1"}],
            "aggregator_plugins": [{"name": "suite_agg1"}, {"name": "suite_agg2"}],
            "prompts": {"user": "Suite user prompt"},
        },
        precedence=4,
    )

    result = merger.merge(prompt_pack, suite_defaults)

    # Verify plugins accumulated
    assert result["row_plugins"] == [
        {"name": "pack_plugin1"},
        {"name": "pack_plugin2"},
        {"name": "suite_plugin1"},
    ]

    assert result["aggregator_plugins"] == [
        {"name": "pack_agg1"},
        {"name": "suite_agg1"},
        {"name": "suite_agg2"},
    ]

    # Verify prompts deep merged
    assert result["prompts"] == {
        "system": "Pack system prompt",
        "user": "Suite user prompt",
    }


def test_full_five_level_precedence_chain():
    """Integration test: Complete 5-level precedence chain with mixed strategies.

    Tests the full merge chain:
    1. System defaults (precedence=1)
    2. Prompt pack (precedence=2)
    3. Profile (precedence=3)
    4. Suite defaults (precedence=4)
    5. Experiment config (precedence=5)

    Verifies APPEND, OVERRIDE, and DEEP_MERGE strategies all work correctly.
    """
    merger = ConfigurationMerger()

    system_defaults = ConfigSource(
        name="system_defaults",
        data={
            "rate_limit": 10,
            "row_plugins": [{"name": "system_plugin"}],
            "llm": {"plugin": "mock", "options": {"temperature": 0.5, "max_tokens": 100}},
        },
        precedence=1,
    )

    prompt_pack = ConfigSource(
        name="prompt_pack",
        data={
            "row_plugins": [{"name": "pack_plugin"}],
            "llm": {"options": {"temperature": 0.7}},  # Override temperature
            "prompts": {"system": "System prompt from pack"},
        },
        precedence=2,
    )

    profile = ConfigSource(
        name="profile",
        data={
            "rate_limit": 50,  # Override from system
            "row_plugins": [{"name": "profile_plugin"}],
            "llm": {"plugin": "azure_openai", "options": {"max_tokens": 200}},  # Override from system  # Override from system
            "prompts": {"user": "User prompt from profile"},
        },
        precedence=3,
    )

    suite_defaults = ConfigSource(
        name="suite_defaults",
        data={
            "row_plugins": [{"name": "suite_plugin"}],
            "sinks": [{"plugin": "csv"}],
        },
        precedence=4,
    )

    experiment = ConfigSource(
        name="experiment",
        data={
            "rate_limit": 100,  # Final override
            "row_plugins": [{"name": "exp_plugin"}],
            "llm": {"options": {"temperature": 0.9}},  # Final override
        },
        precedence=5,
    )

    result = merger.merge(system_defaults, prompt_pack, profile, suite_defaults, experiment)

    # Verify OVERRIDE strategy: highest precedence wins
    assert result["rate_limit"] == 100

    # Verify APPEND strategy: accumulate from all sources in precedence order
    assert result["row_plugins"] == [
        {"name": "system_plugin"},
        {"name": "pack_plugin"},
        {"name": "profile_plugin"},
        {"name": "suite_plugin"},
        {"name": "exp_plugin"},
    ]

    assert result["sinks"] == [{"plugin": "csv"}]

    # Verify DEEP_MERGE strategy: nested dicts merge recursively
    assert result["llm"] == {
        "plugin": "azure_openai",  # From profile (overrides system mock)
        "options": {
            "temperature": 0.9,  # From experiment (final override)
            "max_tokens": 200,  # From profile (overrides system 100)
        },
    }

    assert result["prompts"] == {
        "system": "System prompt from pack",
        "user": "User prompt from profile",
    }

    # Verify trace shows correct sources
    explanation = merger.explain("rate_limit", result)
    assert "experiment" in explanation
    assert "100" in explanation


def test_mixed_base_and_normalized_plugin_names():
    """Test that mixing base names (row_plugins) and normalized names (row_plugin_defs) works.

    In real usage, different precedence levels may use different naming conventions.
    This test verifies they can coexist and accumulate correctly.
    """
    merger = ConfigurationMerger()

    source1 = ConfigSource(
        name="base",
        data={
            "row_plugins": [{"name": "plugin1"}],  # Base name
            "aggregator_plugin_defs": [{"name": "agg1"}],  # Normalized name
        },
        precedence=1,
    )

    source2 = ConfigSource(
        name="override",
        data={
            "row_plugin_defs": [{"name": "plugin2"}],  # Normalized name
            "aggregator_plugins": [{"name": "agg2"}],  # Base name
        },
        precedence=2,
    )

    result = merger.merge(source1, source2)

    # Both row_plugins and row_plugin_defs should accumulate separately
    # (they are different keys in the config)
    assert result["row_plugins"] == [{"name": "plugin1"}]
    assert result["row_plugin_defs"] == [{"name": "plugin2"}]
    assert result["aggregator_plugin_defs"] == [{"name": "agg1"}]
    assert result["aggregator_plugins"] == [{"name": "agg2"}]
