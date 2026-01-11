"""Tests for LLM query transform plugin."""

from __future__ import annotations

import pytest


class TestLLMQueryPluginInit:
    """Test plugin initialization."""

    def test_plugin_instantiation_with_minimal_config(self):
        """Plugin can be instantiated with minimal required config."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {}},
            queries=[
                {
                    "name": "test_query",
                    "inputs": {},
                    "output_key": "test_output",
                }
            ],
            system_prompt="You are a test assistant.",
            user_prompt="Say hello.",
        )

        assert plugin.name == "llm_query"
        assert len(plugin.queries) == 1

    def test_plugin_creates_llm_client(self):
        """Plugin creates LLM client from config."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 42}},
            queries=[{"name": "q1", "inputs": {}, "output_key": "out"}],
            system_prompt="System",
            user_prompt="User",
        )

        assert plugin.llm_client is not None
        assert hasattr(plugin.llm_client, "generate")

    def test_plugin_compiles_prompt_templates(self):
        """Plugin compiles system and user prompt templates."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {}},
            queries=[{"name": "q1", "inputs": {}, "output_key": "out"}],
            system_prompt="You are {{ role }}.",
            user_prompt="Process {{ item }}.",
        )

        assert plugin.system_template is not None
        assert plugin.user_template is not None
        assert plugin.engine is not None

    def test_plugin_creates_executor(self):
        """Plugin creates LLMExecutor with middleware and retry."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {}},
            queries=[{"name": "q1", "inputs": {}, "output_key": "out"}],
            system_prompt="System",
            user_prompt="User",
            retry={"max_attempts": 3, "initial_delay": 1.0},
        )

        assert plugin.executor is not None
        assert plugin.executor.retry_config == {"max_attempts": 3, "initial_delay": 1.0}


class TestLLMQueryPluginTransform:
    """Test plugin transform method."""

    def test_transform_executes_single_query(self):
        """Transform executes a single query and stores response."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 42}},
            queries=[
                {
                    "name": "greeting",
                    "inputs": {"name": "user_name"},
                    "output_key": "greeting_response",
                    "flatten_to_row": True,
                }
            ],
            system_prompt="You are helpful.",
            user_prompt="Greet {{ name }}.",
        )

        row = {"user_name": "Alice", "other_field": "value"}
        context: dict = {}

        result = plugin.transform(row, context)

        # Check context has full response
        assert "greeting_response" in context
        assert "content" in context["greeting_response"]

        # Check row has flattened values
        assert "greeting_response_content" in result

        # Check original row data preserved
        assert result["user_name"] == "Alice"
        assert result["other_field"] == "value"

    def test_transform_executes_multiple_queries(self):
        """Transform executes multiple queries in sequence."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 42}},
            queries=[
                {
                    "name": "capability_circumstance",
                    "inputs": {"case_study": "cs1_text"},
                    "defaults": {"context": "capability"},
                    "output_key": "cap_circ",
                },
                {
                    "name": "capacity_circumstance",
                    "inputs": {"case_study": "cs2_text"},
                    "defaults": {"context": "capacity"},
                    "output_key": "cap_circ_2",
                },
            ],
            system_prompt="Evaluate.",
            user_prompt="Context: {{ context }}. Study: {{ case_study }}.",
        )

        row = {"cs1_text": "Case study 1", "cs2_text": "Case study 2"}
        context: dict = {}

        result = plugin.transform(row, context)

        # Both queries executed
        assert "cap_circ" in context
        assert "cap_circ_2" in context

        # Both flattened to row
        assert "cap_circ_content" in result
        assert "cap_circ_2_content" in result

    def test_query_defaults_override_pack_defaults(self):
        """Query-level defaults override plugin-level pack defaults."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {}},
            queries=[
                {
                    "name": "q1",
                    "inputs": {},
                    "defaults": {"context": "query_context"},  # Should win
                    "output_key": "out",
                }
            ],
            system_prompt="System",
            user_prompt="Context is {{ context }}.",
        )

        # Set pack defaults (normally from prompt pack)
        plugin.pack_defaults = {"context": "pack_context"}

        # Mock the executor to capture the rendered prompt
        captured_prompts = []

        def capture_execute(user_prompt, metadata, system_prompt=None, skip_rate_limit=False):
            captured_prompts.append(user_prompt)
            return {"content": "ok"}

        plugin.executor.execute = capture_execute

        plugin.transform({}, {})

        # Query default should have won
        assert "query_context" in captured_prompts[0]
        assert "pack_context" not in captured_prompts[0]

    def test_flatten_to_row_can_be_disabled(self):
        """When flatten_to_row is False, response only goes to context."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 42}},
            queries=[
                {
                    "name": "q1",
                    "inputs": {},
                    "output_key": "hidden",
                    "flatten_to_row": False,  # Disable flattening
                }
            ],
            system_prompt="System",
            user_prompt="User",
        )

        row = {"existing": "data"}
        context: dict = {}

        result = plugin.transform(row, context)

        # Context has response
        assert "hidden" in context
        assert "content" in context["hidden"]

        # Row does NOT have flattened fields
        assert "hidden_content" not in result
        assert "hidden_raw" not in result

        # Original data preserved
        assert result["existing"] == "data"


class TestLLMQueryPluginPromptPack:
    """Test prompt pack resolution."""

    def test_resolves_prompt_pack(self):
        """Plugin resolves prompts from remote pack URL."""
        from unittest.mock import patch

        # Mock the resolver
        mock_pack = {
            "prompts": {
                "system": "Pack system prompt",
                "user": "Pack user prompt with {{ var }}",
            },
            "prompt_defaults": {"var": "default_value"},
        }

        with patch(
            "elspeth.plugins.transforms.llm_query.resolve_remote_pack"
        ) as mock_resolve:
            mock_resolve.return_value = mock_pack

            from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

            plugin = LLMQueryPlugin(
                llm={"plugin": "mock", "options": {}},
                queries=[{"name": "q1", "inputs": {}, "output_key": "out"}],
                prompt_pack="azuredevops://org/project/repo/prompts",
            )

            mock_resolve.assert_called_once_with("azuredevops://org/project/repo/prompts")
            assert plugin.pack_defaults == {"var": "default_value"}


class TestLLMQueryPluginValidation:
    """Test query validation."""

    def test_empty_queries_raises_error(self):
        """Plugin raises error when queries list is empty."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        with pytest.raises(ValueError) as exc_info:
            LLMQueryPlugin(
                llm={"plugin": "mock", "options": {}},
                queries=[],  # Empty queries
                system_prompt="System",
                user_prompt="User",
            )

        assert "requires at least one query" in str(exc_info.value)

    def test_missing_name_raises_error(self):
        """Plugin raises error when query is missing 'name' field."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        with pytest.raises(ValueError) as exc_info:
            LLMQueryPlugin(
                llm={"plugin": "mock", "options": {}},
                queries=[
                    {
                        # Missing 'name'
                        "inputs": {},
                        "output_key": "out",
                    }
                ],
                system_prompt="System",
                user_prompt="User",
            )

        assert "missing required field 'name'" in str(exc_info.value)
        assert "index 0" in str(exc_info.value)

    def test_missing_output_key_raises_error(self):
        """Plugin raises error when query is missing 'output_key' field."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        with pytest.raises(ValueError) as exc_info:
            LLMQueryPlugin(
                llm={"plugin": "mock", "options": {}},
                queries=[
                    {
                        "name": "my_query",
                        "inputs": {},
                        # Missing 'output_key'
                    }
                ],
                system_prompt="System",
                user_prompt="User",
            )

        assert "missing required field 'output_key'" in str(exc_info.value)
        assert "my_query" in str(exc_info.value)

    def test_duplicate_output_key_raises_error(self):
        """Plugin raises error when queries have duplicate output_keys."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        with pytest.raises(ValueError) as exc_info:
            LLMQueryPlugin(
                llm={"plugin": "mock", "options": {}},
                queries=[
                    {
                        "name": "query_1",
                        "inputs": {},
                        "output_key": "shared_key",
                    },
                    {
                        "name": "query_2",
                        "inputs": {},
                        "output_key": "shared_key",  # Duplicate!
                    },
                ],
                system_prompt="System",
                user_prompt="User",
            )

        assert "Duplicate output_key 'shared_key'" in str(exc_info.value)
        assert "query_2" in str(exc_info.value)

    def test_valid_queries_pass_validation(self):
        """Plugin accepts valid query definitions."""
        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        # Should not raise
        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {}},
            queries=[
                {
                    "name": "query_1",
                    "inputs": {"text": "input_field"},
                    "output_key": "output_1",
                },
                {
                    "name": "query_2",
                    "inputs": {},
                    "output_key": "output_2",
                    "defaults": {"default_var": "value"},
                    "flatten_to_row": False,
                },
            ],
            system_prompt="System",
            user_prompt="User {{ text }} {{ default_var }}",
        )

        assert len(plugin.queries) == 2

    def test_validation_occurs_before_llm_client_creation(self):
        """Validation error should occur before expensive LLM client creation."""
        from unittest.mock import patch

        from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

        # Patch registry.create_llm to track if it's called
        with patch("elspeth.plugins.transforms.llm_query.registry") as mock_registry:
            with pytest.raises(ValueError) as exc_info:
                LLMQueryPlugin(
                    llm={"plugin": "mock", "options": {}},
                    queries=[],  # Empty - should fail validation first
                    system_prompt="System",
                    user_prompt="User",
                )

            assert "requires at least one query" in str(exc_info.value)
            # LLM client should NOT have been created
            mock_registry.create_llm.assert_not_called()
