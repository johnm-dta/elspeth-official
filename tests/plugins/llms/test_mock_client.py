"""Tests for Mock LLM client."""

from __future__ import annotations

from elspeth.plugins.llms.mock import MockLLMClient


class TestMockLLMClientInit:
    """Tests for MockLLMClient initialization."""

    def test_init_with_defaults(self):
        """Client initializes with default seed."""
        client = MockLLMClient()
        assert client.seed == 0

    def test_init_with_custom_seed(self):
        """Client accepts custom seed."""
        client = MockLLMClient(seed=42)
        assert client.seed == 42


class TestMockLLMClientGenerate:
    """Tests for MockLLMClient.generate()."""

    def test_generate_returns_dict_with_required_keys(self):
        """Generate returns dict with content, metrics, and raw."""
        client = MockLLMClient()
        result = client.generate(
            system_prompt="You are a test assistant.",
            user_prompt="Hello, world!",
        )

        assert "content" in result
        assert "metrics" in result
        assert "raw" in result

    def test_generate_content_is_string(self):
        """Generated content is a string."""
        client = MockLLMClient()
        result = client.generate(
            system_prompt="System prompt",
            user_prompt="User prompt",
        )

        assert isinstance(result["content"], str)

    def test_generate_content_includes_mock_marker(self):
        """Generated content includes [mock] marker."""
        client = MockLLMClient()
        result = client.generate(
            system_prompt="System prompt",
            user_prompt="User prompt",
        )

        assert "[mock]" in result["content"]

    def test_generate_content_includes_user_prompt(self):
        """Generated content includes the user prompt."""
        client = MockLLMClient()
        result = client.generate(
            system_prompt="System prompt",
            user_prompt="Unique user prompt text",
        )

        assert "Unique user prompt text" in result["content"]

    def test_generate_metrics_includes_score(self):
        """Metrics include a score value."""
        client = MockLLMClient()
        result = client.generate(
            system_prompt="System",
            user_prompt="User",
        )

        assert "score" in result["metrics"]
        assert isinstance(result["metrics"]["score"], float)

    def test_generate_raw_includes_prompts(self):
        """Raw section includes original prompts."""
        client = MockLLMClient()
        result = client.generate(
            system_prompt="My system prompt",
            user_prompt="My user prompt",
            metadata={"key": "value"},
        )

        assert result["raw"]["system_prompt"] == "My system prompt"
        assert result["raw"]["user_prompt"] == "My user prompt"
        assert result["raw"]["metadata"] == {"key": "value"}


class TestDeterministicBehavior:
    """Tests for deterministic response generation."""

    def test_same_inputs_same_output(self):
        """Same inputs produce same output."""
        client = MockLLMClient(seed=123)

        result1 = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata={"id": 1},
        )
        result2 = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata={"id": 1},
        )

        assert result1["content"] == result2["content"]
        assert result1["metrics"]["score"] == result2["metrics"]["score"]

    def test_different_prompts_different_scores(self):
        """Different prompts produce different scores."""
        client = MockLLMClient()

        result1 = client.generate(
            system_prompt="System A",
            user_prompt="User A",
        )
        result2 = client.generate(
            system_prompt="System B",
            user_prompt="User B",
        )

        # Scores should be different (very unlikely to be equal)
        assert result1["metrics"]["score"] != result2["metrics"]["score"]

    def test_different_seeds_different_scores(self):
        """Different seeds produce different scores for same input."""
        client1 = MockLLMClient(seed=1)
        client2 = MockLLMClient(seed=2)

        result1 = client1.generate(
            system_prompt="Same system",
            user_prompt="Same user",
        )
        result2 = client2.generate(
            system_prompt="Same system",
            user_prompt="Same user",
        )

        assert result1["metrics"]["score"] != result2["metrics"]["score"]

    def test_metadata_affects_score(self):
        """Different metadata produces different scores."""
        client = MockLLMClient()

        result1 = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata={"variant": "A"},
        )
        result2 = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata={"variant": "B"},
        )

        assert result1["metrics"]["score"] != result2["metrics"]["score"]


class TestScoreDistribution:
    """Tests for score distribution range."""

    def test_score_in_expected_range(self):
        """Score should be between 0.4 and 0.9."""
        client = MockLLMClient()

        # Test multiple prompts to verify range
        for i in range(50):
            result = client.generate(
                system_prompt=f"System prompt {i}",
                user_prompt=f"User prompt {i}",
            )
            score = result["metrics"]["score"]
            assert 0.4 <= score <= 0.9, f"Score {score} outside expected range"

    def test_scores_have_variation(self):
        """Scores should vary across different inputs."""
        client = MockLLMClient()
        scores = []

        for i in range(20):
            result = client.generate(
                system_prompt=f"System {i}",
                user_prompt=f"User {i}",
            )
            scores.append(result["metrics"]["score"])

        # Check there's actual variation
        unique_scores = set(scores)
        assert len(unique_scores) > 10, "Expected more variation in scores"


class TestSeedReproducibility:
    """Tests for seed-based reproducibility."""

    def test_seed_ensures_reproducibility(self):
        """Same seed produces reproducible results across instances."""
        client1 = MockLLMClient(seed=42)
        client2 = MockLLMClient(seed=42)

        prompts = [
            ("System A", "User A"),
            ("System B", "User B"),
            ("System C", "User C"),
        ]

        for system, user in prompts:
            result1 = client1.generate(system_prompt=system, user_prompt=user)
            result2 = client2.generate(system_prompt=system, user_prompt=user)

            assert result1["metrics"]["score"] == result2["metrics"]["score"]
            assert result1["content"] == result2["content"]


class TestMockLLMClientSchemas:
    """Tests for MockLLMClient schema declarations."""

    def test_has_input_schema(self):
        """MockLLMClient should have input_schema class attribute."""
        assert hasattr(MockLLMClient, "input_schema")
        assert isinstance(MockLLMClient.input_schema, dict)

    def test_has_output_schema(self):
        """MockLLMClient should have output_schema class attribute."""
        assert hasattr(MockLLMClient, "output_schema")
        assert isinstance(MockLLMClient.output_schema, dict)

    def test_input_schema_requires_prompts(self):
        """Input schema requires system_prompt and user_prompt."""
        schema = MockLLMClient.input_schema
        assert schema["type"] == "object"
        assert "system_prompt" in schema["required"]
        assert "user_prompt" in schema["required"]

    def test_output_schema_requires_content(self):
        """Output schema requires content field."""
        schema = MockLLMClient.output_schema
        assert schema["type"] == "object"
        assert "content" in schema["required"]

    def test_output_schema_includes_metrics(self):
        """Output schema includes metrics definition."""
        schema = MockLLMClient.output_schema
        assert "metrics" in schema["properties"]


class TestMetadataHandling:
    """Tests for metadata parameter handling."""

    def test_none_metadata_handled(self):
        """None metadata is handled gracefully."""
        client = MockLLMClient()
        result = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata=None,
        )

        assert result["raw"]["metadata"] == {}

    def test_empty_metadata_handled(self):
        """Empty metadata dict is handled gracefully."""
        client = MockLLMClient()
        result = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata={},
        )

        assert result["raw"]["metadata"] == {}

    def test_complex_metadata_preserved(self):
        """Complex metadata is preserved in raw output."""
        client = MockLLMClient()
        metadata = {
            "row_id": 123,
            "experiment": "test",
            "nested": {"key": "value"},
        }
        result = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata=metadata,
        )

        assert result["raw"]["metadata"] == metadata
