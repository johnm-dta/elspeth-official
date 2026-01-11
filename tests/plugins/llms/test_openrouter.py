"""Tests for OpenRouter client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from elspeth.plugins.llms.openrouter import OpenRouterClient


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "OpenRouter response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def valid_config():
    """Create a valid configuration."""
    return {
        "api_key": "sk-or-v1-test-key",
        "model": "openai/gpt-4o",
        "temperature": 0.5,
        "max_tokens": 2000,
    }


class TestOpenRouterClientInit:
    """Tests for OpenRouterClient initialization."""

    def test_init_with_valid_config(self, valid_config, mock_openai_client):
        """Client initializes with valid configuration."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        assert client.model == "openai/gpt-4o"
        assert client.temperature == 0.5
        assert client.max_tokens == 2000

    def test_init_with_explicit_model(self, valid_config, mock_openai_client):
        """Explicit model parameter overrides config."""
        client = OpenRouterClient(
            model="anthropic/claude-3-opus",
            config=valid_config,
            client=mock_openai_client,
        )

        assert client.model == "anthropic/claude-3-opus"

    def test_init_model_from_env(self, mock_openai_client, monkeypatch):
        """Model can be resolved from environment variable."""
        monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3-70b")

        config = {"api_key": "test-key"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        assert client.model == "meta-llama/llama-3-70b"

    def test_init_model_from_custom_env(self, mock_openai_client, monkeypatch):
        """Model can be resolved from custom env variable."""
        monkeypatch.setenv("MY_MODEL", "google/gemini-pro")

        config = {"api_key": "test-key", "model_env": "MY_MODEL"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        assert client.model == "google/gemini-pro"

    def test_init_defaults_to_gpt4o_mini(self, mock_openai_client, monkeypatch):
        """Defaults to gpt-4o-mini when no model configured."""
        # Ensure env var doesn't pollute test
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

        config = {"api_key": "test-key"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        assert client.model == "openai/gpt-4o-mini"

    def test_init_without_temperature(self, valid_config, mock_openai_client):
        """Client works without temperature setting."""
        del valid_config["temperature"]
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        assert client.temperature is None

    def test_init_without_max_tokens(self, valid_config, mock_openai_client):
        """Client works without max_tokens setting."""
        del valid_config["max_tokens"]
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        assert client.max_tokens is None


class TestConfigResolution:
    """Tests for configuration resolution."""

    def test_resolve_api_key_from_config(self, mock_openai_client):
        """API key resolved from config."""
        config = {"api_key": "direct-key", "model": "test/model"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        assert client.model == "test/model"

    def test_resolve_api_key_from_env(self, mock_openai_client, monkeypatch):
        """API key resolved from environment."""
        monkeypatch.setenv("MY_API_KEY", "env-key")

        config = {"api_key_env": "MY_API_KEY", "model": "test/model"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        assert client.model == "test/model"

    def test_resolve_missing_api_key_raises(self, mock_openai_client):
        """Missing API key raises ValueError."""
        config = {"model": "test/model"}

        with pytest.raises(ValueError, match=r"missing required.*api_key"):
            OpenRouterClient(config=config)


class TestGenerate:
    """Tests for the generate method."""

    def test_generate_returns_response(self, valid_config, mock_openai_client):
        """Generate returns dict with content and metadata."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        result = client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello!",
        )

        assert "content" in result
        assert result["content"] == "OpenRouter response"
        assert "raw" in result
        assert "metadata" in result

    def test_generate_passes_messages(self, valid_config, mock_openai_client):
        """Generate passes correct messages to API."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        client.generate(
            system_prompt="System message",
            user_prompt="User message",
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System message"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User message"

    def test_generate_passes_model(self, valid_config, mock_openai_client):
        """Generate passes model to API."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "openai/gpt-4o"

    def test_generate_passes_temperature(self, valid_config, mock_openai_client):
        """Generate passes temperature when configured."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.5

    def test_generate_passes_max_tokens(self, valid_config, mock_openai_client):
        """Generate passes max_tokens when configured."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["max_tokens"] == 2000

    def test_generate_omits_temperature_when_none(self, mock_openai_client):
        """Generate omits temperature when not configured."""
        config = {"api_key": "test-key", "model": "test/model"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert "temperature" not in call_args[1]

    def test_generate_preserves_metadata(self, valid_config, mock_openai_client):
        """Generate preserves metadata in response."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        result = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata={"row_id": 456, "variant": "A"},
        )

        assert result["metadata"] == {"row_id": 456, "variant": "A"}

    def test_generate_handles_none_metadata(self, valid_config, mock_openai_client):
        """Generate handles None metadata."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        result = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata=None,
        )

        assert result["metadata"] == {}


class TestOpenRouterHeaders:
    """Tests for OpenRouter-specific headers."""

    def test_generate_adds_site_url_header(self, valid_config, mock_openai_client):
        """Generate adds HTTP-Referer header when site_url configured."""
        valid_config["site_url"] = "https://myapp.example.com"
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        extra_headers = call_args[1].get("extra_headers", {})
        assert extra_headers["HTTP-Referer"] == "https://myapp.example.com"

    def test_generate_adds_app_name_header(self, valid_config, mock_openai_client):
        """Generate adds X-Title header when app_name configured."""
        valid_config["app_name"] = "My Application"
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        extra_headers = call_args[1].get("extra_headers", {})
        assert extra_headers["X-Title"] == "My Application"

    def test_generate_adds_both_headers(self, valid_config, mock_openai_client):
        """Generate adds both headers when configured."""
        valid_config["site_url"] = "https://example.com"
        valid_config["app_name"] = "Test App"
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        extra_headers = call_args[1].get("extra_headers", {})
        assert extra_headers["HTTP-Referer"] == "https://example.com"
        assert extra_headers["X-Title"] == "Test App"

    def test_generate_no_headers_when_not_configured(self, valid_config, mock_openai_client):
        """Generate omits extra_headers when not configured."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert "extra_headers" not in call_args[1]


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_generate_handles_api_error(self, valid_config, mock_openai_client):
        """Generate propagates API errors."""
        mock_openai_client.chat.completions.create.side_effect = Exception("Rate limited")

        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        with pytest.raises(Exception, match="Rate limited"):
            client.generate(system_prompt="System", user_prompt="User")

    def test_generate_handles_empty_choices(self, valid_config, mock_openai_client):
        """Generate handles response with empty choices."""
        mock_response = MagicMock()
        mock_response.choices = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenRouterClient(config=valid_config, client=mock_openai_client)
        result = client.generate(system_prompt="System", user_prompt="User")

        assert result["content"] is None


class TestClientProperty:
    """Tests for the client property."""

    def test_client_property_returns_internal_client(self, valid_config, mock_openai_client):
        """Client property returns the internal client."""
        client = OpenRouterClient(config=valid_config, client=mock_openai_client)

        assert client.client is mock_openai_client


class TestClientCreation:
    """Tests for automatic client creation."""

    @patch("openai.OpenAI")
    def test_creates_openai_client_with_openrouter_base(self, mock_openai_class, valid_config):
        """Creates OpenAI client with OpenRouter base URL."""
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance

        client = OpenRouterClient(config=valid_config)

        mock_openai_class.assert_called_once_with(
            api_key="sk-or-v1-test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert client.client is mock_instance


class TestDefaultModelWarning:
    """Tests related to silent fallback to default model."""

    def test_default_model_is_gpt4o_mini(self, mock_openai_client, monkeypatch):
        """Verify default model is gpt-4o-mini."""
        # Ensure env var doesn't pollute test
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

        config = {"api_key": "test-key"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        assert client.model == "openai/gpt-4o-mini"

    def test_model_priority_explicit_over_config(self, mock_openai_client):
        """Explicit model takes priority over config."""
        config = {"api_key": "test-key", "model": "config-model"}
        client = OpenRouterClient(
            model="explicit-model",
            config=config,
            client=mock_openai_client,
        )

        assert client.model == "explicit-model"

    def test_model_priority_config_over_env(self, mock_openai_client, monkeypatch):
        """Config model takes priority over environment."""
        monkeypatch.setenv("OPENROUTER_MODEL", "env-model")

        config = {"api_key": "test-key", "model": "config-model"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        assert client.model == "config-model"

    def test_model_priority_custom_env_over_default_env(self, mock_openai_client, monkeypatch):
        """Custom env variable takes priority over OPENROUTER_MODEL."""
        monkeypatch.setenv("OPENROUTER_MODEL", "default-env-model")
        monkeypatch.setenv("MY_MODEL", "custom-env-model")

        config = {"api_key": "test-key", "model_env": "MY_MODEL"}
        client = OpenRouterClient(config=config, client=mock_openai_client)

        assert client.model == "custom-env-model"
