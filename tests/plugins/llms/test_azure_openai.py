"""Tests for Azure OpenAI client."""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch

import pytest

from elspeth.plugins.llms.azure_openai import AzureOpenAIClient


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response content"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def valid_config():
    """Create a valid configuration."""
    return {
        "api_key": "test-api-key",
        "api_version": "2024-02-01",
        "azure_endpoint": "https://test.openai.azure.com",
        "deployment": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
    }


class TestAzureOpenAIClientInit:
    """Tests for AzureOpenAIClient initialization."""

    def test_init_with_valid_config(self, valid_config, mock_openai_client):
        """Client initializes with valid configuration."""
        client = AzureOpenAIClient(
            config=valid_config,
            client=mock_openai_client,
        )

        assert client.deployment == "gpt-4"
        assert client.temperature == 0.7
        assert client.max_tokens == 1000

    def test_init_with_explicit_deployment(self, valid_config, mock_openai_client):
        """Explicit deployment parameter overrides config."""
        client = AzureOpenAIClient(
            deployment="gpt-35-turbo",
            config=valid_config,
            client=mock_openai_client,
        )

        assert client.deployment == "gpt-35-turbo"

    def test_init_deployment_from_env(self, mock_openai_client, monkeypatch):
        """Deployment can be resolved from environment variable."""
        monkeypatch.setenv("DMP_AZURE_OPENAI_DEPLOYMENT", "env-deployment")

        config = {
            "api_key": "test-key",
            "api_version": "2024-02-01",
            "azure_endpoint": "https://test.openai.azure.com",
        }

        client = AzureOpenAIClient(config=config, client=mock_openai_client)
        assert client.deployment == "env-deployment"

    def test_init_deployment_from_custom_env(self, mock_openai_client, monkeypatch):
        """Deployment can be resolved from custom env variable."""
        monkeypatch.setenv("MY_DEPLOYMENT", "custom-deployment")

        config = {
            "api_key": "test-key",
            "api_version": "2024-02-01",
            "azure_endpoint": "https://test.openai.azure.com",
            "deployment_env": "MY_DEPLOYMENT",
        }

        client = AzureOpenAIClient(config=config, client=mock_openai_client)
        assert client.deployment == "custom-deployment"

    def test_init_missing_deployment_raises(self, mock_openai_client):
        """Missing deployment raises ValueError."""
        config = {
            "api_key": "test-key",
            "api_version": "2024-02-01",
            "azure_endpoint": "https://test.openai.azure.com",
        }

        with pytest.raises(ValueError, match="missing deployment"):
            AzureOpenAIClient(config=config, client=mock_openai_client)

    def test_init_without_temperature(self, valid_config, mock_openai_client):
        """Client works without temperature setting."""
        del valid_config["temperature"]

        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)
        assert client.temperature is None

    def test_init_without_max_tokens(self, valid_config, mock_openai_client):
        """Client works without max_tokens setting."""
        del valid_config["max_tokens"]

        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)
        assert client.max_tokens is None


class TestConfigResolution:
    """Tests for configuration resolution."""

    def test_resolve_required_from_config(self, mock_openai_client):
        """Required value resolved from config."""
        config = {
            "api_key": "direct-api-key",
            "api_version": "2024-02-01",
            "azure_endpoint": "https://test.openai.azure.com",
            "deployment": "gpt-4",
        }

        client = AzureOpenAIClient(config=config, client=mock_openai_client)
        # If we got here, config resolution worked
        assert client.deployment == "gpt-4"

    def test_resolve_required_from_env(self, mock_openai_client, monkeypatch):
        """Required value resolved from environment via _env suffix."""
        monkeypatch.setenv("MY_API_KEY", "env-api-key")
        monkeypatch.setenv("MY_VERSION", "2024-03-01")
        monkeypatch.setenv("MY_ENDPOINT", "https://env.openai.azure.com")

        config = {
            "api_key_env": "MY_API_KEY",
            "api_version_env": "MY_VERSION",
            "azure_endpoint_env": "MY_ENDPOINT",
            "deployment": "gpt-4",
        }

        client = AzureOpenAIClient(config=config, client=mock_openai_client)
        assert client.deployment == "gpt-4"

    def test_resolve_required_missing_raises(self, mock_openai_client):
        """Missing required config raises ValueError."""
        config = {
            # Missing api_key
            "api_version": "2024-02-01",
            "azure_endpoint": "https://test.openai.azure.com",
            "deployment": "gpt-4",
        }

        with pytest.raises(ValueError, match=r"missing required.*api_key"):
            AzureOpenAIClient(config=config)


class TestGenerate:
    """Tests for the generate method."""

    def test_generate_returns_response(self, valid_config, mock_openai_client):
        """Generate returns dict with content and metadata."""
        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

        result = client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello!",
        )

        assert "content" in result
        assert result["content"] == "Test response content"
        assert "raw" in result
        assert "metadata" in result

    def test_generate_passes_messages(self, valid_config, mock_openai_client):
        """Generate passes correct messages to API."""
        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

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

    def test_generate_passes_deployment(self, valid_config, mock_openai_client):
        """Generate passes deployment as model."""
        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

        client.generate(
            system_prompt="System",
            user_prompt="User",
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4"

    def test_generate_passes_temperature(self, valid_config, mock_openai_client):
        """Generate passes temperature when configured."""
        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

        client.generate(
            system_prompt="System",
            user_prompt="User",
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.7

    def test_generate_passes_max_tokens(self, valid_config, mock_openai_client):
        """Generate passes max_tokens when configured."""
        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

        client.generate(
            system_prompt="System",
            user_prompt="User",
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["max_tokens"] == 1000

    def test_generate_omits_temperature_when_none(self, mock_openai_client):
        """Generate omits temperature when not configured."""
        config = {
            "api_key": "test-key",
            "api_version": "2024-02-01",
            "azure_endpoint": "https://test.openai.azure.com",
            "deployment": "gpt-4",
            # No temperature
        }

        client = AzureOpenAIClient(config=config, client=mock_openai_client)
        client.generate(system_prompt="System", user_prompt="User")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert "temperature" not in call_args[1]

    def test_generate_preserves_metadata(self, valid_config, mock_openai_client):
        """Generate preserves metadata in response."""
        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

        result = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata={"row_id": 123, "experiment": "test"},
        )

        assert result["metadata"] == {"row_id": 123, "experiment": "test"}

    def test_generate_handles_none_metadata(self, valid_config, mock_openai_client):
        """Generate handles None metadata."""
        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

        result = client.generate(
            system_prompt="System",
            user_prompt="User",
            metadata=None,
        )

        assert result["metadata"] == {}


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_generate_handles_api_error(self, valid_config, mock_openai_client):
        """Generate propagates API errors."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

        with pytest.raises(Exception, match="API Error"):
            client.generate(system_prompt="System", user_prompt="User")

    def test_generate_handles_empty_choices(self, valid_config, mock_openai_client):
        """Generate handles response with empty choices."""
        mock_response = MagicMock()
        mock_response.choices = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)
        result = client.generate(system_prompt="System", user_prompt="User")

        # Content should be None when choices are empty
        assert result["content"] is None


class TestClientProperty:
    """Tests for the client property."""

    def test_client_property_returns_internal_client(self, valid_config, mock_openai_client):
        """Client property returns the internal client."""
        client = AzureOpenAIClient(config=valid_config, client=mock_openai_client)

        assert client.client is mock_openai_client


class TestClientCreation:
    """Tests for automatic client creation."""

    @patch("openai.AzureOpenAI")
    def test_creates_azure_openai_client(self, mock_azure_openai_class, valid_config):
        """Creates AzureOpenAI client with correct parameters."""
        mock_instance = MagicMock()
        mock_azure_openai_class.return_value = mock_instance

        client = AzureOpenAIClient(config=valid_config)

        mock_azure_openai_class.assert_called_once_with(
            api_key="test-api-key",
            api_version="2024-02-01",
            azure_endpoint="https://test.openai.azure.com",
            http_client=ANY,
        )
        assert client.client is mock_instance
