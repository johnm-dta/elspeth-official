"""Tests for secrets variable substitution."""

from __future__ import annotations

import pytest

from elspeth.core.secrets import SecretsError, load_secrets, substitute_variables


class TestSubstituteVariables:
    """Tests for substitute_variables function."""

    def test_simple_substitution(self):
        """Test basic ${VAR} substitution."""
        config = {"api_key": "${MY_KEY}"}
        secrets = {"MY_KEY": "secret123"}
        result = substitute_variables(config, secrets)
        assert result == {"api_key": "secret123"}

    def test_nested_substitution(self):
        """Test substitution in nested dicts."""
        config = {
            "datasource": {
                "options": {
                    "connection_string": "${STORAGE_KEY}"
                }
            }
        }
        secrets = {"STORAGE_KEY": "DefaultEndpoints..."}
        result = substitute_variables(config, secrets)
        assert result["datasource"]["options"]["connection_string"] == "DefaultEndpoints..."

    def test_list_substitution(self):
        """Test substitution in list values."""
        config = {"keys": ["${KEY1}", "${KEY2}"]}
        secrets = {"KEY1": "a", "KEY2": "b"}
        result = substitute_variables(config, secrets)
        assert result == {"keys": ["a", "b"]}

    def test_missing_variable_raises(self):
        """Test that missing variables raise SecretsError."""
        config = {"api_key": "${MISSING_KEY}"}
        secrets = {}
        with pytest.raises(SecretsError) as exc_info:
            substitute_variables(config, secrets)
        assert "MISSING_KEY" in str(exc_info.value)

    def test_partial_string_substitution(self):
        """Test ${VAR} within larger string."""
        config = {"url": "https://${HOST}:${PORT}/api"}
        secrets = {"HOST": "example.com", "PORT": "8080"}
        result = substitute_variables(config, secrets)
        assert result == {"url": "https://example.com:8080/api"}

    def test_no_variables_unchanged(self):
        """Test config without variables passes through."""
        config = {"key": "value", "nested": {"inner": 123}}
        result = substitute_variables(config, {})
        assert result == config

    def test_escaped_dollar_sign(self):
        """Test $$ escapes to literal $."""
        config = {"price": "$$100"}
        result = substitute_variables(config, {})
        assert result == {"price": "$100"}


class TestLoadSecrets:
    """Tests for load_secrets function."""

    def test_load_from_file(self, tmp_path):
        """Test loading secrets from YAML file."""
        secrets_file = tmp_path / "secrets.yaml"
        secrets_file.write_text("API_KEY: secret123\nDB_PASS: dbpass", encoding="utf-8")

        secrets = load_secrets(secrets_file)
        assert secrets == {"API_KEY": "secret123", "DB_PASS": "dbpass"}

    def test_missing_file_raises(self, tmp_path):
        """Test that missing file raises SecretsError."""
        with pytest.raises(SecretsError) as exc_info:
            load_secrets(tmp_path / "nonexistent.yaml")
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_yaml_raises(self, tmp_path):
        """Test that invalid YAML raises SecretsError."""
        secrets_file = tmp_path / "secrets.yaml"
        secrets_file.write_text("invalid: yaml: content:", encoding="utf-8")

        with pytest.raises(SecretsError):
            load_secrets(secrets_file)
