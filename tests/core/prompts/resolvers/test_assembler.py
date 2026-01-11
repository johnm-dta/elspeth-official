"""Tests for pack assembler."""

import pytest

from elspeth.core.prompts.resolvers.assembler import DefaultPackAssembler
from elspeth.core.prompts.resolvers.protocols import FetchedFile


class TestDefaultPackAssembler:
    """Tests for DefaultPackAssembler."""

    @pytest.fixture
    def assembler(self):
        return DefaultPackAssembler()

    def test_assemble_config_yaml_only(self, assembler):
        """Test assembly with just config.yaml."""
        files = [
            FetchedFile(
                path="pack/config.yaml",
                content="""
prompts:
  system: "You are a helpful assistant."
  user: "Help me with: {{ task }}"
criteria:
  - name: accuracy
    weight: 1.0
""",
            )
        ]

        result = assembler.assemble(files)

        assert result["prompts"]["system"] == "You are a helpful assistant."
        assert result["prompts"]["user"] == "Help me with: {{ task }}"
        assert result["criteria"] == [{"name": "accuracy", "weight": 1.0}]

    def test_assemble_md_overrides_config(self, assembler):
        """Test that .md files override config.yaml prompts."""
        files = [
            FetchedFile(
                path="pack/config.yaml",
                content="""
prompts:
  system: "Config system prompt"
  user: "Config user prompt"
""",
            ),
            FetchedFile(
                path="pack/system_prompt.md",
                content="# MD System Prompt\nThis overrides config.yaml",
            ),
            FetchedFile(
                path="pack/user_prompt.md",
                content="# MD User Prompt\n{{ text }}",
            ),
        ]

        result = assembler.assemble(files)

        assert result["prompts"]["system"] == "# MD System Prompt\nThis overrides config.yaml"
        assert result["prompts"]["user"] == "# MD User Prompt\n{{ text }}"

    def test_assemble_md_without_config(self, assembler):
        """Test assembly with only .md files."""
        files = [
            FetchedFile(
                path="system_prompt.md",
                content="System prompt content",
            ),
            FetchedFile(
                path="user_prompt.md",
                content="User prompt content",
            ),
        ]

        result = assembler.assemble(files)

        assert result["prompts"]["system"] == "System prompt content"
        assert result["prompts"]["user"] == "User prompt content"

    def test_assemble_criteria_yaml(self, assembler):
        """Test loading criteria from dedicated file."""
        files = [
            FetchedFile(
                path="config.yaml",
                content="prompts:\n  system: test",
            ),
            FetchedFile(
                path="criteria.yaml",
                content="""
- name: accuracy
  weight: 1.0
- name: clarity
  weight: 0.5
""",
            ),
        ]

        result = assembler.assemble(files)

        assert len(result["criteria"]) == 2
        assert result["criteria"][0]["name"] == "accuracy"
        assert result["criteria"][1]["name"] == "clarity"

    def test_assemble_criteria_single_item(self, assembler):
        """Test criteria file with single item (not a list)."""
        files = [
            FetchedFile(
                path="criteria.yml",
                content="""
name: quality
weight: 1.0
""",
            ),
        ]

        result = assembler.assemble(files)

        assert len(result["criteria"]) == 1
        assert result["criteria"][0]["name"] == "quality"

    def test_assemble_defaults_yaml(self, assembler):
        """Test loading prompt_defaults from dedicated file."""
        files = [
            FetchedFile(
                path="config.yaml",
                content="prompts:\n  system: test",
            ),
            FetchedFile(
                path="defaults.yaml",
                content="""
role: analyst
company: ACME Corp
""",
            ),
        ]

        result = assembler.assemble(files)

        assert result["prompt_defaults"]["role"] == "analyst"
        assert result["prompt_defaults"]["company"] == "ACME Corp"

    def test_assemble_preserves_metadata(self, assembler):
        """Test that metadata is preserved in result."""
        files = [
            FetchedFile(
                path="system_prompt.md",
                content="test",
            ),
        ]
        metadata = {"provider": "github", "organization": "myorg"}

        result = assembler.assemble(files, metadata=metadata)

        assert result["_remote_pack_metadata"] == metadata

    def test_assemble_empty_files_returns_empty_dict(self, assembler):
        """Test assembly with no files."""
        result = assembler.assemble([])
        assert result == {}

    def test_assemble_prefers_yaml_over_yml(self, assembler):
        """Test that .yaml files are preferred over .yml."""
        files = [
            FetchedFile(
                path="config.yaml",
                content="prompts:\n  system: from yaml",
            ),
            FetchedFile(
                path="config.yml",
                content="prompts:\n  system: from yml",
            ),
        ]

        result = assembler.assemble(files)

        # Should use config.yaml (first match)
        assert result["prompts"]["system"] == "from yaml"

    def test_assemble_handles_nested_paths(self, assembler):
        """Test that files with nested paths are handled correctly."""
        files = [
            FetchedFile(
                path="deep/nested/path/config.yaml",
                content="prompts:\n  system: nested config",
            ),
            FetchedFile(
                path="another/path/system_prompt.md",
                content="nested system prompt",
            ),
        ]

        result = assembler.assemble(files)

        assert result["prompts"]["system"] == "nested system prompt"

    def test_assemble_config_with_plugins(self, assembler):
        """Test that plugin definitions are preserved."""
        files = [
            FetchedFile(
                path="config.yaml",
                content="""
prompts:
  system: test
  user: test
row_plugins:
  - plugin: score_extractor
    options:
      key: rating
aggregator_plugins:
  - plugin: field_collector
    options:
      output_key: collected
""",
            ),
        ]

        result = assembler.assemble(files)

        assert len(result["row_plugins"]) == 1
        assert result["row_plugins"][0]["plugin"] == "score_extractor"
        assert len(result["aggregator_plugins"]) == 1

    def test_assemble_empty_yaml_file(self, assembler):
        """Test handling of empty YAML files."""
        files = [
            FetchedFile(
                path="config.yaml",
                content="",
            ),
        ]

        result = assembler.assemble(files)

        assert result == {}

    def test_assemble_preserves_partial_prompts_from_config(self, assembler):
        """Test that partial prompts in config.yaml are preserved."""
        files = [
            FetchedFile(
                path="config.yaml",
                content="""
prompts:
  system: "Config system only"
""",
            ),
            FetchedFile(
                path="user_prompt.md",
                content="MD user prompt",
            ),
        ]

        result = assembler.assemble(files)

        # System from config, user from MD file
        assert result["prompts"]["system"] == "Config system only"
        assert result["prompts"]["user"] == "MD user prompt"
