"""Tests for config source path tracking."""


import pytest

from elspeth.config import load_settings


class TestConfigSourcePaths:
    """Tests for tracking source file paths."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create minimal valid config file."""
        config = tmp_path / "settings.yaml"
        config.write_text("""
default:
  datasource:
    plugin: local_csv
    options:
      path: data.csv
  sinks:
    - plugin: csv
      options:
        path: output.csv
""")
        return config

    def test_settings_has_source_paths(self, minimal_config):
        """Settings object exposes source file paths."""
        settings = load_settings(minimal_config)

        assert hasattr(settings, "source_paths")
        assert isinstance(settings.source_paths, list)

    def test_source_paths_includes_settings_file(self, minimal_config):
        """Source paths includes the main settings file."""
        settings = load_settings(minimal_config)

        assert minimal_config in settings.source_paths

    def test_source_paths_includes_prompt_files(self, tmp_path):
        """Source paths includes referenced prompt files."""
        prompts_dir = tmp_path / "prompts" / "test"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "system.md").write_text("System prompt")
        (prompts_dir / "user.md").write_text("User prompt")

        config = tmp_path / "settings.yaml"
        config.write_text(f"""
default:
  datasource:
    plugin: local_csv
    options:
      path: data.csv
  sinks:
    - plugin: csv
      options:
        path: output.csv
  row_plugins:
    - plugin: llm_query
      options:
        llm:
          plugin: mock
        queries:
          - name: test
            prompt_folder: {prompts_dir}
""")
        settings = load_settings(config)

        # Should include prompt files
        assert prompts_dir / "system.md" in settings.source_paths or \
               any("system.md" in str(p) for p in settings.source_paths)
