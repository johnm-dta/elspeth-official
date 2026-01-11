"""Tests for config fingerprint calculation."""



from elspeth.core.sda.fingerprint import compute_config_fingerprint


class TestComputeConfigFingerprint:
    """Tests for compute_config_fingerprint function."""

    def test_returns_sha256_hex_string(self, tmp_path):
        """Fingerprint is a 64-char hex string (SHA256)."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("llm:\n  model: gpt-4\n")

        result = compute_config_fingerprint([config_file])

        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_content_same_fingerprint(self, tmp_path):
        """Same file content produces same fingerprint."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("test: value\n")

        fp1 = compute_config_fingerprint([config_file])
        fp2 = compute_config_fingerprint([config_file])

        assert fp1 == fp2

    def test_different_content_different_fingerprint(self, tmp_path):
        """Different content produces different fingerprint."""
        file1 = tmp_path / "config1.yaml"
        file1.write_text("version: 1\n")

        file2 = tmp_path / "config2.yaml"
        file2.write_text("version: 2\n")

        fp1 = compute_config_fingerprint([file1])
        fp2 = compute_config_fingerprint([file2])

        assert fp1 != fp2

    def test_multiple_files_combined(self, tmp_path):
        """Multiple files are combined into single fingerprint."""
        file1 = tmp_path / "settings.yaml"
        file1.write_text("setting: a\n")

        file2 = tmp_path / "prompts" / "user.md"
        file2.parent.mkdir()
        file2.write_text("User prompt here\n")

        # Single fingerprint for both
        fp_combined = compute_config_fingerprint([file1, file2])

        # Different from individual
        fp1 = compute_config_fingerprint([file1])
        fp2 = compute_config_fingerprint([file2])

        assert fp_combined != fp1
        assert fp_combined != fp2

    def test_order_independent(self, tmp_path):
        """File order doesn't affect fingerprint (sorted internally)."""
        file1 = tmp_path / "a.yaml"
        file1.write_text("a\n")

        file2 = tmp_path / "b.yaml"
        file2.write_text("b\n")

        fp1 = compute_config_fingerprint([file1, file2])
        fp2 = compute_config_fingerprint([file2, file1])

        assert fp1 == fp2

    def test_missing_file_skipped(self, tmp_path):
        """Missing files are skipped without error."""
        existing = tmp_path / "exists.yaml"
        existing.write_text("data\n")

        missing = tmp_path / "missing.yaml"

        # Should not raise
        fp = compute_config_fingerprint([existing, missing])
        assert len(fp) == 64

    def test_empty_paths_returns_empty_hash(self):
        """Empty path list returns hash of empty content."""
        fp = compute_config_fingerprint([])
        assert len(fp) == 64  # SHA256 of empty string
