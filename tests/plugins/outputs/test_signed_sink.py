"""Tests for SignedArtifactSink."""

import json

import pytest

from elspeth.core.security.signing import verify_signature
from elspeth.plugins.outputs.signed import SignedArtifactSink


class TestSignedArtifactSinkBasic:
    """Basic functionality tests."""

    def test_signed_sink_signs_payload_and_collects_artifact(self, tmp_path):
        sink = SignedArtifactSink(
            key="secret",
            algorithm="hmac-sha256",
            base_path=str(tmp_path),
            timestamped=False,
            bundle_name="bundle",
        )

        payload = {"results": [{"row": {"a": 1}}]}
        metadata = {"experiment": "cycle-a"}

        sink.write(payload, metadata=metadata)

        bundle_dir = tmp_path / "bundle"
        signature_path = bundle_dir / "signature.json"
        manifest_path = bundle_dir / "manifest.json"
        results_path = bundle_dir / "results.json"

        assert results_path.exists()
        assert signature_path.exists()
        assert manifest_path.exists()

        signature_content = signature_path.read_text(encoding="utf-8")
        assert "hmac-sha256" in signature_content

    def test_timestamped_bundle_name(self, tmp_path):
        """Timestamped bundles include timestamp in directory name."""
        sink = SignedArtifactSink(
            key="secret",
            base_path=tmp_path,
            timestamped=True,
            bundle_name="test",
        )

        sink.write({"results": []}, metadata={})

        # Should have created a timestamped directory
        dirs = list(tmp_path.iterdir())
        assert len(dirs) == 1
        assert dirs[0].name.startswith("test_")
        assert "T" in dirs[0].name  # ISO timestamp format


class TestSignedArtifactSinkSignatureVerification:
    """Tests for signature correctness and verification."""

    def test_signature_is_verifiable(self, tmp_path):
        """Generated signature can be verified with the same key."""
        key = "my-secret-signing-key"
        sink = SignedArtifactSink(
            key=key,
            algorithm="hmac-sha256",
            base_path=tmp_path,
            timestamped=False,
            bundle_name="verify_test",
        )

        payload = {"results": [{"id": 1, "score": 0.95}]}
        sink.write(payload, metadata={"test": True})

        bundle_dir = tmp_path / "verify_test"
        results_path = bundle_dir / "results.json"
        signature_path = bundle_dir / "signature.json"

        # Read signature
        signature_data = json.loads(signature_path.read_text())
        results_bytes = results_path.read_bytes()

        # Verify signature
        is_valid = verify_signature(
            results_bytes,
            signature_data["signature"],
            key,
            algorithm="hmac-sha256",
        )
        assert is_valid, "Signature should be valid with correct key"

    def test_signature_fails_with_wrong_key(self, tmp_path):
        """Signature verification fails with incorrect key."""
        sink = SignedArtifactSink(
            key="correct-key",
            algorithm="hmac-sha256",
            base_path=tmp_path,
            timestamped=False,
            bundle_name="wrong_key_test",
        )

        sink.write({"results": []}, metadata={})

        bundle_dir = tmp_path / "wrong_key_test"
        results_path = bundle_dir / "results.json"
        signature_path = bundle_dir / "signature.json"

        signature_data = json.loads(signature_path.read_text())
        results_bytes = results_path.read_bytes()

        # Verify with WRONG key
        is_valid = verify_signature(
            results_bytes,
            signature_data["signature"],
            "wrong-key",
            algorithm="hmac-sha256",
        )
        assert not is_valid, "Signature should fail with wrong key"

    def test_tampered_content_fails_verification(self, tmp_path):
        """Modifying content invalidates signature."""
        key = "tamper-test-key"
        sink = SignedArtifactSink(
            key=key,
            algorithm="hmac-sha256",
            base_path=tmp_path,
            timestamped=False,
            bundle_name="tamper_test",
        )

        sink.write({"results": [{"data": "original"}]}, metadata={})

        bundle_dir = tmp_path / "tamper_test"
        signature_path = bundle_dir / "signature.json"

        signature_data = json.loads(signature_path.read_text())

        # Tamper with content
        tampered_bytes = b'{"results": [{"data": "TAMPERED"}]}'

        is_valid = verify_signature(
            tampered_bytes,
            signature_data["signature"],
            key,
            algorithm="hmac-sha256",
        )
        assert not is_valid, "Tampered content should fail verification"

    def test_sha512_algorithm(self, tmp_path):
        """Signing works with SHA-512 algorithm."""
        key = "sha512-key"
        sink = SignedArtifactSink(
            key=key,
            algorithm="hmac-sha512",
            base_path=tmp_path,
            timestamped=False,
            bundle_name="sha512_test",
        )

        sink.write({"results": []}, metadata={})

        bundle_dir = tmp_path / "sha512_test"
        results_path = bundle_dir / "results.json"
        signature_path = bundle_dir / "signature.json"

        signature_data = json.loads(signature_path.read_text())
        assert signature_data["algorithm"] == "hmac-sha512"

        # Verify with SHA-512
        results_bytes = results_path.read_bytes()
        is_valid = verify_signature(
            results_bytes,
            signature_data["signature"],
            key,
            algorithm="hmac-sha512",
        )
        assert is_valid, "SHA-512 signature should be valid"


class TestSignedArtifactSinkKeyResolution:
    """Tests for key resolution from config and environment."""

    def test_key_from_config(self, tmp_path):
        """Key provided directly in config works."""
        sink = SignedArtifactSink(
            key="direct-config-key",
            base_path=tmp_path,
            timestamped=False,
            bundle_name="config_key",
        )

        # Should not raise
        sink.write({"results": []}, metadata={})
        assert (tmp_path / "config_key" / "signature.json").exists()

    def test_key_from_default_env_var(self, tmp_path, monkeypatch):
        """Key resolved from default DMP_SIGNING_KEY env var."""
        monkeypatch.setenv("DMP_SIGNING_KEY", "env-key-value")

        sink = SignedArtifactSink(
            base_path=tmp_path,
            timestamped=False,
            bundle_name="env_key",
            key_env="DMP_SIGNING_KEY",  # Default
        )

        sink.write({"results": []}, metadata={})

        # Verify signature was created with env key
        signature_path = tmp_path / "env_key" / "signature.json"
        assert signature_path.exists()

        # Verify it's valid with the env key
        results_bytes = (tmp_path / "env_key" / "results.json").read_bytes()
        signature_data = json.loads(signature_path.read_text())
        is_valid = verify_signature(
            results_bytes,
            signature_data["signature"],
            "env-key-value",
            algorithm="hmac-sha256",
        )
        assert is_valid

    def test_key_from_custom_env_var(self, tmp_path, monkeypatch):
        """Key resolved from custom env var name."""
        monkeypatch.setenv("MY_CUSTOM_SIGNING_KEY", "custom-env-key")

        sink = SignedArtifactSink(
            base_path=tmp_path,
            timestamped=False,
            bundle_name="custom_env",
            key_env="MY_CUSTOM_SIGNING_KEY",
        )

        sink.write({"results": []}, metadata={})
        assert (tmp_path / "custom_env" / "signature.json").exists()

    def test_missing_key_raises_error(self, tmp_path, monkeypatch):
        """Missing key raises clear error."""
        # Ensure env var is not set
        monkeypatch.delenv("DMP_SIGNING_KEY", raising=False)

        sink = SignedArtifactSink(
            base_path=tmp_path,
            timestamped=False,
            bundle_name="no_key",
            key=None,
            key_env="DMP_SIGNING_KEY",
        )

        with pytest.raises(ValueError, match="Signing key not provided"):
            sink.write({"results": []}, metadata={})

    def test_empty_env_var_raises_error(self, tmp_path, monkeypatch):
        """Empty env var treated as missing."""
        monkeypatch.setenv("DMP_SIGNING_KEY", "")

        sink = SignedArtifactSink(
            base_path=tmp_path,
            timestamped=False,
            bundle_name="empty_key",
            key=None,
            key_env="DMP_SIGNING_KEY",
        )

        with pytest.raises(ValueError, match="Signing key not provided"):
            sink.write({"results": []}, metadata={})


class TestSignedArtifactSinkValidation:
    """Tests for configuration validation."""

    def test_invalid_on_error_raises(self, tmp_path):
        """Invalid on_error value raises ValueError."""
        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            SignedArtifactSink(
                key="test",
                base_path=tmp_path,
                on_error="ignore",  # Invalid
            )

    def test_valid_on_error_abort(self, tmp_path):
        """on_error='abort' is accepted."""
        sink = SignedArtifactSink(
            key="test",
            base_path=tmp_path,
            on_error="abort",
        )
        assert sink.on_error == "abort"


class TestSignedArtifactSinkManifest:
    """Tests for manifest content."""

    def test_manifest_includes_aggregates(self, tmp_path):
        """Manifest includes aggregates from results."""
        sink = SignedArtifactSink(
            key="test",
            base_path=tmp_path,
            timestamped=False,
            bundle_name="manifest_test",
        )

        results = {
            "results": [{"id": 1}],
            "aggregates": {"mean_score": 0.85, "count": 10},
            "cost_summary": {"total_tokens": 1000, "total_cost": 0.05},
        }
        sink.write(results, metadata={"experiment": "test"})

        manifest_path = tmp_path / "manifest_test" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        assert manifest["aggregates"] == {"mean_score": 0.85, "count": 10}
        assert manifest["cost_summary"] == {"total_tokens": 1000, "total_cost": 0.05}
        assert manifest["rows"] == 1
        assert manifest["metadata"] == {"experiment": "test"}
        assert "generated_at" in manifest
        assert "digest" in manifest

    def test_manifest_digest_is_sha256(self, tmp_path):
        """Manifest digest is a valid SHA-256 hash."""
        sink = SignedArtifactSink(
            key="test",
            base_path=tmp_path,
            timestamped=False,
            bundle_name="digest_test",
        )

        sink.write({"results": [{"x": 1}]}, metadata={})

        manifest_path = tmp_path / "digest_test" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # SHA-256 produces 64 hex characters
        assert len(manifest["digest"]) == 64
        assert all(c in "0123456789abcdef" for c in manifest["digest"])
