"""Tests for GitHub and Azure DevOps repository sinks."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
import requests

from elspeth.plugins.outputs.repository import (
    AzureDevOpsRepoSink,
    GitHubRepoSink,
    PreparedFile,
    _default_context,
    _json_bytes,
    _RepoSinkBase,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_default_context_adds_timestamp_fields(self):
        """Default context includes timestamp-based fields."""
        timestamp = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
        metadata = {"experiment": "test-exp", "name": "my-run"}

        context = _default_context(metadata, timestamp)

        assert context["timestamp"] == "20250115T103000Z"
        assert context["date"] == "2025-01-15"
        assert context["time"] == "103000"
        assert context["experiment"] == "test-exp"

    def test_default_context_uses_name_as_experiment_fallback(self):
        """Falls back to 'name' when 'experiment' not in metadata."""
        timestamp = datetime(2025, 1, 1, tzinfo=UTC)
        metadata = {"name": "my-name"}

        context = _default_context(metadata, timestamp)
        assert context["experiment"] == "my-name"

    def test_default_context_uses_default_experiment(self):
        """Falls back to 'experiment' when neither key exists."""
        timestamp = datetime(2025, 1, 1, tzinfo=UTC)
        metadata = {}

        context = _default_context(metadata, timestamp)
        assert context["experiment"] == "experiment"

    def test_json_bytes_serializes_payload(self):
        """Payload is serialized to indented JSON bytes."""
        payload = {"key": "value", "number": 42}
        result = _json_bytes(payload)

        assert isinstance(result, bytes)
        parsed = json.loads(result)
        assert parsed == payload


class TestRepoSinkBase:
    """Tests for base repository sink behavior."""

    def test_init_with_defaults(self):
        """Base sink initializes with defaults."""

        class ConcreteSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                pass

        sink = ConcreteSink()
        assert sink.path_template == "experiments/{experiment}/{timestamp}"
        assert sink.commit_message_template == "Add experiment results for {experiment}"
        assert sink.include_manifest is True
        assert sink.dry_run is True
        assert sink.session is not None

    def test_init_invalid_on_error(self):
        """Rejects invalid on_error value."""

        class ConcreteSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                pass

        with pytest.raises(ValueError, match="on_error must be 'abort'"):
            ConcreteSink(on_error="ignore")

    def test_resolve_prefix_formats_template(self):
        """Path template is correctly resolved."""

        class ConcreteSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                pass

        sink = ConcreteSink(path_template="output/{experiment}/{date}")
        context = {"experiment": "my-exp", "date": "2025-01-15"}

        result = sink._resolve_prefix(context)
        assert result == "output/my-exp/2025-01-15"

    def test_resolve_prefix_missing_placeholder(self):
        """Missing placeholder in template raises error."""

        class ConcreteSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                pass

        sink = ConcreteSink(path_template="output/{missing_key}")

        with pytest.raises(ValueError, match="Missing placeholder 'missing_key'"):
            sink._resolve_prefix({})

    def test_prepare_files_creates_results_and_manifest(self):
        """Files are prepared with results and manifest."""

        class ConcreteSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                pass

        sink = ConcreteSink()
        results = {"results": [{"id": 1}, {"id": 2}], "aggregates": {"count": 2}}
        metadata = {"experiment": "test"}
        timestamp = datetime(2025, 1, 1, tzinfo=UTC)

        files = sink._prepare_files(results, metadata, "output/exp", timestamp)

        assert len(files) == 2
        assert files[0].path == "output/exp/results.json"
        assert files[1].path == "output/exp/manifest.json"

        # Verify results content
        results_parsed = json.loads(files[0].content)
        assert results_parsed["results"] == [{"id": 1}, {"id": 2}]

        # Verify manifest content
        manifest_parsed = json.loads(files[1].content)
        assert manifest_parsed["rows"] == 2
        assert "generated_at" in manifest_parsed

    def test_prepare_files_skips_manifest_when_disabled(self):
        """Manifest is not included when disabled."""

        class ConcreteSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                pass

        sink = ConcreteSink(include_manifest=False)
        results = {"results": []}
        timestamp = datetime(2025, 1, 1, tzinfo=UTC)

        files = sink._prepare_files(results, {}, "output", timestamp)

        assert len(files) == 1
        assert files[0].path == "output/results.json"

    def test_dry_run_records_payload(self):
        """Dry run mode records payload without uploading."""

        class ConcreteSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                raise AssertionError("Should not be called in dry_run mode")

        sink = ConcreteSink(dry_run=True)
        results = {"results": [{"id": 1}]}

        sink.write(results, metadata={"experiment": "test"})

        assert len(sink._last_payloads) == 1
        payload = sink._last_payloads[0]
        assert payload["dry_run"] is True
        assert "commit_message" in payload
        assert len(payload["files"]) == 2  # results + manifest

    def test_live_mode_calls_upload(self):
        """Live mode calls _upload method."""
        upload_called = []

        class ConcreteSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                upload_called.append((files, commit_message))

        sink = ConcreteSink(dry_run=False)
        results = {"results": []}

        sink.write(results, metadata={"experiment": "test"})

        assert len(upload_called) == 1
        files, commit_msg = upload_called[0]
        assert len(files) >= 1
        assert "test" in commit_msg


class TestGitHubRepoSink:
    """Tests for GitHub repository sink."""

    @pytest.fixture
    def mock_session(self):
        """Create mock requests session."""
        session = MagicMock(spec=requests.Session)
        return session

    def test_init_with_required_params(self, mock_session):
        """GitHub sink initializes with required parameters."""
        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            branch="main",
            session=mock_session,
        )

        assert sink.owner == "myorg"
        assert sink.repo == "myrepo"
        assert sink.branch == "main"
        assert sink.base_url == "https://api.github.com"

    def test_init_with_custom_params(self, mock_session):
        """GitHub sink accepts custom parameters."""
        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            branch="feature",
            token_env="MY_GH_TOKEN",
            base_url="https://github.enterprise.com/api/v3",
            session=mock_session,
            dry_run=False,
        )

        assert sink.branch == "feature"
        assert sink.token_env == "MY_GH_TOKEN"
        assert sink.base_url == "https://github.enterprise.com/api/v3"

    def test_headers_includes_token(self, mock_session, monkeypatch):
        """Authorization header is set from environment."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token_123")

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
        )

        headers = sink._headers()

        assert headers["Authorization"] == "Bearer ghp_test_token_123"
        assert headers["Accept"] == "application/vnd.github+json"

    def test_headers_without_token(self, mock_session, monkeypatch):
        """Headers work without token (for public repos)."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
        )

        headers = sink._headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github+json"

    def test_get_existing_sha_for_existing_file(self, mock_session):
        """Returns SHA for existing file."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sha": "abc123def456"}
        mock_session.request.return_value = mock_response

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
        )

        sha = sink._get_existing_sha("path/to/file.json")

        assert sha == "abc123def456"
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[0][0] == "GET"
        assert "contents/path/to/file.json" in call_args[0][1]

    def test_get_existing_sha_for_new_file(self, mock_session):
        """Returns None for non-existing file."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.request.return_value = mock_response

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
        )

        sha = sink._get_existing_sha("new/file.json")

        assert sha is None

    def test_upload_new_file(self, mock_session):
        """Upload creates new file without SHA."""
        # First request: check if file exists (404)
        check_response = MagicMock()
        check_response.status_code = 404

        # Second request: upload file (201)
        upload_response = MagicMock()
        upload_response.status_code = 201

        mock_session.request.side_effect = [
            check_response,
            upload_response,
            check_response,  # For manifest check
            upload_response,  # For manifest upload
        ]

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
            dry_run=False,
        )

        files = [PreparedFile(path="output/results.json", content=b'{"test": true}')]
        sink._upload(files, "Test commit", {}, {}, datetime.now(UTC))

        # Verify PUT request was made
        put_calls = [c for c in mock_session.request.call_args_list if c[0][0] == "PUT"]
        assert len(put_calls) == 1
        call_args = put_calls[0]
        assert "sha" not in call_args[1].get("json", {})

    def test_upload_existing_file_includes_sha(self, mock_session):
        """Upload includes SHA for existing file."""
        # First request: check if file exists (200 with SHA)
        check_response = MagicMock()
        check_response.status_code = 200
        check_response.json.return_value = {"sha": "existing_sha_123"}

        # Second request: upload file (200)
        upload_response = MagicMock()
        upload_response.status_code = 200

        mock_session.request.side_effect = [check_response, upload_response]

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
            dry_run=False,
        )

        files = [PreparedFile(path="output/results.json", content=b'{"test": true}')]
        sink._upload(files, "Update commit", {}, {}, datetime.now(UTC))

        # Verify PUT request includes SHA
        put_calls = [c for c in mock_session.request.call_args_list if c[0][0] == "PUT"]
        assert len(put_calls) == 1
        assert put_calls[0][1]["json"]["sha"] == "existing_sha_123"

    def test_request_raises_on_error_status(self, mock_session):
        """Request method raises on unexpected status codes."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Bad credentials"
        mock_session.request.return_value = mock_response

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
        )

        with pytest.raises(RuntimeError, match="GitHub API call failed \\(401\\)"):
            sink._request("GET", "https://api.github.com/test")

    def test_request_accepts_expected_status(self, mock_session):
        """Request accepts specified expected status codes."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.request.return_value = mock_response

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
        )

        # Should not raise when 404 is expected
        result = sink._request("GET", "https://api.github.com/test", expected_status={200, 404})
        assert result.status_code == 404

    def test_rate_limit_error(self, mock_session):
        """Rate limit error is raised properly."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "API rate limit exceeded"
        mock_session.request.return_value = mock_response

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
        )

        with pytest.raises(RuntimeError, match="429"):
            sink._request("GET", "https://api.github.com/test")


class TestAzureDevOpsRepoSink:
    """Tests for Azure DevOps repository sink."""

    @pytest.fixture
    def mock_session(self):
        """Create mock requests session."""
        session = MagicMock(spec=requests.Session)
        return session

    def test_init_with_required_params(self, mock_session):
        """Azure DevOps sink initializes with required parameters."""
        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
        )

        assert sink.organization == "myorg"
        assert sink.project == "myproject"
        assert sink.repository == "myrepo"
        assert sink.branch == "main"
        assert sink.base_url == "https://dev.azure.com"

    def test_init_with_custom_params(self, mock_session):
        """Azure DevOps sink accepts custom parameters."""
        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            branch="develop",
            token_env="MY_ADO_PAT",
            api_version="7.0",
            session=mock_session,
        )

        assert sink.branch == "develop"
        assert sink.token_env == "MY_ADO_PAT"
        assert sink.api_version == "7.0"

    def test_headers_includes_basic_auth(self, mock_session, monkeypatch):
        """Authorization header uses Basic auth with PAT."""
        monkeypatch.setenv("AZURE_DEVOPS_PAT", "my_pat_token")

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
        )

        headers = sink._headers()

        # Basic auth format is base64(":token")
        expected_auth = base64.b64encode(b":my_pat_token").decode("ascii")
        assert headers["Authorization"] == f"Basic {expected_auth}"

    def test_get_branch_ref_returns_object_id(self, mock_session):
        """Branch ref retrieval returns objectId."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [{"objectId": "abc123def456789"}]
        }
        mock_session.request.return_value = mock_response

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
        )

        ref = sink._get_branch_ref()

        assert ref == "abc123def456789"

    def test_get_branch_ref_not_found(self, mock_session):
        """Branch not found raises error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"value": []}
        mock_session.request.return_value = mock_response

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            branch="nonexistent",
            session=mock_session,
        )

        with pytest.raises(RuntimeError, match="Branch 'nonexistent' not found"):
            sink._get_branch_ref()

    def test_item_exists_returns_true(self, mock_session):
        """Item exists check returns True for 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.request.return_value = mock_response

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
        )

        assert sink._item_exists("path/to/file.json") is True

    def test_item_exists_returns_false(self, mock_session):
        """Item exists check returns False for 404."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.request.return_value = mock_response

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
        )

        assert sink._item_exists("new/file.json") is False

    def test_ensure_path_adds_leading_slash(self, mock_session):
        """Path without leading slash gets one added."""
        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
        )

        assert sink._ensure_path("path/to/file") == "/path/to/file"

    def test_ensure_path_preserves_existing_slash(self, mock_session):
        """Path with leading slash is preserved."""
        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
        )

        assert sink._ensure_path("/already/has/slash") == "/already/has/slash"

    def test_upload_creates_push_with_changes(self, mock_session):
        """Upload creates push request with correct payload."""
        # Branch ref request
        branch_response = MagicMock()
        branch_response.status_code = 200
        branch_response.json.return_value = {"value": [{"objectId": "branch_ref_123"}]}

        # Item exists checks (404 = new files)
        exists_response = MagicMock()
        exists_response.status_code = 404

        # Push request
        push_response = MagicMock()
        push_response.status_code = 201

        mock_session.request.side_effect = [
            branch_response,  # Get branch ref
            exists_response,  # Check file 1
            exists_response,  # Check file 2
            push_response,  # Push
        ]

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
            dry_run=False,
        )

        files = [
            PreparedFile(path="output/results.json", content=b'{"test": true}'),
            PreparedFile(path="output/manifest.json", content=b'{"rows": 1}'),
        ]
        sink._upload(files, "Test commit", {}, {}, datetime.now(UTC))

        # Verify POST request for push
        post_calls = [c for c in mock_session.request.call_args_list if c[0][0] == "POST"]
        assert len(post_calls) == 1

        payload = post_calls[0][1]["json"]
        assert payload["refUpdates"][0]["oldObjectId"] == "branch_ref_123"
        assert payload["commits"][0]["comment"] == "Test commit"
        assert len(payload["commits"][0]["changes"]) == 2

    def test_upload_edit_existing_file(self, mock_session):
        """Upload uses edit changeType for existing files."""
        # Branch ref request
        branch_response = MagicMock()
        branch_response.status_code = 200
        branch_response.json.return_value = {"value": [{"objectId": "branch_ref"}]}

        # Item exists checks (200 = existing)
        exists_response = MagicMock()
        exists_response.status_code = 200

        # Push request
        push_response = MagicMock()
        push_response.status_code = 200

        mock_session.request.side_effect = [
            branch_response,
            exists_response,
            push_response,
        ]

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
            dry_run=False,
        )

        files = [PreparedFile(path="existing/file.json", content=b'{"updated": true}')]
        sink._upload(files, "Update", {}, {}, datetime.now(UTC))

        post_calls = [c for c in mock_session.request.call_args_list if c[0][0] == "POST"]
        change = post_calls[0][1]["json"]["commits"][0]["changes"][0]
        assert change["changeType"] == "edit"

    def test_request_raises_on_error(self, mock_session):
        """Request raises on unexpected status."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_session.request.return_value = mock_response

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
        )

        with pytest.raises(RuntimeError, match="Azure DevOps API call failed \\(403\\)"):
            sink._request("GET", "https://dev.azure.com/test")


class TestDryRunMode:
    """Tests for dry-run mode across both sinks."""

    def test_github_dry_run_does_not_call_api(self):
        """GitHub dry run doesn't make API calls."""
        mock_session = MagicMock(spec=requests.Session)

        sink = GitHubRepoSink(
            owner="myorg",
            repo="myrepo",
            session=mock_session,
            dry_run=True,
        )

        sink.write({"results": []}, metadata={"experiment": "test"})

        # No API calls should be made
        mock_session.request.assert_not_called()

        # Payload should be recorded
        assert len(sink._last_payloads) == 1
        assert sink._last_payloads[0]["dry_run"] is True

    def test_azure_devops_dry_run_does_not_call_api(self):
        """Azure DevOps dry run doesn't make API calls."""
        mock_session = MagicMock(spec=requests.Session)

        sink = AzureDevOpsRepoSink(
            organization="myorg",
            project="myproject",
            repository="myrepo",
            session=mock_session,
            dry_run=True,
        )

        sink.write({"results": []}, metadata={"experiment": "test"})

        # No API calls should be made
        mock_session.request.assert_not_called()

        # Payload should be recorded
        assert len(sink._last_payloads) == 1


class TestCommitMessageFormatting:
    """Tests for commit message template formatting."""

    def test_commit_message_includes_experiment_name(self):
        """Commit message includes experiment name from metadata."""

        class TestSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                self.last_commit_message = commit_message

        sink = TestSink(dry_run=False)
        sink.write({"results": []}, metadata={"experiment": "my-experiment"})

        assert "my-experiment" in sink.last_commit_message

    def test_custom_commit_message_template(self):
        """Custom commit message template is used."""

        class TestSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                self.last_commit_message = commit_message

        sink = TestSink(
            commit_message_template="Deploy {experiment} on {date}",
            dry_run=False,
        )
        sink.write({"results": []}, metadata={"experiment": "prod-run"})

        assert "Deploy prod-run on" in sink.last_commit_message


class TestPathTemplateFormatting:
    """Tests for path template resolution."""

    def test_path_template_with_multiple_placeholders(self):
        """Path template with multiple placeholders works."""

        class TestSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                self.last_files = files

        sink = TestSink(
            path_template="runs/{experiment}/{date}/{time}",
            dry_run=False,
        )
        sink.write({"results": []}, metadata={"experiment": "test-exp"})

        # Files should be in the resolved path
        assert "runs/test-exp/" in sink.last_files[0].path

    def test_path_template_with_custom_metadata(self):
        """Custom metadata fields can be used in path template."""

        class TestSink(_RepoSinkBase):
            def _upload(self, files, commit_message, metadata, context, timestamp):
                self.last_files = files

        sink = TestSink(
            path_template="projects/{project_id}/runs/{run_id}",
            dry_run=False,
        )
        sink.write({"results": []}, metadata={"project_id": "proj-123", "run_id": "run-456"})

        assert "projects/proj-123/runs/run-456" in sink.last_files[0].path
