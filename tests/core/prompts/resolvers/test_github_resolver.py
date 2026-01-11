"""Tests for GitHub resolver."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from elspeth.core.prompts.resolvers.github import GitHubResolver
from elspeth.core.prompts.resolvers.url_parser import parse_pack_url


class TestGitHubResolver:
    """Tests for GitHubResolver."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver with a mock session."""
        resolver = GitHubResolver()
        resolver.session = MagicMock()
        return resolver

    @pytest.fixture
    def parsed_url(self):
        """Parse a sample GitHub URL."""
        return parse_pack_url("github://myorg/prompts/quality-eval@v1.0")

    def test_scheme_property(self, resolver):
        assert resolver.scheme == "github"

    def test_can_resolve_github_url(self, resolver):
        assert resolver.can_resolve("github://org/repo/path") is True

    def test_cannot_resolve_other_url(self, resolver):
        assert resolver.can_resolve("azuredevops://org/proj/repo/path") is False
        assert resolver.can_resolve("http://example.com") is False

    def test_parse_url(self, resolver):
        parsed = resolver.parse_url("github://myorg/myrepo/pack@main")
        assert parsed.scheme == "github"
        assert parsed.organization == "myorg"
        assert parsed.repository == "myrepo"

    def test_headers_without_token(self, resolver):
        with patch.dict("os.environ", {}, clear=True):
            headers = resolver._headers()
            assert "Authorization" not in headers
            assert headers["Accept"] == "application/vnd.github+json"

    def test_headers_with_token(self, resolver):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}):
            resolver._headers_cache = None  # Clear cache
            headers = resolver._headers()
            assert headers["Authorization"] == "Bearer ghp_test123"

    def test_headers_strips_whitespace(self, resolver):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "  ghp_test123  \n"}):
            resolver._headers_cache = None
            headers = resolver._headers()
            assert headers["Authorization"] == "Bearer ghp_test123"

    def test_discover_files_directory(self, resolver, parsed_url):
        """Test discovering files in a directory."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "quality-eval/config.yaml", "type": "file"},
            {"path": "quality-eval/system_prompt.md", "type": "file"},
            {"path": "quality-eval/user_prompt.md", "type": "file"},
            {"path": "quality-eval/subdir", "type": "dir"},  # Should be excluded
        ]
        resolver.session.request.return_value = mock_response

        manifest = resolver.discover_files(parsed_url)

        assert len(manifest.files) == 3
        assert "quality-eval/config.yaml" in manifest.files
        assert manifest.metadata["provider"] == "github"
        assert manifest.metadata["organization"] == "myorg"
        assert manifest.metadata["ref"] == "v1.0"

    def test_discover_files_single_file(self, resolver, parsed_url):
        """Test discovering a single file (not directory)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "path": "quality-eval/config.yaml",
            "type": "file",
        }
        resolver.session.request.return_value = mock_response

        manifest = resolver.discover_files(parsed_url)

        assert len(manifest.files) == 1
        assert manifest.files[0] == "quality-eval/config.yaml"

    def test_discover_files_not_found(self, resolver, parsed_url):
        """Test handling of 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        resolver.session.request.return_value = mock_response

        with pytest.raises(FileNotFoundError):
            resolver.discover_files(parsed_url)

    def test_discover_files_auth_error(self, resolver, parsed_url):
        """Test handling of authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        resolver.session.request.return_value = mock_response

        with pytest.raises(PermissionError, match="authentication failed"):
            resolver.discover_files(parsed_url)

    def test_fetch_file_success(self, resolver, parsed_url):
        """Test successful file fetch."""
        content = "prompts:\n  system: test"
        encoded = base64.b64encode(content.encode()).decode()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": encoded,
            "encoding": "base64",
        }
        resolver.session.request.return_value = mock_response

        fetched = resolver.fetch_file(parsed_url, "quality-eval/config.yaml")

        assert fetched.path == "quality-eval/config.yaml"
        assert fetched.content == content
        assert fetched.encoding == "utf-8"

    def test_fetch_file_not_found(self, resolver, parsed_url):
        """Test handling of file not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        resolver.session.request.return_value = mock_response

        with pytest.raises(FileNotFoundError):
            resolver.fetch_file(parsed_url, "nonexistent.yaml")

    def test_fetch_file_decodes_multiline_base64(self, resolver, parsed_url):
        """Test that base64 with newlines is handled correctly."""
        content = "A long content string that will result in multiline base64"
        # GitHub returns base64 with newlines every 60 chars
        encoded = base64.b64encode(content.encode()).decode()
        encoded_with_newlines = "\n".join(
            encoded[i : i + 60] for i in range(0, len(encoded), 60)
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": encoded_with_newlines,
            "encoding": "base64",
        }
        resolver.session.request.return_value = mock_response

        fetched = resolver.fetch_file(parsed_url, "config.yaml")

        assert fetched.content == content

    def test_request_includes_ref_parameter(self, resolver, parsed_url):
        """Test that requests include the ref parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        resolver.session.request.return_value = mock_response

        resolver.discover_files(parsed_url)

        # Check that ref=v1.0 was passed
        call_args = resolver.session.request.call_args
        assert call_args.kwargs["params"]["ref"] == "v1.0"

    def test_request_network_error(self, resolver, parsed_url):
        """Test handling of network errors."""
        import requests

        resolver.session.request.side_effect = requests.RequestException("Network error")

        with pytest.raises(ConnectionError, match="Failed to connect to GitHub"):
            resolver.discover_files(parsed_url)

    def test_custom_base_url(self):
        """Test resolver with custom base URL (GitHub Enterprise)."""
        resolver = GitHubResolver(base_url="https://github.mycompany.com/api/v3/")
        assert resolver.base_url == "https://github.mycompany.com/api/v3"  # Trailing slash removed

    def test_custom_token_env(self):
        """Test resolver with custom token environment variable."""
        resolver = GitHubResolver(token_env="MY_GITHUB_TOKEN")
        resolver.session = MagicMock()

        with patch.dict("os.environ", {"MY_GITHUB_TOKEN": "custom_token"}):
            headers = resolver._headers()
            assert headers["Authorization"] == "Bearer custom_token"

    def test_headers_cached(self, resolver):
        """Test that headers are cached after first call."""
        with patch.dict("os.environ", {"GITHUB_TOKEN": "token123"}):
            headers1 = resolver._headers()
            headers2 = resolver._headers()
            assert headers1 is headers2


class TestGitHubResolverIntegration:
    """Integration-style tests for GitHubResolver."""

    def test_full_workflow_mocked(self):
        """Test complete resolve workflow with mocked HTTP."""
        resolver = GitHubResolver()
        resolver.session = MagicMock()

        # Mock directory listing and file fetches
        def mock_request(method, url, **kwargs):
            response = MagicMock()
            response.status_code = 200

            # Check for specific file paths in URL (file fetches)
            if "/contents/prompts/config.yaml" in url:
                content = base64.b64encode(b"prompts:\n  user: from config").decode()
                response.json.return_value = {"content": content, "encoding": "base64"}
            elif "/contents/prompts/system_prompt.md" in url:
                content = base64.b64encode(b"# System Prompt\nBe helpful.").decode()
                response.json.return_value = {"content": content, "encoding": "base64"}
            elif url.endswith("/contents/prompts"):
                # Directory listing (no file extension)
                response.json.return_value = [
                    {"path": "prompts/config.yaml", "type": "file"},
                    {"path": "prompts/system_prompt.md", "type": "file"},
                ]
            else:
                response.status_code = 404
                response.text = "Not Found"

            return response

        resolver.session.request.side_effect = mock_request

        parsed = parse_pack_url("github://myorg/prompts/prompts")
        manifest = resolver.discover_files(parsed)

        assert len(manifest.files) == 2

        # Fetch each file
        config = resolver.fetch_file(parsed, "prompts/config.yaml")
        assert "prompts:" in config.content

        system = resolver.fetch_file(parsed, "prompts/system_prompt.md")
        assert "System Prompt" in system.content
