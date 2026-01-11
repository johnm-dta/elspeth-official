"""Tests for Azure DevOps resolver."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from elspeth.core.prompts.resolvers.azuredevops import AzureDevOpsResolver
from elspeth.core.prompts.resolvers.url_parser import parse_pack_url


class TestAzureDevOpsResolver:
    """Tests for AzureDevOpsResolver."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver with a mock session."""
        resolver = AzureDevOpsResolver()
        resolver.session = MagicMock()
        return resolver

    @pytest.fixture
    def parsed_url(self):
        """Parse a sample Azure DevOps URL."""
        return parse_pack_url("azuredevops://myorg/myproj/prompts/quality-eval@release")

    def test_scheme_property(self, resolver):
        assert resolver.scheme == "azuredevops"

    def test_can_resolve_azuredevops_url(self, resolver):
        assert resolver.can_resolve("azuredevops://org/proj/repo/path") is True

    def test_cannot_resolve_other_url(self, resolver):
        assert resolver.can_resolve("github://org/repo/path") is False
        assert resolver.can_resolve("http://example.com") is False

    def test_parse_url(self, resolver):
        parsed = resolver.parse_url("azuredevops://myorg/myproj/myrepo/pack@main")
        assert parsed.scheme == "azuredevops"
        assert parsed.organization == "myorg"
        assert parsed.project == "myproj"
        assert parsed.repository == "myrepo"

    def test_headers_without_token(self, resolver):
        with patch.dict("os.environ", {}, clear=True):
            resolver._headers_cache = None
            headers = resolver._headers()
            assert "Authorization" not in headers
            assert headers["Content-Type"] == "application/json"

    def test_headers_with_token(self, resolver):
        with patch.dict("os.environ", {"AZURE_DEVOPS_PAT": "pat_test123"}):
            resolver._headers_cache = None
            headers = resolver._headers()
            # Should be Basic auth with ":PAT" base64 encoded
            expected_auth = base64.b64encode(b":pat_test123").decode("ascii")
            assert headers["Authorization"] == f"Basic {expected_auth}"

    def test_headers_strips_whitespace(self, resolver):
        with patch.dict("os.environ", {"AZURE_DEVOPS_PAT": "  pat_test123  \n"}):
            resolver._headers_cache = None
            headers = resolver._headers()
            expected_auth = base64.b64encode(b":pat_test123").decode("ascii")
            assert headers["Authorization"] == f"Basic {expected_auth}"

    def test_ensure_path_adds_leading_slash(self, resolver):
        assert resolver._ensure_path("path/to/file") == "/path/to/file"

    def test_ensure_path_preserves_leading_slash(self, resolver):
        assert resolver._ensure_path("/already/has/slash") == "/already/has/slash"

    def test_ensure_path_empty_string(self, resolver):
        assert resolver._ensure_path("") == "/"

    def test_discover_files_directory(self, resolver, parsed_url):
        """Test discovering files in a directory."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [
                {"path": "/quality-eval/config.yaml", "isFolder": False},
                {"path": "/quality-eval/system_prompt.md", "isFolder": False},
                {"path": "/quality-eval/subdir", "isFolder": True},  # Should be excluded
            ]
        }
        resolver.session.request.return_value = mock_response

        manifest = resolver.discover_files(parsed_url)

        assert len(manifest.files) == 2
        assert "quality-eval/config.yaml" in manifest.files
        assert manifest.metadata["provider"] == "azuredevops"
        assert manifest.metadata["organization"] == "myorg"
        assert manifest.metadata["project"] == "myproj"

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

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = content
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

    def test_fetch_file_uses_text_plain_accept(self, resolver, parsed_url):
        """Test that file fetch uses text/plain Accept header."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        resolver.session.request.return_value = mock_response

        resolver.fetch_file(parsed_url, "config.yaml")

        call_args = resolver.session.request.call_args
        assert call_args.kwargs["headers"]["Accept"] == "text/plain"

    def test_request_includes_version_descriptor(self, resolver, parsed_url):
        """Test that requests include version descriptor parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"value": []}
        resolver.session.request.return_value = mock_response

        resolver.discover_files(parsed_url)

        call_args = resolver.session.request.call_args
        params = call_args.kwargs["params"]
        assert params["versionDescriptor.version"] == "release"
        assert params["versionDescriptor.versionType"] == "branch"

    def test_request_network_error(self, resolver, parsed_url):
        """Test handling of network errors."""
        import requests

        resolver.session.request.side_effect = requests.RequestException("Network error")

        with pytest.raises(ConnectionError, match="Failed to connect to Azure DevOps"):
            resolver.discover_files(parsed_url)

    def test_custom_base_url(self):
        """Test resolver with custom base URL."""
        resolver = AzureDevOpsResolver(base_url="https://mycompany.visualstudio.com/")
        assert resolver.base_url == "https://mycompany.visualstudio.com"

    def test_custom_api_version(self):
        """Test resolver with custom API version."""
        resolver = AzureDevOpsResolver(api_version="6.0")
        assert resolver.api_version == "6.0"

    def test_custom_token_env(self):
        """Test resolver with custom token environment variable."""
        resolver = AzureDevOpsResolver(token_env="MY_AZDO_PAT")
        resolver.session = MagicMock()

        with patch.dict("os.environ", {"MY_AZDO_PAT": "custom_pat"}):
            resolver._headers_cache = None
            headers = resolver._headers()
            expected_auth = base64.b64encode(b":custom_pat").decode("ascii")
            assert headers["Authorization"] == f"Basic {expected_auth}"

    def test_headers_cached(self, resolver):
        """Test that headers are cached after first call."""
        with patch.dict("os.environ", {"AZURE_DEVOPS_PAT": "token123"}):
            resolver._headers_cache = None
            headers1 = resolver._headers()
            headers2 = resolver._headers()
            assert headers1 is headers2


class TestAzureDevOpsResolverIntegration:
    """Integration-style tests for AzureDevOpsResolver."""

    def test_full_workflow_mocked(self):
        """Test complete resolve workflow with mocked HTTP."""
        resolver = AzureDevOpsResolver()
        resolver.session = MagicMock()

        # Mock directory listing and file fetch
        def mock_request(method, url, **kwargs):
            response = MagicMock()
            response.status_code = 200

            headers = kwargs.get("headers", {})

            if "recursionLevel" in str(kwargs.get("params", {})):
                # Directory listing
                response.json.return_value = {
                    "value": [
                        {"path": "/prompts/config.yaml", "isFolder": False},
                        {"path": "/prompts/system_prompt.md", "isFolder": False},
                    ]
                }
            elif headers.get("Accept") == "text/plain":
                # File fetch
                if "config.yaml" in str(kwargs.get("params", {}).get("path", "")):
                    response.text = "prompts:\n  user: from config"
                elif "system_prompt.md" in str(kwargs.get("params", {}).get("path", "")):
                    response.text = "# System Prompt\nBe helpful."
                else:
                    response.status_code = 404
                    response.text = "Not Found"
            else:
                response.json.return_value = {}

            return response

        resolver.session.request.side_effect = mock_request

        parsed = parse_pack_url("azuredevops://myorg/myproj/repo/prompts")
        manifest = resolver.discover_files(parsed)

        assert len(manifest.files) == 2

        # Fetch each file
        config = resolver.fetch_file(parsed, "prompts/config.yaml")
        assert "prompts:" in config.content

        system = resolver.fetch_file(parsed, "prompts/system_prompt.md")
        assert "System Prompt" in system.content
