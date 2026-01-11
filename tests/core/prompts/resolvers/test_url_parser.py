"""Tests for URL parsing utilities."""

import pytest

from elspeth.core.prompts.resolvers.url_parser import (
    get_scheme,
    is_remote_pack_url,
    parse_pack_url,
)


class TestIsRemotePackUrl:
    """Tests for is_remote_pack_url function."""

    def test_github_url(self):
        assert is_remote_pack_url("github://org/repo/path") is True

    def test_github_url_with_branch(self):
        assert is_remote_pack_url("github://org/repo/path@main") is True

    def test_azuredevops_url(self):
        assert is_remote_pack_url("azuredevops://org/project/repo/path") is True

    def test_azuredevops_url_with_branch(self):
        assert is_remote_pack_url("azuredevops://org/project/repo/path@release") is True

    def test_local_pack_name(self):
        assert is_remote_pack_url("my_local_pack") is False

    def test_empty_string(self):
        assert is_remote_pack_url("") is False

    def test_none_returns_false(self):
        assert is_remote_pack_url(None) is False  # type: ignore[arg-type]

    def test_non_string_returns_false(self):
        assert is_remote_pack_url(123) is False  # type: ignore[arg-type]

    def test_http_url_not_recognized(self):
        assert is_remote_pack_url("http://example.com") is False

    def test_file_url_not_recognized(self):
        assert is_remote_pack_url("file:///path/to/pack") is False


class TestGetScheme:
    """Tests for get_scheme function."""

    def test_github_scheme(self):
        assert get_scheme("github://org/repo/path") == "github"

    def test_azuredevops_scheme(self):
        assert get_scheme("azuredevops://org/proj/repo/path") == "azuredevops"

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown URL scheme"):
            get_scheme("http://example.com")

    def test_local_name_raises(self):
        with pytest.raises(ValueError, match="Unknown URL scheme"):
            get_scheme("local_pack")


class TestParsePackUrlGitHub:
    """Tests for parsing GitHub URLs."""

    def test_basic_url(self):
        parsed = parse_pack_url("github://myorg/myrepo/prompts/eval")
        assert parsed.scheme == "github"
        assert parsed.organization == "myorg"
        assert parsed.project is None
        assert parsed.repository == "myrepo"
        assert parsed.path == "prompts/eval"
        assert parsed.ref == "main"
        assert parsed.original_url == "github://myorg/myrepo/prompts/eval"

    def test_url_with_branch(self):
        parsed = parse_pack_url("github://myorg/myrepo/prompts/eval@v1.0")
        assert parsed.scheme == "github"
        assert parsed.organization == "myorg"
        assert parsed.repository == "myrepo"
        assert parsed.path == "prompts/eval"
        assert parsed.ref == "v1.0"

    def test_url_with_feature_branch(self):
        parsed = parse_pack_url("github://myorg/myrepo/prompts@feature/new-prompts")
        assert parsed.ref == "feature/new-prompts"

    def test_url_without_path(self):
        parsed = parse_pack_url("github://myorg/myrepo")
        assert parsed.path == ""
        assert parsed.ref == "main"

    def test_url_with_only_branch(self):
        parsed = parse_pack_url("github://myorg/myrepo@develop")
        assert parsed.path == ""
        assert parsed.ref == "develop"

    def test_url_single_folder(self):
        parsed = parse_pack_url("github://myorg/myrepo/prompts")
        assert parsed.path == "prompts"

    def test_url_deep_path(self):
        parsed = parse_pack_url("github://myorg/myrepo/a/b/c/d/e")
        assert parsed.path == "a/b/c/d/e"


class TestParsePackUrlAzureDevOps:
    """Tests for parsing Azure DevOps URLs."""

    def test_basic_url(self):
        parsed = parse_pack_url("azuredevops://myorg/myproj/myrepo/prompts/eval")
        assert parsed.scheme == "azuredevops"
        assert parsed.organization == "myorg"
        assert parsed.project == "myproj"
        assert parsed.repository == "myrepo"
        assert parsed.path == "prompts/eval"
        assert parsed.ref == "main"
        assert parsed.original_url == "azuredevops://myorg/myproj/myrepo/prompts/eval"

    def test_url_with_branch(self):
        parsed = parse_pack_url("azuredevops://myorg/myproj/myrepo/prompts@release/1.0")
        assert parsed.ref == "release/1.0"

    def test_url_without_path(self):
        parsed = parse_pack_url("azuredevops://myorg/myproj/myrepo")
        assert parsed.path == ""
        assert parsed.ref == "main"

    def test_url_with_only_branch(self):
        parsed = parse_pack_url("azuredevops://myorg/myproj/myrepo@develop")
        assert parsed.path == ""
        assert parsed.ref == "develop"


class TestParsePackUrlErrors:
    """Tests for URL parsing error cases."""

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Unrecognized prompt pack URL format"):
            parse_pack_url("http://example.com")

    def test_incomplete_github_url_raises(self):
        with pytest.raises(ValueError, match="Unrecognized prompt pack URL format"):
            parse_pack_url("github://onlyorg")

    def test_local_pack_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognized prompt pack URL format"):
            parse_pack_url("my_local_pack")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unrecognized prompt pack URL format"):
            parse_pack_url("")
