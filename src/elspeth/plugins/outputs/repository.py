"""Result sinks that push artifacts to source control hosting services."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import requests

from elspeth.core.interfaces import ResultSink

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _default_context(metadata: Mapping[str, Any], timestamp: datetime) -> dict[str, Any]:
    context = {k: v for k, v in metadata.items() if isinstance(k, str)}
    context.setdefault("timestamp", timestamp.strftime("%Y%m%dT%H%M%SZ"))
    context.setdefault("date", timestamp.strftime("%Y-%m-%d"))
    context.setdefault("time", timestamp.strftime("%H%M%S"))
    context.setdefault("experiment", metadata.get("experiment") or metadata.get("name") or "experiment")
    return context


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


@dataclass
class PreparedFile:
    path: str
    content: bytes
    content_type: str = "application/json"


@dataclass
class _RepoSinkBase(ResultSink):
    path_template: str = "experiments/{experiment}/{timestamp}"
    commit_message_template: str = "Add experiment results for {experiment}"
    include_manifest: bool = True
    dry_run: bool = True
    session: requests.Session | None = None
    _last_payloads: list[dict[str, Any]] = field(default_factory=list, init=False)
    _artifact_inputs: list[Any] = field(default_factory=list, init=False)
    on_error: str = "abort"

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        if self.on_error != "abort":
            raise ValueError("on_error must be 'abort'")

    def write(self, results: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        timestamp = datetime.now(UTC)
        context = _default_context(metadata, timestamp)
        try:
            prefix = self._resolve_prefix(context)
            files = self._prepare_files(results, metadata, prefix, timestamp)
            commit_message = self.commit_message_template.format(**context)
            payload = {
                "context": context,
                "commit_message": commit_message,
                "files": [
                    {
                        "path": file.path,
                        "size": len(file.content),
                        "content_type": file.content_type,
                    }
                    for file in files
                ],
            }
            if self.dry_run:
                payload["dry_run"] = True
                self._last_payloads.append(payload)
                return
            self._upload(files, commit_message, metadata, context, timestamp)
        except Exception:
            raise

    # ------------------------------------------------------------------ internals
    def _resolve_prefix(self, context: Mapping[str, Any]) -> str:
        try:
            return self.path_template.format(**context)
        except KeyError as exc:  # pragma: no cover - configuration error
            missing = exc.args[0]
            raise ValueError(f"Missing placeholder '{missing}' in path template") from exc

    def _prepare_files(
        self,
        results: dict[str, Any],
        metadata: Mapping[str, Any],
        prefix: str,
        timestamp: datetime,
    ) -> list[PreparedFile]:
        files: list[PreparedFile] = []

        # If we have consumed artifacts, upload those instead of raw results
        if self._artifact_inputs:
            from pathlib import Path
            for artifact in self._artifact_inputs:
                if artifact.path:
                    artifact_path = Path(artifact.path)
                    content = artifact_path.read_bytes()
                    # Determine content type from artifact or file extension
                    content_type = "application/octet-stream"
                    if artifact.metadata and artifact.metadata.get("content_type"):
                        content_type = artifact.metadata["content_type"]
                    elif artifact_path.suffix == ".json":
                        content_type = "application/json"
                    elif artifact_path.suffix == ".zip":
                        content_type = "application/zip"
                    files.append(PreparedFile(
                        path=f"{prefix}/{artifact_path.name}",
                        content=content,
                        content_type=content_type,
                    ))
            return files

        # Default behavior: upload results.json and manifest
        results_path = f"{prefix}/results.json"
        manifest_path = f"{prefix}/manifest.json"
        files.append(PreparedFile(path=results_path, content=_json_bytes(results)))
        if self.include_manifest:
            manifest = {
                "generated_at": timestamp.isoformat(),
                "rows": len(results.get("results", [])),
                "metadata": dict(metadata),
            }
            if "aggregates" in results:
                manifest["aggregates"] = results["aggregates"]
            if "cost_summary" in results:
                manifest["cost_summary"] = results["cost_summary"]
            files.append(PreparedFile(path=manifest_path, content=_json_bytes(manifest)))
        return files

    def prepare_artifacts(self, artifacts: Mapping[str, list[Any]]) -> None:
        """Receive consumed artifacts from upstream sinks.

        Handles split archives by expanding all_parts metadata into separate artifacts.
        """
        from elspeth.core.interfaces import Artifact

        # Deduplicate by path since artifacts are registered by both name and alias
        seen_paths: set[str] = set()
        collected: list[Any] = []
        for values in artifacts.values():
            if values:
                for artifact in values:
                    # Handle split archives - expand all_parts into separate artifacts
                    if artifact.metadata.get("split_archive") and artifact.metadata.get("all_parts"):
                        for part_path in artifact.metadata["all_parts"]:
                            if part_path not in seen_paths:
                                seen_paths.add(part_path)
                                # Create synthetic artifact for each part
                                collected.append(Artifact(
                                    id=f"split_part:{part_path}",
                                    type="file/zip-part",
                                    path=part_path,
                                    metadata={"content_type": "application/octet-stream"},
                                ))
                    elif artifact.path and artifact.path not in seen_paths:
                        seen_paths.add(artifact.path)
                        collected.append(artifact)
        self._artifact_inputs = collected

    # To be implemented by subclasses
    def _upload(
        self,
        files: list[PreparedFile],
        commit_message: str,
        metadata: Mapping[str, Any],
        context: Mapping[str, Any],
        timestamp: datetime,
    ) -> None:
        raise NotImplementedError

    def produces(self):  # pragma: no cover - placeholder for artifact chaining
        return []

    def consumes(self):  # pragma: no cover - placeholder for artifact chaining
        return []

    def finalize(self, artifacts, *, metadata=None):  # pragma: no cover - optional cleanup
        return None


class GitHubRepoSink(_RepoSinkBase):
    """Push experiment artifacts to a GitHub repository via the REST API."""

    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        branch: str = "main",
        token_env: str = "GITHUB_TOKEN",
        base_url: str = "https://api.github.com",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.token_env = token_env
        self.base_url = base_url.rstrip("/")
        self._headers_cache: dict[str, str] | None = None

    # Upload implementation -------------------------------------------------
    def _upload(
        self,
        files: list[PreparedFile],
        commit_message: str,
        metadata: Mapping[str, Any],
        context: Mapping[str, Any],
        timestamp: datetime,
    ) -> None:
        for prepared in files:
            sha = self._get_existing_sha(prepared.path)
            payload = {
                "message": commit_message,
                "branch": self.branch,
                "content": base64.b64encode(prepared.content).decode("ascii"),
            }
            if sha:
                payload["sha"] = sha
            self._request(
                "PUT",
                f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{prepared.path}",
                json=payload,
            )

    # Helpers ----------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        if self._headers_cache is not None:
            return self._headers_cache
        headers = {"Accept": "application/vnd.github+json"}
        token = self._read_token(self.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._headers_cache = headers
        return headers

    def _get_existing_sha(self, path: str) -> str | None:
        response = self._request(
            "GET",
            f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}",
            expected_status={200, 404},
        )
        if response.status_code == 404:
            return None
        data = response.json()
        return data.get("sha")

    def _request(self, method: str, url: str, expected_status: set[int] | None = None, **kwargs: Any):  # type: ignore[return-any]
        expected_status = expected_status or {200, 201}
        response = self.session.request(method, url, headers=self._headers(), **kwargs)
        if response.status_code not in expected_status:
            raise RuntimeError(f"GitHub API call failed ({response.status_code}): {response.text}")
        return response

    @staticmethod
    def _read_token(env_var: str) -> str | None:
        token = os.getenv(env_var)
        return token.strip() if token else None


class AzureDevOpsRepoSink(_RepoSinkBase):
    """Push experiment artifacts to an Azure DevOps Git repository."""

    def __init__(
        self,
        *,
        organization: str,
        project: str,
        repository: str,
        branch: str = "main",
        token_env: str = "AZURE_DEVOPS_PAT",
        api_version: str = "7.1-preview",
        base_url: str = "https://dev.azure.com",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.organization = organization
        self.project = project
        self.repository = repository
        self.branch = branch
        self.token_env = token_env
        self.api_version = api_version
        self.base_url = base_url.rstrip("/")
        self._headers_cache: dict[str, str] | None = None

    # Upload implementation -------------------------------------------------
    # Azure DevOps has ~25MB request size limit
    MAX_BATCH_SIZE = 20 * 1024 * 1024  # 20MB to leave margin for JSON overhead

    def _upload(
        self,
        files: list[PreparedFile],
        commit_message: str,
        metadata: Mapping[str, Any],
        context: Mapping[str, Any],
        timestamp: datetime,
    ) -> None:
        # Batch files by size to avoid Azure DevOps request size limit
        batches = self._batch_files_by_size(files)

        for batch_idx, batch in enumerate(batches):
            branch_ref = self._get_branch_ref()  # Refresh ref for each batch
            changes = []
            for prepared in batch:
                existing = self._item_exists(prepared.path)
                change_type = "edit" if existing else "add"
                # Use base64 for binary content, rawtext for text
                is_binary = prepared.content_type in ("application/zip", "application/octet-stream")
                if is_binary:
                    import base64
                    content = base64.b64encode(prepared.content).decode("ascii")
                    content_type = "base64encoded"
                else:
                    content = prepared.content.decode("utf-8")
                    content_type = "rawtext"
                changes.append(
                    {
                        "changeType": change_type,
                        "item": {"path": self._ensure_path(prepared.path)},
                        "newContent": {
                            "content": content,
                            "contentType": content_type,
                        },
                    }
                )

            # Add batch suffix to commit message if multiple batches
            msg = commit_message if len(batches) == 1 else f"{commit_message} (part {batch_idx + 1}/{len(batches)})"

            payload = {
                "refUpdates": [
                    {
                        "name": f"refs/heads/{self.branch}",
                        "oldObjectId": branch_ref,
                    }
                ],
                "commits": [
                    {
                        "comment": msg,
                        "changes": changes,
                    }
                ],
            }
            url = (
                f"{self.base_url}/{self.organization}/{self.project}/_apis/git"
                f"/repositories/{self.repository}/pushes?api-version={self.api_version}"
            )
            self._request("POST", url, json=payload, expected_status={200, 201})

    def _batch_files_by_size(self, files: list[PreparedFile]) -> list[list[PreparedFile]]:
        """Split files into batches that fit within Azure DevOps size limits."""
        batches: list[list[PreparedFile]] = []
        current_batch: list[PreparedFile] = []
        current_size = 0

        for prepared in files:
            file_size = len(prepared.content)
            # Base64 encoding increases size by ~33%
            is_binary = prepared.content_type in ("application/zip", "application/octet-stream")
            if is_binary:
                file_size = int(file_size * 1.37)  # Base64 overhead + JSON escaping

            # If single file exceeds limit, it goes in its own batch
            if file_size > self.MAX_BATCH_SIZE:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_size = 0
                batches.append([prepared])
                continue

            # If adding this file would exceed limit, start new batch
            if current_size + file_size > self.MAX_BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []
                current_size = 0

            current_batch.append(prepared)
            current_size += file_size

        if current_batch:
            batches.append(current_batch)

        return batches

    # Helpers ----------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        if self._headers_cache is not None:
            return self._headers_cache
        headers = {"Content-Type": "application/json"}
        token = self._read_token(self.token_env)
        if token:
            auth = base64.b64encode(f":{token}".encode()).decode("ascii")
            headers["Authorization"] = f"Basic {auth}"
        self._headers_cache = headers
        return headers

    def _get_branch_ref(self) -> str:
        url = (
            f"{self.base_url}/{self.organization}/{self.project}/_apis/git"
            f"/repositories/{self.repository}/refs?filter=heads/{self.branch}"
            f"&api-version={self.api_version}"
        )
        response = self._request("GET", url, expected_status={200})
        data = response.json()
        if not data.get("value"):
            raise RuntimeError(f"Branch '{self.branch}' not found")
        return data["value"][0]["objectId"]

    def _item_exists(self, path: str) -> bool:
        url = (
            f"{self.base_url}/{self.organization}/{self.project}/_apis/git"
            f"/repositories/{self.repository}/items?path={self._ensure_path(path)}"
            f"&includeContentMetadata=true&api-version={self.api_version}"
        )
        response = self._request("GET", url, expected_status={200, 404})
        return response.status_code == 200

    def _request(self, method: str, url: str, expected_status: set[int] | None = None, **kwargs: Any):  # type: ignore[return-any]
        expected_status = expected_status or {200, 201}
        response = self.session.request(method, url, headers=self._headers(), **kwargs)
        if response.status_code not in expected_status:
            raise RuntimeError(f"Azure DevOps API call failed ({response.status_code}): {response.text}")
        return response

    def _ensure_path(self, path: str) -> str:
        if not path.startswith("/"):
            return f"/{path}"
        return path

    @staticmethod
    def _read_token(env_var: str) -> str | None:
        token = os.getenv(env_var)
        return token.strip() if token else None


import os

# --- Plugin Registration ---
from elspeth.core.registry import ARTIFACTS_SECTION_SCHEMA, ON_ERROR_ENUM, registry

GITHUB_REPO_SCHEMA = {
    "type": "object",
    "properties": {
        "path_template": {"type": "string"},
        "commit_message_template": {"type": "string"},
        "include_manifest": {"type": "boolean"},
        "owner": {"type": "string"},
        "repo": {"type": "string"},
        "branch": {"type": "string"},
        "token_env": {"type": "string"},
        "dry_run": {"type": "boolean"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["owner", "repo"],
    "additionalProperties": True,
}

AZURE_DEVOPS_REPO_SCHEMA = {
    "type": "object",
    "properties": {
        "path_template": {"type": "string"},
        "commit_message_template": {"type": "string"},
        "include_manifest": {"type": "boolean"},
        "organization": {"type": "string"},
        "project": {"type": "string"},
        "repository": {"type": "string"},
        "branch": {"type": "string"},
        "token_env": {"type": "string"},
        "api_version": {"type": "string"},
        "base_url": {"type": "string"},
        "dry_run": {"type": "boolean"},
        "artifacts": ARTIFACTS_SECTION_SCHEMA,
        "security_level": {"type": "string"},
        "on_error": ON_ERROR_ENUM,
    },
    "required": ["organization", "project", "repository"],
    "additionalProperties": True,
}

registry.register_sink("github_repo", GitHubRepoSink, GITHUB_REPO_SCHEMA)
registry.register_sink("azure_devops_repo", AzureDevOpsRepoSink, AZURE_DEVOPS_REPO_SCHEMA)
