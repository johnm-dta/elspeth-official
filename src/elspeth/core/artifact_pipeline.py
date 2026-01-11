"""Artifact dependency resolution for result sinks."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from elspeth.core.artifacts import validate_artifact_type
from elspeth.core.interfaces import Artifact, ArtifactDescriptor, ResultSink
from elspeth.core.security import is_security_level_allowed, normalize_security_level

VALID_REQUEST_MODES = {"single", "all"}
logger = logging.getLogger(__name__)


@dataclass
@dataclass
class ArtifactRequest:
    token: str
    mode: str = "single"


class ArtifactRequestParser:
    @staticmethod
    def parse(entry: Any) -> ArtifactRequest:
        if isinstance(entry, ArtifactRequest):
            ArtifactRequestParser._validate(entry.mode)
            return entry
        if isinstance(entry, str):
            ArtifactRequestParser._validate("single")
            return ArtifactRequest(token=entry, mode="single")
        if isinstance(entry, Mapping):
            token = entry.get("token") or entry.get("name")
            if not token:
                raise ValueError("Artifact consume entry requires 'token'")
            mode = entry.get("mode", "single")
            ArtifactRequestParser._validate(mode)
            return ArtifactRequest(token=token, mode=mode)
        raise ValueError(f"Unsupported consume declaration: {entry!r}")

    @staticmethod
    def _validate(mode: str) -> None:
        if mode not in VALID_REQUEST_MODES:
            raise ValueError(f"Unsupported artifact request mode '{mode}'")


@dataclass
class SinkBinding:
    """Container tying a sink instance to its configuration metadata."""

    id: str
    plugin: str
    sink: ResultSink
    artifact_config: dict[str, Any]
    original_index: int
    produces: list[ArtifactDescriptor] = field(default_factory=list)
    consumes: list[ArtifactRequest] = field(default_factory=list)
    security_level: str | None = None


class ArtifactStore:
    """Holds produced artifacts for downstream sinks."""

    def __init__(self) -> None:
        self._by_id: dict[str, Artifact] = {}
        self._by_alias: dict[str, Artifact] = {}
        self._by_type: dict[str, list[Artifact]] = defaultdict(list)

    def register(self, binding: SinkBinding, descriptor: ArtifactDescriptor, artifact: Artifact) -> None:
        artifact_id = artifact.id or f"{binding.id}:{descriptor.name}"
        artifact.id = artifact_id
        artifact.produced_by = binding.id
        artifact.persist = descriptor.persist or artifact.persist
        artifact.schema_id = artifact.schema_id or descriptor.schema_id
        level = artifact.security_level or descriptor.security_level or binding.security_level
        artifact.security_level = normalize_security_level(level)
        self._by_id[artifact_id] = artifact

        alias_key = descriptor.alias or descriptor.name
        if alias_key:
            self._by_alias[alias_key] = artifact

        self._by_type[descriptor.type].append(artifact)

    def get_by_alias(self, alias: str) -> Artifact | None:
        return self._by_alias.get(alias)

    def get_by_type(self, type_name: str) -> list[Artifact]:
        return list(self._by_type.get(type_name, []))

    def resolve_requests(self, requests: Iterable[ArtifactRequest]) -> dict[str, list[Artifact]]:
        resolved: dict[str, list[Artifact]] = {}
        for request in requests:
            token = request.token
            if not token:
                continue
            if token.startswith("@"):  # alias lookup
                alias = token[1:]
                artifact = self.get_by_alias(alias)
                selected = [artifact] if artifact else []
            else:
                try:
                    validate_artifact_type(token)
                except ValueError:
                    continue
                selected = self.get_by_type(token)

            if request.mode == "single" and selected:
                selected = selected[:1]

            key = token
            resolved[key] = selected
            if token.startswith("@"):  # convenience alias without '@'
                resolved[token[1:]] = selected
        return resolved

    def items(self) -> Iterable[tuple[str, Artifact]]:
        return self._by_id.items()


class ArtifactPipeline:
    """Resolves sink execution order based on declared artifact dependencies."""

    def __init__(self, bindings: list[SinkBinding]) -> None:
        self._bindings = [self._prepare_binding(binding) for binding in bindings]
        self._ordered_bindings = self._resolve_order(self._bindings)

    @staticmethod
    def _prepare_binding(binding: SinkBinding) -> SinkBinding:
        binding.security_level = normalize_security_level(binding.security_level)
        artifact_section = binding.artifact_config or {}
        produces_config = artifact_section.get("produces", []) or []
        for entry in produces_config:
            descriptor = ArtifactDescriptor(
                name=entry["name"],
                type=entry["type"],
                schema_id=entry.get("schema_id"),
                persist=entry.get("persist", False),
                alias=entry.get("alias"),
                security_level=normalize_security_level(entry.get("security_level")),
            )
            validate_artifact_type(descriptor.type)
            binding.produces.append(descriptor)

        produces_method = getattr(binding.sink, "produces", None)
        if callable(produces_method):
            for descriptor in produces_method() or []:
                validate_artifact_type(descriptor.type)
                descriptor.security_level = normalize_security_level(descriptor.security_level)
                binding.produces.append(descriptor)

        consumes_config = list(artifact_section.get("consumes", []) or [])
        consumes_method = getattr(binding.sink, "consumes", None)
        if callable(consumes_method):
            for token in consumes_method() or []:
                consumes_config.append(token)
        binding.consumes = [ArtifactRequestParser.parse(entry) for entry in consumes_config]
        return binding

    @staticmethod
    def _enforce_dependency_security(consumer: SinkBinding, producer: SinkBinding) -> None:
        if not consumer.security_level:
            return
        if not is_security_level_allowed(producer.security_level, consumer.security_level):
            raise PermissionError(f"Sink '{consumer.id}' cannot depend on '{producer.id}' due to security level mismatch")

    @staticmethod
    def _resolve_order(bindings: list[SinkBinding]) -> list[SinkBinding]:
        if not bindings:
            return []

        by_id = {binding.id: binding for binding in bindings}
        producers_by_name: dict[str, SinkBinding] = {}
        producers_by_type: dict[str, list[SinkBinding]] = defaultdict(list)

        for binding in bindings:
            for descriptor in binding.produces:
                key = descriptor.alias or descriptor.name
                if key and key not in producers_by_name:
                    producers_by_name[key] = binding
                producers_by_type[descriptor.type].append(binding)

        dependencies: dict[str, set[str]] = {binding.id: set() for binding in bindings}
        dependents: dict[str, set[str]] = {binding.id: set() for binding in bindings}

        for binding in bindings:
            for request in binding.consumes:
                token = request.token
                if not token:
                    continue
                matched_ids: list[str] = []
                if token.startswith("@"):  # alias/name match
                    key = token[1:]
                    producer_binding = producers_by_name.get(key)
                    if producer_binding:
                        ArtifactPipeline._enforce_dependency_security(binding, producer_binding)
                        matched_ids.append(producer_binding.id)
                else:
                    try:
                        validate_artifact_type(token)
                    except ValueError:
                        continue
                    for producer_binding in producers_by_type.get(token, []):
                        ArtifactPipeline._enforce_dependency_security(binding, producer_binding)
                        matched_ids.append(producer_binding.id)
                for producer_id in matched_ids:
                    if producer_id == binding.id:
                        continue
                    dependencies[binding.id].add(producer_id)
                    dependents[producer_id].add(binding.id)

        ready: deque[SinkBinding] = deque(
            sorted(
                [binding for binding in bindings if not dependencies[binding.id]],
                key=lambda b: b.original_index,
            )
        )

        ordered: list[SinkBinding] = []

        while ready:
            current = ready.popleft()
            ordered.append(current)
            for dependent_id in dependents[current.id]:
                deps = dependencies[dependent_id]
                if current.id in deps:
                    deps.remove(current.id)
                    if not deps:
                        dependent_binding = by_id[dependent_id]
                        ready.append(dependent_binding)
                        ready = deque(sorted(ready, key=lambda b: b.original_index))

        if len(ordered) != len(bindings):
            raise ValueError("Sink artifact dependencies contain a cycle or unresolved reference")

        return ordered

    def execute(self, payload: dict[str, Any], metadata: Mapping[str, Any] | None = None) -> ArtifactStore:
        store = ArtifactStore()
        for binding in self._ordered_bindings:
            consumed = store.resolve_requests(binding.consumes)

            clearance = binding.security_level
            if clearance:
                for artifacts in consumed.values():
                    for artifact in artifacts:
                        if not is_security_level_allowed(artifact.security_level, clearance):
                            raise PermissionError(
                                f"Sink '{binding.id}' with clearance '{clearance}' cannot consume "
                                f"artifact '{artifact.id}' at level '{normalize_security_level(artifact.security_level)}'"
                            )

            prepare = getattr(binding.sink, "prepare_artifacts", None)
            if callable(prepare):
                prepare(consumed)

            binding.sink.write(payload, metadata=dict(metadata) if metadata is not None else None)

            produced: dict[str, Artifact] = {}
            collector = getattr(binding.sink, "collect_artifacts", None)
            if callable(collector):
                produced = collector() or {}

            for descriptor in binding.produces:
                key = descriptor.name
                candidate = produced.get(key)
                if not candidate and descriptor.alias:
                    candidate = produced.get(descriptor.alias)
                if candidate:
                    store.register(binding, descriptor, candidate)

            finalize = getattr(binding.sink, "finalize", None)
            if callable(finalize):
                finalize(dict(store.items()), metadata=metadata)

        return store
