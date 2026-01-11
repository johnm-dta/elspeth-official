"""LLM query transform plugin for chained LLM calls."""

from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from elspeth.core.controls.registry import create_cost_tracker, create_rate_limiter
from elspeth.core.llm.registry import create_middleware
from elspeth.core.prompts import PromptEngine
from elspeth.core.prompts.resolvers import resolve_remote_pack
from elspeth.core.registry import registry
from elspeth.core.sda.llm_executor import LLMExecutor
from elspeth.core.sda.plugin_registry import register_transform_plugin

logger = logging.getLogger(__name__)


class LLMQueryPlugin:
    """Transform plugin that executes LLM queries.

    Each plugin instance is self-contained with its own:
    - LLM client
    - Middleware chain
    - Retry config
    - Prompt templates

    Queries map row fields to template variables and store
    responses in context with optional flattening to row.
    """

    name = "llm_query"

    def __init__(
        self,
        llm: dict[str, Any],
        queries: list[dict[str, Any]],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        prompt_pack: str | None = None,
        middlewares: list[dict[str, Any]] | None = None,
        retry: dict[str, Any] | None = None,
        rate_limiter: dict[str, Any] | None = None,
        cost_tracker: dict[str, Any] | None = None,
        parallel_queries: bool = True,
        max_parallel: int = 5,
    ) -> None:
        """Initialize LLM query plugin.

        Args:
            llm: LLM client config {plugin: str, options: dict}
            queries: List of query definitions
            system_prompt: System prompt template (inline)
            user_prompt: User prompt template (inline)
            prompt_pack: Remote prompt pack URL (alternative to inline)
            middlewares: Middleware chain config
            retry: Retry config
            rate_limiter: Rate limiter config (controls launch pacing)
            cost_tracker: Cost tracker config
            parallel_queries: Execute queries in parallel (default: True)
            max_parallel: Max concurrent queries (default: 5)
        """
        self.llm_config = llm
        self.queries = queries
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._prompt_pack = prompt_pack
        self._middlewares_config = middlewares or []
        self._retry_config = retry
        self._rate_limiter_config = rate_limiter
        self._cost_tracker_config = cost_tracker
        self._parallel_queries = parallel_queries
        self._max_parallel = max_parallel

        # Validate queries
        self._validate_queries()

        # Create LLM client
        self.llm_client = registry.create_llm(
            llm["plugin"],
            llm.get("options", {}),
        )

        # Create prompt engine and compile templates
        self.engine = PromptEngine()
        self.pack_defaults: dict[str, Any] = {}
        self.pack_config: dict[str, Any] = {}  # Full pack config for lookups

        # Resolve prompts (pack or inline)
        system_prompt_str, user_prompt_str = self._resolve_prompts()
        self.system_template = self.engine.compile(system_prompt_str, name="llm_query:system")
        self.user_template = self.engine.compile(user_prompt_str, name="llm_query:user")

        # Build middleware chain
        self.middlewares = self._build_middlewares()

        # Create LLM executor
        self.executor = LLMExecutor(
            llm_client=self.llm_client,
            middlewares=self.middlewares,
            retry_config=self._retry_config,
            rate_limiter=self._build_rate_limiter(),
            cost_tracker=self._build_cost_tracker(),
        )

    def _validate_queries(self) -> None:
        """Validate query definitions.

        Raises:
            ValueError: If queries are invalid
        """
        if not self.queries:
            raise ValueError("llm_query plugin requires at least one query")

        seen_output_keys: set[str] = set()

        for i, query in enumerate(self.queries):
            # Check required fields
            if "name" not in query:
                raise ValueError(f"Query at index {i} missing required field 'name'")
            if "output_key" not in query:
                raise ValueError(
                    f"Query '{query.get('name', f'index {i}')}' missing required field 'output_key'"
                )

            # Check for duplicate output_keys
            output_key = query["output_key"]
            if output_key in seen_output_keys:
                raise ValueError(
                    f"Duplicate output_key '{output_key}' in query '{query['name']}'. "
                    "Each query must have a unique output_key to avoid context collisions."
                )
            seen_output_keys.add(output_key)

    def _build_middlewares(self) -> list:
        """Build middleware chain from config.

        Returns:
            List of middleware instances
        """
        middlewares = []
        for mw_config in self._middlewares_config:
            mw = create_middleware(mw_config)
            if mw:
                middlewares.append(mw)
        return middlewares

    def _build_rate_limiter(self):
        """Build rate limiter from config."""
        if not self._rate_limiter_config:
            return None
        return create_rate_limiter(self._rate_limiter_config)

    def _build_cost_tracker(self):
        """Build cost tracker from config."""
        if not self._cost_tracker_config:
            return None
        return create_cost_tracker(self._cost_tracker_config)

    def _resolve_prompts(self) -> tuple[str, str]:
        """Resolve prompts from pack or inline config.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if self._prompt_pack:
            pack = resolve_remote_pack(self._prompt_pack)
            prompts = pack.get("prompts", {})
            self.pack_defaults = pack.get("prompt_defaults", {})
            self.pack_config = pack  # Store full pack for lookups
            return (
                prompts.get("system", self._system_prompt or ""),
                prompts.get("user", self._user_prompt or ""),
            )

        return self._system_prompt or "", self._user_prompt or ""

    def _execute_single_query(
        self,
        query: dict[str, Any],
        row: dict[str, Any],
        rendered_system: str,
        skip_rate_limit: bool = False,
    ) -> tuple[str, dict[str, Any], bool]:
        """Execute a single query and return (output_key, response, flatten_to_row).

        Args:
            query: Query definition
            row: Current row data (read-only in parallel context)
            rendered_system: Pre-rendered system prompt
            skip_rate_limit: Skip rate limiter (for pre-acquired slots)

        Returns:
            Tuple of (output_key, response_dict)
        """
        query_name = query["name"]
        output_key = query["output_key"]
        logger.debug("Executing query '%s' -> output_key '%s'", query_name, output_key)

        # Build render context: pack_defaults < query.defaults < mapped inputs
        render_ctx: dict[str, Any] = {**self.pack_defaults}
        render_ctx.update(query.get("defaults", {}))

        # Map inputs from row to template variables
        for template_var, row_field in query.get("inputs", {}).items():
            render_ctx[template_var] = row.get(row_field)

        # Process lookups: resolve values from pack config dicts
        for target_var, lookup_config in query.get("lookups", {}).items():
            source_dict_name = lookup_config.get("from")
            key_var = lookup_config.get("key")
            nested_key_var = lookup_config.get("nested_key")
            default = lookup_config.get("default", "")

            if source_dict_name and key_var:
                source_dict = self.pack_config.get(source_dict_name, {})
                key_value = render_ctx.get(key_var, "")
                result = source_dict.get(key_value, {})

                # Handle nested lookup if specified
                if nested_key_var and isinstance(result, dict):
                    nested_key_value = render_ctx.get(nested_key_var, "")
                    result = result.get(nested_key_value, default)
                elif not isinstance(result, str):
                    result = default

                render_ctx[target_var] = result if result else default
                logger.debug(
                    "Lookup '%s': %s[%s]%s = %s",
                    target_var,
                    source_dict_name,
                    key_value,
                    f"[{nested_key_var}]" if nested_key_var else "",
                    str(render_ctx[target_var])[:50] if render_ctx[target_var] else "(empty)",
                )

        # Render user prompt
        user_prompt = self.engine.render(self.user_template, render_ctx)

        # Execute LLM call
        response = self.executor.execute(
            user_prompt=user_prompt,
            metadata={"query": query_name},
            system_prompt=rendered_system,
            skip_rate_limit=skip_rate_limit,
        )

        logger.debug(
            "Query '%s' completed: content_length=%d",
            query_name,
            len(response.get("content", "")),
        )

        return output_key, response, query.get("flatten_to_row", True)

    def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute LLM queries and update row/context.

        Args:
            row: Current row data
            context: Shared context for inter-plugin communication

        Returns:
            Updated row data with flattened response fields
        """
        # Render system prompt once (shared across queries)
        rendered_system = self.engine.render(self.system_template, row)

        if self._parallel_queries and len(self.queries) > 1:
            # Parallel execution with rate-limited launches
            results: list[tuple[str, dict[str, Any], bool]] = []
            max_workers = min(self._max_parallel, len(self.queries))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for query in self.queries:
                    # Pre-acquire rate limit slot before launching
                    # This blocks until a slot is available, pacing the launches
                    if self.executor.rate_limiter:
                        with self.executor.rate_limiter.acquire():
                            pass  # Slot acquired and registered in sliding window

                    # Copy context per-task to propagate contextvars (e.g., landscape)
                    # to worker threads. Each ctx.run() needs its own copy since
                    # a Context can only be entered once at a time.
                    ctx = contextvars.copy_context()
                    future = executor.submit(
                        ctx.run,
                        self._execute_single_query,
                        query,
                        row,
                        rendered_system,
                        True,  # skip_rate_limit - Already acquired above
                    )
                    futures[future] = query

                for future in as_completed(futures):
                    query = futures[future]
                    try:
                        output_key, response, flatten = future.result()
                        results.append((output_key, response, flatten))
                    except Exception as e:
                        logger.error("Query '%s' failed: %s", query.get("name"), e)
                        raise

            # Apply results to context and row
            for output_key, response, flatten in results:
                context[output_key] = response
                if flatten:
                    for key, value in response.items():
                        if key not in ("metrics", "retry"):
                            row[f"{output_key}_{key}"] = value
        else:
            # Sequential execution (single query or parallel disabled)
            for query in self.queries:
                output_key, response, flatten = self._execute_single_query(
                    query, row, rendered_system
                )
                context[output_key] = response
                if flatten:
                    for key, value in response.items():
                        if key not in ("metrics", "retry"):
                            row[f"{output_key}_{key}"] = value

        return row


def _llm_query_factory(options: dict[str, Any]) -> LLMQueryPlugin:
    """Factory function for creating LLMQueryPlugin instances."""
    return LLMQueryPlugin(**options)


# Register plugin
register_transform_plugin("llm_query", _llm_query_factory)
