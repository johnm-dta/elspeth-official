"""Command-line entry point for local experimentation.

For now the CLI focuses on hydrating experiment input data from Azure Blob
Storage using the configuration profiles defined in ``config/blob_store.yaml``.
Future work will layer in the experiment runner once additional modules land.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from elspeth.config import load_settings
from elspeth.core.config_merger import ConfigSource, ConfigurationMerger
from elspeth.core.diagnostics import DiagnosticsWriter
from elspeth.core.landscape import RunLandscape
from elspeth.core.orchestrator import SDAOrchestrator
from elspeth.core.sda import SDASuite
from elspeth.core.secrets import SecretsError
from elspeth.core.validation import ConfigurationError, validate_settings, validate_suite
from elspeth.orchestrators import ExperimentalOrchestrator, StandardOrchestrator
from elspeth.plugins.outputs.csv_file import CsvResultSink

logger = logging.getLogger(__name__)


def load_dotenv() -> None:
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if not env_file.exists():
        return

    try:
        with env_file.open() as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse key=value pairs
                if "=" in line:
                    key, value = line.split("=", 1)
                    # Only set if not already in environment (env vars take precedence)
                    if key not in os.environ:
                        os.environ[key] = value
        logger.debug(f"Loaded environment variables from {env_file}")
    except Exception as e:
        logger.warning(f"Failed to load .env file: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DMP data bootstrap CLI")
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to orchestrator settings YAML",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Settings profile to load",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Limit processing to first N rows (useful for testing)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional local path to persist the downloaded dataset",
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        help="Override suite root directory (if unset, uses settings)",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Force single experiment run even if suite settings exist",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set logging verbosity",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable metrics/statistical plugins from the loaded settings",
    )
    parser.add_argument(
        "--live-outputs",
        action="store_true",
        help="Allow sinks to perform live writes (disables repo dry-run modes)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Clear checkpoint file and start fresh (ignores previous progress)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint (continue where you left off)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved configuration and exit (no execution)",
    )
    parser.add_argument(
        "--explain-config",
        type=str,
        metavar="KEY",
        help="Explain source of specific config key (e.g., 'rate_limiter' or 'llm.options.temperature')",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="BUNDLE",
        help="Verify a signed archive bundle (path to .zip file)",
    )
    parser.add_argument(
        "--verify-key-env",
        type=str,
        default="DMP_ARCHIVE_SIGNING_KEY",
        help="Environment variable containing the verification key (default: DMP_ARCHIVE_SIGNING_KEY)",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        help="Path to secrets YAML file for variable substitution",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def format_preview(df: pd.DataFrame, head: int) -> str:
    """Convert the dataframe preview into a printable string."""

    preview = df.head(head) if head > 0 else df.head(0)
    with pd.option_context("display.max_columns", None):
        return str(preview.to_string(index=False))


def _flatten_value(target: dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else key
            _flatten_value(target, next_prefix, inner)
    else:
        target[prefix] = value


def _result_to_row(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record.get("row") or {})

    def consume_response(prefix: str, response: Mapping[str, Any] | None) -> None:
        if not response:
            return
        content = response.get("content")
        if content is not None:
            row[prefix] = content
        metrics = response.get("metrics")
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                _flatten_value(row, f"{prefix}_metric_{key}", value)

    consume_response("llm_content", record.get("response"))
    for name, response in (record.get("responses") or {}).items():
        consume_response(f"llm_{name}", response)

    for key, value in (record.get("metrics") or {}).items():
        _flatten_value(row, f"metric_{key}", value)

    retry_info = record.get("retry")
    if retry_info:
        row["retry_attempts"] = retry_info.get("attempts")
        row["retry_max_attempts"] = retry_info.get("max_attempts")
        history = retry_info.get("history")
        if history:
            row["retry_history"] = json.dumps(history)

    if "security_level" in record:
        row["security_level"] = record["security_level"]

    return row


def _print_configuration(args: argparse.Namespace, settings) -> None:
    """Print resolved configuration for debugging."""
    import yaml

    # Build configuration dictionary for display
    config_dict = {
        "datasource": {
            "plugin": type(settings.datasource).__name__,
        },
        "prompts": settings.orchestrator_config.llm_prompt,
        "row_plugins": settings.orchestrator_config.transform_plugin_defs,
        "aggregator_plugins": settings.orchestrator_config.aggregation_transform_defs,
        "baseline_plugins": settings.orchestrator_config.baseline_plugin_defs,
        "sinks": [{"plugin": type(sink).__name__} for sink in settings.sinks],
    }

    # Add optional llm field
    if settings.llm:
        config_dict["llm"] = {"plugin": type(settings.llm).__name__}

    # Add optional fields
    if settings.orchestrator_config.prompt_fields:
        config_dict["prompt_fields"] = settings.orchestrator_config.prompt_fields
    if settings.orchestrator_config.prompt_aliases:
        config_dict["prompt_aliases"] = settings.orchestrator_config.prompt_aliases
    if settings.orchestrator_config.criteria:
        config_dict["criteria"] = settings.orchestrator_config.criteria
    if settings.orchestrator_config.concurrency_config:
        config_dict["concurrency"] = settings.orchestrator_config.concurrency_config
    if settings.orchestrator_config.retry_config:
        config_dict["retry"] = settings.orchestrator_config.retry_config
    if settings.orchestrator_config.checkpoint_config:
        config_dict["checkpoint"] = settings.orchestrator_config.checkpoint_config
    if settings.orchestrator_config.halt_condition_config:
        config_dict["early_stop"] = settings.orchestrator_config.halt_condition_config
    if settings.orchestrator_config.halt_condition_plugin_defs:
        config_dict["early_stop_plugins"] = settings.orchestrator_config.halt_condition_plugin_defs
    if settings.orchestrator_config.llm_middleware_defs:
        config_dict["llm_middlewares"] = settings.orchestrator_config.llm_middleware_defs
    if settings.orchestrator_config.prompt_defaults:
        config_dict["prompt_defaults"] = settings.orchestrator_config.prompt_defaults
    if settings.rate_limiter:
        config_dict["rate_limiter"] = type(settings.rate_limiter).__name__
    if settings.cost_tracker:
        config_dict["cost_tracker"] = type(settings.cost_tracker).__name__
    if settings.suite_root:
        config_dict["suite_root"] = str(settings.suite_root)
    if settings.suite_defaults:
        config_dict["suite_defaults"] = settings.suite_defaults
    if settings.prompt_pack:
        config_dict["prompt_pack"] = settings.prompt_pack

    print("# Resolved Configuration")
    print(f"# Loaded from: {args.settings} (profile: {args.profile})")
    print()

    if args.explain_config:
        # Explain specific key
        print(f"Configuration key: {args.explain_config}")
        print()
        print("Note: Full explain() support requires preserving ConfigurationMerger instance.")
        print("Currently showing resolved value only:")
        print()

        # Navigate to the key value
        keys = args.explain_config.split(".")
        value = config_dict
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                print(f"{args.explain_config} = <not found>")
                return

        print(f"{args.explain_config} = {value}")
    else:
        # Print full config
        print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))


def _run_verify(args: argparse.Namespace) -> None:
    """Verify a signed archive bundle."""
    import json

    from elspeth.core.security.verify import verify_bundle_cli

    # Check if bundle uses RSA (doesn't need key) or HMAC (needs key)
    signature_path = args.verify.with_suffix(".signature.json")
    algorithm = "hmac-sha256"  # default

    if signature_path.exists():
        try:
            sig_data = json.loads(signature_path.read_text())
            algorithm = sig_data.get("algorithm", "hmac-sha256")
        except Exception:
            pass

    key = None
    if algorithm.startswith("hmac"):
        # HMAC requires a key
        key = os.environ.get(args.verify_key_env)
        if not key:
            print("Error: HMAC verification requires a signing key")
            print(f"Set it with: export {args.verify_key_env}=\"your-key-here\"")
            raise SystemExit(1)
    else:
        # RSA uses embedded public key
        print("Note: RSA-signed bundle - using embedded public key for verification")

    verify_bundle_cli(str(args.verify), key)


def run(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    # Handle --verify (standalone operation, doesn't need settings)
    if args.verify:
        _run_verify(args)
        return

    # Get secrets path if provided
    secrets_path = getattr(args, 'secrets', None)

    # Create diagnostics writer
    diagnostics = DiagnosticsWriter(Path("diagnostics"))

    try:
        settings_report = validate_settings(args.settings, profile=args.profile)
        for warning in settings_report.warnings:
            logger.warning(warning.format())
        settings_report.raise_if_errors()

        settings = load_settings(args.settings, profile=args.profile, secrets_path=secrets_path)

        # Handle --print-config and --explain-config
        if args.print_config or args.explain_config:
            _print_configuration(args, settings)
            return
        if args.disable_metrics:
            _strip_metrics_plugins(settings)
        _configure_sink_dry_run(settings, enable_live=args.live_outputs)
        suite_root = args.suite_root or settings.suite_root

        if suite_root and not args.single_run:
            suite_validation = validate_suite(suite_root)
            for warning in suite_validation.report.warnings:
                logger.warning(warning.format())
            suite_validation.report.raise_if_errors()
            _run_suite(args, settings, suite_root, preflight=suite_validation.preflight)
        else:
            _run_single(args, settings)

    except SecretsError as e:
        diagnostics.write_error("SecretsError", str(e), {"settings_file": args.settings})
        diagnostics.write_stack_trace()
        logger.error("Secrets error: %s", e)
        logger.error("Diagnostics saved to: diagnostics/")
        raise SystemExit(1) from e
    except ConfigurationError as e:
        diagnostics.write_error("ConfigurationError", str(e), {"settings_file": args.settings})
        diagnostics.write_stack_trace()
        logger.error("Configuration error: %s", e)
        logger.error("Diagnostics saved to: diagnostics/")
        raise SystemExit(1) from e
    except Exception as e:
        diagnostics.write_error(type(e).__name__, str(e), {"settings_file": args.settings})
        diagnostics.write_stack_trace()
        logger.error("Unexpected error: %s", e)
        logger.error("Diagnostics saved to: diagnostics/")
        raise


def _run_single(args: argparse.Namespace, settings) -> None:
    # Check for existing checkpoint and require explicit choice
    checkpoint_config = settings.orchestrator_config.checkpoint_config
    if checkpoint_config:
        checkpoint_path = Path(checkpoint_config.get("path", "checkpoint.jsonl"))
        if checkpoint_path.exists() and not args.restart and not args.resume:
            # Count completed rows to show progress
            try:
                with open(checkpoint_path) as f:
                    completed_count = sum(1 for _ in f)
            except Exception:
                completed_count = "unknown"
            print(f"\n⚠️  Checkpoint data detected: {checkpoint_path}")
            print(f"   {completed_count} rows already processed")
            print(f"\nOptions:")
            print(f"   --resume   Resume from checkpoint (continue where you left off)")
            print(f"   --restart  Clear checkpoint and start fresh")
            print()
            raise SystemExit(1)

    # Handle --restart: clear checkpoint file
    if args.restart:
        if checkpoint_config:
            checkpoint_path = Path(checkpoint_config.get("path", "checkpoint.jsonl"))
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Cleared checkpoint file: %s", checkpoint_path)
                # Also clear checkpoint meta file if exists
                meta_path = Path(str(checkpoint_path) + ".meta")
                if meta_path.exists():
                    meta_path.unlink()
                    logger.info("Cleared checkpoint meta file: %s", meta_path)
        else:
            logger.warning("--restart specified but no checkpoint configured")

    logger.info("Running single experiment")

    # Set up landscape for artifact capture (config, inputs, LLM logs)
    landscape_config = getattr(settings, "landscape_config", {})
    landscape_path = landscape_config.get("path")
    landscape = RunLandscape(
        base_path=Path(landscape_path) if landscape_path else None,
        persist=landscape_config.get("persist", False),
        capture_llm_calls=landscape_config.get("capture_llm_calls", True),
        clean_before_run=landscape_config.get("clean_before_run", False),
    )

    orchestrator = SDAOrchestrator(
        datasource=settings.datasource,
        sinks=settings.sinks,
        config=settings.orchestrator_config,
        llm_client=settings.llm,
        rate_limiter=settings.rate_limiter,
        cost_tracker=settings.cost_tracker,
        row_limit=args.head,
    )

    with landscape:
        payload = orchestrator.run()

    for failure in payload.get("failures", []):
        retry = failure.get("retry") or {}
        attempts = retry.get("attempts")
        logger.error(
            "Row processing failed after %s attempts: %s",
            attempts if attempts is not None else 1,
            failure.get("error"),
        )

    rows = [_result_to_row(result) for result in payload["results"]]
    df = pd.DataFrame(rows)

    if args.output_csv:
        output_path: Path = args.output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved dataset to %s", output_path)

    # Print preview of results (show all if --head was used, otherwise first 5)
    if not df.empty:
        preview_rows = args.head if args.head else 5
        print(format_preview(df, preview_rows))


def _clone_suite_sinks(base_sinks: list, experiment_name: str) -> list:
    cloned = []
    for sink in base_sinks:
        if isinstance(sink, CsvResultSink):
            base_path = Path(sink.path)
            new_path = base_path.with_name(f"{experiment_name}_{base_path.name}")
            cloned.append(CsvResultSink(path=str(new_path), overwrite=True))
        else:
            cloned.append(sink)
    return cloned


def _run_suite(args: argparse.Namespace, settings, suite_root: Path, *, preflight: dict | None = None) -> None:
    """Run suite with appropriate orchestrator based on orchestrator_type."""
    logger.info("Running suite at %s", suite_root)
    suite = SDASuite.load(suite_root)
    df = settings.datasource.load()

    # Determine orchestrator type (default to experimental for backward compatibility)
    orchestrator_type = getattr(settings, "orchestrator_type", "experimental")
    if orchestrator_type == "standard":
        suite_runner = StandardOrchestrator(
            suite=suite,
            sinks=settings.sinks,
        )
        logger.info("Using StandardOrchestrator (simple sequential execution)")
    elif orchestrator_type == "experimental":
        suite_runner = ExperimentalOrchestrator(
            suite=suite,
            sinks=settings.sinks,
        )
        logger.info("Using ExperimentalOrchestrator (baseline comparison)")
    else:
        raise ValueError(f"Unknown orchestrator_type: {orchestrator_type}. Must be 'standard' or 'experimental'")

    # Use ConfigurationMerger for defaults
    merger = ConfigurationMerger()

    # Source 1: orchestrator config
    orch_config_data = {
        "prompt_system": settings.orchestrator_config.llm_prompt.get("system", ""),
        "prompt_template": settings.orchestrator_config.llm_prompt.get("user", ""),
        "prompt_fields": settings.orchestrator_config.prompt_fields,
        "criteria": settings.orchestrator_config.criteria,
        "prompt_packs": settings.prompt_packs,
    }

    # Add optional fields if present
    if settings.orchestrator_config.prompt_pack:
        orch_config_data["prompt_pack"] = settings.orchestrator_config.prompt_pack
    if settings.orchestrator_config.transform_plugin_defs:
        orch_config_data["row_plugin_defs"] = settings.orchestrator_config.transform_plugin_defs
    if settings.orchestrator_config.aggregation_transform_defs:
        orch_config_data["aggregator_plugin_defs"] = settings.orchestrator_config.aggregation_transform_defs
    if settings.orchestrator_config.baseline_plugin_defs:
        orch_config_data["baseline_plugin_defs"] = settings.orchestrator_config.baseline_plugin_defs
    if settings.orchestrator_config.sink_defs:
        orch_config_data["sink_defs"] = settings.orchestrator_config.sink_defs
    if settings.orchestrator_config.llm_middleware_defs:
        orch_config_data["llm_middleware_defs"] = settings.orchestrator_config.llm_middleware_defs
    if settings.orchestrator_config.prompt_defaults:
        orch_config_data["prompt_defaults"] = settings.orchestrator_config.prompt_defaults
    if settings.orchestrator_config.concurrency_config:
        orch_config_data["concurrency_config"] = settings.orchestrator_config.concurrency_config
    if settings.orchestrator_config.halt_condition_plugin_defs:
        orch_config_data["early_stop_plugin_defs"] = settings.orchestrator_config.halt_condition_plugin_defs
    if settings.orchestrator_config.halt_condition_config:
        orch_config_data["early_stop_config"] = settings.orchestrator_config.halt_condition_config

    # Source 2: suite_defaults (with key normalization)
    suite_defaults = settings.suite_defaults or {}
    suite_defaults_normalized = dict(suite_defaults)

    # Normalize plugin key names for consistency
    if "row_plugins" in suite_defaults_normalized:
        suite_defaults_normalized["row_plugin_defs"] = suite_defaults_normalized.pop("row_plugins")
    if "aggregator_plugins" in suite_defaults_normalized:
        suite_defaults_normalized["aggregator_plugin_defs"] = suite_defaults_normalized.pop("aggregator_plugins")
    if "baseline_plugins" in suite_defaults_normalized:
        suite_defaults_normalized["baseline_plugin_defs"] = suite_defaults_normalized.pop("baseline_plugins")
    if "llm_middlewares" in suite_defaults_normalized:
        suite_defaults_normalized["llm_middleware_defs"] = suite_defaults_normalized.pop("llm_middlewares")
    if "concurrency" in suite_defaults_normalized:
        suite_defaults_normalized["concurrency_config"] = suite_defaults_normalized.pop("concurrency")
    if "early_stop_plugins" in suite_defaults_normalized:
        suite_defaults_normalized["early_stop_plugin_defs"] = suite_defaults_normalized.pop("early_stop_plugins")
    if "early_stop" in suite_defaults_normalized:
        suite_defaults_normalized["early_stop_config"] = suite_defaults_normalized.pop("early_stop")
    if "sinks" in suite_defaults_normalized:
        suite_defaults_normalized["sink_defs"] = suite_defaults_normalized.pop("sinks")

    sources = [
        ConfigSource(name="orchestrator", data=orch_config_data, precedence=1),
        ConfigSource(name="suite_defaults", data=suite_defaults_normalized, precedence=2),
    ]

    defaults = merger.merge(*sources)

    # Add runtime instances (not part of merge)
    if settings.rate_limiter:
        defaults["rate_limiter"] = settings.rate_limiter
    if settings.cost_tracker:
        defaults["cost_tracker"] = settings.cost_tracker

    results = suite_runner.run(
        df,
        defaults=defaults,
        sink_factory=lambda exp: _clone_suite_sinks(settings.sinks, exp.name),
        preflight_info=preflight,
    )

    for name, entry in results.items():
        logger.info("Experiment %s completed with %s rows", name, len(entry["payload"]["results"]))


def _strip_metrics_plugins(settings) -> None:
    """Remove metrics plugins from settings and prompt packs when disabled."""

    row_names = {"score_extractor"}
    agg_names = {"score_stats", "score_recommendation"}
    baseline_names = {"score_delta"}

    def _filter(defs, names):
        if not defs:
            return defs
        return [entry for entry in defs if entry.get("name") not in names]

    cfg = settings.orchestrator_config
    cfg.row_plugin_defs = _filter(cfg.row_plugin_defs, row_names)
    cfg.aggregator_plugin_defs = _filter(cfg.aggregator_plugin_defs, agg_names)
    cfg.baseline_plugin_defs = _filter(cfg.baseline_plugin_defs, baseline_names)

    defaults = settings.suite_defaults or {}
    if "row_plugins" in defaults:
        defaults["row_plugins"] = _filter(defaults.get("row_plugins"), row_names)
    if "aggregator_plugins" in defaults:
        defaults["aggregator_plugins"] = _filter(defaults.get("aggregator_plugins"), agg_names)
    if "baseline_plugins" in defaults:
        defaults["baseline_plugins"] = _filter(defaults.get("baseline_plugins"), baseline_names)

    for pack in settings.prompt_packs.values():
        if isinstance(pack, dict):
            if "row_plugins" in pack:
                pack["row_plugins"] = _filter(pack.get("row_plugins"), row_names)
            if "aggregator_plugins" in pack:
                pack["aggregator_plugins"] = _filter(pack.get("aggregator_plugins"), agg_names)
            if "baseline_plugins" in pack:
                pack["baseline_plugins"] = _filter(pack.get("baseline_plugins"), baseline_names)


def _configure_sink_dry_run(settings, enable_live: bool) -> None:
    """Toggle dry-run behaviour for sinks supporting remote writes."""

    dry_run = not enable_live

    for sink in settings.sinks:
        if hasattr(sink, "dry_run"):
            sink.dry_run = dry_run

    def _update_defs(defs):
        if not defs:
            return defs
        updated = []
        for entry in defs:
            options = dict(entry.get("options", {}))
            if entry.get("plugin") in {"github_repo", "azure_devops_repo"} or "dry_run" in options:
                options["dry_run"] = dry_run
            updated.append({"plugin": entry.get("plugin"), "options": options})
        return updated

    config = settings.orchestrator_config
    config.sink_defs = _update_defs(config.sink_defs)

    suite_defaults = settings.suite_defaults or {}
    if "sinks" in suite_defaults:
        suite_defaults["sinks"] = _update_defs(suite_defaults.get("sinks"))

    for pack in settings.prompt_packs.values():
        if isinstance(pack, dict) and pack.get("sinks"):
            pack["sinks"] = _update_defs(pack.get("sinks"))


def main(argv: Iterable[str] | None = None) -> None:
    # Load .env file if it exists
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
