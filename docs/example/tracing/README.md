# Tracing Demo (OpenRouter)

Run a small workflow that emits LLM traces via the `tracing` middleware.

Prereqs:
- `OPENROUTER_API_KEY` set
- `OPENROUTER_MODEL` set (e.g. `openai/gpt-4o-mini`)
- Optional: `opentelemetry-sdk` installed and an OTEL exporter configured if you want OTEL spans instead of JSONL.

Run:
```bash
elspeth --settings example/tracing/settings.yaml
```

Outputs:
- Results CSV: `example/tracing/output/results.csv`
- Traces: `example/tracing/output/llm_traces.jsonl` (when `sink: jsonl`), or OTEL spans if `sink: otel` and exporters are configured.
