# Configuration Precedence Guide

This document explains how Elspeth merges configuration from multiple sources with clear precedence rules.

## Precedence Levels

Configuration is merged from 5 levels, from lowest to highest precedence:

| Level | Precedence | Source | Example |
|-------|------------|--------|---------|
| 1 | Lowest | System defaults | Built-in defaults |
| 2 | Low | Prompt pack | `prompt_packs.quality-evaluation` |
| 3 | Medium | Profile | `production:` section in YAML |
| 4 | High | Suite defaults | `suite_defaults:` section |
| 5 | Highest | Experiment config | Individual experiment settings |

**Rule:** Higher precedence overrides lower precedence (except for lists, see Merge Strategies below)

## Merge Strategies

Different configuration keys use different merge strategies:

### Override (Most Keys)

Scalar values use **override** strategy - higher precedence completely replaces lower precedence.

**Example:**
```yaml
# Prompt pack (precedence 2)
rate_limiter:
  requests_per_minute: 10

# Profile (precedence 3)
rate_limiter:
  requests_per_minute: 50

# Result: 50 (profile overrides pack)
```

### Append (Plugin Lists)

List keys use **append** strategy - values accumulate from all sources.

**Keys using append:**
- `row_plugins`
- `aggregator_plugins`
- `baseline_plugins`
- `llm_middlewares`
- `sinks`
- `early_stop_plugins`

**Example:**
```yaml
# Prompt pack (precedence 2)
row_plugins:
  - name: score_extractor

# Profile (precedence 3)
row_plugins:
  - name: custom_metric

# Result: [score_extractor, custom_metric] (both included)
```

### Deep Merge (Nested Dicts)

Nested dictionary keys use **deep merge** strategy - recursively merge nested structures.

**Keys using deep merge:**
- `llm`
- `datasource`
- `prompts`
- `orchestrator`
- `concurrency`
- `retry`
- `checkpoint`
- `early_stop`

**Example:**
```yaml
# Prompt pack (precedence 2)
llm:
  plugin: azure_openai
  options:
    temperature: 0.5
    max_tokens: 500

# Profile (precedence 3)
llm:
  options:
    temperature: 0.7  # Override
    top_p: 0.9        # Add new

# Result:
llm:
  plugin: azure_openai
  options:
    temperature: 0.7    # From profile (override)
    max_tokens: 500     # From pack (preserved)
    top_p: 0.9          # From profile (added)
```

## Debugging Configuration

### Print Resolved Configuration

```bash
elspeth --settings config.yaml --profile production --print-config
```

Shows the final merged configuration without running experiments.

### Explain Specific Key

```bash
elspeth --settings config.yaml --explain-config rate_limiter
```

Shows the resolved value for a specific key. For nested keys, use dot notation:

```bash
elspeth --settings config.yaml --explain-config llm.options.temperature
```

## Common Mistakes

### Mistake 1: Unexpected Override

**Problem:** Set rate limit in profile, but prompt pack value is used

**Cause:** Prompt packs have lower precedence than profiles

**Solution:** Check precedence order - profile (3) should override pack (2). If the pack value is being used, verify your YAML structure is correct.

### Mistake 2: Plugins Not Accumulating

**Problem:** Profile plugins replace pack plugins instead of adding to them

**Cause:** Using wrong YAML syntax

**Correct:**
```yaml
row_plugins:  # List syntax
  - name: plugin1
```

**Incorrect:**
```yaml
row_plugins: {name: plugin1}  # Dict syntax (will override)
```

### Mistake 3: Nested Config Confusion

**Problem:** Can't tell if deep merge or override is happening

**Solution:** Use `--print-config` to see final merged result:

```bash
elspeth --settings config.yaml --print-config
```

### Mistake 4: Suite Defaults Not Working

**Problem:** Suite defaults aren't being applied to experiments

**Cause:** Suite defaults require proper key names

**Solution:** Ensure suite_defaults uses correct key names:
- Use `row_plugin_defs` or `row_plugins` (both work)
- Use `concurrency_config` or `concurrency` (both work)

## How Configuration Flows

### Single Experiment Run

```
System Defaults (1)
  ↓
Prompt Pack (2)
  ↓
Profile (3)
  ↓
[Final Configuration]
```

### Suite Run

```
System Defaults (1)
  ↓
Prompt Pack (2)
  ↓
Profile (3)
  ↓
Suite Defaults (4)
  ↓
Experiment Config (5)
  ↓
[Final Configuration for Each Experiment]
```

## Troubleshooting

### Configuration Not Loading

1. **Check file paths**: Ensure `--settings` points to correct file
2. **Check profile name**: Verify profile exists in YAML
3. **Check YAML syntax**: Invalid YAML fails silently sometimes

### Values Not Merging As Expected

1. **Run with --print-config**: See actual merged result
2. **Check data types**: Lists use append, scalars use override
3. **Check precedence**: Higher numbers override lower
4. **Check key names**: Some keys have aliases (row_plugins vs row_plugin_defs)

### Prompt Pack Not Applied

1. **Verify prompt_pack name**: Must match key in `prompt_packs` section
2. **Check precedence**: Profile values (3) override pack values (2)
3. **Check list merging**: Plugins should accumulate, not replace

## Implementation Details

Configuration merging is implemented in `src/elspeth/core/config_merger.py`.

The `ConfigurationMerger` class provides:
- `merge(*sources)`: Merge multiple config sources with precedence
- `explain(key, config)`: Explain where a config value comes from
- Documented merge strategies for each key type

### Key Classes

```python
from elspeth.core.config_merger import (
    ConfigurationMerger,
    ConfigSource,
    MergeStrategy
)

# Create merger
merger = ConfigurationMerger()

# Define sources
pack = ConfigSource(name="prompt_pack", data={...}, precedence=2)
profile = ConfigSource(name="profile", data={...}, precedence=3)

# Merge
result = merger.merge(pack, profile)

# Explain
explanation = merger.explain("rate_limiter", result)
```

## Advanced Topics

### Custom Merge Strategies

To add custom merge strategies, modify `MERGE_STRATEGIES` in `ConfigurationMerger`:

```python
MERGE_STRATEGIES = {
    "my_custom_key": MergeStrategy.DEEP_MERGE,
    # ... other strategies
}
```

### Preserving Merge History

For debugging, the merger maintains `_merge_trace` showing source of each value:

```python
merger = ConfigurationMerger()
result = merger.merge(source1, source2)

# Check trace
for entry in merger._merge_trace:
    print(f"{entry['key']} from {entry['source']}")
```

## See Also

- [Environment Variables](ENVIRONMENT_VARIABLES.md)
- [Migration Guide](MIGRATION_TO_SRC_LAYOUT.md)
- [OpenRouter Setup](OPENROUTER_SETUP.md)
