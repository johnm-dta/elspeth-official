# FieldCollector Plugin

**Type:** Meta-Plugin (Aggregation)
**Category:** Data Structure Transformation
**Mode:** Object -> Collection

---

## Overview

FieldCollector is a meta-plugin that transposes row-based data into columnar format (collection type). It converts an array of row objects into a dictionary of field arrays, enabling efficient batch analytics.

**Key characteristics:**
- Transforms data structure (not content)
- Enables collection-mode analytics
- Runs during aggregation phase
- Output stored in aggregates dict (not final result)

---

## Schema

### Input Schema

```python
input_schema = {
    "type": "array",
    "items": {"type": "object"}
}
```

**Accepts:** Array of row objects (all processed rows)

### Output Schema

```python
output_schema = {
    "type": "collection"
}
```

**Produces:** Collection type (dict of arrays)

**Note:** `item_schema` omitted because field types are dynamic (inferred from input rows)

---

## Configuration

### Required Options

- **`output_key`** (string): Key where collection will be stored in `aggregates` dict
  - Other plugins reference this key via their `input_key` option
  - Must be unique within aggregation pipeline

### Optional Options

- **`exclude_fields`** (array of strings): Fields to exclude from collection
  - Default: `[]` (collect all fields)
  - Use case: Exclude verbose fields (prompts, responses, metadata)

---

## Configuration Example

```yaml
aggregation_plugins:
  - plugin: field_collector
    options:
      output_key: collected_scores
      exclude_fields:
        - prompt
        - llm_response
        - debug_info
```

---

## Behavior

### Basic Transposition

**Input (3 rows):**
```python
[
    {"score": 10.5, "name": "Alice", "category": "A"},
    {"score": 12.3, "name": "Bob", "category": "B"},
    {"score": 9.8, "name": "Carol", "category": "A"}
]
```

**Output (collection):**
```python
{
    "score": [10.5, 12.3, 9.8],
    "name": ["Alice", "Bob", "Carol"],
    "category": ["A", "B", "A"]
}
```

### With Field Exclusion

**Input:**
```python
[
    {"score": 10.5, "name": "Alice", "prompt": "Rate this..."},
    {"score": 12.3, "name": "Bob", "prompt": "Rate this..."}
]
```

**Configuration:**
```yaml
exclude_fields: [prompt]
```

**Output:**
```python
{
    "score": [10.5, 12.3],
    "name": ["Alice", "Bob"]
    # "prompt" excluded
}
```

### Nested Field Flattening

FieldCollector recursively flattens nested dictionaries:

**Input:**
```python
[
    {"id": 1, "scores": {"quality": 0.8, "accuracy": 0.9}},
    {"id": 2, "scores": {"quality": 0.7, "accuracy": 0.95}}
]
```

**Output:**
```python
{
    "id": [1, 2],
    "quality": [0.8, 0.7],      # Flattened from scores.quality
    "accuracy": [0.9, 0.95]     # Flattened from scores.accuracy
}
```

### Missing Fields

Rows with missing fields get `None` values:

**Input:**
```python
[
    {"score": 10.5, "name": "Alice"},
    {"score": 12.3},  # Missing "name"
    {"score": 9.8, "name": "Carol"}
]
```

**Output:**
```python
{
    "score": [10.5, 12.3, 9.8],
    "name": ["Alice", None, "Carol"]
}
```

---

## Usage Patterns

### Pattern 1: Single Analyzer

Collect once, analyze once:

```yaml
aggregation_plugins:
  - plugin: field_collector
    options:
      output_key: scores

  - plugin: score_stats_analyzer
    options:
      input_key: scores
      score_field: delta
```

### Pattern 2: Multiple Analyzers (Recommended)

Collect once, multiple analyses:

```yaml
aggregation_plugins:
  - plugin: field_collector
    options:
      output_key: scores

  # All analyzers read from same collection
  - plugin: score_stats_analyzer
    options:
      input_key: scores
      score_field: delta

  - plugin: score_distribution_analyzer
    options:
      input_key: scores
      score_field: delta

  - plugin: score_variant_ranking_analyzer
    options:
      input_key: scores
      score_field: delta
```

**Benefit:** Efficient - collection happens once, shared by all analyzers.

### Pattern 3: Selective Collection

Collect only needed fields:

```yaml
aggregation_plugins:
  - plugin: field_collector
    options:
      output_key: minimal_data
      exclude_fields:
        - prompt
        - response
        - llm_response
        - metadata
```

---

## Validation

### Compile-Time Validation

Pipeline validator checks:
- `output_key` is unique (no duplicate keys in aggregates)
- Plugins following FieldCollector expect collection type

### Runtime Validation

- Input must be array of objects
- All row objects must be dicts
- Field types consistent across rows (numbers don't mix with strings)

### Error Messages

**Missing output_key:**
```
Configuration Error: FieldCollector

Missing required option: output_key

Add to configuration:
  - plugin: field_collector
    options:
      output_key: "collected_data"
```

**Type inconsistency:**
```
Runtime Validation Error (FieldCollector):

Field: score
Row 5: Expected number, got string ("12.3")
Previous rows had: number

Suggestions:
  - Check data source for type inconsistencies
  - Add type conversion plugin before FieldCollector
  - Update row plugins to ensure consistent output types
```

---

## Performance

- **Time complexity:** O(rows x fields)
- **Memory:** O(rows x fields x field_size)
- **Optimization:** Exclude unused fields to reduce memory

**Example memory usage:**
- 1000 rows x 20 fields x 8 bytes (float) = ~160 KB
- 1000 rows x 5 fields (after exclusion) = ~40 KB (4x reduction)

---

## API Reference

### Python API

```python
from elspeth.plugins.transforms.field_collector import FieldCollector

# Create instance
collector = FieldCollector({
    "output_key": "collected_data",
    "exclude_fields": ["prompt"]
})

# Aggregate rows
results = [
    {"score": 10.5, "name": "Alice"},
    {"score": 12.3, "name": "Bob"}
]
aggregates = {}

collection = collector.aggregate(results, aggregates)
# Returns: {"score": [10.5, 12.3], "name": ["Alice", "Bob"]}

# Collection also stored in aggregates
assert aggregates["collected_data"] == collection
```

---

## See Also

- **FieldExpander**: Inverse operation (collection->object)
- **Metrics Analyzers**: Plugins that consume collections
- **Schema System**: `docs/schema-driven-execution.md`
- **Migration Guide**: `docs/migration-v3.0-metrics.md`
