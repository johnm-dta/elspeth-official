# FieldExpander Plugin

**Type:** Meta-Plugin (Aggregation)
**Category:** Data Structure Transformation
**Mode:** Collection -> Object

---

## Overview

FieldExpander is a meta-plugin that transposes columnar data back into flat object fields. It converts a collection (dict of arrays) into a flattened object with indexed field names, enabling CSV output of collection data.

**Key characteristics:**
- Inverse of FieldCollector
- Transforms data structure (not content)
- Flattens collection arrays into indexed object fields
- Runs during aggregation phase

---

## Schema

### Input Schema

```python
input_schema = {
    "type": "collection",
    "item_schema": {"type": "object"}
}
```

**Accepts:** Collection type (dict of arrays)

### Output Schema

```python
output_schema = {
    "type": "array",
    "items": {"type": "object"}
}
```

**Produces:** Array of flat objects (for CSV compatibility)

---

## Configuration

### Required Options

- **`input_key`** (string): Key where collection is stored in `aggregates` dict
  - References output from another plugin (e.g., FieldCollector, multi-field analyzer)

---

## Configuration Example

```yaml
aggregation_plugins:
  - plugin: field_collector
    options:
      output_key: stats_collection

  - plugin: multi_field_stats_analyzer
    options:
      input_key: stats_collection
      fields: [delta, ratio]
      output_key: computed_stats  # Produces collection

  - plugin: field_expander
    options:
      input_key: computed_stats  # Expand collection to flat object
```

---

## Behavior

### Basic Expansion

**Input (collection):**
```python
{
    "mean": [10.5, 12.3],
    "std": [1.2, 0.8]
}
```

**Output (flat object):**
```python
{
    "mean_0": 10.5,
    "mean_1": 12.3,
    "std_0": 1.2,
    "std_1": 0.8
}
```

**Field naming:** `{field_name}_{index}`

### Single-Value Arrays

**Input:**
```python
{
    "total_mean": [42.5],
    "total_std": [3.2]
}
```

**Output:**
```python
{
    "total_mean_0": 42.5,
    "total_std_0": 3.2
}
```

### Empty Arrays

**Input:**
```python
{
    "scores": [],
    "names": []
}
```

**Output:**
```python
{}  # No fields added
```

---

## Usage Patterns

### Pattern 1: Collection -> CSV Output

Enable CSV export of collection data:

```yaml
aggregation_plugins:
  - plugin: multi_field_stats_analyzer
    options:
      input_key: collected_data
      fields: [baseline_score, variant_score]
      output_key: stats_collection

  - plugin: field_expander
    options:
      input_key: stats_collection

sinks:
  - plugin: csv
    options:
      path: output/stats.csv
      # CSV can now write flattened fields
```

**Output CSV:**
```
baseline_score_mean_0,baseline_score_std_0,variant_score_mean_0,variant_score_std_0
10.5,1.2,12.3,0.8
```

### Pattern 2: Multiple Collections

Expand multiple collections:

```yaml
aggregation_plugins:
  - plugin: field_expander
    options:
      input_key: stats_collection_1

  - plugin: field_expander
    options:
      input_key: stats_collection_2
```

**Result:** Both collections flattened into same final object

---

## Validation

### Compile-Time Validation

Pipeline validator checks:
- `input_key` references existing collection in aggregates
- Plugin providing collection outputs collection type

### Runtime Validation

- Input must be dict of arrays (collection type)
- All arrays must have valid length
- Field names must be valid identifiers

### Error Messages

**Missing input_key:**
```
Runtime Error: FieldExpander

input_key 'stats_collection' not found in aggregates.
Available keys: ['collected_data', 'baseline_stats']

Suggestions:
  - Check output_key of previous plugin matches input_key
  - Ensure plugin providing collection runs before FieldExpander
  - Verify aggregation pipeline order
```

**Invalid collection type:**
```
Runtime Validation Error: FieldExpander

Expected collection type (dict of arrays)
Got: object

Input from: stats_analyzer (aggregates['stats_collection'])

Suggestions:
  - Check stats_analyzer output_schema declares collection type
  - Verify stats_analyzer actually returns dict of arrays
  - Use FieldCollector if you need to create collection first
```

---

## Performance

- **Time complexity:** O(fields x array_length)
- **Memory:** O(fields x array_length x field_size)
- **Optimization:** Minimal overhead (simple iteration)

**Example:**
- Collection: 10 fields x 5 values = 50 scalar values
- Output object: 50 indexed fields
- Memory: Same as input (no duplication)

---

## API Reference

### Python API

```python
from elspeth.plugins.transforms.field_expander import FieldExpander

# Create instance
expander = FieldExpander({
    "input_key": "stats_collection"
})

# Setup aggregates with collection
aggregates = {
    "stats_collection": {
        "mean": [10.5, 12.3],
        "std": [1.2, 0.8]
    }
}

# Expand collection
result = expander.aggregate([], aggregates)
# Returns: [{"mean_0": 10.5, "mean_1": 12.3, "std_0": 1.2, "std_1": 0.8}]
```

---

## See Also

- **FieldCollector**: Inverse operation (object->collection)
- **CSV Sink**: Primary use case for FieldExpander
- **Schema System**: `docs/schema-driven-execution.md`
