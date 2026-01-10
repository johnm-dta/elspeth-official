# Data Reshape Plugins

**Type:** Transform Plugins
**Category:** Data Manipulation
**Modes:** Row (RowDataReshape) and Collection (AggregateDataReshape)

---

## Overview

Data reshape plugins provide configurable data transformations without writing custom code. They apply operations like `flatten`, `rename`, `filter_fields`, `cast` to restructure data.

**Two variants:**
1. **RowDataReshape**: Row-by-row transformations (object mode)
2. **AggregateDataReshape**: Collection-level transformations (collection mode)

---

## RowDataReshape

### Schema

**Input:** `{"type": "object"}`
**Output:** `{"type": "object"}` (dynamic based on operations)

### Configuration

```yaml
row_plugins:
  - plugin: row_data_reshape
    options:
      operations:
        - operation: flatten
          field: nested_data
        - operation: rename
          from: old_field
          to: new_field
        - operation: filter_fields
          keep: [field1, field2, field3]
```

### Operations

#### 1. Flatten

Flatten nested object to top-level fields:

**Input:**
```python
{"id": 1, "scores": {"quality": 0.8, "accuracy": 0.9}}
```

**Operation:**
```yaml
- operation: flatten
  field: scores
```

**Output:**
```python
{"id": 1, "quality": 0.8, "accuracy": 0.9}
```

#### 2. Rename

Rename field:

**Input:**
```python
{"old_name": "value", "other": "data"}
```

**Operation:**
```yaml
- operation: rename
  from: old_name
  to: new_name
```

**Output:**
```python
{"new_name": "value", "other": "data"}
```

#### 3. Filter Fields (Keep)

Keep only specified fields:

**Input:**
```python
{"a": 1, "b": 2, "c": 3, "d": 4}
```

**Operation:**
```yaml
- operation: filter_fields
  keep: [a, c]
```

**Output:**
```python
{"a": 1, "c": 3}
```

#### 4. Exclude Fields

Remove specified fields:

**Input:**
```python
{"a": 1, "b": 2, "c": 3}
```

**Operation:**
```yaml
- operation: exclude_fields
  fields: [b]
```

**Output:**
```python
{"a": 1, "c": 3}
```

#### 5. Extract

Extract nested field to top-level:

**Input:**
```python
{"metadata": {"user_id": 123, "timestamp": "2025-11-23"}, "score": 10}
```

**Operation:**
```yaml
- operation: extract
  from: metadata.user_id
  to: user_id
```

**Output:**
```python
{"metadata": {...}, "score": 10, "user_id": 123}
```

#### 6. Cast

Convert field type:

**Input:**
```python
{"score": "10.5", "count": "42"}
```

**Operation:**
```yaml
- operation: cast
  field: score
  type: float
- operation: cast
  field: count
  type: int
```

**Output:**
```python
{"score": 10.5, "count": 42}
```

**Supported types:** `int`, `float`, `str`, `bool`

---

## AggregateDataReshape

### Schema

**Input:** `{"type": "collection"}`
**Output:** `{"type": "collection"}` (dynamic)

### Configuration

```yaml
aggregation_plugins:
  - plugin: aggregate_data_reshape
    options:
      input_key: collected_data
      output_key: reshaped_data
      operations:
        - operation: rename
          from: old_field
          to: new_field
        - operation: filter_fields
          keep: [field1, field2]
```

### Operations

Same operations as RowDataReshape, but applied to collection structure:

**Input (collection):**
```python
{
    "old_name": [1, 2, 3],
    "other": [4, 5, 6]
}
```

**Operation:**
```yaml
- operation: rename
  from: old_name
  to: new_name
```

**Output (collection):**
```python
{
    "new_name": [1, 2, 3],
    "other": [4, 5, 6]
}
```

---

## Operation Chaining

Operations execute in sequence:

```yaml
operations:
  - operation: flatten
    field: nested_scores
  - operation: rename
    from: nested_scores_quality
    to: quality_score
  - operation: cast
    field: quality_score
    type: float
  - operation: filter_fields
    keep: [quality_score, accuracy_score]
```

**Execution order matters:**
1. Flatten creates `nested_scores_quality`
2. Rename changes to `quality_score`
3. Cast converts to float
4. Filter keeps only specified fields

---

## Use Cases

### Use Case 1: Clean LLM JSON Output

```yaml
row_plugins:
  - plugin: row_data_reshape
    options:
      operations:
        - operation: flatten
          field: llm_response  # Flatten JSON response
        - operation: exclude_fields
          fields: [prompt, raw_response, metadata]
        - operation: rename
          from: llm_response_rating
          to: score
```

### Use Case 2: Prepare Data for CSV Export

```yaml
aggregation_plugins:
  - plugin: aggregate_data_reshape
    options:
      input_key: collected_data
      output_key: csv_ready_data
      operations:
        - operation: filter_fields
          keep: [score, delta, recommendation]
        - operation: rename
          from: recommendation
          to: suggested_action
```

### Use Case 3: Type Conversion Pipeline

```yaml
row_plugins:
  - plugin: row_data_reshape
    options:
      operations:
        - operation: cast
          field: score
          type: float
        - operation: cast
          field: timestamp
          type: str
        - operation: cast
          field: is_verified
          type: bool
```

---

## See Also

- **Operations Library**: `src/elspeth/core/operations.py`
- **RowDataReshape**: `src/elspeth/plugins/transforms/row_data_reshape.py`
- **AggregateDataReshape**: `src/elspeth/plugins/transforms/aggregate_data_reshape.py`
