# Plugin Documentation

Comprehensive documentation for all elspeth-simple plugins.

---

## Core Plugins

### Meta-Plugins

Meta-plugins transform data structure (not content):

- **[FieldCollector](field-collector.md)** - Row -> Collection transposition
- **[FieldExpander](field-expander.md)** - Collection -> Object transposition

### Data Manipulation

- **[Data Reshape Plugins](data-reshape.md)** - Configurable transformations (flatten, rename, filter, cast)
  - RowDataReshape (row mode)
  - AggregateDataReshape (collection mode)

### Metrics & Analytics

- **[Metrics Analyzers](metrics-analyzers.md)** - Statistical analysis (v3.0)
  - ScoreStatsAnalyzer
  - ScoreRecommendationAnalyzer
  - ScoreVariantRankingAnalyzer
  - ScoreDistributionAnalyzer
  - ScoreAgreementAnalyzer
  - ScorePowerAnalyzer

---

## Plugin Types

### Row Plugins (Object Mode)

Process individual rows:
- Input: `{"type": "object"}`
- Output: `{"type": "object"}`
- Examples: score_delta, score_ratio, data reshape

### Aggregation Plugins (Collection Mode)

Process all rows collectively:
- Input: `{"type": "collection"}` or `{"type": "array"}`
- Output: `{"type": "object"}` or `{"type": "collection"}`
- Examples: FieldCollector, metrics analyzers

---

## Quick Reference

### Meta-Plugin Pattern

```yaml
aggregation_plugins:
  # 1. Collect rows -> collection
  - plugin: field_collector
    options:
      output_key: collected_data

  # 2. Analyze collection -> object
  - plugin: score_stats_analyzer
    options:
      input_key: collected_data
      score_field: delta

  # 3. (Optional) Expand collection -> flat object
  - plugin: field_expander
    options:
      input_key: some_collection
```

### Data Reshape Pattern

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
```

---

## See Also

- **Architecture**: `docs/schema-driven-execution.md`
- **Migration**: `docs/migration-v3.0-metrics.md`
- **Examples**: `example/meta-plugins/`, `example/experimental/`
- **CLAUDE.md**: Developer guide with plugin patterns
