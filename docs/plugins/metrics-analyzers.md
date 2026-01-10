# Metrics Analyzers (v3.0)

**Type:** Aggregation Plugins
**Category:** Statistical Analysis
**Mode:** Collection -> Object

---

## Overview

Metrics analyzers are v3.0 collection-mode plugins that compute statistical metrics on columnar data. They replace v2.x aggregators with separation of concerns: FieldCollector handles collection, analyzers handle computation.

**Key changes from v2.x:**
- **Renamed:** `*_aggregator` -> `*_analyzer`
- **Separated:** Collection logic moved to FieldCollector
- **Input:** Read from `aggregates[input_key]` (not from results directly)
- **Schemas:** All analyzers declare input_schema and output_schema

---

## Plugin Catalog

### Aggregation Analyzers (metrics/aggregation.py)

#### 1. ScoreStatsAnalyzer

Compute descriptive statistics on score arrays.

**Configuration:**
```yaml
- plugin: score_stats_analyzer
  options:
    input_key: collected_data
    score_field: delta
    output_prefix: stats
```

**Input schema:**
```python
{
    "type": "collection",
    "item_schema": {
        "type": "object",
        "required": ["delta"],
        "properties": {"delta": {"type": "number"}}
    }
}
```

**Output:**
```python
{
    "stats_mean": 5.2,
    "stats_std": 1.8,
    "stats_min": 2.1,
    "stats_max": 9.4,
    "stats_count": 100
}
```

---

#### 2. ScoreRecommendationAnalyzer

Recommend baseline or variant based on score comparison.

**Configuration:**
```yaml
- plugin: score_recommendation_analyzer
  options:
    input_key: collected_data
    score_field: delta
    delta_threshold: 0.05  # Minimum delta for recommendation
    confidence_threshold: 0.8  # P-value threshold for "high" confidence
```

**Output:**
```python
{
    "recommendation": "variant",  # or "baseline" or "inconclusive"
    "confidence": "high",  # or "medium" or "low"
    "delta": 1.2,
    "reason": "Variant scores significantly higher (p<0.05)"
}
```

**Recommendation logic:**
- `delta > delta_threshold` and `p < confidence_threshold` -> variant (high confidence)
- `delta < -delta_threshold` and `p < confidence_threshold` -> baseline (high confidence)
- Otherwise -> inconclusive

---

#### 3. ScoreVariantRankingAnalyzer

Rank variants by score (highest to lowest).

**Configuration:**
```yaml
- plugin: score_variant_ranking_analyzer
  options:
    input_key: collected_data
    score_field: delta
    variant_id_field: variant_name
```

**Output:**
```python
{
    "ranking": ["variant_c", "variant_a", "variant_b"],
    "top_variant": "variant_c",
    "top_score": 12.5,
    "score_spread": 3.2  # max - min
}
```

---

#### 4. ScoreAgreementAnalyzer

Measure agreement between baseline and variant scores.

**Configuration:**
```yaml
- plugin: score_agreement_analyzer
  options:
    input_key: collected_data
    baseline_field: baseline_score
    variant_field: variant_score
```

**Output:**
```python
{
    "agreement_rate": 0.85,  # % of rows where baseline ~ variant
    "correlation": 0.92,  # Pearson correlation
    "mean_absolute_error": 0.5
}
```

---

### Distribution Analyzers (metrics/distribution.py)

#### 5. ScoreDistributionAnalyzer

Test score distribution properties (normality, outliers).

**Configuration:**
```yaml
- plugin: score_distribution_analyzer
  options:
    input_key: collected_data
    score_field: delta
    output_prefix: dist
```

**Output:**
```python
{
    "dist_normality_pvalue": 0.23,  # Shapiro-Wilk test
    "dist_is_normal": true,  # p > 0.05
    "dist_skewness": 0.12,
    "dist_kurtosis": -0.05,
    "dist_outlier_count": 2,
    "dist_outlier_indices": [5, 42]
}
```

---

### Power Analysis (metrics/power_analysis.py)

#### 6. ScorePowerAnalyzer

Compute statistical power for detecting effect.

**Configuration:**
```yaml
- plugin: score_power_analyzer
  options:
    input_key: collected_data
    score_field: delta
    effect_size_threshold: 0.5  # Cohen's d
```

**Output:**
```python
{
    "power": 0.85,  # Probability of detecting effect
    "sample_size": 100,
    "effect_size": 0.6,
    "is_adequately_powered": true  # power > 0.8
}
```

---

## Common Configuration Patterns

### Pattern 1: Basic Stats

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

### Pattern 2: Comprehensive Analysis

```yaml
aggregation_plugins:
  - plugin: field_collector
    options:
      output_key: scores

  - plugin: score_stats_analyzer
    options:
      input_key: scores
      score_field: delta
      output_prefix: delta_stats

  - plugin: score_distribution_analyzer
    options:
      input_key: scores
      score_field: delta
      output_prefix: delta_dist

  - plugin: score_recommendation_analyzer
    options:
      input_key: scores
      score_field: delta
```

**Result:** Stats, distribution tests, and recommendation in one pipeline.

### Pattern 3: Multi-Field Analysis

```yaml
aggregation_plugins:
  - plugin: field_collector
    options:
      output_key: all_scores

  - plugin: score_stats_analyzer
    options:
      input_key: all_scores
      score_field: baseline_score
      output_prefix: baseline

  - plugin: score_stats_analyzer
    options:
      input_key: all_scores
      score_field: variant_score
      output_prefix: variant

  - plugin: score_agreement_analyzer
    options:
      input_key: all_scores
      baseline_field: baseline_score
      variant_field: variant_score
```

---

## Migration from v2.x

### Before (v2.x)

```yaml
aggregator_plugins:
  - plugin: score_stats_aggregator
    options:
      score_field: delta
```

### After (v3.0)

```yaml
aggregation_plugins:
  - plugin: field_collector
    options:
      output_key: collected_data

  - plugin: score_stats_analyzer
    options:
      input_key: collected_data
      score_field: delta
```

**Changes:**
1. Add FieldCollector before analyzer
2. Rename `aggregator` -> `analyzer`
3. Add `input_key` referencing FieldCollector's `output_key`

---

## See Also

- **FieldCollector**: Required for collection-mode analytics
- **Module Structure**: `src/elspeth/plugins/transforms/metrics/`
- **Migration Guide**: `docs/migration-v3.0-metrics.md`
- **Schema System**: `docs/schema-driven-execution.md`
