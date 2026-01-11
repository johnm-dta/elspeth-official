# Experimental Orchestrator Example - Prompt A/B Testing

This example demonstrates the experimental orchestrator with statistical A/B testing of different prompting strategies.

## Overview

**Question:** Which prompting strategy yields the highest ratings from an LLM?

We test 3 different approaches for rating programming languages (1-10 scale):
- **Baseline (Neutral):** Objective, balanced evaluation
- **Variant A (Enthusiastic):** Highlights strengths and positives
- **Variant B (Critical):** Focuses on weaknesses and limitations

The experimental orchestrator automatically:
1. Identifies the baseline from cycle metadata
2. Runs baseline first, then variants
3. Applies statistical comparison plugins
4. Generates comparison reports

## Running the Example

```bash
# From project root
elspeth --settings example/experimental/settings.yaml
```

## Architecture

### Experimental Orchestrator Configuration

```yaml
orchestrator_type: experimental  # Use ExperimentalOrchestrator
```

This triggers:
- Baseline identification from `metadata.is_baseline`
- Baseline-first execution order
- Automatic comparison analysis

### Cycle Structure

```
example/experimental/cycles/
├── baseline-neutral/          # Baseline (neutral prompt)
│   ├── config.json           # metadata: {is_baseline: true}
│   ├── system_prompt.md
│   └── user_prompt.md
├── variant-enthusiastic/      # Variant A (positive framing)
│   ├── config.json           # metadata: {is_baseline: false}
│   ├── system_prompt.md
│   └── user_prompt.md
└── variant-critical/          # Variant B (critical framing)
    ├── config.json           # metadata: {is_baseline: false}
    ├── system_prompt.md
    └── user_prompt.md
```

### Statistical Plugins

**Transform Plugins** (extract scores from LLM responses):
```yaml
row_plugins:
  - plugin: score_extractor
    options:
      key: rating  # Extract "rating" field from JSON
```

**Aggregation Plugins** (compute statistics per cycle):
```yaml
aggregator_plugins:
  - plugin: score_stats  # mean, median, std dev, min, max
```

**Baseline Comparison Plugins** (compare variants to baseline):
```yaml
baseline_plugins:
  - plugin: score_delta         # Simple mean difference
  - plugin: score_significance  # t-tests, effect sizes
  - plugin: score_practical     # Practical significance
```

## Interpreting Results

### Output Files

**1. CSV Results** (`example/experimental/output/results.csv`)

Contains all LLM responses with extracted ratings.

**2. Console Output**

```
INFO: Using ExperimentalOrchestrator (baseline comparison)
INFO: Experiment baseline-neutral completed with 10 rows
INFO: Experiment variant-enthusiastic completed with 10 rows
INFO: Experiment variant-critical completed with 10 rows
```

### Result Structure

The orchestrator returns a dict keyed by cycle name:

```python
{
  "baseline-neutral": {
    "payload": {
      "results": [...],  # All 10 language ratings
      "aggregates": {
        "score_stats": {
          "overall": {
            "mean": 7.2,      # Average rating
            "median": 7.0,
            "std": 1.1,
            "min": 5.0,
            "max": 9.0,
            "count": 10
          }
        }
      }
    },
    "config": {...}
  },

  "variant-enthusiastic": {
    "payload": {
      "results": [...],
      "aggregates": {...},
      "baseline_comparison": {  # ← Comparison to baseline
        "score_delta": {
          "overall": 1.5  # 1.5 points higher than baseline
        },
        "score_significance": {
          "overall": {
            "baseline_mean": 7.2,
            "variant_mean": 8.7,
            "mean_difference": 1.5,
            "effect_size": 1.36,  # Cohen's d
            "t_stat": 4.12,
            "p_value": 0.002,     # Statistically significant!
          }
        },
        "score_practical": {
          "overall": {
            "mean_difference": 1.5,
            "meaningful_change_rate": 0.8,  # 80% meaningfully different
            "success_delta": 0.3,            # 30% more "high scores"
          }
        }
      }
    },
    "config": {...}
  },

  "variant-critical": {
    "payload": {
      "results": [...],
      "aggregates": {...},
      "baseline_comparison": {
        "score_delta": {
          "overall": -2.1  # 2.1 points LOWER than baseline
        },
        "score_significance": {
          "overall": {
            "baseline_mean": 7.2,
            "variant_mean": 5.1,
            "mean_difference": -2.1,
            "effect_size": -1.91,
            "p_value": 0.0008,  # Significantly lower!
          }
        }
      }
    },
    "config": {...}
  }
}
```

### Reading the Statistics

**Score Delta** (simple comparison):
- Positive number = variant scored higher than baseline
- Negative number = variant scored lower than baseline

**Statistical Significance** (t-test):
- `p_value < 0.05` = statistically significant difference
- `effect_size`:
  - Small: 0.2
  - Medium: 0.5
  - Large: 0.8+

**Practical Significance**:
- `meaningful_change_rate` = % of items with meaningful difference (>1 point)
- `success_delta` = Change in % of "high scores" (≥7.0)

### Example Interpretation

Given the results above:

**Baseline (Neutral):**
- Mean rating: 7.2/10
- Std dev: 1.1

**Variant A (Enthusiastic):**
- Mean rating: 8.7/10 (+1.5 vs baseline)
- Effect size: 1.36 (large!)
- p-value: 0.002 (highly significant)
- **Conclusion:** Enthusiastic prompts yield significantly higher ratings

**Variant B (Critical):**
- Mean rating: 5.1/10 (-2.1 vs baseline)
- Effect size: -1.91 (large negative!)
- p-value: 0.0008 (highly significant)
- **Conclusion:** Critical prompts yield significantly lower ratings

**Winner:** Variant A (Enthusiastic) scores highest

## Key Insights

1. **Prompt framing matters:** Same task, different framing yields 3.6-point spread (5.1 to 8.7)

2. **Statistical validation:** With only 10 samples, we can detect large effects with confidence

3. **Practical implications:** If you want higher scores, frame questions positively!

## Customizing the Example

### Try Different Prompts

Edit the `*_prompt.md` files in each cycle directory to test your own hypotheses.

### Add More Variants

```bash
cp -r example/experimental/cycles/baseline-neutral example/experimental/cycles/variant-technical
# Edit config.json to set is_baseline: false
# Edit prompts to focus on technical merit
```

### Change the Dataset

Replace `data/programming_languages.csv` with your own data. The LLM will rate whatever you provide!

### Adjust Statistical Thresholds

```yaml
baseline_plugins:
  - plugin: score_practical
    options:
      threshold: 0.5         # Minimum meaningful difference (default: 1.0)
      success_threshold: 8.0  # What counts as "success" (default: 7.0)
```

## Notes on the Refactor

**Why separate orchestrators?**

Previously, experiment concepts (`is_baseline`, `hypothesis`, comparison logic) were hard-coded into the core suite runner. Now:

- **StandardOrchestrator:** Pure SDA execution (no experiment concepts)
- **ExperimentalOrchestrator:** A/B testing as a plugin

This example demonstrates that **statistical experimentation is opt-in**, not mandatory for the platform.

**About the 1500-line metrics.py file:**

Yes, it needs refactoring! The comparison plugins (`score_delta`, `score_significance`, `score_practical`, etc.) should probably be split into separate files. But for now, they work!

## Further Reading

- **Statistical comparison plugins:** `src/elspeth/plugins/transforms/metrics.py`
- **Experimental orchestrator:** `src/elspeth/orchestrators/experimental.py`
- **Standard orchestrator:** `src/elspeth/orchestrators/standard.py`
