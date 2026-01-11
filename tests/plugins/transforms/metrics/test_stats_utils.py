"""Tests for stats_utils shared statistical utilities."""


import numpy as np
import pytest

from elspeth.plugins.transforms.metrics.stats_utils import (
    benjamini_hochberg,
    calculate_cliffs_delta,
    collect_paired_scores_by_criterion,
    collect_scores_by_criterion,
    extract_scores,
    safe_mean,
    safe_std,
    validate_collection_key,
)


class TestValidateCollectionKey:
    """Tests for validate_collection_key()."""

    def test_returns_collection_when_key_exists(self):
        """Returns collection for valid key."""
        collection = {"score": [0.7, 0.8, 0.9]}
        aggregates = {"my_key": collection}

        result = validate_collection_key("my_key", aggregates)

        assert result is collection

    def test_raises_key_error_when_missing(self):
        """Raises KeyError with helpful message for missing key."""
        aggregates = {"other_key": {}}

        with pytest.raises(KeyError) as exc_info:
            validate_collection_key("missing_key", aggregates)

        error_msg = str(exc_info.value)
        assert "missing_key" in error_msg
        assert "not found in aggregates" in error_msg
        assert "Available keys" in error_msg
        assert "FieldCollector" in error_msg

    def test_lists_available_keys_in_error(self):
        """Error message lists available keys."""
        aggregates = {"key_a": {}, "key_b": {}, "key_c": {}}

        with pytest.raises(KeyError) as exc_info:
            validate_collection_key("missing", aggregates)

        error_msg = str(exc_info.value)
        assert "key_a" in error_msg
        assert "key_b" in error_msg
        assert "key_c" in error_msg


class TestExtractScores:
    """Tests for extract_scores()."""

    def test_extracts_scores_from_collection(self):
        """Returns scores array from collection."""
        collection = {"score": [0.7, 0.8, 0.9], "id": [1, 2, 3]}

        result = extract_scores(collection, "score")

        assert result == [0.7, 0.8, 0.9]

    def test_raises_value_error_for_missing_field(self):
        """Raises ValueError with helpful message for missing field."""
        collection = {"id": [1, 2, 3], "name": ["a", "b", "c"]}

        with pytest.raises(ValueError) as exc_info:
            extract_scores(collection, "missing_field")

        error_msg = str(exc_info.value)
        assert "missing_field" in error_msg
        assert "not found in collection" in error_msg
        assert "Available fields" in error_msg

    def test_filters_by_flag_field(self):
        """Filters scores by truthy flag values."""
        collection = {
            "score": [0.7, 0.8, 0.9, 0.6],
            "include": [True, False, True, False],
        }

        result = extract_scores(collection, "score", flag_field="include")

        assert result == [0.7, 0.9]

    def test_flag_field_missing_does_not_filter(self):
        """Missing flag_field returns all scores."""
        collection = {"score": [0.7, 0.8, 0.9]}

        result = extract_scores(collection, "score", flag_field="nonexistent")

        assert result == [0.7, 0.8, 0.9]

    def test_flag_field_none_does_not_filter(self):
        """None flag_field returns all scores."""
        collection = {"score": [0.7, 0.8, 0.9]}

        result = extract_scores(collection, "score", flag_field=None)

        assert result == [0.7, 0.8, 0.9]


class TestSafeMean:
    """Tests for safe_mean()."""

    def test_computes_mean_for_valid_values(self):
        """Computes mean correctly."""
        result = safe_mean([1.0, 2.0, 3.0])
        assert result == 2.0

    def test_returns_none_for_empty_list(self):
        """Returns None for empty list."""
        result = safe_mean([])
        assert result is None

    def test_handles_single_value(self):
        """Handles single value."""
        result = safe_mean([5.0])
        assert result == 5.0


class TestSafeStd:
    """Tests for safe_std()."""

    def test_computes_std_for_valid_values(self):
        """Computes standard deviation correctly."""
        result = safe_std([1.0, 2.0, 3.0])
        assert result is not None
        assert result > 0

    def test_returns_none_for_empty_list(self):
        """Returns None for empty list."""
        result = safe_std([])
        assert result is None

    def test_returns_none_for_single_value(self):
        """Returns None for single value (can't compute std)."""
        result = safe_std([5.0])
        assert result is None

    def test_respects_ddof_parameter(self):
        """Uses ddof parameter correctly."""
        values = [1.0, 2.0, 3.0]
        result_ddof_1 = safe_std(values, ddof=1)
        result_ddof_0 = safe_std(values, ddof=0)

        assert result_ddof_1 is not None
        assert result_ddof_0 is not None
        assert result_ddof_1 > result_ddof_0  # Sample std > population std


class TestCollectScoresByCriterion:
    """Tests for collect_scores_by_criterion()."""

    def test_collects_scores_by_name(self):
        """Collects scores grouped by criterion name."""
        payload = {
            "results": [
                {"metrics": {"scores": {"quality": 0.8, "accuracy": 0.9}}},
                {"metrics": {"scores": {"quality": 0.7, "accuracy": 0.85}}},
            ]
        }

        result = collect_scores_by_criterion(payload)

        assert "quality" in result
        assert "accuracy" in result
        assert result["quality"] == [0.8, 0.7]
        assert result["accuracy"] == [0.9, 0.85]

    def test_handles_missing_results(self):
        """Handles payload without results key."""
        result = collect_scores_by_criterion({})
        assert result == {}

    def test_handles_none_results(self):
        """Handles None results."""
        result = collect_scores_by_criterion({"results": None})
        assert result == {}

    def test_skips_none_values(self):
        """Skips None score values."""
        payload = {
            "results": [
                {"metrics": {"scores": {"quality": 0.8}}},
                {"metrics": {"scores": {"quality": None}}},
                {"metrics": {"scores": {"quality": 0.9}}},
            ]
        }

        result = collect_scores_by_criterion(payload)

        assert result["quality"] == [0.8, 0.9]

    def test_skips_nan_values(self):
        """Skips NaN score values."""
        payload = {
            "results": [
                {"metrics": {"scores": {"quality": 0.8}}},
                {"metrics": {"scores": {"quality": float("nan")}}},
                {"metrics": {"scores": {"quality": 0.9}}},
            ]
        }

        result = collect_scores_by_criterion(payload)

        assert result["quality"] == [0.8, 0.9]

    def test_converts_string_numbers(self):
        """Converts numeric strings to floats."""
        payload = {
            "results": [
                {"metrics": {"scores": {"quality": "0.8"}}},
                {"metrics": {"scores": {"quality": "0.9"}}},
            ]
        }

        result = collect_scores_by_criterion(payload)

        assert result["quality"] == [0.8, 0.9]

    def test_skips_non_numeric_strings(self):
        """Skips non-numeric string values."""
        payload = {
            "results": [
                {"metrics": {"scores": {"quality": 0.8}}},
                {"metrics": {"scores": {"quality": "invalid"}}},
            ]
        }

        result = collect_scores_by_criterion(payload)

        assert result["quality"] == [0.8]

    def test_handles_missing_metrics(self):
        """Handles records without metrics."""
        payload = {
            "results": [
                {"metrics": {"scores": {"quality": 0.8}}},
                {},  # No metrics
                {"metrics": None},  # None metrics
            ]
        }

        result = collect_scores_by_criterion(payload)

        assert result["quality"] == [0.8]

    def test_handles_non_dict_scores(self):
        """Handles non-dict scores gracefully."""
        payload = {
            "results": [
                {"metrics": {"scores": {"quality": 0.8}}},
                {"metrics": {"scores": [1, 2, 3]}},  # List instead of dict
            ]
        }

        result = collect_scores_by_criterion(payload)

        assert result["quality"] == [0.8]


class TestCollectPairedScoresByCriterion:
    """Tests for collect_paired_scores_by_criterion()."""

    def test_collects_paired_scores(self):
        """Collects paired baseline/variant scores."""
        baseline = {
            "results": [
                {"metrics": {"scores": {"quality": 0.7}}},
                {"metrics": {"scores": {"quality": 0.8}}},
            ]
        }
        variant = {
            "results": [
                {"metrics": {"scores": {"quality": 0.75}}},
                {"metrics": {"scores": {"quality": 0.85}}},
            ]
        }

        result = collect_paired_scores_by_criterion(baseline, variant)

        assert "quality" in result
        assert result["quality"] == [(0.7, 0.75), (0.8, 0.85)]

    def test_handles_mismatched_lengths(self):
        """Uses shorter length when results differ."""
        baseline = {
            "results": [
                {"metrics": {"scores": {"quality": 0.7}}},
                {"metrics": {"scores": {"quality": 0.8}}},
                {"metrics": {"scores": {"quality": 0.9}}},
            ]
        }
        variant = {
            "results": [
                {"metrics": {"scores": {"quality": 0.75}}},
            ]
        }

        result = collect_paired_scores_by_criterion(baseline, variant)

        assert result["quality"] == [(0.7, 0.75)]

    def test_only_pairs_matching_criteria(self):
        """Only pairs criteria present in both."""
        baseline = {
            "results": [
                {"metrics": {"scores": {"quality": 0.7, "speed": 0.5}}},
            ]
        }
        variant = {
            "results": [
                {"metrics": {"scores": {"quality": 0.75}}},  # No speed
            ]
        }

        result = collect_paired_scores_by_criterion(baseline, variant)

        assert "quality" in result
        assert "speed" not in result

    def test_skips_nan_pairs(self):
        """Skips pairs with NaN values."""
        baseline = {
            "results": [
                {"metrics": {"scores": {"quality": 0.7}}},
                {"metrics": {"scores": {"quality": float("nan")}}},
            ]
        }
        variant = {
            "results": [
                {"metrics": {"scores": {"quality": 0.75}}},
                {"metrics": {"scores": {"quality": 0.85}}},
            ]
        }

        result = collect_paired_scores_by_criterion(baseline, variant)

        # Only first pair should be included
        assert result["quality"] == [(0.7, 0.75)]


class TestCalculateCliffsDelta:
    """Tests for calculate_cliffs_delta()."""

    def test_returns_zero_for_empty_groups(self):
        """Returns (0.0, 'no_data') for empty groups."""
        delta, interpretation = calculate_cliffs_delta([], [1, 2, 3])
        assert delta == 0.0
        assert interpretation == "no_data"

        delta, interpretation = calculate_cliffs_delta([1, 2, 3], [])
        assert delta == 0.0
        assert interpretation == "no_data"

    def test_negligible_effect(self):
        """Detects negligible effect size."""
        # Groups with substantial overlap to get negligible effect
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        group2 = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]

        delta, interpretation = calculate_cliffs_delta(group1, group2)

        # High overlap means negligible effect
        assert abs(delta) < 0.147
        assert interpretation == "negligible"

    def test_small_effect(self):
        """Detects small effect size."""
        # Groups with some but not much overlap
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [1.5, 2.5, 3.5, 4.5, 5.5]

        delta, interpretation = calculate_cliffs_delta(group1, group2)

        # Partial overlap gives small effect
        assert 0.147 <= abs(delta) < 0.33
        assert interpretation == "small"

    def test_medium_effect(self):
        """Detects medium effect size."""
        # Groups with shift=1.0 give delta=0.36 (medium range: 0.33-0.474)
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [2.0, 3.0, 4.0, 5.0, 6.0]  # Shift by 1

        delta, interpretation = calculate_cliffs_delta(group1, group2)

        # 0.36 is in medium range
        assert 0.33 <= abs(delta) < 0.474
        assert interpretation == "medium"

    def test_large_effect(self):
        """Detects large effect size."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [10.0, 11.0, 12.0, 13.0, 14.0]  # No overlap

        delta, interpretation = calculate_cliffs_delta(group1, group2)

        assert abs(delta) >= 0.474
        assert interpretation == "large"

    def test_direction_matters(self):
        """Delta sign reflects direction."""
        group1 = [1.0, 2.0, 3.0]
        group2 = [4.0, 5.0, 6.0]

        delta1, _ = calculate_cliffs_delta(group1, group2)
        delta2, _ = calculate_cliffs_delta(group2, group1)

        # Should be opposite signs
        assert delta1 > 0  # group2 > group1
        assert delta2 < 0  # group1 < group2
        assert abs(delta1 + delta2) < 0.001  # Magnitudes equal

    def test_identical_groups(self):
        """Identical groups have negligible effect."""
        group = [1.0, 2.0, 3.0, 4.0, 5.0]

        delta, interpretation = calculate_cliffs_delta(group, group)

        assert delta == 0.0
        assert interpretation == "negligible"


class TestBenjaminiHochberg:
    """Tests for benjamini_hochberg()."""

    def test_empty_list(self):
        """Returns empty list for empty input."""
        result = benjamini_hochberg([])
        assert result == []

    def test_single_value(self):
        """Single value returned unchanged (capped at 1.0)."""
        result = benjamini_hochberg([0.5])
        assert result == [0.5]

    def test_already_significant(self):
        """Already significant values remain significant."""
        result = benjamini_hochberg([0.01, 0.02, 0.03])

        # All should still be < 0.05 after correction
        for adj_p in result:
            assert adj_p < 0.1

    def test_preserves_order_relationship(self):
        """Adjusted p-values maintain relative ordering."""
        p_values = [0.01, 0.05, 0.10, 0.20]

        result = benjamini_hochberg(p_values)

        # Verify monotonicity (after adjustment, smaller raw p should have smaller adjusted)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_caps_at_one(self):
        """Adjusted p-values capped at 1.0."""
        result = benjamini_hochberg([0.9, 0.95, 0.99])

        for adj_p in result:
            assert adj_p <= 1.0

    def test_known_correction(self):
        """Verifies against known FDR correction result."""
        # Example: 3 tests with p-values
        p_values = [0.01, 0.04, 0.03]

        result = benjamini_hochberg(p_values)

        # After BH correction:
        # Sorted: 0.01 (rank 1), 0.03 (rank 2), 0.04 (rank 3)
        # Adjusted: 0.01*3/1=0.03, 0.03*3/2=0.045, 0.04*3/3=0.04
        # Then enforce monotonicity from back: [0.03, 0.04, 0.04]
        # Map back to original order: [0.03, 0.04, 0.04]
        assert abs(result[0] - 0.03) < 0.001  # First p-value
        assert abs(result[2] - 0.04) < 0.001  # Third p-value (0.03 gets 0.04 after monotonicity)


class TestEdgeCases:
    """Edge case tests."""

    def test_extract_scores_empty_collection(self):
        """Extract scores from empty field."""
        collection = {"score": [], "id": []}

        result = extract_scores(collection, "score")

        assert result == []

    def test_collect_scores_empty_results(self):
        """Collect scores with empty results list."""
        payload = {"results": []}

        result = collect_scores_by_criterion(payload)

        assert result == {}

    def test_cliffs_delta_with_ties(self):
        """Cliff's delta handles tied values."""
        group1 = [1.0, 1.0, 1.0]
        group2 = [1.0, 2.0, 2.0]

        delta, _ = calculate_cliffs_delta(group1, group2)

        # Should compute despite ties
        assert isinstance(delta, float)

    def test_safe_mean_with_numpy_array(self):
        """safe_mean works with numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])

        result = safe_mean(list(arr))

        assert result == 2.0
