import pytest

from elspeth.plugins.transforms.field_collector import FieldCollector


def test_field_collector_transposes_rows_to_columns():
    """FieldCollector should transpose row-based data to columnar format."""
    config = {"output_key": "collected"}

    collector = FieldCollector(config)

    # Input: array of row objects
    results = [
        {"id": 1, "score": 0.75, "delta": 0.15},
        {"id": 2, "score": 0.82, "delta": 0.22},
        {"id": 3, "score": 0.91, "delta": 0.08}
    ]

    # Execute aggregation
    collection = collector.aggregate(results, aggregates={})

    # Output: columnar format (object with array values)
    expected = {
        "id": [1, 2, 3],
        "score": [0.75, 0.82, 0.91],
        "delta": [0.15, 0.22, 0.08]
    }

    assert collection == expected


def test_field_collector_config_schema():
    """FieldCollector should have required config schema."""
    from elspeth.core.validation import validate_schema

    # Valid config
    valid_config = {"output_key": "collected"}
    errors = list(validate_schema(valid_config, FieldCollector.config_schema))
    assert len(errors) == 0

    # Invalid config (missing output_key)
    invalid_config = {}
    errors = list(validate_schema(invalid_config, FieldCollector.config_schema))
    assert len(errors) > 0
    assert "required" in errors[0].message.lower()


def test_field_collector_has_schemas():
    """FieldCollector should have input and output schemas."""
    assert hasattr(FieldCollector, 'config_schema')
    assert hasattr(FieldCollector, 'input_schema')
    assert hasattr(FieldCollector, 'output_schema')

    # Should be collection type
    assert FieldCollector.output_schema['type'] == 'collection'


def test_field_collector_excludes_fields():
    """FieldCollector should exclude specified fields."""
    config = {
        "output_key": "collected",
        "exclude_fields": ["prompt", "response"]
    }

    collector = FieldCollector(config)

    results = [
        {"id": 1, "score": 0.75, "prompt": "Question 1", "response": "Answer 1"},
        {"id": 2, "score": 0.82, "prompt": "Question 2", "response": "Answer 2"}
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should only have id and score
    assert "id" in collection
    assert "score" in collection
    assert "prompt" not in collection
    assert "response" not in collection

    assert collection == {
        "id": [1, 2],
        "score": [0.75, 0.82]
    }


def test_field_collector_validates_type_consistency():
    """FieldCollector should detect type changes across rows."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    # Row 0: score is number, Row 1: score is string
    results = [
        {"id": 1, "score": 0.75},
        {"id": 2, "score": "invalid"}  # Type changed!
    ]

    with pytest.raises(TypeError) as exc_info:
        collector.aggregate(results, aggregates={})

    error_msg = str(exc_info.value)
    assert "type inconsistency" in error_msg.lower()
    assert "score" in error_msg
    assert "first non-none" in error_msg.lower() or "row 0" in error_msg.lower()
    assert "row 1" in error_msg.lower()


def test_field_collector_handles_missing_fields():
    """FieldCollector uses None for missing fields in rows."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    # Row 0 has score, Row 1 missing score
    results = [
        {"id": 1, "score": 0.75, "delta": 0.15},
        {"id": 2, "delta": 0.22}  # Missing score!
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should collect all fields with None for missing values
    assert collection["id"] == [1, 2]
    assert collection["score"] == [0.75, None]
    assert collection["delta"] == [0.15, 0.22]


def test_field_collector_handles_empty_results():
    """FieldCollector should handle empty results gracefully."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    results = []

    collection = collector.aggregate(results, aggregates={})

    assert collection == {}


def test_field_collector_flattens_nested_metrics():
    """FieldCollector should flatten nested metrics dict to top-level arrays."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    # Real SDA result structure with nested metrics
    results = [
        {
            "row": {"id": 1, "text": "Sample 1"},
            "response": {"content": "Response 1"},
            "metrics": {
                "extracted_score": 0.75,
                "score_flags": True,
                "delta": 0.15
            }
        },
        {
            "row": {"id": 2, "text": "Sample 2"},
            "response": {"content": "Response 2"},
            "metrics": {
                "extracted_score": 0.82,
                "score_flags": True,
                "delta": 0.22
            }
        },
        {
            "row": {"id": 3, "text": "Sample 3"},
            "response": {"content": "Response 3"},
            "metrics": {
                "extracted_score": 0.91,
                "score_flags": False,
                "delta": 0.08
            }
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should have flattened metric fields as top-level arrays
    assert "extracted_score" in collection
    assert "score_flags" in collection
    assert "delta" in collection

    # Verify values
    assert collection["extracted_score"] == [0.75, 0.82, 0.91]
    assert collection["score_flags"] == [True, True, False]
    assert collection["delta"] == [0.15, 0.22, 0.08]

    # Should also have top-level fields
    assert "row" in collection
    assert "response" in collection

    # Should NOT have metrics as a nested object
    assert "metrics" not in collection


def test_field_collector_excludes_nested_metric_fields():
    """FieldCollector should exclude nested metric fields when specified."""
    config = {
        "output_key": "collected",
        "exclude_fields": ["delta", "score_flags"]
    }
    collector = FieldCollector(config)

    results = [
        {
            "row": {"id": 1},
            "metrics": {
                "extracted_score": 0.75,
                "score_flags": True,
                "delta": 0.15
            }
        },
        {
            "row": {"id": 2},
            "metrics": {
                "extracted_score": 0.82,
                "score_flags": True,
                "delta": 0.22
            }
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should have extracted_score but not excluded fields
    assert "extracted_score" in collection
    assert "delta" not in collection
    assert "score_flags" not in collection
    assert collection["extracted_score"] == [0.75, 0.82]


def test_field_collector_handles_missing_metrics_dict():
    """FieldCollector should handle results without metrics dict."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    # Some rows have metrics, some don't
    results = [
        {"id": 1, "score": 0.75},
        {"id": 2, "score": 0.82}
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should work fine with top-level fields only
    assert collection == {
        "id": [1, 2],
        "score": [0.75, 0.82]
    }


def test_field_collector_handles_missing_metric_fields():
    """FieldCollector uses None for missing metric fields."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    # Row 0 has extracted_score in metrics, Row 1 doesn't
    results = [
        {
            "row": {"id": 1},
            "metrics": {"extracted_score": 0.75}
        },
        {
            "row": {"id": 2},
            "metrics": {}  # Missing extracted_score!
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should collect with None for missing values
    assert collection["row"] == [{"id": 1}, {"id": 2}]
    assert collection["extracted_score"] == [0.75, None]


def test_field_collector_top_level_takes_precedence():
    """FieldCollector should prefer top-level fields over metrics fields."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    # Field exists at both top-level and in metrics
    results = [
        {
            "score": 999,  # Top-level
            "metrics": {"score": 0.75}  # Nested
        },
        {
            "score": 888,  # Top-level
            "metrics": {"score": 0.82}  # Nested
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should use top-level values
    assert collection["score"] == [999, 888]


def test_field_collector_late_arriving_fields():
    """FieldCollector collects fields that appear in later rows."""
    collector = FieldCollector({"output_key": "collected"})

    # Row 0 has only "score", row 1 adds "score_std"
    results = [
        {
            "row": {"id": 1},
            "metrics": {"score": 0.5}
        },
        {
            "row": {"id": 2},
            "metrics": {"score": 0.6, "score_std": 0.02}
        },
        {
            "row": {"id": 3},
            "metrics": {"score": 0.7, "score_std": 0.03}
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should collect both fields, with None for missing values
    assert "score" in collection
    assert "score_std" in collection
    assert collection["score"] == [0.5, 0.6, 0.7]
    assert collection["score_std"] == [None, 0.02, 0.03]


def test_field_collector_late_arriving_top_level_fields():
    """FieldCollector handles late-arriving top-level fields."""
    collector = FieldCollector({"output_key": "collected"})

    # Row 0 has "id" and "score", row 1 adds "category"
    results = [
        {"id": 1, "score": 0.5},
        {"id": 2, "score": 0.6, "category": "A"},
        {"id": 3, "score": 0.7, "category": "B"}
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should collect all fields with None for missing
    assert collection["id"] == [1, 2, 3]
    assert collection["score"] == [0.5, 0.6, 0.7]
    assert collection["category"] == [None, "A", "B"]


def test_field_collector_intermittent_fields():
    """FieldCollector handles fields that appear/disappear intermittently."""
    collector = FieldCollector({"output_key": "collected"})

    # Field appears in row 0, missing in row 1, back in row 2
    results = [
        {"metrics": {"score": 0.5, "flag": True}},
        {"metrics": {"score": 0.6}},
        {"metrics": {"score": 0.7, "flag": False}},
        {"metrics": {"score": 0.8}}
    ]

    collection = collector.aggregate(results, aggregates={})

    assert collection["score"] == [0.5, 0.6, 0.7, 0.8]
    assert collection["flag"] == [True, None, False, None]


def test_field_collector_flattens_nested_metric_dicts():
    """FieldCollector should recursively flatten nested metric dicts (e.g., scores: {quality: 0.8})."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    # Simulates ScoreExtractorPlugin output with nested scores dict
    results = [
        {
            "id": 1,
            "metrics": {
                "scores": {
                    "quality": 0.75,
                    "accuracy": 0.82
                },
                "score_flags": {
                    "quality": True,
                    "accuracy": True
                }
            }
        },
        {
            "id": 2,
            "metrics": {
                "scores": {
                    "quality": 0.88,
                    "accuracy": 0.91
                },
                "score_flags": {
                    "quality": True,
                    "accuracy": True
                }
            }
        },
        {
            "id": 3,
            "metrics": {
                "scores": {
                    "quality": 0.62,
                    "accuracy": 0.70
                },
                "score_flags": {
                    "quality": False,
                    "accuracy": True
                }
            }
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should flatten nested scores dict into separate arrays
    assert "quality" in collection
    assert "accuracy" in collection

    # Verify values extracted from nested path
    assert collection["quality"] == [0.75, 0.88, 0.62]
    assert collection["accuracy"] == [0.82, 0.91, 0.70]

    # Should NOT have scores as array of dicts
    assert "scores" not in collection

    # Should still have top-level fields
    assert "id" in collection
    assert collection["id"] == [1, 2, 3]


def test_field_collector_flattens_deeply_nested_metrics():
    """FieldCollector should handle deeply nested metric dicts."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    results = [
        {
            "id": 1,
            "metrics": {
                "analysis": {
                    "sentiment": {
                        "positive": 0.7,
                        "negative": 0.2
                    }
                }
            }
        },
        {
            "id": 2,
            "metrics": {
                "analysis": {
                    "sentiment": {
                        "positive": 0.85,
                        "negative": 0.1
                    }
                }
            }
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should flatten deeply nested values
    assert "positive" in collection
    assert "negative" in collection

    assert collection["positive"] == [0.7, 0.85]
    assert collection["negative"] == [0.2, 0.1]

    # Should NOT have intermediate dict levels
    assert "analysis" not in collection
    assert "sentiment" not in collection


def test_field_collector_handles_sparse_nested_metrics():
    """FieldCollector handles nested metrics that appear in some rows but not others."""
    config = {"output_key": "collected"}
    collector = FieldCollector(config)

    results = [
        {
            "id": 1,
            "metrics": {
                "scores": {
                    "quality": 0.75,
                    "accuracy": 0.82
                }
            }
        },
        {
            "id": 2,
            "metrics": {
                # No scores dict in this row
                "count": 5
            }
        },
        {
            "id": 3,
            "metrics": {
                "scores": {
                    "quality": 0.88,
                    "accuracy": None  # Explicit None
                }
            }
        }
    ]

    collection = collector.aggregate(results, aggregates={})

    # Should have fields from nested scores
    assert "quality" in collection
    assert "accuracy" in collection
    assert "count" in collection

    # Missing nested dict should result in None
    assert collection["quality"] == [0.75, None, 0.88]
    assert collection["accuracy"] == [0.82, None, None]
    assert collection["count"] == [None, 5, None]
