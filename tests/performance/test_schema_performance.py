"""Performance tests for schema validation system."""

import time

from elspeth.core.validation import validate_schema
from elspeth.plugins.transforms.field_collector import FieldCollector


class TestValidationOverhead:
    """Benchmark validation overhead."""

    def test_object_validation_overhead(self):
        """Object validation should add <1ms per object."""
        schema = {
            "type": "object",
            "required": ["score", "name"],
            "properties": {
                "score": {"type": "number"},
                "name": {"type": "string"},
            }
        }

        data = {"score": 0.85, "name": "test"}

        # Warmup
        for _ in range(100):
            list(validate_schema(data, schema))

        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            errors = list(validate_schema(data, schema))
            assert len(errors) == 0
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000
        print(f"\nObject validation: {avg_ms:.3f}ms per call")

        # Target: <1ms per validation
        assert avg_ms < 1.0, f"Validation too slow: {avg_ms:.3f}ms"

    def test_collection_validation_overhead(self):
        """Collection validation should scale linearly."""
        schema = {
            "type": "collection",
            "item_schema": {
                "type": "object",
                "properties": {"score": {"type": "number"}}
            }
        }

        # Small collection
        small_data = {"score": [0.8] * 10}

        # Large collection
        large_data = {"score": [0.8] * 1000}

        # Benchmark small
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            list(validate_schema(small_data, schema))
        small_time = time.perf_counter() - start

        # Benchmark large
        start = time.perf_counter()
        for _ in range(iterations):
            list(validate_schema(large_data, schema))
        large_time = time.perf_counter() - start

        # Should scale roughly linearly (100x data = ~100x time, with overhead)
        ratio = large_time / small_time
        print(f"\nSmall (10 items): {(small_time/iterations)*1000:.3f}ms")
        print(f"Large (1000 items): {(large_time/iterations)*1000:.3f}ms")
        print(f"Scaling ratio: {ratio:.1f}x (expected ~100x)")

        # Allow 200x max (overhead is acceptable)
        assert ratio < 200, f"Scaling too poor: {ratio:.1f}x"


class TestFieldCollectorPerformance:
    """Benchmark FieldCollector performance."""

    def test_collector_scales_with_rows(self):
        """FieldCollector should scale linearly with row count."""
        collector = FieldCollector({"output_key": "data"})

        # Small dataset
        small_rows = [{"score": 0.8, "name": f"item_{i}"} for i in range(100)]

        # Large dataset
        large_rows = [{"score": 0.8, "name": f"item_{i}"} for i in range(10000)]

        # Benchmark small
        start = time.perf_counter()
        for _ in range(10):
            collector.aggregate(small_rows, {})
        small_time = time.perf_counter() - start

        # Benchmark large
        start = time.perf_counter()
        for _ in range(10):
            collector.aggregate(large_rows, {})
        large_time = time.perf_counter() - start

        ratio = large_time / small_time
        print(f"\n100 rows: {(small_time/10)*1000:.3f}ms")
        print(f"10000 rows: {(large_time/10)*1000:.3f}ms")
        print(f"Scaling ratio: {ratio:.1f}x (expected ~100x)")

        # Should scale roughly linearly (100x data = ~100x time)
        assert ratio < 150, f"Scaling too poor: {ratio:.1f}x"

    def test_collector_memory_efficiency(self):
        """FieldCollector shouldn't duplicate data excessively."""
        import sys

        rows = [{"score": float(i), "name": f"item_{i}"} for i in range(1000)]

        collector = FieldCollector({"output_key": "data"})
        collection = collector.aggregate(rows, {})

        # Rough size check - collection shouldn't be >2x row data
        row_size = sys.getsizeof(rows)
        collection_size = sys.getsizeof(collection)

        # Collection stores references, not copies
        # So size should be reasonable
        print(f"\nRows size: {row_size} bytes")
        print(f"Collection size: {collection_size} bytes")

        # Collection overhead should be reasonable
        # (This is a rough check - actual memory is complex)
        assert collection_size < row_size * 3, "Collection too large"
