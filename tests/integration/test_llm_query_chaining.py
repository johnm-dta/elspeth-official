"""Integration test for LLM query plugin chaining."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from elspeth.core.sda.runner import SDARunner
from elspeth.plugins.outputs.csv_file import CsvResultSink
from elspeth.plugins.transforms.llm_query import LLMQueryPlugin

if TYPE_CHECKING:
    from pathlib import Path


class MockSink:
    """Mock sink for testing."""

    def __init__(self):
        self.written_payload: dict[str, Any] | None = None
        self.written_metadata: dict[str, Any] | None = None

    def write(self, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        self.written_payload = payload
        self.written_metadata = metadata

    def produces(self) -> list[str]:
        return []

    def consumes(self) -> list[str]:
        return []

    def finalize(
        self, artifacts: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        pass


class TestLLMQueryChaining:
    """Test chaining multiple llm_query plugins."""

    def test_two_pass_llm_workflow(self):
        """Second llm_query plugin can use outputs from first.

        This test verifies the core use case:
        Pass 1: Extract key points from text
        Pass 2: Generate summary using key points from Pass 1
        """
        # First plugin extracts key points
        plugin1 = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 1}},
            queries=[
                {
                    "name": "extract_key_points",
                    "inputs": {"text": "input_text"},
                    "output_key": "key_points",
                    "flatten_to_row": True,  # Flatten so Pass 2 can access
                }
            ],
            system_prompt="You extract key points from text.",
            user_prompt="Extract key points from: {{ text }}",
        )

        # Second plugin generates summary using key points
        plugin2 = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 2}},
            queries=[
                {
                    "name": "generate_summary",
                    "inputs": {
                        "key_points": "key_points_content",  # From plugin1 flattened output
                        "original_text": "input_text",
                    },
                    "output_key": "summary",
                    "flatten_to_row": True,
                }
            ],
            system_prompt="You generate summaries from key points.",
            user_prompt="Generate summary from: {{ key_points }}. Original: {{ original_text }}",
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin1, plugin2],
            prompt_fields=["input_text"],
        )

        df = pd.DataFrame([{"input_text": "This is a long article about technology..."}])

        result = runner.run(df)

        # Verify both plugins executed
        assert len(result["results"]) == 1
        record = result["results"][0]

        # Verify plugin1 output is present and flattened
        assert "key_points_content" in record["row"]
        assert isinstance(record["row"]["key_points_content"], str)

        # Verify plugin2 output is present and flattened
        assert "summary_content" in record["row"]
        assert isinstance(record["row"]["summary_content"], str)

        # Verify context has full responses from both plugins
        assert "key_points" in record["context"]
        assert "summary" in record["context"]
        assert "content" in record["context"]["key_points"]
        assert "content" in record["context"]["summary"]

    def test_context_communication_between_plugins(self):
        """Plugins can communicate through context dict."""
        # Plugin1 stores intermediate data in context (no flatten)
        plugin1 = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 42}},
            queries=[
                {
                    "name": "evaluate_criteria",
                    "inputs": {"item": "item_name"},
                    "output_key": "criteria_scores",
                    "flatten_to_row": False,  # Keep only in context
                }
            ],
            system_prompt="Evaluate against criteria.",
            user_prompt="Evaluate: {{ item }}",
        )

        # Plugin2 reads from context to make final decision
        call_count = {"n": 0}

        class ContextAwarePlugin:
            """Plugin that reads from context."""

            name = "context_reader"

            def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
                call_count["n"] += 1
                # Read data from plugin1 via context
                criteria = context.get("criteria_scores", {})
                # Mock returns metrics.score derived from hash
                score = criteria.get("metrics", {}).get("score", 0)
                row["decision"] = "PASS" if score >= 0.5 else "FAIL"
                row["used_context"] = True
                return row

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin1, ContextAwarePlugin()],
            prompt_fields=["item_name"],
        )

        df = pd.DataFrame([{"item_name": "Product A"}])

        result = runner.run(df)

        assert len(result["results"]) == 1
        record = result["results"][0]

        # Plugin1 didn't flatten (shouldn't be in row)
        assert "criteria_scores_content" not in record["row"]

        # But context has the data
        assert "criteria_scores" in record["context"]
        assert "metrics" in record["context"]["criteria_scores"]

        # Plugin2 read context and made decision
        assert record["row"]["decision"] in ("PASS", "FAIL")
        assert record["row"]["used_context"] is True
        assert call_count["n"] == 1

    def test_multiple_queries_in_single_plugin(self):
        """Single llm_query plugin with multiple queries."""
        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 123}},
            queries=[
                {
                    "name": "clarity_check",
                    "inputs": {"text": "input_text"},
                    "defaults": {"criterion": "clarity"},
                    "output_key": "clarity",
                },
                {
                    "name": "accuracy_check",
                    "inputs": {"text": "input_text"},
                    "defaults": {"criterion": "accuracy"},
                    "output_key": "accuracy",
                },
                {
                    "name": "completeness_check",
                    "inputs": {"text": "input_text"},
                    "defaults": {"criterion": "completeness"},
                    "output_key": "completeness",
                },
            ],
            system_prompt="Evaluate text quality.",
            user_prompt="Evaluate {{ criterion }} of: {{ text }}",
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin],
            prompt_fields=["input_text"],
        )

        df = pd.DataFrame([{"input_text": "Sample text for evaluation"}])

        result = runner.run(df)

        record = result["results"][0]

        # All three queries should be in context
        assert "clarity" in record["context"]
        assert "accuracy" in record["context"]
        assert "completeness" in record["context"]

        # All should be flattened to row with content
        assert "clarity_content" in record["row"]
        assert "accuracy_content" in record["row"]
        assert "completeness_content" in record["row"]

    def test_three_plugin_chain(self):
        """Chain of three llm_query plugins."""
        # Plugin 1: Extract entities
        plugin1 = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 10}},
            queries=[{"name": "extract", "inputs": {"doc": "document"}, "output_key": "entities"}],
            system_prompt="Extract entities.",
            user_prompt="Extract from: {{ doc }}",
        )

        # Plugin 2: Classify entities
        plugin2 = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 20}},
            queries=[
                {
                    "name": "classify",
                    "inputs": {"entities": "entities_content"},  # From plugin1
                    "output_key": "classified",
                }
            ],
            system_prompt="Classify entities.",
            user_prompt="Classify: {{ entities }}",
        )

        # Plugin 3: Generate insights
        plugin3 = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 30}},
            queries=[
                {
                    "name": "insights",
                    "inputs": {
                        "entities": "entities_content",
                        "classes": "classified_content",
                    },
                    "output_key": "final",
                }
            ],
            system_prompt="Generate insights.",
            user_prompt="Insights for {{ entities }} classified as {{ classes }}",
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin1, plugin2, plugin3],
            prompt_fields=["document"],
        )

        df = pd.DataFrame([{"document": "Article about big tech companies..."}])

        result = runner.run(df)

        record = result["results"][0]

        # All three contexts should be present
        assert "entities" in record["context"]
        assert "classified" in record["context"]
        assert "final" in record["context"]

        # Final insight should be in row
        assert "final_content" in record["row"]
        assert isinstance(record["row"]["final_content"], str)

    def test_chain_with_error_in_middle_plugin(self):
        """Error in middle plugin should be captured as failure."""

        class FailingPlugin:
            """Plugin that raises an error."""

            name = "failing_plugin"

            def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
                raise ValueError("Simulated failure in middle plugin")

        plugin1 = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 1}},
            queries=[{"name": "q1", "inputs": {}, "output_key": "first"}],
            system_prompt="System",
            user_prompt="User",
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin1, FailingPlugin()],
            prompt_fields=["text"],
        )

        df = pd.DataFrame([{"text": "Test"}])

        result = runner.run(df)

        # Should have a failure, not a result
        assert len(result["results"]) == 0
        assert len(result["failures"]) == 1
        assert "Simulated failure in middle plugin" in result["failures"][0]["error"]

    def test_full_sda_pipeline_with_llm_query(self):
        """Full end-to-end test with checkpointing and sink write."""
        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 99}},
            queries=[
                {
                    "name": "sentiment_analysis",
                    "inputs": {"review": "review_text"},
                    "output_key": "sentiment",
                }
            ],
            system_prompt="Analyze sentiment.",
            user_prompt="Analyze: {{ review }}",
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin],
            prompt_fields=["id", "review_text"],
        )

        df = pd.DataFrame(
            [
                {"id": "1", "review_text": "Great product!"},
                {"id": "2", "review_text": "Terrible experience."},
                {"id": "3", "review_text": "It's okay, nothing special."},
            ]
        )

        result = runner.run(df)

        # All rows processed
        assert len(result["results"]) == 3
        assert result["metadata"]["rows"] == 3
        assert result["metadata"]["row_count"] == 3

        # Sink received payload
        assert sink.written_payload is not None
        assert len(sink.written_payload["results"]) == 3

        # Each record has sentiment analysis
        for record in result["results"]:
            assert "sentiment_content" in record["row"]
            # Mock returns metrics.score from hash
            assert "metrics" in record["context"]["sentiment"]

    def test_llm_query_with_rate_limiter(self):
        """LLM query plugin with rate limiter config."""
        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 1}},
            queries=[{"name": "q1", "inputs": {}, "output_key": "out"}],
            system_prompt="System",
            user_prompt="User",
            rate_limiter={"plugin": "fixed_window", "options": {"requests_per_minute": 60}},
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin],
            prompt_fields=["text"],
        )

        df = pd.DataFrame([{"text": f"Row {i}"} for i in range(5)])

        result = runner.run(df)

        # All rows should process (rate limiter shouldn't block mock)
        assert len(result["results"]) == 5


class TestLLMQueryConfigPatterns:
    """Test various configuration patterns for llm_query plugin."""

    def test_flatten_disabled_preserves_row(self):
        """When flatten_to_row=False, row only has original data."""
        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 50}},
            queries=[
                {
                    "name": "internal_calc",
                    "inputs": {},
                    "output_key": "internal",
                    "flatten_to_row": False,  # Don't expose in row
                }
            ],
            system_prompt="System",
            user_prompt="User",
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin],
            prompt_fields=["value"],
        )

        df = pd.DataFrame([{"value": 42}])

        result = runner.run(df)

        record = result["results"][0]

        # Original field preserved
        assert record["row"]["value"] == 42

        # Flattened fields NOT in row
        assert "internal_content" not in record["row"]

        # But context has full response
        assert "internal" in record["context"]
        assert "content" in record["context"]["internal"]

    def test_query_defaults_merge_with_row_data(self):
        """Query defaults provide fallback values for template vars."""
        captured_prompts: list[str] = []

        class CapturingPlugin(LLMQueryPlugin):
            """Subclass that captures rendered prompts."""

            def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
                # Capture the rendered prompt before execution
                for query in self.queries:
                    render_ctx: dict[str, Any] = {**self.pack_defaults}
                    render_ctx.update(query.get("defaults", {}))
                    for template_var, row_field in query.get("inputs", {}).items():
                        render_ctx[template_var] = row.get(row_field)
                    captured_prompts.append(self.engine.render(self.user_template, render_ctx))
                return super().transform(row, context)

        plugin = CapturingPlugin(
            llm={"plugin": "mock", "options": {"seed": 1}},
            queries=[
                {
                    "name": "q1",
                    "inputs": {"text": "input_text"},
                    "defaults": {"language": "English", "format": "JSON"},  # Defaults
                    "output_key": "out",
                }
            ],
            system_prompt="System",
            user_prompt="Process {{ text }} in {{ language }} as {{ format }}.",
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin],
            prompt_fields=["input_text"],
        )

        df = pd.DataFrame([{"input_text": "Hello world"}])

        runner.run(df)

        # Defaults should be used in rendered prompt
        assert len(captured_prompts) == 1
        assert "English" in captured_prompts[0]
        assert "JSON" in captured_prompts[0]
        assert "Hello world" in captured_prompts[0]

    def test_multiple_rows_processed_independently(self):
        """Each row gets its own context - no cross-contamination."""
        plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 77}},
            queries=[
                {
                    "name": "process",
                    "inputs": {"text": "text"},
                    "output_key": "result",
                }
            ],
            system_prompt="Process text.",
            user_prompt="Process: {{ text }}",
        )

        sink = MockSink()

        runner = SDARunner(
            sinks=[sink],
            transform_plugins=[plugin],
            prompt_fields=["id", "text"],
        )

        df = pd.DataFrame(
            [
                {"id": "1", "text": "First row"},
                {"id": "2", "text": "Second row"},
                {"id": "3", "text": "Third row"},
            ]
        )

        result = runner.run(df)

        # All rows processed
        assert len(result["results"]) == 3

        # Each row has its own context (not shared)
        for i, record in enumerate(result["results"]):
            assert "result" in record["context"]
            # Each row's original id preserved
            assert record["row"]["id"] == str(i + 1)


class TestLLMQueryEndToEnd:
    """End-to-end tests with real file I/O."""

    def test_two_llm_plugins_chained_to_csv_output(self, tmp_path: Path):
        """End-to-end test: two LLM plugins chained together with CSV output.

        This test verifies the complete workflow:
        1. Input data with text to process
        2. First LLM plugin extracts key points from text
        3. Second LLM plugin generates summary using key points from first plugin
        4. Results written to CSV with all columns (input, intermediate, final)
        """
        csv_output_path = tmp_path / "results.csv"

        # First plugin: Extract key points from the input text
        extract_plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 100}},
            queries=[
                {
                    "name": "extract_key_points",
                    "inputs": {"text": "article_text"},
                    "output_key": "key_points",
                    "flatten_to_row": True,  # Makes key_points_content available for next plugin
                }
            ],
            system_prompt="You are an expert at extracting key points from articles.",
            user_prompt="Extract the main key points from the following text:\n\n{{ text }}",
        )

        # Second plugin: Generate summary using key points from first plugin
        summarize_plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 200}},
            queries=[
                {
                    "name": "generate_summary",
                    "inputs": {
                        "key_points": "key_points_content",  # Output from first plugin
                        "original": "article_text",  # Original input for context
                    },
                    "output_key": "summary",
                    "flatten_to_row": True,
                }
            ],
            system_prompt="You are an expert summarizer.",
            user_prompt="Based on these key points:\n{{ key_points }}\n\nAnd the original text:\n{{ original }}\n\nGenerate a concise summary.",
        )

        # CSV sink to write results
        csv_sink = CsvResultSink(path=str(csv_output_path), overwrite=True)

        # Create runner with both plugins chained
        runner = SDARunner(
            sinks=[csv_sink],
            transform_plugins=[extract_plugin, summarize_plugin],
            prompt_fields=["id", "article_text", "category"],
        )

        # Input data
        input_df = pd.DataFrame(
            [
                {
                    "id": "article_001",
                    "article_text": "Artificial intelligence is transforming healthcare. Machine learning models can now detect diseases earlier than human doctors. This technology promises to save millions of lives.",
                    "category": "technology",
                },
                {
                    "id": "article_002",
                    "article_text": "Climate change is accelerating faster than predicted. Sea levels are rising and extreme weather events are becoming more common. Urgent action is needed.",
                    "category": "environment",
                },
                {
                    "id": "article_003",
                    "article_text": "The global economy is showing signs of recovery. Employment rates are improving and consumer spending is up. However, inflation remains a concern.",
                    "category": "economics",
                },
            ]
        )

        # Run the pipeline
        result = runner.run(input_df)

        # Verify all rows processed successfully
        assert len(result["results"]) == 3
        assert len(result.get("failures", [])) == 0

        # Verify CSV file was created
        assert csv_output_path.exists(), "CSV output file should be created"

        # Read back the CSV and verify all columns are present
        output_df = pd.read_csv(csv_output_path)

        # Check row count
        assert len(output_df) == 3, f"Expected 3 rows, got {len(output_df)}"

        # Check that original input columns are present
        assert "id" in output_df.columns, "Original 'id' column should be in output"
        assert "article_text" in output_df.columns, "Original 'article_text' column should be in output"
        assert "category" in output_df.columns, "Original 'category' column should be in output"

        # Check that first plugin outputs are present (flattened)
        assert "key_points_content" in output_df.columns, "First plugin 'key_points_content' should be in output"

        # Check that second plugin outputs are present (flattened)
        assert "summary_content" in output_df.columns, "Second plugin 'summary_content' should be in output"

        # Verify data integrity - IDs should match input
        assert list(output_df["id"]) == ["article_001", "article_002", "article_003"]

        # Verify categories preserved
        assert list(output_df["category"]) == ["technology", "environment", "economics"]

        # Verify LLM outputs are non-empty strings
        for idx, row in output_df.iterrows():
            assert isinstance(row["key_points_content"], str), f"Row {idx}: key_points_content should be string"
            assert len(row["key_points_content"]) > 0, f"Row {idx}: key_points_content should not be empty"
            assert isinstance(row["summary_content"], str), f"Row {idx}: summary_content should be string"
            assert len(row["summary_content"]) > 0, f"Row {idx}: summary_content should not be empty"

        # Print columns for visibility (useful when running test directly)
        print(f"\nCSV Output Columns: {list(output_df.columns)}")
        print(f"Sample row:\n{output_df.iloc[0].to_dict()}")

    def test_three_stage_pipeline_with_csv_output(self, tmp_path: Path):
        """End-to-end test: three LLM plugins in sequence with CSV output.

        Pipeline:
        1. Extract entities from text
        2. Classify extracted entities
        3. Generate insights based on entities and classifications
        """
        csv_output_path = tmp_path / "three_stage_results.csv"

        # Stage 1: Extract entities
        extract_plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 10}},
            queries=[
                {
                    "name": "extract_entities",
                    "inputs": {"doc": "document"},
                    "output_key": "entities",
                    "flatten_to_row": True,
                }
            ],
            system_prompt="Extract named entities from documents.",
            user_prompt="Extract all named entities from: {{ doc }}",
        )

        # Stage 2: Classify entities
        classify_plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 20}},
            queries=[
                {
                    "name": "classify_entities",
                    "inputs": {"entities": "entities_content"},
                    "output_key": "classifications",
                    "flatten_to_row": True,
                }
            ],
            system_prompt="Classify entities by type.",
            user_prompt="Classify these entities: {{ entities }}",
        )

        # Stage 3: Generate insights
        insights_plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 30}},
            queries=[
                {
                    "name": "generate_insights",
                    "inputs": {
                        "entities": "entities_content",
                        "classes": "classifications_content",
                        "original": "document",
                    },
                    "output_key": "insights",
                    "flatten_to_row": True,
                }
            ],
            system_prompt="Generate business insights.",
            user_prompt="Generate insights from entities: {{ entities }}, classifications: {{ classes }}, original: {{ original }}",
        )

        csv_sink = CsvResultSink(path=str(csv_output_path), overwrite=True)

        runner = SDARunner(
            sinks=[csv_sink],
            transform_plugins=[extract_plugin, classify_plugin, insights_plugin],
            prompt_fields=["id", "document", "source"],
        )

        input_df = pd.DataFrame(
            [
                {
                    "id": "doc_1",
                    "document": "Apple Inc. announced new products in Cupertino. Tim Cook presented the iPhone 15.",
                    "source": "news",
                },
                {
                    "id": "doc_2",
                    "document": "Microsoft Azure expanded to new regions. Satya Nadella spoke at the conference.",
                    "source": "press_release",
                },
            ]
        )

        result = runner.run(input_df)

        # Verify success
        assert len(result["results"]) == 2
        assert len(result.get("failures", [])) == 0
        assert csv_output_path.exists()

        # Read and verify CSV
        output_df = pd.read_csv(csv_output_path)
        assert len(output_df) == 2

        # All expected columns present
        expected_columns = [
            "id",
            "document",
            "source",
            "entities_content",
            "classifications_content",
            "insights_content",
        ]
        for col in expected_columns:
            assert col in output_df.columns, f"Column '{col}' should be in output"

        # All LLM outputs should be non-empty
        for col in ["entities_content", "classifications_content", "insights_content"]:
            for idx, val in enumerate(output_df[col]):
                assert isinstance(val, str) and len(val) > 0, f"Row {idx}, column {col} should have content"

    def test_chained_plugins_with_selective_flattening(self, tmp_path: Path):
        """Test that flatten_to_row=False keeps data only in context, not CSV.

        Verifies that:
        - Intermediate results with flatten_to_row=False don't appear in CSV
        - Final results with flatten_to_row=True do appear in CSV
        - Context-only data can still be used by subsequent plugins
        """
        csv_output_path = tmp_path / "selective_flatten.csv"

        # First plugin: Internal processing (NOT flattened to row)
        internal_plugin = LLMQueryPlugin(
            llm={"plugin": "mock", "options": {"seed": 50}},
            queries=[
                {
                    "name": "internal_analysis",
                    "inputs": {"text": "input_text"},
                    "output_key": "internal",
                    "flatten_to_row": False,  # Keep only in context
                }
            ],
            system_prompt="Internal analysis.",
            user_prompt="Analyze internally: {{ text }}",
        )

        # Second plugin: Reads from context and produces final output
        # Note: Since internal is not flattened, we need to access it differently
        # For this test, we'll use a plugin that reads from context directly

        class ContextReaderPlugin:
            """Plugin that reads from context and produces final output."""

            name = "context_reader"

            def transform(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
                # Read internal analysis from context
                internal_result = context.get("internal", {})
                internal_content = internal_result.get("content", "no_content")

                # Add final output to row (this WILL appear in CSV)
                row["final_output"] = f"Processed: {internal_content[:50]}..."
                row["has_internal_context"] = "internal" in context
                return row

        csv_sink = CsvResultSink(path=str(csv_output_path), overwrite=True)

        runner = SDARunner(
            sinks=[csv_sink],
            transform_plugins=[internal_plugin, ContextReaderPlugin()],
            prompt_fields=["id", "input_text"],
        )

        input_df = pd.DataFrame(
            [
                {"id": "1", "input_text": "Sample text for processing"},
                {"id": "2", "input_text": "Another sample text"},
            ]
        )

        result = runner.run(input_df)

        assert len(result["results"]) == 2
        assert csv_output_path.exists()

        output_df = pd.read_csv(csv_output_path)

        # Original columns present
        assert "id" in output_df.columns
        assert "input_text" in output_df.columns

        # Internal plugin output NOT in CSV (flatten_to_row=False)
        assert "internal_content" not in output_df.columns, "internal_content should NOT be in CSV"

        # Final output IS in CSV
        assert "final_output" in output_df.columns, "final_output should be in CSV"
        assert "has_internal_context" in output_df.columns

        # Verify context was passed (all rows should have had internal context)
        for val in output_df["has_internal_context"]:
            assert val is True or val == "True", "Context should have been available"
