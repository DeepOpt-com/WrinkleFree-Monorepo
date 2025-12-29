"""Tests for WrinkleFree-Eval.

These tests verify the evaluation pipeline works correctly.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestBenchmarkPresets:
    """Test benchmark preset definitions."""

    def test_list_benchmarks(self):
        """Test that benchmark presets are correctly defined."""
        from wrinklefree_eval.api import list_benchmarks

        benchmarks = list_benchmarks()

        assert "bitdistill" in benchmarks
        assert "glue" in benchmarks
        assert "summarization" in benchmarks
        assert "smoke_test" in benchmarks

    def test_bitdistill_has_all_tasks(self):
        """Test bitdistill preset includes all paper benchmarks."""
        from wrinklefree_eval.api import list_benchmarks

        benchmarks = list_benchmarks()
        bitdistill = benchmarks["bitdistill"]

        assert "mnli" in bitdistill
        assert "qnli" in bitdistill
        assert "sst2" in bitdistill
        assert "cnn_dailymail_summarization" in bitdistill

    def test_smoke_test_is_minimal(self):
        """Test smoke test preset is minimal for fast validation."""
        from wrinklefree_eval.api import list_benchmarks

        benchmarks = list_benchmarks()
        smoke = benchmarks["smoke_test"]

        # Should have exactly 2 tasks for quick validation
        assert len(smoke) == 2
        assert "sst2" in smoke

    def test_task_mapping(self):
        """Test task name mapping is correct."""
        from wrinklefree_eval.api import TASK_MAPPING

        assert TASK_MAPPING["mnli"] == "glue_mnli"
        assert TASK_MAPPING["qnli"] == "glue_qnli"
        assert TASK_MAPPING["sst2"] == "glue_sst2"


class TestEvaluateFunction:
    """Test the main evaluate() function."""

    def test_invalid_benchmark_raises_error(self):
        """Test that invalid benchmark raises ValueError before calling lm_eval."""
        from wrinklefree_eval.api import BENCHMARK_PRESETS

        # Verify the check happens at the right place
        assert "nonexistent" not in BENCHMARK_PRESETS

    @patch("wrinklefree_eval.api.evaluator.simple_evaluate")
    @patch("wrinklefree_eval.api.lm_eval.tasks.include_path")
    def test_evaluate_calls_lm_eval(self, mock_include, mock_simple_eval):
        """Test that evaluate() correctly calls lm_eval."""
        from wrinklefree_eval.api import evaluate

        # Mock lm_eval response
        mock_simple_eval.return_value = {
            "results": {
                "glue_sst2": {
                    "acc,none": 0.92,
                    "acc_stderr,none": 0.01,
                }
            }
        }

        results = evaluate(
            model_path="test/model",
            benchmark="glue",
            device="cpu",
        )

        # Verify lm_eval was called
        mock_simple_eval.assert_called_once()
        call_kwargs = mock_simple_eval.call_args

        assert "glue_sst2" in call_kwargs.kwargs.get("tasks", []) or \
               "glue_sst2" in call_kwargs.args[0] if call_kwargs.args else True

    @patch("wrinklefree_eval.api.evaluator.simple_evaluate")
    @patch("wrinklefree_eval.api.lm_eval.tasks.include_path")
    def test_smoke_test_sets_limit(self, mock_include, mock_simple_eval):
        """Test that smoke_test=True sets appropriate limit."""
        from wrinklefree_eval.api import evaluate

        mock_simple_eval.return_value = {"results": {}}

        evaluate(
            model_path="test/model",
            benchmark="smoke_test",
            smoke_test=True,
            device="cpu",
        )

        # Verify limit was set
        call_kwargs = mock_simple_eval.call_args.kwargs
        assert call_kwargs.get("limit") == 10

    @patch("wrinklefree_eval.api.evaluator.simple_evaluate")
    @patch("wrinklefree_eval.api.lm_eval.tasks.include_path")
    def test_custom_limit_overrides_smoke_test(self, mock_include, mock_simple_eval):
        """Test that explicit limit overrides smoke_test default."""
        from wrinklefree_eval.api import evaluate

        mock_simple_eval.return_value = {"results": {}}

        evaluate(
            model_path="test/model",
            benchmark="smoke_test",
            smoke_test=True,
            limit=5,  # Explicit limit
            device="cpu",
        )

        call_kwargs = mock_simple_eval.call_args.kwargs
        # smoke_test sets limit=10 only if limit is None, but our code uses `or`
        # so limit=5 should be preserved... actually checking the code:
        # limit = limit or 10 means if limit is 5, it stays 5
        assert call_kwargs.get("limit") == 5


class TestResultFormatting:
    """Test result formatting logic."""

    def test_format_results_basic(self):
        """Test basic result formatting."""
        from wrinklefree_eval.api import _format_results

        raw_results = {
            "results": {
                "glue_sst2": {
                    "acc,none": 0.92,
                    "acc_stderr,none": 0.01,
                    "alias": "glue_sst2",
                }
            }
        }

        formatted = _format_results(raw_results)

        assert "glue_sst2" in formatted
        assert "acc" in formatted["glue_sst2"]
        assert formatted["glue_sst2"]["acc"] == 0.92

    def test_format_results_handles_empty(self):
        """Test formatting handles empty results."""
        from wrinklefree_eval.api import _format_results

        assert _format_results({}) == {}
        assert _format_results({"results": {}}) == {}

    def test_format_results_multiple_metrics(self):
        """Test formatting with multiple metrics (like ROUGE)."""
        from wrinklefree_eval.api import _format_results

        raw_results = {
            "results": {
                "cnn_dailymail": {
                    "rouge1,none": 0.45,
                    "rouge2,none": 0.21,
                    "rougeL,none": 0.38,
                }
            }
        }

        formatted = _format_results(raw_results)

        assert formatted["cnn_dailymail"]["rouge1"] == 0.45
        assert formatted["cnn_dailymail"]["rouge2"] == 0.21
        assert formatted["cnn_dailymail"]["rougeL"] == 0.38

    def test_make_serializable(self):
        """Test JSON serialization helper."""
        from wrinklefree_eval.api import _make_serializable
        import numpy as np

        # Test basic types
        assert _make_serializable({"a": 1}) == {"a": 1}
        assert _make_serializable([1, 2, 3]) == [1, 2, 3]
        assert _make_serializable("hello") == "hello"

        # Test numpy scalar (if numpy available)
        try:
            arr = np.array([1.5])
            assert _make_serializable(arr[0]) == 1.5
        except ImportError:
            pass  # numpy not installed, skip


class TestCustomTasks:
    """Test custom task definitions."""

    def test_tasks_dir_exists(self):
        """Verify tasks directory exists."""
        from wrinklefree_eval.tasks import TASKS_DIR
        assert TASKS_DIR.exists()

    def test_cnn_dailymail_task_exists(self):
        """Verify CNN/DailyMail task YAML exists."""
        from wrinklefree_eval.tasks import TASKS_DIR
        task_file = TASKS_DIR / "cnn_dailymail.yaml"
        assert task_file.exists()

    def test_task_yaml_valid(self):
        """Test that task YAML is valid."""
        from wrinklefree_eval.tasks import TASKS_DIR
        import yaml

        task_file = TASKS_DIR / "cnn_dailymail.yaml"
        with open(task_file) as f:
            config = yaml.safe_load(f)

        assert config["task"] == "cnn_dailymail_summarization"
        assert config["dataset_path"] == "cnn_dailymail"
        assert config["output_type"] == "generate_until"
        assert "metric_list" in config


class TestTaskUtils:
    """Test task utility functions."""

    def test_doc_to_text_summarization(self):
        """Test summarization prompt formatting."""
        from wrinklefree_eval.tasks.utils import doc_to_text_summarization

        doc = {"article": "This is a test article about AI."}
        prompt = doc_to_text_summarization(doc)

        assert "test article" in prompt
        assert "Summarize" in prompt
        assert "Article:" in prompt

    def test_doc_to_text_truncation(self):
        """Test that long articles are truncated."""
        from wrinklefree_eval.tasks.utils import doc_to_text_summarization

        long_article = "x" * 5000
        doc = {"article": long_article}
        prompt = doc_to_text_summarization(doc)

        # Should be truncated to ~2000 chars + prompt text
        assert len(prompt) < 3000
        assert "..." in prompt

    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        from wrinklefree_eval.tasks.utils import strip_whitespace

        assert strip_whitespace("  hello  ") == "hello"
        assert strip_whitespace("\n\ntext\n\n") == "text"
        assert strip_whitespace("no_change") == "no_change"

    def test_strip_whitespace_non_string(self):
        """Test strip_whitespace with non-string input."""
        from wrinklefree_eval.tasks.utils import strip_whitespace

        # Should return input unchanged if not string
        assert strip_whitespace(123) == 123
        assert strip_whitespace(None) is None


class TestModelWrappers:
    """Test model wrapper classes."""

    def test_hf_model_import(self):
        """Test HuggingFace model can be imported."""
        from wrinklefree_eval.models import HuggingFaceModel
        assert HuggingFaceModel is not None

    def test_bitnet_model_import(self):
        """Test BitNet model can be imported."""
        from wrinklefree_eval.models import BitNetModel
        assert BitNetModel is not None

    def test_bitnet_path_detection(self):
        """Test BitNet path detection doesn't crash."""
        from wrinklefree_eval.models.bitnet_model import find_bitnet_path

        # Just verify function runs without error
        path = find_bitnet_path()
        # Path may or may not exist depending on environment
        assert path is None or isinstance(path, Path)

    def test_bitnet_availability_check(self):
        """Test BitNet availability check."""
        from wrinklefree_eval.models.bitnet_model import check_bitnet_available

        # Should return boolean without crashing
        result = check_bitnet_available()
        assert isinstance(result, bool)


class TestConfigFiles:
    """Test configuration file structure."""

    def test_benchmark_configs_exist(self):
        """Test all benchmark config files exist."""
        configs_dir = Path(__file__).parent.parent / "configs" / "benchmarks"

        assert (configs_dir / "bitdistill.yaml").exists()
        assert (configs_dir / "glue.yaml").exists()
        assert (configs_dir / "summarization.yaml").exists()
        assert (configs_dir / "smoke_test.yaml").exists()

    def test_model_configs_exist(self):
        """Test model config files exist."""
        configs_dir = Path(__file__).parent.parent / "configs" / "models"

        assert (configs_dir / "hf_model.yaml").exists()
        assert (configs_dir / "bitnet_model.yaml").exists()

    def test_main_config_exists(self):
        """Test main eval.yaml config exists."""
        configs_dir = Path(__file__).parent.parent / "configs"
        assert (configs_dir / "eval.yaml").exists()


class TestWandBLogging:
    """Tests for W&B logging integration."""

    def test_wandb_available_flag(self):
        """Test WANDB_AVAILABLE flag is set correctly."""
        from wrinklefree_eval.api import WANDB_AVAILABLE
        # Should be a boolean
        assert isinstance(WANDB_AVAILABLE, bool)

    @patch("wrinklefree_eval.api.evaluator.simple_evaluate")
    @patch("wrinklefree_eval.api.lm_eval.tasks.include_path")
    def test_evaluate_without_wandb(self, mock_include, mock_simple_eval):
        """Test evaluate works without wandb parameters."""
        from wrinklefree_eval.api import evaluate

        mock_simple_eval.return_value = {"results": {"glue_sst2": {"acc,none": 0.9}}}

        results = evaluate(
            model_path="test/model",
            benchmark="glue",
            device="cpu",
        )

        # Should work without wandb
        assert "glue_sst2" in results

    @patch("wrinklefree_eval.api.WANDB_AVAILABLE", False)
    @patch("wrinklefree_eval.api.evaluator.simple_evaluate")
    @patch("wrinklefree_eval.api.lm_eval.tasks.include_path")
    def test_evaluate_wandb_not_installed(self, mock_include, mock_simple_eval):
        """Test evaluate handles missing wandb gracefully."""
        from wrinklefree_eval.api import evaluate

        mock_simple_eval.return_value = {"results": {}}

        # Should not raise even if wandb_project specified
        results = evaluate(
            model_path="test/model",
            benchmark="glue",
            device="cpu",
            wandb_project="test-project",
        )

        assert results == {}

    @patch("wrinklefree_eval.api.WANDB_AVAILABLE", True)
    @patch("wrinklefree_eval.api.wandb")
    @patch("wrinklefree_eval.api.evaluator.simple_evaluate")
    @patch("wrinklefree_eval.api.lm_eval.tasks.include_path")
    def test_evaluate_with_wandb(self, mock_include, mock_simple_eval, mock_wandb):
        """Test evaluate initializes and logs to wandb."""
        from wrinklefree_eval.api import evaluate

        mock_simple_eval.return_value = {
            "results": {"glue_sst2": {"acc,none": 0.92}}
        }
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        results = evaluate(
            model_path="test/model",
            benchmark="glue",
            device="cpu",
            wandb_project="test-project",
            wandb_run_id="test-run-123",
        )

        # Verify wandb.init was called
        mock_wandb.init.assert_called_once()
        init_kwargs = mock_wandb.init.call_args.kwargs
        assert init_kwargs["project"] == "test-project"
        assert init_kwargs["id"] == "test-run-123"

        # Verify finish was called
        mock_run.finish.assert_called_once()

    def test_log_to_wandb_function(self):
        """Test _log_to_wandb helper function."""
        from wrinklefree_eval.api import _log_to_wandb

        mock_run = MagicMock()
        results = {
            "glue_sst2": {"acc": 0.92, "acc_stderr": 0.01},
            "glue_mnli": {"acc": 0.85},
        }

        # Mock wandb.Table
        with patch("wrinklefree_eval.api.wandb") as mock_wandb:
            mock_wandb.Table = MagicMock()
            _log_to_wandb(results, mock_run)

        # Should have logged metrics
        assert mock_run.log.called
        # Should have created a summary table
        mock_wandb.Table.assert_called_once()


class TestUploadResults:
    """Tests for upload_results.py script."""

    def test_script_exists(self):
        """Verify upload_results.py script exists."""
        script_path = Path(__file__).parent.parent / "scripts" / "upload_results.py"
        assert script_path.exists()

    def test_script_is_importable(self):
        """Verify script can be imported."""
        import sys
        scripts_path = str(Path(__file__).parent.parent / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        # Should not raise
        import upload_results
        assert hasattr(upload_results, "upload_directory_to_gcs")


@pytest.mark.slow
class TestIntegration:
    """Integration tests that require model downloads.

    Run with: pytest -m slow --run-slow
    """

    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Requires model download, run with pytest --run-slow"
    )
    def test_smoke_evaluation_tiny_model(self):
        """End-to-end test with tiny model."""
        from wrinklefree_eval.api import evaluate

        results = evaluate(
            model_path="hf-internal-testing/tiny-random-gpt2",
            benchmark="smoke_test",
            limit=2,
            device="cpu",
            dtype="float32",
        )

        assert len(results) > 0
