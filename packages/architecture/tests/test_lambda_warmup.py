"""Tests for Lambda Warmup scheduler."""

import pytest
import math

from bitnet_arch.quantization.lambda_warmup import (
    LambdaWarmup,
    get_global_lambda_warmup,
    set_global_lambda_warmup,
    get_current_lambda,
)


class TestLambdaWarmup:
    """Test LambdaWarmup scheduler."""

    def test_init_defaults(self):
        """Test default initialization."""
        warmup = LambdaWarmup()
        assert warmup.warmup_steps == 1000
        assert warmup.schedule == "linear"
        assert warmup.min_lambda == 0.0
        assert warmup.max_lambda == 1.0
        assert warmup.lambda_val == 0.0

    def test_linear_schedule(self):
        """Test linear warmup schedule."""
        warmup = LambdaWarmup(warmup_steps=100, schedule="linear")

        # At step 0, lambda should be 0
        assert warmup.lambda_val == 0.0

        # At step 50, lambda should be 0.5
        for _ in range(50):
            warmup.step()
        assert abs(warmup.lambda_val - 0.5) < 0.01

        # At step 100, lambda should be 1.0
        for _ in range(50):
            warmup.step()
        assert warmup.lambda_val == 1.0

    def test_cosine_schedule(self):
        """Test cosine warmup schedule."""
        warmup = LambdaWarmup(warmup_steps=100, schedule="cosine")

        # At step 0
        assert warmup.lambda_val == 0.0

        # At step 50 (halfway), cosine gives 0.5
        for _ in range(50):
            warmup.step()
        expected = (1 - math.cos(math.pi * 0.5)) / 2
        assert abs(warmup.lambda_val - expected) < 0.01

    def test_warmup_complete(self):
        """Test is_warmup_complete method."""
        warmup = LambdaWarmup(warmup_steps=10)

        assert not warmup.is_warmup_complete()

        for _ in range(10):
            warmup.step()

        assert warmup.is_warmup_complete()

    def test_lambda_clamps_at_max(self):
        """Test that lambda doesn't exceed max after warmup."""
        warmup = LambdaWarmup(warmup_steps=10)

        for _ in range(100):  # Way more than warmup_steps
            warmup.step()

        assert warmup.lambda_val == 1.0

    def test_custom_min_max(self):
        """Test custom min/max lambda values."""
        warmup = LambdaWarmup(
            warmup_steps=100,
            min_lambda=0.5,
            max_lambda=0.9,
        )

        assert warmup.lambda_val == 0.5

        for _ in range(100):
            warmup.step()

        assert warmup.lambda_val == 0.9

    def test_state_dict(self):
        """Test state serialization."""
        warmup = LambdaWarmup(warmup_steps=100)
        for _ in range(30):
            warmup.step()

        state = warmup.state_dict()

        assert state["current_step"] == 30
        assert "lambda" in state
        assert state["warmup_steps"] == 100

    def test_load_state_dict(self):
        """Test state restoration."""
        warmup1 = LambdaWarmup(warmup_steps=100)
        for _ in range(30):
            warmup1.step()
        state = warmup1.state_dict()

        warmup2 = LambdaWarmup(warmup_steps=100)
        warmup2.load_state_dict(state)

        assert warmup2.current_step == 30
        assert warmup2.lambda_val == warmup1.lambda_val


class TestGlobalLambda:
    """Test global lambda management."""

    def teardown_method(self):
        """Reset global state after each test."""
        set_global_lambda_warmup(None)

    def test_get_current_lambda_no_warmup(self):
        """Test default lambda when no warmup set."""
        set_global_lambda_warmup(None)
        assert get_current_lambda() == 1.0

    def test_get_current_lambda_with_warmup(self):
        """Test lambda with warmup set."""
        warmup = LambdaWarmup(warmup_steps=100)
        set_global_lambda_warmup(warmup)

        assert get_current_lambda() == 0.0

        for _ in range(50):
            warmup.step()

        assert abs(get_current_lambda() - 0.5) < 0.01

    def test_get_global_lambda_warmup(self):
        """Test getting global warmup instance."""
        assert get_global_lambda_warmup() is None

        warmup = LambdaWarmup()
        set_global_lambda_warmup(warmup)

        assert get_global_lambda_warmup() is warmup
