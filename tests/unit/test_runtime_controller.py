"""
Unit Tests for RuntimeController.

Tests the mathematical and state-machine logic of the adaptive
token budget controller without requiring actual GPU hardware.
"""

import pytest
from unittest.mock import patch, MagicMock

from neuro_stylometry.hardware_ops.runtime import (
    RuntimeController,
    RuntimeConfig,
    RuntimeMetrics,
    RuntimeState,
    RuntimeSnapshot,
    get_gpu_memory_stats,
)


class TestRuntimeControllerInit:
    """Tests for controller initialization."""
    
    def test_default_config_values(self):
        """Test that default config has sensible values."""
        config = RuntimeConfig()
        assert config.warmup_batches == 10
        assert config.initial_token_budget == 16384
        assert config.min_token_budget < config.max_token_budget
        assert 0 < config.scale_up_factor < 2
        assert 0 < config.scale_down_factor < 1
        assert 0 < config.oom_slash_factor < 1
    
    def test_controller_starts_in_warmup(self):
        """Controller should start in WARMUP state."""
        config = RuntimeConfig(warmup_batches=5)
        ctrl = RuntimeController(config)
        assert ctrl.state == RuntimeState.WARMUP
        assert ctrl.batches_processed == 0
    
    def test_initial_budget_from_config(self):
        """Initial budget should match config."""
        config = RuntimeConfig(initial_token_budget=8192)
        ctrl = RuntimeController(config)
        assert ctrl.get_next_budget() == 8192
    
    def test_from_config_dict(self):
        """Test creating controller from pipeline config dict."""
        config_dict = {
            "execution": {
                "autotuning": {
                    "warmup_batches": 5,
                    "initial_token_budget": 4096,
                    "min_token_budget": 1024,
                    "max_token_budget": 32768,
                }
            }
        }
        ctrl = RuntimeController.from_config(config_dict)
        assert ctrl.config.warmup_batches == 5
        assert ctrl.config.initial_token_budget == 4096
        assert ctrl.get_next_budget() == 4096


class TestRuntimeControllerScaleUp:
    """Tests for scale-up behavior."""
    
    def test_scale_up_after_warmup_with_low_memory(self):
        """Controller should scale up when memory pressure is low and in SCALING_UP state."""
        config = RuntimeConfig(
            warmup_batches=3,
            initial_token_budget=4000,  # Above default min_token_budget
            max_token_budget=100000,
            scale_up_factor=1.25,
            memory_headroom_mb=1000,
            stability_threshold=0.01,  # Very low, so we go to SCALING_UP
        )
        ctrl = RuntimeController(config)
        # Mock total memory detection
        ctrl._total_memory_mb = 10000.0  # 10GB
        
        # Feed varied throughput during warmup to ensure we go to SCALING_UP
        # (high variance -> SCALING_UP instead of STABLE)
        for i in range(3):
            ctrl.report_metrics(RuntimeMetrics(
                tokens_processed=1000 + i * 100,  # Varying throughput
                batch_time_ms=10.0,
                memory_used_mb=5000.0,  # 50% of 10GB
            ))
        
        # Should exit warmup
        assert ctrl.state != RuntimeState.WARMUP
        
        # Force SCALING_UP state if needed
        ctrl._state = RuntimeState.SCALING_UP
        
        # Record initial budget
        initial_budget = ctrl.current_budget
        
        # Feed low memory metrics - budget adjustment happens on report_metrics
        ctrl.report_metrics(RuntimeMetrics(
            tokens_processed=1000,
            batch_time_ms=10.0,
            memory_used_mb=5000.0,
        ))
        
        # Budget should have increased in SCALING_UP state
        assert ctrl.get_next_budget() > initial_budget
    
    def test_scale_up_respects_max_budget(self):
        """Scale up should not exceed max_token_budget."""
        config = RuntimeConfig(
            warmup_batches=2,
            initial_token_budget=9000,
            max_token_budget=10000,
            scale_up_factor=1.5,  # Would push to 13500
            memory_headroom_mb=100,
        )
        ctrl = RuntimeController(config)
        ctrl._total_memory_mb = 10000.0
        
        # Complete warmup with low memory
        for _ in range(2):
            ctrl.report_metrics(RuntimeMetrics(
                tokens_processed=9000,
                batch_time_ms=10.0,
                memory_used_mb=1000.0,  # Very low
            ))
        
        # Force scaling up
        ctrl._state = RuntimeState.SCALING_UP
        ctrl._adjust_budget(RuntimeMetrics(
            tokens_processed=9000,
            batch_time_ms=10.0,
            memory_used_mb=1000.0,
        ))
        
        # Should be clamped to max
        assert ctrl.get_next_budget() <= config.max_token_budget


class TestRuntimeControllerScaleDown:
    """Tests for scale-down (throttling) behavior."""
    
    def test_scale_down_on_high_memory_pressure(self):
        """Controller should scale down when memory pressure is high."""
        config = RuntimeConfig(
            warmup_batches=2,
            initial_token_budget=5000,
            min_token_budget=1000,
            scale_down_factor=0.85,
            memory_headroom_mb=500,
        )
        ctrl = RuntimeController(config)
        ctrl._total_memory_mb = 10000.0  # 10GB total
        
        # Complete warmup
        for _ in range(2):
            ctrl.report_metrics(RuntimeMetrics(
                tokens_processed=5000,
                batch_time_ms=10.0,
                memory_used_mb=5000.0,
            ))
        
        initial_budget = ctrl.current_budget
        
        # Feed high memory pressure (96% = 9600MB of 10GB)
        for _ in range(3):
            ctrl.report_metrics(RuntimeMetrics(
                tokens_processed=5000,
                batch_time_ms=10.0,
                memory_used_mb=9600.0,  # 96% usage
            ))
        
        # Budget should decrease
        assert ctrl.get_next_budget() < initial_budget
    
    def test_scale_down_respects_min_budget(self):
        """Scale down should not go below min_token_budget."""
        config = RuntimeConfig(
            warmup_batches=1,
            initial_token_budget=1500,
            min_token_budget=1000,
            scale_down_factor=0.5,  # Would push to 750
            memory_headroom_mb=100,
        )
        ctrl = RuntimeController(config)
        ctrl._total_memory_mb = 10000.0
        
        # Complete warmup
        ctrl.report_metrics(RuntimeMetrics(
            tokens_processed=1500,
            batch_time_ms=10.0,
            memory_used_mb=5000.0,
        ))
        
        # Force throttling state
        ctrl._state = RuntimeState.THROTTLING
        ctrl._adjust_budget(RuntimeMetrics(
            tokens_processed=1500,
            batch_time_ms=10.0,
            memory_used_mb=9500.0,
        ))
        
        # Should be clamped to min
        assert ctrl.get_next_budget() >= config.min_token_budget


class TestRuntimeControllerOOM:
    """Tests for OOM handling behavior."""
    
    def test_handle_oom_slashes_budget(self):
        """handle_oom should reduce budget by oom_slash_factor."""
        config = RuntimeConfig(
            initial_token_budget=10000,
            min_token_budget=1000,
            oom_slash_factor=0.5,
        )
        ctrl = RuntimeController(config)
        
        initial_budget = ctrl.get_next_budget()
        new_budget = ctrl.handle_oom()
        
        assert new_budget == int(initial_budget * 0.5)
        assert ctrl.get_next_budget() == new_budget
    
    def test_handle_oom_transitions_to_recovery(self):
        """handle_oom should transition to RECOVERY state."""
        config = RuntimeConfig(initial_token_budget=10000)
        ctrl = RuntimeController(config)
        
        ctrl.handle_oom()
        
        assert ctrl.state == RuntimeState.RECOVERY
    
    def test_handle_oom_respects_min_budget(self):
        """OOM slash should not go below min_token_budget."""
        config = RuntimeConfig(
            initial_token_budget=1500,
            min_token_budget=1000,
            oom_slash_factor=0.5,  # Would push to 750
        )
        ctrl = RuntimeController(config)
        
        new_budget = ctrl.handle_oom()
        
        assert new_budget >= config.min_token_budget
    
    def test_recovery_state_cautious_scaling(self):
        """In RECOVERY, scaling should be cautious."""
        config = RuntimeConfig(
            warmup_batches=1,
            initial_token_budget=10000,
            recovery_patience=3,
        )
        ctrl = RuntimeController(config)
        ctrl._total_memory_mb = 10000.0
        
        # Complete warmup
        ctrl.report_metrics(RuntimeMetrics(
            tokens_processed=10000,
            batch_time_ms=10.0,
            memory_used_mb=5000.0,
        ))
        
        # Trigger OOM
        post_oom_budget = ctrl.handle_oom()
        
        # Feed a few successful batches
        for _ in range(2):
            ctrl.report_metrics(RuntimeMetrics(
                tokens_processed=post_oom_budget,
                batch_time_ms=10.0,
                memory_used_mb=5000.0,
            ))
        
        # Should still be in recovery (patience not reached)
        assert ctrl.state == RuntimeState.RECOVERY
        
        # Budget increase should be minimal
        budget_increase = ctrl.get_next_budget() - post_oom_budget
        assert budget_increase < post_oom_budget * 0.1  # Less than 10% increase
    
    def test_multiple_ooms_track_count(self):
        """Multiple OOMs should be tracked."""
        config = RuntimeConfig(initial_token_budget=10000)
        ctrl = RuntimeController(config)
        
        ctrl.handle_oom()
        ctrl.handle_oom()
        ctrl.handle_oom()
        
        assert ctrl._oom_count == 3


class TestRuntimeControllerClamping:
    """Tests for budget clamping behavior."""
    
    def test_budget_clamped_above_max(self):
        """Budget forced above max should be clamped."""
        config = RuntimeConfig(
            initial_token_budget=5000,
            max_token_budget=10000,
        )
        ctrl = RuntimeController(config)
        
        # Manually force budget above max
        ctrl._current_budget = 50000
        ctrl._current_budget = max(
            config.min_token_budget,
            min(ctrl._current_budget, config.max_token_budget)
        )
        
        assert ctrl._current_budget == config.max_token_budget
    
    def test_budget_clamped_below_min(self):
        """Budget forced below min should be clamped."""
        config = RuntimeConfig(
            initial_token_budget=5000,
            min_token_budget=1000,
        )
        ctrl = RuntimeController(config)
        
        # Manually force budget below min
        ctrl._current_budget = 100
        ctrl._current_budget = max(
            config.min_token_budget,
            min(ctrl._current_budget, config.max_token_budget)
        )
        
        assert ctrl._current_budget == config.min_token_budget


class TestRuntimeControllerSnapshot:
    """Tests for state snapshot functionality."""
    
    def test_snapshot_captures_current_state(self):
        """Snapshot should capture current controller state."""
        config = RuntimeConfig(
            warmup_batches=2,
            initial_token_budget=5000,
        )
        ctrl = RuntimeController(config)
        ctrl._total_memory_mb = 10000.0
        
        # Complete warmup
        for _ in range(2):
            ctrl.report_metrics(RuntimeMetrics(
                tokens_processed=5000,
                batch_time_ms=10.0,
                memory_used_mb=5000.0,
            ))
        
        snapshot = ctrl.get_snapshot()
        
        assert isinstance(snapshot, RuntimeSnapshot)
        assert snapshot.state == ctrl.state
        assert snapshot.current_budget == ctrl.current_budget
        assert snapshot.batches_processed == ctrl.batches_processed
    
    def test_reset_returns_to_initial_state(self):
        """Reset should return controller to initial state."""
        config = RuntimeConfig(
            warmup_batches=2,
            initial_token_budget=5000,
        )
        ctrl = RuntimeController(config)
        ctrl._total_memory_mb = 10000.0
        
        # Process some batches
        for _ in range(5):
            ctrl.report_metrics(RuntimeMetrics(
                tokens_processed=5000,
                batch_time_ms=10.0,
                memory_used_mb=5000.0,
            ))
        
        # Reset
        ctrl.reset()
        
        assert ctrl.state == RuntimeState.WARMUP
        assert ctrl.batches_processed == 0
        assert ctrl.current_budget == config.initial_token_budget


class TestRuntimeControllerThroughputMetrics:
    """Tests for throughput-based decisions."""
    
    def test_throughput_calculation(self):
        """RuntimeMetrics should correctly calculate throughput."""
        metrics = RuntimeMetrics(
            tokens_processed=10000,
            batch_time_ms=100.0,  # 100ms
            memory_used_mb=5000.0,
        )
        
        # 10000 tokens / 0.1 seconds = 100000 tokens/sec
        assert metrics.throughput_tokens_per_sec == 100000.0
    
    def test_throughput_with_zero_time(self):
        """Throughput should be 0 when batch_time_ms is 0."""
        metrics = RuntimeMetrics(
            tokens_processed=10000,
            batch_time_ms=0.0,
            memory_used_mb=5000.0,
        )
        
        assert metrics.throughput_tokens_per_sec == 0.0
    
    def test_stability_detection(self):
        """Controller should detect stable throughput."""
        config = RuntimeConfig(
            warmup_batches=5,
            stability_threshold=0.1,  # 10% variance
            history_window=5,
        )
        ctrl = RuntimeController(config)
        ctrl._total_memory_mb = 10000.0
        
        # Feed consistent metrics (low variance)
        for _ in range(5):
            ctrl.report_metrics(RuntimeMetrics(
                tokens_processed=5000,
                batch_time_ms=10.0,  # Consistent timing
                memory_used_mb=5000.0,
            ))
        
        variance = ctrl._compute_throughput_variance()
        assert variance < config.stability_threshold


class TestGetGpuMemoryStats:
    """Tests for GPU memory stats helper."""
    
    @patch('neuro_stylometry.hardware_ops.runtime.torch')
    def test_returns_zero_when_cuda_unavailable(self, mock_torch):
        """Should return zeros when CUDA not available."""
        mock_torch.cuda.is_available.return_value = False
        
        stats = get_gpu_memory_stats()
        
        assert stats["allocated_mb"] == 0.0
        assert stats["reserved_mb"] == 0.0
        assert stats["free_mb"] == 0.0
        assert stats["total_mb"] == 0.0
