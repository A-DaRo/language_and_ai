"""
Unit Tests for OOMGuard.

Tests the OOM protection wrappers with mocked exceptions,
verifying retry logic, budget adjustment, and error propagation.
"""

import pytest
from unittest.mock import patch, MagicMock, call

from neuro_stylometry.hardware_ops.oom_guard import (
    is_cuda_oom,
    clear_cuda_cache,
    execute_with_oom_protection,
    oom_guarded_context,
    oom_protected,
    OOMEvent,
    OOMRecoveryError,
)
from neuro_stylometry.hardware_ops.runtime import RuntimeController, RuntimeConfig


class MockCUDAOOMError(RuntimeError):
    """Mock CUDA OOM error for testing."""
    def __init__(self):
        super().__init__("CUDA out of memory. Tried to allocate 2.00 GiB")


class TestIsCudaOOM:
    """Tests for OOM detection."""
    
    def test_detects_standard_oom_message(self):
        """Should detect standard CUDA OOM error message."""
        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert is_cuda_oom(error) is True
    
    def test_detects_lowercase_oom_message(self):
        """Should detect lowercase OOM message."""
        error = RuntimeError("cuda error: out of memory")
        assert is_cuda_oom(error) is True
    
    def test_detects_cudnn_internal_error(self):
        """Should detect cuDNN internal errors (often caused by OOM)."""
        error = RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")
        assert is_cuda_oom(error) is True
    
    def test_detects_cublas_alloc_failed(self):
        """Should detect cuBLAS allocation failure."""
        error = RuntimeError("cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED")
        assert is_cuda_oom(error) is True
    
    def test_does_not_detect_unrelated_runtime_error(self):
        """Should not detect unrelated RuntimeError."""
        error = RuntimeError("Some other error")
        assert is_cuda_oom(error) is False
    
    def test_does_not_detect_value_error(self):
        """Should not detect ValueError."""
        error = ValueError("Invalid value")
        assert is_cuda_oom(error) is False
    
    def test_does_not_detect_type_error(self):
        """Should not detect TypeError."""
        error = TypeError("Type mismatch")
        assert is_cuda_oom(error) is False


class TestClearCudaCache:
    """Tests for cache clearing functionality."""
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.torch')
    @patch('neuro_stylometry.hardware_ops.oom_guard.gc')
    def test_clears_cache_when_cuda_available(self, mock_gc, mock_torch):
        """Should clear CUDA cache and run GC when CUDA available."""
        mock_torch.cuda.is_available.return_value = True
        
        clear_cuda_cache()
        
        mock_gc.collect.assert_called()
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.synchronize.assert_called_once()
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.torch')
    @patch('neuro_stylometry.hardware_ops.oom_guard.gc')
    def test_only_gc_when_cuda_unavailable(self, mock_gc, mock_torch):
        """Should only run GC when CUDA not available."""
        mock_torch.cuda.is_available.return_value = False
        
        clear_cuda_cache()
        
        mock_torch.cuda.empty_cache.assert_not_called()


class TestExecuteWithOOMProtection:
    """Tests for the execute_with_oom_protection wrapper."""
    
    def test_passthrough_on_success(self):
        """Should pass through result on success."""
        def success_func():
            return "success"
        
        result = execute_with_oom_protection(success_func)
        
        assert result == "success"
    
    def test_function_called_once_on_success(self):
        """Function should be called exactly once on success."""
        mock_func = MagicMock(return_value="success")
        
        execute_with_oom_protection(mock_func)
        
        assert mock_func.call_count == 1
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.clear_cuda_cache')
    @patch('neuro_stylometry.hardware_ops.oom_guard.get_memory_snapshot')
    def test_retry_on_oom_then_success(self, mock_snapshot, mock_clear):
        """Should retry on OOM and succeed on second attempt."""
        mock_snapshot.return_value = (5000.0, 8000.0)
        
        call_count = [0]
        def oom_then_success():
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockCUDAOOMError()
            return "success"
        
        result = execute_with_oom_protection(oom_then_success, retry_limit=3)
        
        assert result == "success"
        assert call_count[0] == 2
        mock_clear.assert_called()
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.clear_cuda_cache')
    @patch('neuro_stylometry.hardware_ops.oom_guard.get_memory_snapshot')
    def test_controller_handle_oom_called_on_oom(self, mock_snapshot, mock_clear):
        """Controller's handle_oom should be called on OOM."""
        mock_snapshot.return_value = (5000.0, 8000.0)
        
        config = RuntimeConfig(initial_token_budget=10000, oom_slash_factor=0.5)
        controller = RuntimeController(config)
        
        call_count = [0]
        def oom_then_success():
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockCUDAOOMError()
            return "success"
        
        initial_budget = controller.current_budget
        execute_with_oom_protection(oom_then_success, controller=controller)
        
        # Budget should be slashed
        assert controller.current_budget < initial_budget
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.clear_cuda_cache')
    @patch('neuro_stylometry.hardware_ops.oom_guard.get_memory_snapshot')
    def test_raises_recovery_error_after_retry_limit(self, mock_snapshot, mock_clear):
        """Should raise OOMRecoveryError after retry_limit exhausted."""
        mock_snapshot.return_value = (5000.0, 8000.0)
        
        def always_oom():
            raise MockCUDAOOMError()
        
        with pytest.raises(OOMRecoveryError) as exc_info:
            execute_with_oom_protection(always_oom, retry_limit=3)
        
        assert len(exc_info.value.oom_events) == 4  # Initial + 3 retries
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.clear_cuda_cache')
    @patch('neuro_stylometry.hardware_ops.oom_guard.get_memory_snapshot')
    def test_on_oom_callback_invoked(self, mock_snapshot, mock_clear):
        """on_oom callback should be invoked on each OOM."""
        mock_snapshot.return_value = (5000.0, 8000.0)
        
        oom_events = []
        def track_oom(event: OOMEvent):
            oom_events.append(event)
        
        call_count = [0]
        def oom_then_success():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise MockCUDAOOMError()
            return "success"
        
        execute_with_oom_protection(
            oom_then_success,
            retry_limit=3,
            on_oom=track_oom,
        )
        
        assert len(oom_events) == 2
        assert all(isinstance(e, OOMEvent) for e in oom_events)
    
    def test_reraises_non_oom_exception(self):
        """Should reraise non-OOM exceptions."""
        def raise_value_error():
            raise ValueError("Not an OOM error")
        
        with pytest.raises(ValueError, match="Not an OOM error"):
            execute_with_oom_protection(raise_value_error)
    
    def test_does_not_reraise_non_oom_when_disabled(self):
        """Should return None for non-OOM if reraise_non_oom=False."""
        def raise_value_error():
            raise ValueError("Not an OOM error")
        
        result = execute_with_oom_protection(
            raise_value_error,
            reraise_non_oom=False,
        )
        
        assert result is None


class TestOOMGuardedContext:
    """Tests for the oom_guarded_context context manager."""
    
    def test_normal_execution_no_oom(self):
        """Context should work normally without OOM."""
        with oom_guarded_context() as guard:
            result = 1 + 1
        
        assert not guard.oom_occurred
        assert len(guard.oom_events) == 0
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.clear_cuda_cache')
    @patch('neuro_stylometry.hardware_ops.oom_guard.get_memory_snapshot')
    def test_catches_oom_and_sets_flag(self, mock_snapshot, mock_clear):
        """Context should catch OOM and set flag."""
        mock_snapshot.return_value = (5000.0, 8000.0)
        
        with oom_guarded_context() as guard:
            raise MockCUDAOOMError()
        
        assert guard.oom_occurred
        assert len(guard.oom_events) == 1
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.clear_cuda_cache')
    @patch('neuro_stylometry.hardware_ops.oom_guard.get_memory_snapshot')
    def test_controller_budget_adjusted_on_oom(self, mock_snapshot, mock_clear):
        """Controller budget should be adjusted when OOM caught."""
        mock_snapshot.return_value = (5000.0, 8000.0)
        
        config = RuntimeConfig(initial_token_budget=10000)
        controller = RuntimeController(config)
        initial_budget = controller.current_budget
        
        with oom_guarded_context(controller) as guard:
            raise MockCUDAOOMError()
        
        assert guard.oom_occurred
        assert controller.current_budget < initial_budget
        assert guard.final_budget == controller.current_budget
    
    def test_reraises_non_oom_exception(self):
        """Should reraise non-OOM exceptions."""
        with pytest.raises(ValueError, match="Not OOM"):
            with oom_guarded_context() as guard:
                raise ValueError("Not OOM")


class TestOOMProtectedDecorator:
    """Tests for the @oom_protected decorator."""
    
    def test_decorated_function_runs_normally(self):
        """Decorated function should run normally without OOM."""
        class MyClass:
            @oom_protected(retry_limit=3)
            def my_method(self, x):
                return x * 2
        
        obj = MyClass()
        result = obj.my_method(5)
        
        assert result == 10
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.clear_cuda_cache')
    @patch('neuro_stylometry.hardware_ops.oom_guard.get_memory_snapshot')
    def test_decorator_retries_on_oom(self, mock_snapshot, mock_clear):
        """Decorator should retry on OOM."""
        mock_snapshot.return_value = (5000.0, 8000.0)
        
        class MyClass:
            def __init__(self):
                self.call_count = 0
            
            @oom_protected(retry_limit=3)
            def my_method(self):
                self.call_count += 1
                if self.call_count == 1:
                    raise MockCUDAOOMError()
                return "success"
        
        obj = MyClass()
        result = obj.my_method()
        
        assert result == "success"
        assert obj.call_count == 2
    
    @patch('neuro_stylometry.hardware_ops.oom_guard.clear_cuda_cache')
    @patch('neuro_stylometry.hardware_ops.oom_guard.get_memory_snapshot')
    def test_decorator_uses_controller_from_class(self, mock_snapshot, mock_clear):
        """Decorator should use controller from class attribute."""
        mock_snapshot.return_value = (5000.0, 8000.0)
        
        class MyClass:
            def __init__(self):
                self._runtime_controller = RuntimeController(
                    RuntimeConfig(initial_token_budget=10000)
                )
                self.call_count = 0
            
            @oom_protected(retry_limit=3, controller_attr="_runtime_controller")
            def my_method(self):
                self.call_count += 1
                if self.call_count == 1:
                    raise MockCUDAOOMError()
                return "success"
        
        obj = MyClass()
        initial_budget = obj._runtime_controller.current_budget
        
        obj.my_method()
        
        # Controller should have been notified of OOM
        assert obj._runtime_controller.current_budget < initial_budget


class TestOOMEvent:
    """Tests for OOMEvent dataclass."""
    
    def test_oom_event_creation(self):
        """OOMEvent should capture all relevant information."""
        event = OOMEvent(
            budget_before=10000,
            budget_after=5000,
            memory_allocated_mb=8000.0,
            memory_reserved_mb=9000.0,
            retry_count=1,
            exception_msg="CUDA out of memory",
            traceback_str="...",
        )
        
        assert event.budget_before == 10000
        assert event.budget_after == 5000
        assert event.memory_allocated_mb == 8000.0
        assert event.retry_count == 1


class TestOOMRecoveryError:
    """Tests for OOMRecoveryError exception."""
    
    def test_recovery_error_contains_events(self):
        """OOMRecoveryError should contain OOM event history."""
        events = [
            OOMEvent(10000, 5000, 8000, 9000, 0, "OOM 1", "..."),
            OOMEvent(5000, 2500, 8000, 9000, 1, "OOM 2", "..."),
        ]
        
        error = OOMRecoveryError("Recovery failed", events)
        
        assert len(error.oom_events) == 2
        assert "Recovery failed" in str(error)
