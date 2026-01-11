# Self-Optimizing Runtime: Autotuning Interface

> **Author**: Auto-generated  
> **Date**: January 2026  
> **Status**: Phase A Implementation Complete, Phase D Extensions Planned

## Overview

The Self-Optimizing Runtime provides adaptive GPU batching that automatically adjusts token budgets based on real-time performance feedback. This eliminates the need for manual batch size tuning across different hardware configurations (8GB laptop GPUs to 80GB datacenter H100s).

### Key Benefits

- **Zero manual tuning**: Works across hardware profiles without config changes
- **OOM resilience**: Automatic recovery with budget slashing
- **Throughput optimization**: Feedback loop maximizes GPU utilization
- **Memory awareness**: Adjusts proactively to memory pressure

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RuntimeController                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ State       │  │ Feedback     │  │ Budget              │    │
│  │ Machine     │──│ Loop         │──│ Adjuster            │    │
│  │ (5 states)  │  │ (PID-like)   │  │ (scale up/down)     │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
         │                    ▲
         │ get_next_budget()  │ report_metrics()
         ▼                    │
┌─────────────────────────────────────────────────────────────────┐
│                    DynamicBatchIterator                         │
│  - Queries controller for token budget each iteration           │
│  - Builds variable-size batches to fit budget                   │
│  - Integrates with global sorting for minimal padding           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OOM Guard                                  │
│  - execute_with_oom_protection(): Automatic retry               │
│  - oom_guarded_context(): Manual control                        │
│  - Aggressive cache clearing on OOM                              │
└─────────────────────────────────────────────────────────────────┘
```

## State Machine

The controller operates in five states:

| State | Description | Behavior |
|-------|-------------|----------|
| **WARMUP** | Initial calibration | Collects baseline metrics for `warmup_batches` iterations |
| **SCALING_UP** | Aggressive growth | Increases budget by `scale_up_factor` (default 1.25) |
| **STABLE** | Optimal operation | Minor adjustments (±2%) based on memory pressure |
| **THROTTLING** | Memory pressure | Decreases budget by `scale_down_factor` (default 0.85) |
| **RECOVERY** | Post-OOM caution | Cautious scaling after OOM with patience period |

### State Transitions

```
            ┌─────────────────────────────────────────┐
            │                WARMUP                    │
            │  (warmup_batches iterations)            │
            └──────────────────┬──────────────────────┘
                               │ variance < threshold
                               ▼
            ┌─────────────────────────────────────────┐
            │               STABLE                     │
            │  (throughput stabilized)                │
            └───┬───────────────────────────────┬─────┘
    mem > 85%   │                               │ trend positive
                ▼                               ▼ mem < 60%
┌───────────────────────┐       ┌───────────────────────────────┐
│     THROTTLING        │       │         SCALING_UP            │
│  (reduce budget)      │       │  (increase budget)            │
└───────────┬───────────┘       └───────────────┬───────────────┘
            │ mem < 70%                         │ mem > 90%
            └───────────────────────────────────┘
                               │
                          OOM Event
                               │
                               ▼
            ┌─────────────────────────────────────────┐
            │              RECOVERY                    │
            │  (recovery_patience batches)            │
            └─────────────────────────────────────────┘
```

## Quick Start

### Basic Usage (Phase A GLiNER Inference)

```python
from neuro_stylometry.hardware_ops import RuntimeController, RuntimeMetrics, CUDATimer
from neuro_stylometry.pollution_guard.global_sort import DynamicBatchIterator, flatten_chunks

# Create controller from config
controller = RuntimeController.from_config(pipeline_config)

# Flatten chunks from post_chunked column
flattened = flatten_chunks(table["post_chunked"])

# Iterate with dynamic batching
for batch_indices in DynamicBatchIterator(flattened, controller):
    batch_texts = [flattened.texts[i] for i in batch_indices]
    
    # Time the inference
    with CUDATimer() as timer:
        results = model.inference(batch_texts)
    
    # Report metrics for feedback loop
    tokens = sum(flattened.token_counts[i] for i in batch_indices)
    controller.report_metrics(RuntimeMetrics(
        tokens_processed=tokens,
        batch_time_ms=timer.elapsed_ms,
        memory_used_mb=torch.cuda.memory_allocated() / 1e6,
        batch_size=len(batch_indices),
    ))
```

### With OOM Protection

```python
from neuro_stylometry.hardware_ops import execute_with_oom_protection

def run_batch(batch):
    return model.inference(batch)

# Automatic retry with budget adjustment
result = execute_with_oom_protection(
    lambda: run_batch(batch_texts),
    controller=controller,
    retry_limit=3,
)
```

### Telemetry Collection

```python
from neuro_stylometry.hardware_ops import TelemetryCollector

collector = TelemetryCollector.get_instance()

for batch in batches:
    with CUDATimer() as timer:
        result = model(batch)
    
    collector.record_batch(
        batch_size=len(batch),
        tokens=batch.num_tokens,
        elapsed_ms=timer.elapsed_ms,
    )

# Get summary
summary = collector.get_summary()
print(f"Throughput: {summary.avg_throughput_tokens_per_sec:.0f} tok/s")
print(f"Peak memory: {summary.peak_memory_mb:.0f} MB")
```

## Configuration

### YAML Configuration

```yaml
# conf/hpc/pipeline.yaml
execution:
  autotuning:
    enabled: true
    warmup_batches: 10             # Batches before adjusting budget
    initial_token_budget: 32768    # Starting budget (~256 * 128 tokens)
    min_token_budget: 4096         # Minimum budget (safety floor)
    max_token_budget: 524288       # Maximum budget (H100 can handle this)
    memory_headroom_mb: 2048       # Keep 2GB free for safety
    scale_up_factor: 1.25          # 25% increase when scaling up
    scale_down_factor: 0.85        # 15% decrease when throttling
    oom_slash_factor: 0.5          # 50% reduction after OOM
    stability_threshold: 0.1       # 10% variance = stable
    history_window: 10             # Batches for moving average
    recovery_patience: 5           # Batches in recovery before scaling
```

### Programmatic Configuration

```python
from neuro_stylometry.hardware_ops import RuntimeController, RuntimeConfig

config = RuntimeConfig(
    warmup_batches=10,
    initial_token_budget=16384,
    min_token_budget=2048,
    max_token_budget=262144,
    memory_headroom_mb=1024,
    scale_up_factor=1.25,
    scale_down_factor=0.85,
)

controller = RuntimeController(config)
```

### Hardware-Specific Presets

| Hardware | initial_budget | max_budget | memory_headroom |
|----------|---------------|------------|-----------------|
| RTX 3060 (12GB) | 8,192 | 65,536 | 1,024 MB |
| RTX 4090 (24GB) | 16,384 | 131,072 | 2,048 MB |
| A100 (40GB) | 32,768 | 262,144 | 4,096 MB |
| H100 (80GB) | 65,536 | 524,288 | 8,192 MB |

## API Reference

### RuntimeController

```python
class RuntimeController:
    """Self-optimizing runtime controller for GPU inference."""
    
    def __init__(self, config: RuntimeConfig): ...
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> RuntimeController:
        """Create from pipeline config dict."""
    
    def get_next_budget(self) -> int:
        """Get token budget for next batch."""
    
    def report_metrics(self, metrics: RuntimeMetrics) -> None:
        """Report batch metrics for feedback loop."""
    
    def handle_oom(self) -> int:
        """Handle OOM by slashing budget. Returns new budget."""
    
    def get_snapshot(self) -> RuntimeSnapshot:
        """Get current state for logging."""
    
    def reset(self) -> None:
        """Reset to initial state."""
```

### RuntimeMetrics

```python
@dataclass
class RuntimeMetrics:
    tokens_processed: int      # Total tokens in batch
    batch_time_ms: float       # Wall-clock time (ms)
    memory_used_mb: float      # GPU memory after batch (MB)
    memory_peak_mb: float      # Peak memory during batch (optional)
    batch_size: int            # Number of sequences
```

### DynamicBatchIterator

```python
class DynamicBatchIterator:
    """Adaptive batch iterator with controller feedback."""
    
    def __init__(
        self,
        flattened: FlattenedChunks,
        controller: RuntimeController = None,
        fallback_batch_size: int = 128,
        min_batch_size: int = 1,
        max_batch_size: int = 2048,
    ): ...
    
    def __iter__(self) -> Generator[List[int], None, None]:
        """Yield batches of chunk indices."""
    
    @property
    def progress(self) -> float:
        """Progress fraction [0, 1]."""
    
    @property
    def remaining_chunks(self) -> int:
        """Chunks not yet yielded."""
```

### OOM Protection Functions

```python
def execute_with_oom_protection(
    func: Callable[[], T],
    controller: RuntimeController = None,
    retry_limit: int = 3,
    on_oom: Callable[[OOMEvent], None] = None,
) -> T:
    """Execute with automatic OOM retry."""

@contextmanager
def oom_guarded_context(
    controller: RuntimeController = None,
) -> Generator[OOMGuardState, None, None]:
    """Context manager for OOM handling."""
```

## Phase D Extensions

The autotuning infrastructure is designed to extend seamlessly to Phase D transformer training. Here's how Phase D developers should integrate:

### Training Loop Integration

```python
from neuro_stylometry.hardware_ops import RuntimeController, RuntimeMetrics

class PhaseD Trainer:
    def __init__(self, config):
        # Create controller for training
        self.controller = RuntimeController.from_config(config)
        
        # Phase D may want different scaling factors for training
        self.controller.config.scale_up_factor = 1.1  # More conservative
        self.controller.config.oom_slash_factor = 0.6  # Less aggressive slash
    
    def train_epoch(self, dataloader):
        for batch in self._dynamic_batches(dataloader):
            with CUDATimer() as timer:
                loss = self._train_step(batch)
            
            self.controller.report_metrics(RuntimeMetrics(
                tokens_processed=batch.num_tokens,
                batch_time_ms=timer.elapsed_ms,
                memory_used_mb=torch.cuda.memory_allocated() / 1e6,
            ))
    
    def _dynamic_batches(self, dataloader):
        """Wrap dataloader with dynamic batching."""
        # Implementation depends on dataloader type
        # Key: query controller.get_next_budget() before each batch
        pass
```

### Gradient Accumulation Support

Phase D can use token budgets to determine gradient accumulation steps:

```python
def compute_accumulation_steps(self, target_batch_tokens: int) -> int:
    """Compute accumulation steps to reach target batch size."""
    current_budget = self.controller.get_next_budget()
    if current_budget >= target_batch_tokens:
        return 1
    return math.ceil(target_batch_tokens / current_budget)
```

### Mixed Precision Integration

```python
from neuro_stylometry.hardware_ops import execute_with_oom_protection

@torch.cuda.amp.autocast(dtype=torch.bfloat16)
def train_step_with_oom_guard(self, batch):
    return execute_with_oom_protection(
        lambda: self._forward_backward(batch),
        controller=self.controller,
        retry_limit=2,  # Training may want fewer retries
    )
```

### Distributed Training (Future)

The current implementation is single-GPU. For Phase D distributed training:

1. **Per-rank controllers**: Each rank maintains its own RuntimeController
2. **Synchronized budgets**: All-reduce token budgets before batching
3. **Global OOM handling**: Any rank OOM triggers global budget reduction

```python
# Future distributed extension (placeholder)
class DistributedRuntimeController(RuntimeController):
    def sync_budget(self, world_size: int):
        """All-reduce to find minimum safe budget across ranks."""
        pass
```

### Checkpointing Controller State

For long training runs, save/restore controller state:

```python
def save_checkpoint(self, path):
    checkpoint = {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "controller_budget": self.controller.current_budget,
        "controller_state": self.controller.state.value,
    }
    torch.save(checkpoint, path)

def load_checkpoint(self, path):
    checkpoint = torch.load(path)
    self.controller._current_budget = checkpoint["controller_budget"]
    # Note: State resets to WARMUP for safety after load
```

## Monitoring & Debugging

### Logging

Enable debug logging for detailed autotuning info:

```python
import logging
logging.getLogger("neuro_stylometry.hardware_ops.runtime").setLevel(logging.DEBUG)
```

### Telemetry Export

Export metrics for analysis:

```python
collector = TelemetryCollector.get_instance()
metrics = collector.export_metrics()

import pandas as pd
df = pd.DataFrame(metrics)
df.to_csv("inference_telemetry.csv")
```

### Controller Snapshots

Log controller state periodically:

```python
for i, batch in enumerate(batches):
    # ... process batch ...
    
    if i % 100 == 0:
        snapshot = controller.get_snapshot()
        logger.info(
            f"Step {i}: state={snapshot.state.value}, "
            f"budget={snapshot.current_budget}, "
            f"throughput={snapshot.throughput_avg:.0f} tok/s"
        )
```

## Troubleshooting

### OOM Despite Autotuning

If you still hit OOM:

1. **Reduce `initial_token_budget`**: Start lower
2. **Increase `memory_headroom_mb`**: More safety margin
3. **Check for memory leaks**: Ensure tensors are freed
4. **Increase `oom_slash_factor`**: More aggressive reduction (e.g., 0.3)

### Throughput Not Improving

If throughput plateaus early:

1. **Increase `max_token_budget`**: Allow larger batches
2. **Reduce `stability_threshold`**: Stay in SCALING_UP longer
3. **Check GPU utilization**: Use `nvidia-smi` to verify

### Budget Oscillating

If budget jumps up and down:

1. **Increase `warmup_batches`**: More calibration time
2. **Increase `history_window`**: Smoother averaging
3. **Reduce `scale_up_factor`**: More gradual scaling

## Files Reference

| File | Description |
|------|-------------|
| [runtime.py](../src/neuro_stylometry/hardware_ops/runtime.py) | RuntimeController, RuntimeConfig, RuntimeMetrics |
| [oom_guard.py](../src/neuro_stylometry/hardware_ops/oom_guard.py) | OOM protection wrappers |
| [telemetry.py](../src/neuro_stylometry/hardware_ops/telemetry.py) | CUDATimer, TelemetryCollector |
| [global_sort.py](../src/neuro_stylometry/pollution_guard/global_sort.py) | DynamicBatchIterator |
| [hpc/pipeline.yaml](../conf/hpc/pipeline.yaml) | HPC autotuning config |
| [laptop/pipeline.yaml](../conf/laptop/pipeline.yaml) | Laptop autotuning config |

---

*For questions or issues, consult the codebase or raise an issue on the repository.*
