**Current State**
- Graph capture inputs already use `detach().clone()` in `src/neuro_stylometry/hardware_ops/cuda_graphs.py`
- `compute_loss_compiled` is a debug stub and returns a constant loss in `src/neuro_stylometry/stylometry_net/classification_head.py`
- Manual CUDA-graph training is not wired into the Phase D training loop in `src/neuro_stylometry/training/trainer.py`
- Config toggles for compile and cudagraphs are not enforced in the training path
- Graph replay always clones outputs, adding extra device copies per step

**Key Targets**
- `src/neuro_stylometry/hardware_ops/cuda_graphs.py` for capture, replay, and GraphAwareTraining
- `src/neuro_stylometry/stylometry_net/classification_head.py` for compiled loss correctness
- `src/neuro_stylometry/training/trainer.py` for hot loop integration and compile controls
- `src/neuro_stylometry/phase_d_pipeline.py` for wiring config toggles into runtime
- `src/neuro_stylometry/stylometry_net/phase_d_dataset.py` for AOT and pre-padded data flow
- `conf/hpc/phase_d.yaml` for performance defaults and guardrails

**Phase 1 Correctness**
- Remove the debug bypass in `compute_loss_compiled` and restore real loss computation
- Ensure label ordering matches `head.task_order` before capture and replay
- Enforce `detach().clone()` on inputs and labels before capture and preserve requires_grad
- Warm up on a side stream and call `zero_grad(set_to_none=True)` before capture
- Use `capture_error_mode="relaxed"` only for debug runs and keep default for production

**Phase 2 Integration**
- Add a `use_cuda_graph_training` path in `PhaseDTrainer._train` that uses GraphAwareTraining
- Honor `torch_compile_disable_cudagraphs` by switching compile mode or config flags
- Respect `compile_train_step` by compiling a full step function when enabled
- Thread `torch_compile_dynamic` and `torch_compile_backend` into `torch.compile` calls
- Enforce a clear precedence between manual graphs and torch.compile graphs

**Phase 3 Profiling**
- Add short `torch.profiler` runs with CPU and CUDA activities and a schedule
- Instrument NVTX ranges for data load, H2D, forward, backward, optimizer, and step
- Emit telemetry snapshots to JSONL for tokens per second, ms per batch, and memory
- Compare eager vs compile vs manual graphs with identical shapes and batch sizes
- Add a minimal repro for capture errors to isolate stream history contamination

**Phase 4 Data Pipeline**
- Validate AOT columns and `tokenized_text_field` alignment at startup
- Enforce pre-padded AOT path to hit FastCollator zero-copy branch when available
- Tune `num_workers`, `prefetch_factor`, and `pin_memory` to avoid GPU starvation
- Verify DevicePrefetcher overlap with compute and eliminate CPU syncs in hot loop
- Add light checks for collation latency and CPU bottlenecks during runs

**Phase 5 Shape and Bucketing**
- Align `quantize_step` with CUDA-graph bucket sizes and dataset length distribution
- Adjust `min_bucket_usage` to avoid capturing graphs for one-off shapes
- Log `ShapeBucketer` stats for padding efficiency and bucket hits per epoch
- Tune token budget to stabilize shapes and reduce graph churn
- Revisit `seq_len_buckets` based on observed length histograms

**Phase 6 Replay Overhead**
- Add an option to return internal buffers to avoid output clones in training paths
- Keep cloning as default for safety and external call sites
- Ensure outputs are not mutated in place across steps when sharing buffers
- Measure replay overhead before and after to validate impact
- Document safe usage in `GraphCache` API docs

**Phase 7 Validation**
- Unit test manual graph capture on a toy model with gradients enabled
- Integration test RoBERTa forward plus backward capture without error 906
- Compare loss values and gradients against eager within tolerance
- Validate throughput uplift and GPU utilization improvements in profiler
- Add fallback to eager path when capture fails to keep runs stable

**Acceptance Criteria**
- No `cudaErrorStreamCaptureImplicit` in graph training path under static shapes
- Loss curves and metrics match eager within tolerance for fixed inputs
- Manual graphs provide a measurable throughput improvement
- Profiler shows data loading is not the primary bottleneck
- Config toggles behave as documented across laptop and HPC configs

**Rollout**
- Enable on small runs first and collect capture stats and errors
- Ramp token budget and batch sizes incrementally to full throughput
- Keep automatic fallback to eager or compile when capture fails
- Store profiling artifacts per run for regression tracking
- Update HPC config notes to reflect final behavior and caveats
# CUDA Graph Capture Suggestions (Repo-Specific)

This document expands on CUDA graph capture guidance for the `neuro_stylometry`
codebase. It includes explicit instructions, code snippets, and justifications.
It follows the repo style guide in `.github/copilot-instructions.md` by:
- anchoring guidance to concrete file paths
- respecting Phase A -> Phase D data contracts
- keeping instructions aligned with existing pipeline structure

## Scope and Goals

Goal: maximize throughput while avoiding `cudaErrorStreamCaptureImplicit` and
other graph-capture failures when training or running inference on CUDA.

Primary targets in this repo:
- Graph capture and replay: `src/neuro_stylometry/hardware_ops/cuda_graphs.py`
- Training loop: `src/neuro_stylometry/training/trainer.py`
- Loss head: `src/neuro_stylometry/stylometry_net/classification_head.py`
- Phase D configs: `conf/base/phase_d.yaml`, `conf/hpc/phase_d.yaml`

## Hard Requirements (Do Not Skip)

1. Static shapes only (bucket or pre-pad)
   - Why: CUDA graphs require fixed shapes; dynamic shapes cause capture churn
     and cache misses.
   - Do this: ensure shapes are bucketed or pre-padded before capture.
   - Repo hooks:
     - Bucketing: `ShapeBucketer` in `src/neuro_stylometry/hardware_ops/cuda_graphs.py`
     - AOT pre-padding: `PhaseDDataset` + `FastCollator` in
       `src/neuro_stylometry/stylometry_net/phase_d_dataset.py`

2. Clean buffers: `detach().clone()` order matters
   - Why: `clone().detach()` preserves legacy stream history and can trigger
     `cudaErrorStreamCaptureImplicit` during backward.
   - Do this: always detach first, then clone.
   - Good pattern:
     ```python
     def _prepare_capture_buffer(t: torch.Tensor) -> torch.Tensor:
         clean = t.detach().clone()
         if t.requires_grad:
             clean.requires_grad_(True)
         return clean
     ```
   - Repo hook: `GraphCache.capture` already uses `detach().clone()` in
     `src/neuro_stylometry/hardware_ops/cuda_graphs.py`.

3. Capture forward + loss + backward together
   - Why: backward may synchronize with the stream that created the forward
     graph; capturing only forward can create illegal implicit dependencies.
   - Do this: capture the entire training step in one graph, then keep
     optimizer step outside capture.
   - Pattern:
     ```python
     # inside graph capture
     outputs = model(...)
     logits = head(...)
     loss = head.compute_loss(...)
     (loss / accum_steps).backward()
     # optimizer.step() stays outside capture
     ```
   - Repo hook: `GraphAwareTraining._train_step_fn` in
     `src/neuro_stylometry/hardware_ops/cuda_graphs.py`

4. Warmup on a side stream before capture
   - Why: ensures kernels, memory pools, and graph state are stable.
   - Do this: run warmup iterations in a dedicated stream before capture.
   - Pattern:
     ```python
     capture_stream = torch.cuda.Stream()
     capture_stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(capture_stream):
         for _ in range(warmup_iters):
             _ = forward_fn(**buffers)
     torch.cuda.current_stream().wait_stream(capture_stream)
     ```
   - Repo hook: `GraphCache.capture` already creates a capture stream and warms
     up on it.

5. Avoid implicit legacy stream dependencies
   - Why: PyTorch autograd will sync with the stream that produced tensors
     if it sees history. This is illegal during capture (error 906).
   - Do this:
     - Clean inputs with `detach().clone()` before capture.
     - Avoid pre-capture ops on the default stream for tensors you will capture.
     - Ensure labels are also detached and cloned before capture.

## Repo-Specific Integration Checklist

### GraphCache Capture (`src/neuro_stylometry/hardware_ops/cuda_graphs.py`)
- Keep `detach().clone()` for all input buffers.
- If inputs can require gradients, re-enable them after clone.
- Warmup on capture stream before graph capture.
- Capture with a dedicated pool when possible to reduce fragmentation.
- Use `torch.cuda.synchronize()` before and after warmup to eliminate
  cross-stream hazards.

### GraphAwareTraining (`src/neuro_stylometry/hardware_ops/cuda_graphs.py`)
- Ensure labels are passed in a stable order; use `head.task_order`.
- Keep `_train_step_fn` free of side effects and global state.
- Always capture `loss.backward()` within the graph.
- Keep optimizer steps outside capture.

### MultiTaskHead Loss (`src/neuro_stylometry/stylometry_net/classification_head.py`)
- Remove any debug-only loss bypasses before using capture in production.
- Justification: a dummy loss trains the wrong objective and invalidates
  performance measurements.
- Good loss path (example):
  ```python
  def compute_loss_compiled(self, logits, labels, ignore_index=-1):
      first = logits[0]
      total = torch.zeros((), device=first.device, dtype=first.dtype)
      valid_flag = torch.zeros((), device=first.device, dtype=torch.int32)
      for task_logits, task_labels in zip(logits, labels):
          valid_mask = task_labels != ignore_index
          per_sample = F.cross_entropy(
              task_logits, task_labels, reduction="none", ignore_index=ignore_index
          )
          loss_sum = (per_sample * valid_mask).sum()
          denom = valid_mask.sum().clamp(min=1)
          total = total + (loss_sum / denom)
          valid_flag = valid_flag + (valid_mask.sum() > 0).to(valid_flag.dtype)
      return total, (valid_flag > 0).to(valid_flag.dtype)
  ```

### Phase D Trainer (`src/neuro_stylometry/training/trainer.py`)
- Respect config flags:
  - `torch_compile_disable_cudagraphs` should disable cudagraph capture.
    Use `torch_compile_mode="default"` when this flag is true.
  - `compile_train_step` should decide whether to compile just the model or
    a full training step function.
  - `use_cuda_graph_training` should route to GraphAwareTraining when enabled.
- Justification: HPC config expects these flags to affect behavior.

### Dataset and Collation (`src/neuro_stylometry/stylometry_net/phase_d_dataset.py`)
- Prefer pre-padded AOT datasets for stable shapes.
- Use `FastCollator` zero-copy path when `is_pre_padded` is true.
- Ensure `tokenized_text_field` matches `text_field` to avoid silent fallback
  to JIT tokenization.

## Decision Guide: torch.compile vs Manual Graphs

Use torch.compile (default):
- If you need maximum stability with fewer capture errors.
- If your shapes vary but are snapped to grid (quantize step).
- Recommended for most training runs in `conf/hpc/phase_d.yaml`.

Use manual graphs:
- When you control shapes tightly and want full capture of
  forward + loss + backward.
- When you can tolerate stricter constraints and want maximum throughput.

## Troubleshooting: Error 906 (`cudaErrorStreamCaptureImplicit`)

Symptoms:
- "operation would have resulted in a disallowed implicit dependency..."

Likely causes:
- Inputs or labels carry legacy stream history.
- `clone().detach()` used instead of `detach().clone()`.
- Autograd graph references non-captured stream ops.

Fix checklist:
- Verify `detach().clone()` usage for all buffers.
- Ensure `loss.backward()` is inside capture.
- Avoid any pre-capture ops on default stream for capture inputs.
- Warm up on a side stream before capture.

## Performance Validation Checklist

Data pipeline:
- Confirm AOT mode is active (presence of `input_ids` column).
- Ensure pre-padding enabled for zero-copy collation.
- Increase `num_workers`, `prefetch_factor`, and `pin_memory` on HPC.

GPU utilization:
- Use `TelemetryCollector` to track tokens/s and ms/batch.
- Add short `torch.profiler` runs to identify CPU vs GPU stalls.
- Use `torch.compiler.cudagraph_mark_step_begin()` for compiled models.

Graph cache health:
- Track `GraphCache.get_stats()` and `ShapeBucketer.get_stats()`.
- Aim for high `hit_rate` and stable `padding_efficiency_pct`.

## Minimal Reference Snippets

Manual capture pattern:
```python
capture_stream = torch.cuda.Stream()
buffers = {k: v.detach().clone() for k, v in inputs.items()}
with torch.cuda.stream(capture_stream):
    for _ in range(warmup):
        _ = forward_fn(**buffers)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=capture_stream):
    outputs = forward_fn(**buffers)
```

Compile gating pattern:
```python
mode = "default" if torch_compile_disable_cudagraphs else "reduce-overhead"
model = torch.compile(model, mode=mode, fullgraph=False)
```

## Source Alignment

This guidance builds on:
- CUDA graph rules in `Docs_from_SM/Generated/CUDA Programming Guide.md`
- PyTorch graph capture issues (error 906) and repo-local findings
- The repo's execution conventions in `.github/copilot-instructions.md`

