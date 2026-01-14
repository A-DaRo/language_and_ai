# cudaErrorStreamCaptureImplicit Fix Assessment Report

**Date:** January 14, 2026  
**Repository:** `A-DaRo/language_and_ai` (branch: `interim_py`)  
**Scope:** Assessment of CUDA graph capture error 906 (`cudaErrorStreamCaptureImplicit`) mitigation

---

## Executive Summary

The repository **has correctly implemented the primary fix** for the `cudaErrorStreamCaptureImplicit` (error 906) issue. The buffer creation order has been changed from `clone().detach()` to `detach().clone()`, which eliminates autograd/stream history contamination. Additional stream context management and synchronization patterns are also implemented.

However, **several secondary issues remain** that may affect production stability:
1. Debug bypasses in `forward_cls()` and `forward_compiled()` that skip actual computation
2. Lack of explicit backward stream context control inside the captured closure
3. No explicit `capture_error_mode` configuration for debugging vs. production

---

## 1. Primary Fix: Buffer Creation Order

### 1.1 Issue Description

The original bug involved using `clone().detach()` instead of `detach().clone()` when preparing input buffers for CUDA graph capture:

```python
# WRONG: Preserves legacy stream traces
input_buffers = {name: tensor.clone().detach() ...}

# CORRECT: Clears autograd history BEFORE cloning
input_buffers = {name: tensor.detach().clone() ...}
```

### 1.2 Implementation Status: ✅ FIXED

**Location:** [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L38-L42)

The repository implements a dedicated helper function:

```python
def _prepare_capture_buffer(tensor: Tensor) -> Tensor:
    clean = tensor.detach().clone()
    if tensor.requires_grad:
        clean.requires_grad_(True)
    return clean
```

**Key characteristics:**
- `detach()` is called **first** to sever the autograd graph and eliminate stream history
- `clone()` creates a fresh copy without legacy stream traces
- `requires_grad` is explicitly restored after cloning when needed

### 1.3 Usage in GraphCache.capture()

**Location:** [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L221-L227)

```python
# Allocate input buffers (copy input shapes) on the capture stream.
# Use detach().clone() to ensure clean buffer creation and preserve requires_grad.
with torch.cuda.stream(self._capture_stream):
    input_buffers = {
        name: _prepare_capture_buffer(tensor)
        for name, tensor in input_tensors.items()
    }
```

The comment explicitly acknowledges the fix rationale.

---

## 2. Stream Context Management

### 2.1 Capture Stream Isolation

**Location:** [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L214-L219)

```python
# Create capture stream if needed
if self._capture_stream is None:
    self._capture_stream = torch.cuda.Stream()

# Ensure capture stream waits on any prior work before cloning buffers.
self._capture_stream.wait_stream(torch.cuda.current_stream())
torch.cuda.synchronize()
```

**Assessment:** ✅ CORRECT

This implementation:
1. Creates a dedicated capture stream for isolation
2. Synchronizes the capture stream with the current stream before buffer allocation
3. Performs a global `synchronize()` to ensure all prior operations are complete

### 2.2 Warmup on Side Stream

**Location:** [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L231-L237)

```python
# Warmup iterations (stabilizes CUDA state)
with torch.cuda.stream(self._capture_stream):
    for _ in range(self.config.warmup_iterations):
        _ = forward_fn(**input_buffers)

# Synchronize before capture to avoid cross-stream hazards.
torch.cuda.synchronize()
```

**Assessment:** ✅ CORRECT

Follows the PyTorch recommended pattern of warming up on a side stream before capture.

### 2.3 Graph Capture with Explicit Stream

**Location:** [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L239-L252)

```python
graph = torch.cuda.CUDAGraph()

capture_kwargs = {"stream": self._capture_stream}
if self._capture_pool is not None:
     capture_kwargs["pool"] = self._capture_pool

with torch.cuda.graph(graph, **capture_kwargs):
    output_dict = forward_fn(**input_buffers)
```

**Assessment:** ✅ CORRECT

- Explicit stream context is passed to graph capture
- Dedicated memory pool is used when available to reduce fragmentation

---

## 3. GraphAwareTraining Implementation

### 3.1 Train Step Function

**Location:** [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L994-L1016)

```python
def _train_step_fn(
    self,
    *,
    model: Any,
    head: Any,
    accum_steps: int,
    label_keys: list[str],
) -> Callable[..., Dict[str, Tensor]]:
    ignore_index = self.ignore_index

    def _step(input_ids: Tensor, attention_mask: Tensor, **label_inputs: Tensor):
        labels = tuple(label_inputs[key] for key in label_keys)
        cls_embedding = model.forward_cls(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = head.forward_compiled(cls_embedding)
        loss, valid_flag = head.compute_loss_compiled(
            logits,
            labels,
            ignore_index=ignore_index,
        )
        (loss / accum_steps).backward()
        return {"loss": loss, "valid": valid_flag}

    return _step
```

**Assessment:** ✅ STRUCTURALLY CORRECT

The training step correctly:
1. Captures forward, loss computation, and backward in a single closure
2. Uses graph-safe methods (`forward_cls`, `forward_compiled`, `compute_loss_compiled`)
3. Keeps optimizer step outside the captured function

### 3.2 Trainer Integration

**Location:** [trainer.py](src/neuro_stylometry/training/trainer.py#L621-L655)

The trainer correctly instantiates `GraphAwareTraining` when `use_cuda_graph_training` is enabled and handles conflicts with `torch.compile` modes.

---

## 4. Remaining Issues

### 4.1 ⚠️ Debug Bypass in `forward_cls`

**Location:** [transformer.py](src/neuro_stylometry/stylometry_net/transformer.py#L183-L197)

```python
def forward_cls(self, ...):
    ...
    # encoder_outputs = self.model.encoder(
    #     embedding_output,
    #     attention_mask=extended_attention_mask,
    #     ...
    # )
    # return encoder_outputs.last_hidden_state[:, 0, :]  # CLS only
    return embedding_output[:, 0, :]  # Fake CLS from embeddings for debugging
```

**Impact:** CRITICAL
- The encoder pass is commented out
- Returns embedding layer output instead of actual transformer output
- Model will **not train correctly** with this bypass active

### 4.2 ⚠️ Debug Bypass in `forward_compiled`

**Location:** [classification_head.py](src/neuro_stylometry/stylometry_net/classification_head.py#L34)

```python
def forward_compiled(self, cls_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # cls_embedding = self.dropout(cls_embedding)  # DEBUG: Disable dropout
    logits = []
    ...
```

**Impact:** MODERATE
- Dropout is disabled, affecting regularization
- May lead to overfitting during training

### 4.3 No Explicit `capture_error_mode` Configuration

**Location:** [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L250)

```python
with torch.cuda.graph(graph, **capture_kwargs):
```

**Impact:** LOW
- Uses default `capture_error_mode='global'` (strictest)
- For debugging, `capture_error_mode='relaxed'` would help identify additional issues
- Recommendation: Add config option for debug runs

### 4.4 No Explicit Backward Stream Control

**Location:** [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L1012-L1013)

```python
(loss / accum_steps).backward()
return {"loss": loss, "valid": valid_flag}
```

**Impact:** LOW (mitigated by buffer fix)
- The `backward()` call has no explicit stream context
- However, since input buffers are properly cleaned with `detach().clone()`, the legacy stream dependency issue should not occur
- The entire step runs within the capture stream context from `GraphCache.capture()`

---

## 5. Verification Evidence

### 5.1 Scripts Using Correct Pattern

**Location:** [profile_manual_cudagraphs.py](scripts/profile_manual_cudagraphs.py#L134-L140)

```python
input_buffers = {
    "input_ids": input_ids.detach().clone(),
    "attention_mask": mask.detach().clone(),
}
# Labels (clones)
raw_labels = {
    k: v.detach().clone() for k,v in labels.items()
}
```

This script demonstrates the raw capture pattern working with the RoBERTa model, confirming:
1. Model is graph-safe
2. `detach().clone()` pattern is consistently used
3. Both inputs and labels are properly cleaned

### 5.2 Documentation

**Location:** [cuda_suggestions.md](Docs_from_SM/Generated/cuda_suggestions.md#L109-L121)

```markdown
2. Clean buffers: `detach().clone()` order matters
   - Why: `clone().detach()` preserves legacy stream history and can trigger
     `cudaErrorStreamCaptureImplicit` during backward.
   - Do this: always detach first, then clone.
   ...
   - Repo hook: `GraphCache.capture` already uses `detach().clone()` in
     `src/neuro_stylometry/hardware_ops/cuda_graphs.py`.
```

The documentation explicitly references the fix and provides rationale.

---

## 6. Technical Flow Summary

### 6.1 Entry Points

| Entry Point | File | Line | Purpose |
|-------------|------|------|---------|
| `GraphAwareTraining.run()` | [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L1018) | 1018 | Main training entry for graph-accelerated training |
| `GraphCache.capture()` | [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L176) | 176 | Graph capture with buffer preparation |
| `GraphCache.run_with_graph()` | [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L303) | 303 | Unified capture/replay interface |
| `PhaseDTrainer._train()` | [trainer.py](src/neuro_stylometry/training/trainer.py#L594) | 594 | Training loop with graph integration |

### 6.2 Data Flow

```
PhaseDTrainer._train()
    │
    ├──[If use_cuda_graph_training]
    │   │
    │   └── GraphAwareTraining.run()
    │       │
    │       ├── ShapeBucketer.find_bucket() → Determine bucket for shape
    │       ├── ShapeBucketer.pad_tensors() → Pad to bucket dimensions
    │       │
    │       └── GraphCache.run_with_graph()
    │           │
    │           ├──[Cache Miss]
    │           │   └── GraphCache.capture()
    │           │       ├── _prepare_capture_buffer() → detach().clone() + requires_grad
    │           │       ├── Warmup on capture_stream
    │           │       ├── torch.cuda.graph() capture
    │           │       │   └── _train_step_fn()
    │           │       │       ├── model.forward_cls()
    │           │       │       ├── head.forward_compiled()
    │           │       │       ├── head.compute_loss_compiled()
    │           │       │       └── loss.backward()
    │           │       └── Return CapturedGraph
    │           │
    │           └──[Cache Hit]
    │               └── GraphCache.replay()
    │                   ├── Copy inputs to buffers
    │                   └── graph.replay()
    │
    └──[Else Standard Path]
        └── Direct forward/backward without graph capture
```

### 6.3 Key Fix Locations

| Fix | File | Lines | Status |
|-----|------|-------|--------|
| `_prepare_capture_buffer` helper | [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L38-L42) | 38-42 | ✅ Implemented |
| Buffer allocation in capture | [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L221-L227) | 221-227 | ✅ Implemented |
| Capture stream creation | [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L214-L219) | 214-219 | ✅ Implemented |
| Warmup synchronization | [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L231-L238) | 231-238 | ✅ Implemented |
| Graph-safe forward | [transformer.py](src/neuro_stylometry/stylometry_net/transformer.py#L153-L197) | 153-197 | ⚠️ Debug bypass |
| Graph-safe loss | [classification_head.py](src/neuro_stylometry/stylometry_net/classification_head.py#L75-L98) | 75-98 | ✅ Implemented |

---

## 7. Recommendations

### 7.1 Critical (Must Fix)

1. **Remove debug bypass in `forward_cls`**
   - Uncomment the encoder forward pass in [transformer.py#L183-L195](src/neuro_stylometry/stylometry_net/transformer.py#L183-L195)
   - Return actual CLS embedding from encoder output

### 7.2 High Priority

2. **Re-enable dropout in `forward_compiled`**
   - Uncomment dropout line in [classification_head.py#L34](src/neuro_stylometry/stylometry_net/classification_head.py#L34)

3. **Add integration test for graph training**
   - Test that captures forward+backward without error 906
   - Verify loss gradients match eager execution

### 7.3 Low Priority

4. **Add `capture_error_mode` configuration**
   - Allow `'relaxed'` mode for debugging
   - Default to `'global'` for production

5. **Add explicit stream context documentation**
   - Document that backward occurs within capture stream context
   - Explain why no additional stream control is needed after buffer fix

---

## 8. Conclusion

The **core fix for `cudaErrorStreamCaptureImplicit`** has been correctly implemented. The `detach().clone()` buffer preparation pattern eliminates legacy stream traces that would cause error 906 during backward pass.

The implementation follows PyTorch best practices:
- ✅ Buffer creation order: `detach()` before `clone()`
- ✅ Dedicated capture stream with proper synchronization
- ✅ Warmup iterations before capture
- ✅ Complete forward+loss+backward captured together
- ✅ Optimizer step kept outside capture

**Production readiness:** ⚠️ NOT YET READY due to debug bypasses in model forward pass. Once these are removed, the graph capture implementation should function correctly.

---

## References

- [PyTorch Issue #85660](https://github.com/pytorch/pytorch/issues/85660) - functorch CUDA Graph failure
- [PyTorch Issue #67285](https://github.com/pytorch/pytorch/issues/67285) - Legacy stream dependency error
- [PyTorch Discussion #34173](https://discuss.pytorch.org/t/difference-between-detach-clone-and-clone-detach/34173) - `detach().clone()` vs `clone().detach()`
- [PyTorch Blog: Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [NVIDIA CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
