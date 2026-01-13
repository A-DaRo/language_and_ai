# Phase D CUDA Graph Training Implementation Plan

## Executive Summary

Analysis of the Phase D training path reveals several high-priority issues that render the CUDA graph training infrastructure effectively unusable and leave significant throughput headroom on modern GPUs (RTX 5090/A100/H100). This document validates the findings and provides a comprehensive implementation plan.

---

## Validated Findings

### 🔴 HIGH: Manual CUDA-graph training is unusable

**Status: CONFIRMED**

**Evidence:**

1. **Missing `forward_cls` method** - [cuda_graphs.py#L943](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L943) calls `model.forward_cls()`:
   ```python
   cls_embedding = model.forward_cls(
       input_ids=input_ids,
       attention_mask=attention_mask,
   )
   ```
   However, [transformer.py](src/neuro_stylometry/stylometry_net/transformer.py) only defines `forward()`, not `forward_cls()`. This will raise `AttributeError` at runtime.

2. **Trainer never switches to `GraphAwareTraining`** - The `use_cuda_graph_training` config flag is:
   - Defined in `PhaseDTrainConfig` ([trainer.py#L124](src/neuro_stylometry/training/trainer.py#L124))
   - Passed through `PhaseDTrainConfig` in phase_d_pipeline.py ([phase_d_pipeline.py#L160](src/neuro_stylometry/phase_d_pipeline.py#L160))
   - **BUT NEVER USED** in the actual `_train()` method ([trainer.py#L566-L900](src/neuro_stylometry/training/trainer.py#L566-L900))
   
   The training loop always uses the standard forward pass, never instantiating `GraphAwareTraining` or calling its `run()` method.

3. **Config passed but unused** - `cuda_graph_training` dict is set in config but never consumed.

---

### 🔴 HIGH: CUDA graph memory-pool controls are ignored

**Status: CONFIRMED**

**Evidence:**

1. **Config defines pool settings** - [cuda_graphs.py#L47-54](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L47-L54):
   ```python
   capture_pool_size_mb: int = 256
   use_cuda_graph_memory_pool: bool = True
   ```

2. **Capture ignores pool** - [cuda_graphs.py#L200-204](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L200-L204):
   ```python
   graph = torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph, stream=self._capture_stream):
       output_dict = forward_fn(**input_buffers)
   ```
   
   The `pool=` parameter is **never passed** to `torch.cuda.graph()`. Per the CUDA Programming Guide (Section 4.2.5), using a dedicated memory pool for graph captures stabilizes allocations and reduces fragmentation.

**Impact:** Each graph capture uses the default CUDA allocator pool, which can cause:
- Memory fragmentation across captures
- Inconsistent allocation patterns
- Potential OOM during repeated capture/replay cycles

---

### 🟡 MEDIUM: HPC config flags are no-ops

**Status: CONFIRMED**

| Flag | Location | Issue |
|------|----------|-------|
| `compile_train_step` | [phase_d.yaml#L45](conf/hpc/phase_d.yaml#L45) | Set to `true` but never referenced in trainer. Full train step compilation not implemented. |
| `torch_compile_disable_cudagraphs` | [phase_d.yaml#L48](conf/hpc/phase_d.yaml#L48) | **Partially implemented** - switches torch.compile mode to "default" in `_build_model()` but doesn't pass `options={"triton.cudagraphs": False}` to fully disable. |
| `use_pre_padded` | [phase_d.yaml#L41](conf/hpc/phase_d.yaml#L41) | Defined but never checked. `FastCollator` doesn't branch on this; pre-padding detection relies on dataset attributes. |

**Evidence:** Search for `compile_train_step` usage:
```
# Only appears in config dataclass definition, never in logic
use_cuda_graph_training: bool = False  # Manual CUDA-graph training path
```

---

### 🟡 MEDIUM: FP8 kernels won't activate

**Status: CONFIRMED**

**Evidence:**

1. **TransformerEngine autocast is used** - [trainer.py#L421-426](src/neuro_stylometry/training/trainer.py#L421-L426):
   ```python
   if self._precision == "fp8" and is_fp8_available():
       return _TE_MODULE.fp8_autocast(
           enabled=True,
           fp8_recipe=get_fp8_recipe(),
       )
   ```

2. **Model uses standard HuggingFace layers** - [transformer.py#L46-53](src/neuro_stylometry/stylometry_net/transformer.py#L46-L53):
   ```python
   self.model = AutoModel.from_pretrained(
       model_name,
       attn_implementation="sdpa",
   )
   ```

3. **FP8 kernels require TransformerEngine modules** - `te.fp8_autocast` only triggers FP8 execution for:
   - `te.Linear`
   - `te.LayerNorm`
   - `te.TransformerLayer`
   
   Standard `nn.Linear` and HuggingFace attention layers will **fall back to BF16/FP16** inside the autocast, not FP8.

**Impact:** FP8 precision setting is misleading - actual throughput will be BF16-equivalent despite config claiming FP8.

---

### 🟢 LOW/MEDIUM: Autotuning disabled despite comment

**Status: CONFIRMED**

**Evidence:** [phase_d.yaml#L66-67](conf/hpc/phase_d.yaml#L66-L67):
```yaml
execution:
  autotuning:
    enabled: false  # Let runtime find optimal budget  <-- Comment is wrong!
```

The comment says "Let runtime find optimal budget" but `enabled: false` means autotuning is OFF. This may be intentional for stability with CUDA graphs (fixed shapes), but the comment is misleading.

---

## Implementation Plan

### Phase 1: Wire CUDA-graph Training (Priority: HIGH)

#### 1.1 Add `forward_cls` to `AffineGuardTransformer`

```python
# transformer.py - Add after forward() method
def forward_cls(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Optimized forward for CUDA graph capture - returns only CLS embedding.
    
    Excludes optional outputs to ensure static graph structure.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    embedding_output = self.model.embeddings(input_ids)
    if self.affine_guard is not None:
        embedding_output = self.affine_guard(embedding_output)

    extended_attention_mask = self.model.get_extended_attention_mask(
        attention_mask, input_ids.shape
    )

    encoder_outputs = self.model.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    return encoder_outputs.last_hidden_state[:, 0, :]  # CLS only
```

#### 1.2 Wire `use_cuda_graph_training` into `PhaseDTrainer._train()`

```python
# trainer.py - In _train() method, after model/head initialization

# Initialize CUDA graph training if enabled
self._use_cuda_graph_training = bool(self.config.use_cuda_graph_training)
self._graph_aware_trainer: Optional[GraphAwareTraining] = None

if self._use_cuda_graph_training and self.device.type == "cuda":
    from ..hardware_ops.cuda_graphs import (
        create_graph_aware_training_from_config,
    )
    self._graph_aware_trainer = create_graph_aware_training_from_config(
        self.config.cuda_graph_training,
        default_max_seq_len=self.config.max_length,
    )
    
    # Disable torch.compile when using manual CUDA graphs (conflict)
    if self._use_torch_compile:
        logger.warning(
            "Disabling torch.compile (use_cuda_graph_training=True takes precedence)"
        )
        self._use_torch_compile = False
        
    logger.info("GraphAwareTraining enabled for CUDA graph capture")
```

Then modify the training step:

```python
# Inside the training loop
if self._graph_aware_trainer is not None and self._graph_aware_trainer.is_enabled:
    # CUDA graph training path
    labels_tuple = tuple(labels[k] for k in sorted(labels.keys()))
    loss, valid, used_graph = self._graph_aware_trainer.run(
        model=model,
        head=head,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels_tuple,
        accum_steps=accum_steps,
        pad_token_id=1,  # RoBERTa
    )
    # ... handle loss/valid
else:
    # Standard training path (existing code)
    ...
```

---

### Phase 2: Add CUDA Graph Pool Support (Priority: HIGH)

#### 2.1 Create shared memory pool in `GraphCache.__init__`

```python
# cuda_graphs.py - In GraphCache.__init__

# Create dedicated memory pool for graph captures
self._capture_pool: Optional[Tuple[int, int]] = None
if self.config.use_cuda_graph_memory_pool and self._cuda_available:
    try:
        # Get memory pool handle from CUDA allocator
        # Note: pool_size_mb is a hint; actual allocation is dynamic
        pool = torch.cuda.graph_pool_handle()
        self._capture_pool = pool
        logger.info(
            f"CUDA graph memory pool initialized "
            f"(hint: {self.config.capture_pool_size_mb}MB)"
        )
    except Exception as e:
        logger.warning(f"Failed to create CUDA graph pool: {e}")
```

#### 2.2 Pass pool to `torch.cuda.graph()` during capture

```python
# cuda_graphs.py - In GraphCache.capture()

# Capture the graph with pool (if available)
graph = torch.cuda.CUDAGraph()

capture_kwargs = {"stream": self._capture_stream}
if self._capture_pool is not None:
    capture_kwargs["pool"] = self._capture_pool

with torch.cuda.graph(graph, **capture_kwargs):
    output_dict = forward_fn(**input_buffers)
```

---

### Phase 3: Honor Config Flags (Priority: MEDIUM)

#### 3.1 Implement `compile_train_step`

```python
# trainer.py - After model compilation

if self.config.compile_train_step and self._use_torch_compile:
    def _compiled_train_step(input_ids, attention_mask, labels):
        with self._get_autocast_context():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs["cls_embedding"].clone()
            logits = head(cls_embedding)
            loss = head.compute_loss(logits, labels)
        if loss is not None:
            (loss / accum_steps).backward()
        return loss
    
    self._compiled_train_step = torch.compile(
        _compiled_train_step,
        mode=self._torch_compile_mode,
        fullgraph=False,
    )
    logger.info("Full train step compiled with torch.compile")
```

#### 3.2 Properly disable CUDA graphs in torch.compile

```python
# trainer.py - In _build_model()

compile_options = {}
if self._torch_compile_disable_cudagraphs:
    compile_options["options"] = {"triton.cudagraphs": False}
    effective_mode = "default"  # Also switch mode
else:
    effective_mode = self._torch_compile_mode

model = torch.compile(
    model,
    mode=effective_mode,
    fullgraph=False,
    **compile_options,
)
```

#### 3.3 Add `use_pre_padded` check in `FastCollator`

```python
# phase_d_dataset.py - In FastCollator.__call__()

# When use_pre_padded=True, skip padding entirely (zero-copy)
if getattr(self, 'use_pre_padded', False) and all(
    b.get('pre_padded', False) for b in batch
):
    # Stack directly without padding
    return self._zero_copy_collate(batch)
```

---

### Phase 4: FP8 Validation/Alternative (Priority: MEDIUM)

#### Option A: Use TransformerEngine modules (Full FP8)

```python
# transformer.py - Replace HF model with TE layers

if fp8_enabled and _TE_AVAILABLE:
    # Build transformer from TransformerEngine components
    self.encoder = te.TransformerLayer(
        hidden_size=config.hidden_size,
        ffn_hidden_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        ...
    )
```

**Pros:** True FP8 execution, 2-3x throughput
**Cons:** Requires architecture rewrite, different checkpoints

#### Option B: Use torchao FP8 quantization (Simpler)

```python
# trainer.py - After model load

from torchao.quantization import quantize, float8_dynamic_activation_float8_weight

if self._precision == "fp8":
    model = quantize(model, float8_dynamic_activation_float8_weight())
```

**Pros:** Works with existing HF models
**Cons:** May have different numeric behavior

#### Option C: Document BF16 as "production FP8" (Pragmatic)

Update config comments to clarify FP8 falls back to BF16 with HF models. Recommend BF16 + SDPA/Flash for production throughput.

---

### Phase 5: Autotuning & Config Cleanup (Priority: LOW)

#### 5.1 Fix misleading comment

```yaml
# conf/hpc/phase_d.yaml
execution:
  autotuning:
    enabled: false  # Disabled for CUDA graph stability (fixed shapes required)
```

#### 5.2 Consider enabling for non-graph mode

```yaml
# When use_cuda_graph_training: false
autotuning:
  enabled: true  # Dynamic budget optimization
  
# When use_cuda_graph_training: true  
autotuning:
  enabled: false  # Fixed shapes required for graph reuse
```

---

## Testing Plan

### Unit Tests

1. **`test_forward_cls_exists`**: Verify `AffineGuardTransformer.forward_cls` returns CLS tensor
2. **`test_graph_pool_creation`**: Verify `GraphCache` creates pool when configured
3. **`test_graph_capture_uses_pool`**: Verify captures pass `pool=` parameter
4. **`test_cuda_graph_training_wired`**: Verify trainer uses `GraphAwareTraining` when flag set

### Integration Tests

1. **`test_cuda_graph_training_e2e`**: Full training run with `use_cuda_graph_training: true`
2. **`test_config_flags_honored`**: Verify all config flags affect runtime behavior

### Benchmark

```python
# scripts/benchmark_cuda_graph_training.py

def benchmark_phase_d_throughput():
    """Compare throughput: baseline vs torch.compile vs manual CUDA graphs."""
    configs = [
        ("baseline", {"use_torch_compile": False}),
        ("compile_default", {"use_torch_compile": True, "torch_compile_mode": "default"}),
        ("compile_reduce_overhead", {"use_torch_compile": True, "torch_compile_mode": "reduce-overhead"}),
        ("manual_cuda_graphs", {"use_cuda_graph_training": True}),
    ]
    
    for name, overrides in configs:
        # Run N batches, measure throughput
        throughput = run_benchmark(overrides)
        print(f"{name}: {throughput:.1f} samples/sec")
```

---

## Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1: Wire CUDA-graph training | 4-6 hours | Medium (backward compat) |
| Phase 2: Graph pool support | 2-3 hours | Low |
| Phase 3: Honor config flags | 3-4 hours | Low |
| Phase 4: FP8 validation | 2-8 hours | Medium (option dependent) |
| Phase 5: Config cleanup | 1 hour | Low |
| Testing | 4-6 hours | - |
| **Total** | **16-28 hours** | - |

---

## Recommended Implementation Order

1. **Phase 2** (Graph pool) - Quick win, no API changes
2. **Phase 1** (Wire CUDA-graph training) - Core functionality
3. **Phase 3** (Config flags) - Consistency
4. **Phase 5** (Config cleanup) - Documentation
5. **Phase 4** (FP8) - Optional, depends on throughput requirements

---

## Appendix: Key File Locations

| Component | File |
|-----------|------|
| CUDA Graph Cache | [cuda_graphs.py](src/neuro_stylometry/hardware_ops/cuda_graphs.py) |
| GraphAwareTraining | [cuda_graphs.py#L883](src/neuro_stylometry/hardware_ops/cuda_graphs.py#L883) |
| Transformer Model | [transformer.py](src/neuro_stylometry/stylometry_net/transformer.py) |
| Trainer | [trainer.py](src/neuro_stylometry/training/trainer.py) |
| Pipeline | [phase_d_pipeline.py](src/neuro_stylometry/phase_d_pipeline.py) |
| HPC Config | [conf/hpc/phase_d.yaml](conf/hpc/phase_d.yaml) |
| Classification Head | [classification_head.py](src/neuro_stylometry/stylometry_net/classification_head.py) |
