# CUDA Graph Tensor Overwrite Error Investigation

**Date:** January 13, 2026  
**Status:** Under Investigation  
**Severity:** Critical (Blocking Phase D Training)

---

## Executive Summary

Phase D training crashes with a CUDA Graph tensor overwrite error when using `torch.compile` with `mode="reduce-overhead"`. The error occurs during `loss.backward()` because CUDA Graphs reuse memory addresses, causing stale tensor access when the backward pass tries to read output tensors from the forward pass.

**Attempted fixes with `torch.compiler.cudagraph_mark_step_begin()` have not resolved the issue.**

---

## Error Details

### Stack Trace
```
File "trainer.py", line 713, in _train
    loss_value, did_step = execute_with_oom_protection(
File "oom_guard.py", line 184, in execute_with_oom_protection
    return func()
File "trainer.py", line 683, in run_step
    loss.backward()
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten 
by a subsequent run. Stack trace: 
File "classification_head.py", line 30, in forward
    return {task: head(cls_embedding) for task, head in self.heads.items()}.
```

### Error Message Recommendations
The error suggests two solutions:
1. Clone tensors outside of `torch.compile()`
2. Call `torch.compiler.cudagraph_mark_step_begin()` before each model invocation

---

## Architecture Analysis

### Component Chain
```
PhaseDTrainer._train()
    └── execute_with_oom_protection(run_step)  # OOM wrapper
        └── run_step()
            ├── torch.compiler.cudagraph_mark_step_begin()  ← Currently called here
            ├── model(input_ids, attention_mask)  # Compiled AffineGuardTransformer
            │   └── Returns dict with "cls_embedding"
            ├── head(outputs["cls_embedding"])   # Compiled MultiTaskHead
            │   └── Returns Dict[str, Tensor]
            ├── head.compute_loss(logits, labels)
            │   └── Iterates over logits dict
            └── loss.backward()  ← Error occurs here
```

### Compilation Setup (trainer.py:371-392)
```python
# Both model AND head are separately compiled:
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
head = torch.compile(head, mode="reduce-overhead", fullgraph=False)
```

**Key Observation:** Two separately compiled modules means two separate CUDA Graph captures.

---

## Root Cause Analysis

### Why `cudagraph_mark_step_begin()` Isn't Working

The `cudagraph_mark_step_begin()` function marks boundaries between "steps" to tell PyTorch when tensor memory can be safely reused. However, there are several reasons it may not work in this architecture:

#### 1. Two Separate Compiled Graphs
The model and head are compiled **separately**, creating two independent CUDA Graph captures. When calling `cudagraph_mark_step_begin()` once:
- It may only affect the first graph (model)
- The head's graph may not respect the same boundary

#### 2. Dictionary Return Type Complexity
The `MultiTaskHead.forward()` returns a dictionary:
```python
def forward(self, cls_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
    cls_embedding = self.dropout(cls_embedding)
    return {task: head(cls_embedding) for task, head in self.heads.items()}
```
- CUDA Graphs capture static memory patterns
- Dictionary comprehensions with dynamic iteration may cause graph breaks
- Each head output tensor gets a fixed memory address in the graph

#### 3. Timing of mark_step_begin
The call may need to happen at a different point:
- **Before** data is moved to device?
- **Between** model and head calls?
- The correct timing is unclear with two separate graphs

#### 4. Gradient Tape vs Forward Tensors
CUDA Graphs capture both forward and backward kernels. The error suggests:
- The forward pass's output tensors are being overwritten
- But the backward pass still references the old memory
- This implies the graph is being re-executed (warmup? second iteration?) before backward completes

---

## Affected Files

| File | Component | Role |
|------|-----------|------|
| `src/neuro_stylometry/training/trainer.py` | `PhaseDTrainer._train()` | Training loop with OOM protection |
| `src/neuro_stylometry/training/trainer.py` | `PhaseDTrainer._build_model()` | Applies `torch.compile` to model+head |
| `src/neuro_stylometry/stylometry_net/classification_head.py` | `MultiTaskHead.forward()` | Returns dict of logits (error origin) |
| `src/neuro_stylometry/stylometry_net/transformer.py` | `AffineGuardTransformer.forward()` | Returns dict with cls_embedding |
| `src/neuro_stylometry/hardware_ops/oom_guard.py` | `execute_with_oom_protection()` | Wraps run_step |

---

## Proposed Solutions (In Order of Preference)

### Solution 1: Disable torch.compile for Head Only
**Rationale:** The head is a simple linear layer - compilation overhead likely outweighs benefits.

```python
# trainer.py:_build_model()
if self._use_torch_compile and self.device.type == "cuda":
    model = torch.compile(model, mode=self._torch_compile_mode, fullgraph=False)
    # Do NOT compile the head - simple linear layers don't benefit much
    # and dict return type causes CUDA Graph issues
    # head = torch.compile(head, ...)  # REMOVED
```

**Pros:**
- Simple change
- Head is cheap anyway (few parameters)
- Eliminates multi-graph interaction issues

**Cons:**
- Loses minor optimization on head

---

### Solution 2: Clone cls_embedding Before Head
**Rationale:** Explicitly break the CUDA Graph tensor dependency.

```python
# trainer.py:run_step()
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
# Clone to break CUDA Graph memory reuse dependency
cls_embedding = outputs["cls_embedding"].clone()
logits = head(cls_embedding)
```

**Pros:**
- Directly addresses the error message recommendation
- Keeps compilation on both modules

**Cons:**
- Small memory overhead from clone
- May need to clone in other locations too

---

### Solution 3: Use "default" Compile Mode
**Rationale:** "reduce-overhead" aggressively uses CUDA Graphs; "default" is more conservative.

```python
# trainer.py:_build_model() or config
torch_compile_mode: str = "default"  # Instead of "reduce-overhead"
```

**Pros:**
- Still gets compilation benefits
- Avoids CUDA Graph complexity

**Cons:**
- Less optimization for small batches
- May still have issues

---

### Solution 4: Mark Step Between Model and Head
**Rationale:** If two graphs need separate boundaries, mark between them.

```python
# trainer.py:run_step()
if self._use_torch_compile and self.device.type == "cuda":
    torch.compiler.cudagraph_mark_step_begin()

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Mark step AGAIN before head to separate the two graphs
if self._use_torch_compile and self.device.type == "cuda":
    torch.compiler.cudagraph_mark_step_begin()

logits = head(outputs["cls_embedding"])
```

**Pros:**
- Properly handles two-graph scenario

**Cons:**
- May not work if graphs are already captured
- Overhead of multiple boundary markers

---

### Solution 5: Compile Model and Head as Single Unit
**Rationale:** Avoid multi-graph issues by treating them as one.

```python
# Create a wrapper module
class CombinedModelHead(nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.head = head
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(outputs["cls_embedding"])

# Compile the combined module
combined = CombinedModelHead(model, head)
combined = torch.compile(combined, mode="reduce-overhead", fullgraph=False)
```

**Pros:**
- Single graph capture
- Cleaner execution

**Cons:**
- Requires architectural refactor
- May affect checkpoint loading

---

### Solution 6: Disable torch.compile Entirely (Fallback)
**Rationale:** If optimization causes instability, correctness takes priority.

```python
# conf/hpc/phase_d.yaml or conf/laptop/phase_d.yaml
optimization:
  use_torch_compile: false
```

**Pros:**
- Guaranteed to work
- No code changes needed

**Cons:**
- Loses compilation benefits
- Should be last resort

---

## Recommended Fix Implementation

**Primary Recommendation: Solution 1 (Don't compile head)**

This is the least invasive fix with the highest probability of success:

```python
# File: src/neuro_stylometry/training/trainer.py
# Location: _build_model() method, lines 371-392

def _build_model(self, *, use_affine_guard: bool) -> tuple[AffineGuardTransformer, MultiTaskHead]:
    # ... model creation code ...
    
    # Apply torch.compile for kernel optimization (CUDA Graphs)
    if self._use_torch_compile and self.device.type == "cuda":
        logger.info(
            f"Applying torch.compile to model (mode={self._torch_compile_mode})"
        )
        try:
            # Compile the transformer backbone for CUDA Graph caching
            model = torch.compile(
                model,
                mode=self._torch_compile_mode,
                fullgraph=False,
            )
            # NOTE: Do NOT compile the classification head.
            # The head returns a Dict[str, Tensor] which causes CUDA Graph
            # tensor overwrite issues during backward pass. The head is also
            # a simple linear layer that doesn't benefit much from compilation.
            logger.info("torch.compile applied to transformer (head excluded)")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without: {e}")
    
    return model, head
```

**If Solution 1 doesn't work, try Solution 2 (clone cls_embedding):**

```python
# File: src/neuro_stylometry/training/trainer.py
# Location: run_step() inner function

with autocast_ctx:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Clone to break CUDA Graph tensor dependency
    cls_embedding = outputs["cls_embedding"].clone()
    logits = head(cls_embedding)
    loss = head.compute_loss(logits, labels)
```

---

## Testing Verification

After implementing fix, verify with:

```bash
# Run Phase D with minimal data
python scripts/run_phase_d.py \
    --dataset artifacts/phase_a/clean_dataset.arrow \
    --output artifacts/phase_d_test \
    --artifacts-dir artifacts/phase_a \
    --mode laptop

# Check for:
# 1. No CUDA Graph errors
# 2. Training completes successfully
# 3. Metrics are reasonable
```

---

## Related PyTorch Issues

- [PyTorch #96854](https://github.com/pytorch/pytorch/issues/96854) - CUDA Graph tensor aliasing issues
- [PyTorch #106236](https://github.com/pytorch/pytorch/issues/106236) - cudagraph_mark_step_begin behavior
- [PyTorch Docs](https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html) - CUDA Graph trees documentation

---

## Appendix: Full Code Context

### MultiTaskHead.forward() (classification_head.py:28-30)
```python
def forward(self, cls_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
    cls_embedding = self.dropout(cls_embedding)
    return {task: head(cls_embedding) for task, head in self.heads.items()}
```

### AffineGuardTransformer.forward() (transformer.py:90-130)
```python
def forward(self, input_ids, attention_mask=None, ...):
    embedding_output = self.model.embeddings(input_ids)
    if self.affine_guard is not None:
        embedding_output = self.affine_guard(embedding_output)
    # ... encoder forward ...
    return {
        "last_hidden_state": sequence_output,
        "cls_embedding": cls_output,
    }
```

### Compilation Code (trainer.py:371-392)
```python
if self._use_torch_compile and self.device.type == "cuda":
    model = torch.compile(model, mode=self._torch_compile_mode, fullgraph=False)
    head = torch.compile(head, mode=self._torch_compile_mode, fullgraph=False)
```

---

## Document History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-13 | Copilot | Initial investigation document |
