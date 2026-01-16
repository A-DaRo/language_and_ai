## Plan: Fix VRAM Accumulation in Phase D Training Pipeline

The analysis identified 15-25 GB VRAM waste by epoch 100 due to unchecked buffer growth, pipeline depth, budget ratcheting, and missing cleanup. This plan addresses the critical memory leaks through config tuning, explicit cleanup, and architectural safeguards to enable stable 300-epoch training on 24GB GPUs.

### Steps

1. **Apply immediate config tuning** in phase_d.yaml: Lower `optimization.token_budget` from 8192→4096 (conservative start), re-enable `execution.autotuning.enabled: true` with strict ceiling `max_token_budget: 65536` (down from 393216 to prevent runaway scaling), reduce `data_loader.prefetch_factor` from 16→4 (cuts pipeline from 384→96 batches), reduce `training.gradient_accumulation_steps` from 2→1 (halves gradient memory). Expected impact: **~10 GB VRAM** recovered immediately.

2. **Add epoch-end CUDA cleanup** in trainer.py: After `batch_pbar.close()` at end of epoch loop, insert explicit `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`, and `gc.collect()` with debug logging. This prevents 2-4 GB cache fragmentation accumulation across 300 epochs. Also clear telemetry via `self._telemetry.clear()` to release 400KB BatchMetrics buffer per epoch.

3. **Implement aggressive tensor cleanup in training loop** in trainer.py: After `loss.backward()` and optional metrics logging, add explicit `del loss, outputs, logits, cls_embedding` to release computation graph before next batch. Reduces transient memory spikes by ~100 MB and prevents autograd graph retention under high-throughput training.

4. **Add budget growth ceiling to prevent ratcheting** in trainer.py: Modify epoch-start budget reset to cap `successful_budget` at `4 × initial_token_budget` (prevents unbounded growth from 8KB→384KB). Reset `self._runtime_controller._successful_budget = capped_successful` each epoch to prevent monotonic increase. Saves 3-10 GB VRAM in late epochs by preventing single batches from reaching 384KB tokens.

5. **Optimize RuntimeController history trimming** in runtime.py: Replace list slicing `self._throughput_history = self._throughput_history[-window:]` with in-place deletion `del self._throughput_history[:-window]` for all three history buffers. Eliminates 14.4 MB memory churn from 60,000+ list slice operations creating temporary objects during 300-epoch training.

6. **Add DevicePrefetcher reset method and epoch-level call** in phase_d_dataset.py: Implement `DevicePrefetcher.reset()` method that clears `self._next_batch = None` and synchronizes CUDA stream. Call from trainer at epoch end (after line 1208) when `use_device_prefetch` is active. Releases ~500 MB held in prefetcher's CUDA stream across epochs.

7. **Update torch.compile config for stability** in phase_d.yaml: Change `torch_compile_mode: "reduce-overhead"` → `"default"` (kernel fusion only, no CUDA Graphs), set `torch_compile_dynamic: false` (force static shapes to prevent recompilation), keep `torch_compile_disable_cudagraphs: true`. Trades 20% throughput (100→80 it/s) for eliminating 800 MB kernel cache growth and CUDA Graph tensor aliasing bugs.

8. **Add validation logging for memory tracking** in trainer.py: At training start, log initial VRAM baseline via `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`. At each epoch end (after cleanup), log current memory stats with delta from baseline. Add warning if reserved memory grows >20% from epoch to epoch, indicating cleanup failure.

### Further Considerations

1. **OOM recovery interaction with adaptive budgets**: Current `execution.oom_recovery.epoch_level_protection: true` may conflict with budget ceiling - if OOM occurs, should we slash budget below ceiling or disable adaptive entirely for that epoch? Recommend adding config flag `oom_recovery.disable_adaptive_on_oom: true` to lock budget at last successful value after OOM event.

2. **Telemetry singleton persistence across runs**: `TelemetryCollector.get_instance()` returns same object if training script reruns in Jupyter/interactive session. Should we add `TelemetryCollector.reset_singleton()` method called at pipeline start, or is per-epoch `clear()` sufficient? Consider adding `max_epochs` parameter to telemetry that auto-clears when exceeded.

3. **Prefetch factor tuning by GPU generation**: `prefetch_factor: 4` is conservative for RTX 5090 (900 GB/s memory bandwidth). A100/H100 may benefit from higher values (8-12). Consider auto-tuning based on GPU memory bandwidth detection or making it a function of `batch_size` (e.g., `prefetch_factor = max(2, 32 // batch_size)`).