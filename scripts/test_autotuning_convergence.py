#!/usr/bin/env python
"""
Manual Integration Test: Autotuning Convergence Test

This script verifies that the autotuning system finds the optimal
batch size by starting with a very low budget and observing scale-up.

Goal: Verify the PID/Heuristic controller finds the optimum.

Configuration:
- Sets initial_token_budget very low (512 tokens)
- Target VRAM utilization ~90%

Expected Behavior:
1. Observe logs showing "Scaling Up" messages
2. Watch nvidia-smi: VRAM usage should climb step-by-step
3. VRAM usage should stabilize around 80-90%
4. Throughput (samples/sec) should increase as batches get larger

Usage:
    python scripts/test_autotuning_convergence.py --data-path artifacts/data/sobr.arrow

    # In another terminal, monitor GPU:
    watch -n 1 nvidia-smi

Requirements:
    - CUDA-capable GPU
    - Installed neuro_stylometry package
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Enable debug logging for autotuning
logging.getLogger("neuro_stylometry.hardware_ops.runtime").setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description="Autotuning Convergence Test")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("artifacts/data/sobr.arrow"),
        help="Path to SOBR Arrow dataset",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=5000,
        help="Number of chunks to process (default: 5000)",
    )
    parser.add_argument(
        "--initial-budget",
        type=int,
        default=512,  # Very low
        help="Initial token budget (default: 512 - very low)",
    )
    parser.add_argument(
        "--target-vram",
        type=float,
        default=0.85,
        help="Target VRAM utilization (default: 0.85)",
    )
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This test requires a GPU.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    
    from neuro_stylometry.hardware_ops.runtime import (
        RuntimeController,
        RuntimeConfig,
        RuntimeMetrics,
        RuntimeState,
    )
    from neuro_stylometry.hardware_ops.telemetry import CUDATimer, TelemetryCollector
    from neuro_stylometry.pollution_guard.global_sort import (
        DynamicBatchIterator,
        FlattenedChunks,
    )
    
    # Create synthetic test data
    logger.info(f"Creating synthetic test data ({args.num_chunks} chunks)...")
    np.random.seed(42)
    
    texts = [f"Test chunk {i} " * 10 for i in range(args.num_chunks)]
    token_counts = np.random.randint(50, 250, size=args.num_chunks).astype(np.int16)
    doc_offsets = np.array([0, args.num_chunks], dtype=np.int64)
    chunk_metadata = [(0, i) for i in range(args.num_chunks)]
    
    flattened = FlattenedChunks(
        texts=texts,
        token_counts=token_counts,
        doc_offsets=doc_offsets,
        chunk_metadata=chunk_metadata,
    )
    
    logger.info(f"Total chunks: {args.num_chunks}")
    logger.info(f"Total tokens: {token_counts.sum():,}")
    logger.info(f"Avg tokens/chunk: {token_counts.mean():.1f}")
    
    # Compute memory headroom based on target utilization
    target_memory_mb = gpu_memory_gb * 1024 * args.target_vram
    headroom_mb = gpu_memory_gb * 1024 * (1 - args.target_vram)
    
    # Configure controller with very low initial budget
    config = RuntimeConfig(
        warmup_batches=5,
        initial_token_budget=args.initial_budget,
        min_token_budget=256,
        max_token_budget=10_000_000,  # 10M - high ceiling
        scale_up_factor=1.3,  # 30% increase when scaling
        scale_down_factor=0.85,
        memory_headroom_mb=headroom_mb,
        stability_threshold=0.1,
        history_window=10,
    )
    
    controller = RuntimeController(config)
    logger.info(f"Initial token budget: {args.initial_budget}")
    logger.info(f"Target VRAM utilization: {args.target_vram * 100:.0f}%")
    logger.info(f"Memory headroom: {headroom_mb:.0f} MB")
    
    # Reset telemetry
    TelemetryCollector.reset_instance()
    telemetry = TelemetryCollector.get_instance()
    
    # Create a realistic-ish GPU workload
    class SimulatedEncoder:
        """Simulated encoder that uses GPU memory."""
        
        def __init__(self, hidden_size: int = 768):
            self.device = torch.device("cuda")
            self.hidden_size = hidden_size
            # Create a small "model" tensor
            self.weights = torch.randn(hidden_size, hidden_size, device=self.device)
        
        def encode(self, batch_size: int, seq_len: int) -> torch.Tensor:
            """Simulate encoding a batch."""
            # Create input tensor
            x = torch.randn(batch_size, seq_len, self.hidden_size, device=self.device)
            # Simple matrix multiply to simulate computation
            with torch.no_grad():
                y = torch.matmul(x, self.weights)
            del x
            return y.mean()
    
    encoder = SimulatedEncoder()
    
    # Run the test
    logger.info("=" * 60)
    logger.info("Starting Autotuning Convergence Test")
    logger.info("=" * 60)
    logger.info("Watch nvidia-smi in another terminal to see VRAM climb")
    logger.info("=" * 60)
    
    iterator = DynamicBatchIterator(
        flattened=flattened,
        controller=controller,
        min_batch_size=1,
        max_batch_size=1000,
    )
    
    # Track metrics
    budget_history = []
    throughput_history = []
    vram_history = []
    state_changes = []
    
    last_state = controller.state
    start_time = time.time()
    
    for batch_idx, batch_indices in enumerate(iterator):
        batch_tokens = sum(int(flattened.token_counts[i]) for i in batch_indices)
        batch_size = len(batch_indices)
        avg_seq_len = batch_tokens // batch_size if batch_size > 0 else 0
        
        # Time the operation
        with CUDATimer() as timer:
            result = encoder.encode(batch_size, avg_seq_len)
        
        # Get memory usage
        memory_mb = torch.cuda.memory_allocated() / (1024**2)
        memory_pct = memory_mb / (gpu_memory_gb * 1024) * 100
        
        # Report to controller
        controller.report_metrics(RuntimeMetrics(
            tokens_processed=batch_tokens,
            batch_time_ms=timer.elapsed_ms,
            memory_used_mb=memory_mb,
            batch_size=batch_size,
        ))
        
        # Record telemetry
        telemetry.record_batch(
            batch_size=batch_size,
            tokens=batch_tokens,
            elapsed_ms=timer.elapsed_ms,
            memory_mb=memory_mb,
        )
        
        # Track history
        budget_history.append(controller.current_budget)
        throughput_history.append(batch_tokens / timer.elapsed_ms * 1000)
        vram_history.append(memory_pct)
        
        # Detect state changes
        if controller.state != last_state:
            state_changes.append((batch_idx, last_state.value, controller.state.value))
            logger.info(f"State transition: {last_state.value} -> {controller.state.value}")
            last_state = controller.state
        
        # Log progress every 20 batches
        if batch_idx % 20 == 0:
            logger.info(
                f"Batch {batch_idx:4d}: "
                f"size={batch_size:4d}, "
                f"budget={controller.current_budget:8,}, "
                f"VRAM={memory_pct:5.1f}%, "
                f"throughput={batch_tokens / timer.elapsed_ms * 1000:,.0f} tok/s, "
                f"state={controller.state.value}"
            )
        
        # Clean up
        del result
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    
    # Final summary
    logger.info("=" * 60)
    logger.info("Autotuning Convergence Test Complete")
    logger.info("=" * 60)
    
    summary = telemetry.get_summary()
    
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Total batches: {summary.total_batches}")
    logger.info(f"Total tokens: {summary.total_tokens:,}")
    logger.info(f"Avg throughput: {summary.avg_throughput_tokens_per_sec:,.0f} tok/s")
    logger.info(f"Peak VRAM: {max(vram_history):.1f}%")
    logger.info(f"Final VRAM: {vram_history[-1]:.1f}%")
    logger.info(f"Initial budget: {args.initial_budget}")
    logger.info(f"Final budget: {controller.current_budget:,}")
    logger.info(f"Budget increase: {controller.current_budget / args.initial_budget:.1f}x")
    logger.info(f"Final state: {controller.state.value}")
    
    # State transitions
    logger.info("\nState transitions:")
    for batch_idx, from_state, to_state in state_changes:
        logger.info(f"  Batch {batch_idx}: {from_state} -> {to_state}")
    
    # Convergence analysis
    logger.info("\nConvergence Analysis:")
    
    # Check if budget increased
    if budget_history[-1] > budget_history[0]:
        logger.info("✓ Budget scaled up from initial value")
    else:
        logger.warning("✗ Budget did not scale up")
    
    # Check if throughput improved
    early_throughput = np.mean(throughput_history[:10]) if len(throughput_history) >= 10 else throughput_history[0]
    late_throughput = np.mean(throughput_history[-10:]) if len(throughput_history) >= 10 else throughput_history[-1]
    
    if late_throughput > early_throughput:
        improvement = (late_throughput - early_throughput) / early_throughput * 100
        logger.info(f"✓ Throughput improved by {improvement:.1f}%")
    else:
        logger.warning("✗ Throughput did not improve")
    
    # Check VRAM utilization
    final_vram = np.mean(vram_history[-10:]) if len(vram_history) >= 10 else vram_history[-1]
    if final_vram > args.target_vram * 100 * 0.7:  # Within 70% of target
        logger.info(f"✓ VRAM utilization reached {final_vram:.1f}% (target: {args.target_vram * 100:.0f}%)")
    else:
        logger.warning(f"? VRAM utilization low: {final_vram:.1f}% (target: {args.target_vram * 100:.0f}%)")


if __name__ == "__main__":
    main()
