#!/usr/bin/env python
"""
Manual Integration Test: VRAM Squeeze Test

This script deliberately triggers OOM to verify real GPU error handling.
Run this on a machine with a GPU to verify the autotuning system's
OOM recovery works in practice.

Goal: Verify real GPU metric collection and OOM recovery.

Configuration:
- Sets initial_token_budget absurdly high to force immediate OOM
- Uses a small data subset for quick testing

Expected Behavior:
1. Logs show "OOM Detected"
2. Logs show "Slashed budget to X"
3. Logs show "Garbage collection triggered"
4. Pipeline continues and finishes successfully

Usage:
    python scripts/test_vram_squeeze.py --data-path artifacts/data/sobr.arrow

Requirements:
    - CUDA-capable GPU
    - Installed neuro_stylometry package
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import pyarrow as pa
import pyarrow.feather as feather

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Enable debug logging for autotuning components
logging.getLogger("neuro_stylometry.hardware_ops.runtime").setLevel(logging.DEBUG)
logging.getLogger("neuro_stylometry.hardware_ops.oom_guard").setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description="VRAM Squeeze Test")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("artifacts/data/sobr.arrow"),
        help="Path to SOBR Arrow dataset",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1000,
        help="Number of chunks to process (default: 1000)",
    )
    parser.add_argument(
        "--initial-budget",
        type=int,
        default=50_000_000,  # 50M tokens - way too high
        help="Initial token budget (default: 50M to force OOM)",
    )
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This test requires a GPU.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Import after logging setup
    from neuro_stylometry.hardware_ops.runtime import (
        RuntimeController,
        RuntimeConfig,
        RuntimeMetrics,
        RuntimeState,
    )
    from neuro_stylometry.hardware_ops.oom_guard import execute_with_oom_protection
    from neuro_stylometry.pollution_guard.global_sort import (
        DynamicBatchIterator,
        FlattenedChunks,
    )
    import numpy as np
    
    # Create synthetic test data (simpler than loading real data)
    logger.info(f"Creating synthetic test data ({args.num_chunks} chunks)...")
    np.random.seed(42)
    
    texts = [f"This is test chunk number {i} with some text content." for i in range(args.num_chunks)]
    token_counts = np.random.randint(50, 300, size=args.num_chunks).astype(np.int16)
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
    
    # Configure controller with absurdly high initial budget
    config = RuntimeConfig(
        warmup_batches=2,
        initial_token_budget=args.initial_budget,
        min_token_budget=10_000,
        max_token_budget=100_000_000,
        oom_slash_factor=0.3,  # Aggressive slash
        recovery_patience=3,
        memory_headroom_mb=500,
    )
    
    controller = RuntimeController(config)
    logger.info(f"Initial token budget: {args.initial_budget:,} (designed to cause OOM)")
    
    # Create a dummy model that allocates GPU memory
    class DummyGPUModel:
        """Dummy model that allocates memory based on input size."""
        
        def __init__(self):
            self.device = torch.device("cuda")
        
        def __call__(self, batch_tokens: int) -> torch.Tensor:
            """Allocate memory proportional to batch tokens."""
            # Allocate ~1KB per token (adjust to trigger OOM)
            # This is artificial but demonstrates the principle
            elements = batch_tokens * 256  # 256 floats = 1KB per token
            tensor = torch.randn(elements, device=self.device)
            # Do some computation to ensure allocation happens
            result = tensor.mean()
            del tensor
            return result
    
    model = DummyGPUModel()
    
    # Run the test loop
    logger.info("=" * 60)
    logger.info("Starting VRAM Squeeze Test")
    logger.info("=" * 60)
    
    iterator = DynamicBatchIterator(
        flattened=flattened,
        controller=controller,
        min_batch_size=1,
        max_batch_size=500,
    )
    
    processed_chunks = 0
    oom_events = 0
    
    for batch_idx, batch_indices in enumerate(iterator):
        batch_tokens = sum(int(flattened.token_counts[i]) for i in batch_indices)
        
        logger.info(
            f"Batch {batch_idx}: {len(batch_indices)} chunks, "
            f"{batch_tokens:,} tokens, budget={controller.current_budget:,}"
        )
        
        def run_batch():
            return model(batch_tokens)
        
        try:
            result = execute_with_oom_protection(
                run_batch,
                controller=controller,
                retry_limit=5,
            )
            
            # Report metrics
            memory_mb = torch.cuda.memory_allocated() / (1024**2)
            controller.report_metrics(RuntimeMetrics(
                tokens_processed=batch_tokens,
                batch_time_ms=10.0,
                memory_used_mb=memory_mb,
                batch_size=len(batch_indices),
            ))
            
            processed_chunks += len(batch_indices)
            
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            oom_events += 1
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Summary
    logger.info("=" * 60)
    logger.info("VRAM Squeeze Test Complete")
    logger.info("=" * 60)
    logger.info(f"Processed chunks: {processed_chunks}/{args.num_chunks}")
    logger.info(f"OOM events detected: {controller._oom_count}")
    logger.info(f"Final budget: {controller.current_budget:,}")
    logger.info(f"Final state: {controller.state.value}")
    
    # Verdict
    if processed_chunks == args.num_chunks:
        logger.info("✓ SUCCESS: All chunks processed despite OOM")
    else:
        logger.warning(f"✗ INCOMPLETE: Only {processed_chunks}/{args.num_chunks} processed")
    
    if controller._oom_count > 0:
        logger.info("✓ OOM Recovery: System recovered from OOM events")
    else:
        logger.warning("? No OOM detected - increase initial_budget or reduce GPU memory")


if __name__ == "__main__":
    main()
