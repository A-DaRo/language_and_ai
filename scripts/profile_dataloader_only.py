#!/usr/bin/env python3
"""Profile DataLoader overhead in isolation."""

import sys
import torch
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from neuro_stylometry.stylometry_net.phase_d_dataset import (
    PhaseDDataset, FastCollator, DevicePrefetcher
)
from neuro_stylometry.data_engine.bucketing import QuantizedBucketSampler


def main():
    device = torch.device('cuda')
    
    # Load real dataset
    print('Loading dataset...', flush=True)
    arrow_path = Path('/workspace/language_and_ai/artifacts/phase_d/tokenized_dataset_post_masked.arrow')
    
    dataset = PhaseDDataset(
        arrow_path, 
        text_field='post_masked',
        split='train',
    )
    print(f'  Dataset: {len(dataset)} samples', flush=True)
    print(f'  AOT mode: {dataset.is_aot_mode}', flush=True)
    
    # Get lengths from dataset (this is slow for 228k samples)
    print('Extracting lengths...', flush=True)
    # Try to get lengths efficiently from the arrow metadata if available
    sample = dataset[0]
    if 'token_count' in sample:
        # Fast path - read directly from arrow table
        import pyarrow.feather as feather
        table = feather.read_table(arrow_path, memory_map=True, columns=['token_count'])
        lengths = np.array(table['token_count'].to_pylist(), dtype=np.int32)
        # Filter to train split size
        lengths = lengths[:len(dataset)]
    else:
        # Slow path - iterate
        lengths = np.array([dataset.get_length(i) for i in range(min(10000, len(dataset)))], dtype=np.int32)
    
    print(f'  Lengths extracted: {len(lengths)} samples', flush=True)
    
    # Create DataLoader with real config
    collator = FastCollator(max_length=512, pad_token_id=1, quantize_step=32)
    
    sampler = QuantizedBucketSampler(
        lengths=lengths,
        token_budget=32768,
        max_length=512,
        quantize_step=32,
        shuffle=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    # Test 1: Pure DataLoader iteration (no GPU at all)
    print('\n=== TEST 1: Pure DataLoader iteration (no compute) ===', flush=True)
    start = time.perf_counter()
    batch_count = 0
    sample_count = 0
    for batch in loader:
        batch_count += 1
        sample_count += batch['input_ids'].size(0)
        if batch_count >= 100:
            break
    elapsed = time.perf_counter() - start
    print(f'  Batches: {batch_count}, Samples: {sample_count}', flush=True)
    print(f'  Time: {elapsed:.2f}s, Rate: {batch_count/elapsed:.1f} batch/s', flush=True)
    
    # Test 2: DataLoader + H2D transfer only (no model)
    print('\n=== TEST 2: DataLoader + H2D transfer (no compute) ===', flush=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch_count = 0
    sample_count = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = {k: v.to(device, non_blocking=True) for k, v in batch['labels'].items()}
        torch.cuda.synchronize()  # Ensure transfer completes
        batch_count += 1
        sample_count += input_ids.size(0)
        if batch_count >= 100:
            break
    elapsed = time.perf_counter() - start
    print(f'  Batches: {batch_count}, Samples: {sample_count}', flush=True)
    print(f'  Time: {elapsed:.2f}s, Rate: {batch_count/elapsed:.1f} batch/s', flush=True)
    
    # Test 3: DataLoader + DevicePrefetcher (no compute)
    print('\n=== TEST 3: DataLoader + DevicePrefetcher (no compute) ===', flush=True)
    prefetcher = DevicePrefetcher(loader, device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch_count = 0
    sample_count = 0
    for batch in prefetcher:
        batch_count += 1
        sample_count += batch['input_ids'].size(0)
        if batch_count >= 100:
            break
    elapsed = time.perf_counter() - start
    print(f'  Batches: {batch_count}, Samples: {sample_count}', flush=True)
    print(f'  Time: {elapsed:.2f}s, Rate: {batch_count/elapsed:.1f} batch/s', flush=True)
    
    # Test 4: tqdm overhead test
    print('\n=== TEST 4: DataLoader + Prefetch + tqdm (no compute) ===', flush=True)
    from tqdm import tqdm
    prefetcher = DevicePrefetcher(loader, device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch_count = 0
    sample_count = 0
    for batch in tqdm(prefetcher, total=100, desc="Test"):
        batch_count += 1
        sample_count += batch['input_ids'].size(0)
        if batch_count >= 100:
            break
    elapsed = time.perf_counter() - start
    print(f'  Batches: {batch_count}, Samples: {sample_count}', flush=True)
    print(f'  Time: {elapsed:.2f}s, Rate: {batch_count/elapsed:.1f} batch/s', flush=True)
    
    print('\n=== SUMMARY ===', flush=True)
    print(f'If raw model does 6.4 it/s and DataLoader is slower,', flush=True)
    print(f'that indicates DataLoader is the bottleneck.', flush=True)
    print(f'Expected: DataLoader should be much faster than 6.4 batch/s.', flush=True)
    
    print('\nDone!', flush=True)


if __name__ == '__main__':
    main()
