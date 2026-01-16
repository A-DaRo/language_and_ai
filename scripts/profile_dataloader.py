#!/usr/bin/env python3
"""Profile DataLoader overhead vs pure compute."""

import sys
import torch
import time
import yaml
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

from neuro_stylometry.stylometry_net.classification_head import MultiTaskHead
from neuro_stylometry.stylometry_net.phase_d_dataset import (
    PhaseDDataset, FastCollator, DevicePrefetcher
)
from neuro_stylometry.data_engine.bucketing import QuantizedBucketSampler


def main():
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load real dataset
    print('Loading dataset...', flush=True)
    arrow_path = Path('/workspace/language_and_ai/artifacts/phase_d/tokenized_dataset_post_masked.arrow')
    
    with open('/workspace/language_and_ai/conf/hpc/phase_d.yaml') as f:
        config = yaml.safe_load(f)
    
    dataset = PhaseDDataset(
        arrow_path, 
        text_field='post_masked',  # Use masked text field
        split='train',  # Training split only
    )
    print(f'  Dataset: {len(dataset)} samples', flush=True)
    print(f'  AOT mode: {dataset.is_aot_mode}', flush=True)
    
    # Load model
    print('\nLoading model...', flush=True)
    encoder = AutoModel.from_pretrained(
        'distilbert-base-uncased', 
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa'
    ).to(device)
    encoder = torch.compile(encoder, mode='default', fullgraph=False)
    
    num_labels = {'author': 24, 'stance': 2, 'topic': 5, 'region': 4, 
                  'age': 6, 'gender': 3, 'education': 5, 'income': 5}
    head = MultiTaskHead(
        hidden_dim=768, 
        num_labels_per_task=num_labels, 
        dropout=0.1
    ).to(device, dtype=torch.bfloat16)
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), 
        lr=1e-5
    )
    
    # Create DataLoader with real config
    collator = FastCollator(max_length=512, pad_token_id=1, quantize_step=32)
    
    # Get lengths from dataset - using get_length for each sample
    lengths = np.array([dataset.get_length(i) for i in range(len(dataset))], dtype=np.int32)
    
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
    
    # Warmup model
    print('\nWarming up model...', flush=True)
    sample_batch = next(iter(loader))
    for _ in range(5):
        input_ids = sample_batch['input_ids'].to(device)
        attention_mask = sample_batch['attention_mask'].to(device)
        labels = {k: v.to(device) for k, v in sample_batch['labels'].items()}
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = encoder(input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :].clone()
            logits = head(cls)
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    
    # Test 1: Pure DataLoader iteration (no model)
    print('\n=== TEST 1: Pure DataLoader iteration (no compute) ===', flush=True)
    torch.cuda.synchronize()
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
    
    # Test 2: DataLoader + H2D transfer only
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
    
    # Test 4: Full pipeline with DevicePrefetcher
    print('\n=== TEST 4: Full pipeline (DataLoader + Prefetch + Model + Optimizer) ===', flush=True)
    prefetcher = DevicePrefetcher(loader, device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch_count = 0
    sample_count = 0
    for batch in prefetcher:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = encoder(input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :].clone()
            logits = head(cls)
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        batch_count += 1
        sample_count += input_ids.size(0)
        if batch_count >= 100:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f'  Batches: {batch_count}, Samples: {sample_count}', flush=True)
    print(f'  Time: {elapsed:.2f}s, Rate: {batch_count/elapsed:.1f} batch/s', flush=True)
    print(f'  Samples/s: {sample_count/elapsed:.0f}', flush=True)
    
    print('\nDone!', flush=True)


if __name__ == '__main__':
    main()
