#!/usr/bin/env python3
"""Profile with actual AffineGuardTransformer + MultiTaskHead."""

import sys
import torch
import time
from pathlib import Path
from transformers import AutoModel
from neuro_stylometry.stylometry_net.transformer import AffineGuardTransformer
from neuro_stylometry.stylometry_net.classification_head import MultiTaskHead


def main():
    # Clear any stale CUDA allocations and restart CUDA context
    import gc
    import os
    gc.collect()
    torch.cuda.empty_cache()
    
    # Enable memory-efficient settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Loading AffineGuardTransformer...', flush=True)
    model = AffineGuardTransformer.from_phase_a(
        artifacts_dir='/workspace/language_and_ai/artifacts/phase_a',
        model_name='roberta-base',
        max_length=512,
        freeze_projection=True,
    ).to(device)
    
    print('Applying torch.compile...', flush=True)
    model = torch.compile(model, mode='default', fullgraph=False)
    
    num_labels = {'author': 24, 'stance': 2, 'topic': 5, 'region': 4, 
                  'age': 6, 'gender': 3, 'education': 5, 'income': 5}
    head = MultiTaskHead(
        hidden_dim=model.config.hidden_size, 
        num_labels_per_task=num_labels, 
        dropout=0.1
    ).to(device)

    batch_size, seq_len = 64, 512  # Full training batch size
    input_ids = torch.randint(0, 50265, (batch_size, seq_len), device=device)  # RoBERTa vocab
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = {k: torch.randint(0, v, (batch_size,), device=device) 
              for k, v in num_labels.items()}

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), 
        lr=1e-5
    )

    # Warmup phase (includes compilation)
    print('\n=== WARMUP PHASE (includes compilation) ===', flush=True)
    for i in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids, attention_mask=attention_mask)
            cls_embedding = out['cls_embedding'].clone()
            logits = head(cls_embedding)
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'  Warmup iter {i}: {elapsed*1000:.1f}ms ({1/elapsed:.2f} it/s)', flush=True)

    # Steady state benchmark
    print('\n=== STEADY STATE BENCHMARK (100 iterations) ===', flush=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    num_iters = 100
    for _ in range(num_iters):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids, attention_mask=attention_mask)
            cls_embedding = out['cls_embedding'].clone()
            logits = head(cls_embedding)
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    throughput_its = num_iters / elapsed
    throughput_samples = batch_size * num_iters / elapsed
    
    print(f'\n=== RESULTS ===', flush=True)
    print(f'  Model: AffineGuardTransformer (roberta-base) + MultiTaskHead', flush=True)
    print(f'  Batch size: {batch_size} samples x {seq_len} tokens', flush=True)
    print(f'  Total time: {elapsed:.2f}s for {num_iters} iterations', flush=True)
    print(f'  Throughput: {throughput_its:.1f} it/s ({throughput_samples:.0f} samples/s)', flush=True)
    print(f'  Expected training time per epoch: {1787 / throughput_its:.1f}s (1787 batches)', flush=True)
    
    print('\nDone!', flush=True)


if __name__ == '__main__':
    main()
