#!/usr/bin/env python3
"""Profile torch.compile overhead and steady state performance."""

import sys
import torch
import time
from transformers import AutoModel
from neuro_stylometry.stylometry_net.classification_head import MultiTaskHead


def main():
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Loading model...', flush=True)
    encoder = AutoModel.from_pretrained(
        'distilbert-base-uncased', 
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa'
    ).to(device)
    
    print('Applying torch.compile...', flush=True)
    encoder = torch.compile(encoder, mode='default', fullgraph=False)

    num_labels = {'author': 24, 'stance': 2, 'topic': 5, 'region': 4, 
                  'age': 6, 'gender': 3, 'education': 5, 'income': 5}
    head = MultiTaskHead(
        hidden_dim=768, 
        num_labels_per_task=num_labels, 
        dropout=0.1
    ).to(device, dtype=torch.bfloat16)

    batch_size, seq_len = 64, 512
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = {k: torch.randint(0, v, (batch_size,), device=device) 
              for k, v in num_labels.items()}

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), 
        lr=1e-5
    )

    # Warmup phase (includes compilation)
    print('\n=== WARMUP PHASE (includes compilation) ===', flush=True)
    for i in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = encoder(input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :].clone()
            logits = head(cls)
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
            out = encoder(input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :].clone()
            logits = head(cls)
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    throughput_its = num_iters / elapsed
    throughput_samples = batch_size * num_iters / elapsed
    
    print(f'\n=== RESULTS ===', flush=True)
    print(f'  Batch size: {batch_size} samples x {seq_len} tokens', flush=True)
    print(f'  Total time: {elapsed:.2f}s for {num_iters} iterations', flush=True)
    print(f'  Throughput: {throughput_its:.1f} it/s ({throughput_samples:.0f} samples/s)', flush=True)
    print(f'  Expected training time per epoch: {1787 / throughput_its:.1f}s (1787 batches)', flush=True)
    
    print('\nDone!', flush=True)


if __name__ == '__main__':
    main()
