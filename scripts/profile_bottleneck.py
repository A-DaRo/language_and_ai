#!/usr/bin/env python3
"""Profile training bottleneck: compare raw model throughput vs full training loop."""

import torch
import time
from transformers import AutoModel
from neuro_stylometry.stylometry_net.classification_head import MultiTaskHead
from tqdm import tqdm

def main():
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Loading model...')
    encoder = AutoModel.from_pretrained(
        'distilbert-base-uncased', 
        torch_dtype=torch.bfloat16, 
        attn_implementation='sdpa'
    ).to(device)

    num_labels = {
        'author': 24, 'stance': 2, 'topic': 5, 'region': 4, 
        'age': 6, 'gender': 3, 'education': 5, 'income': 5
    }
    head = MultiTaskHead(
        hidden_dim=768, 
        num_labels_per_task=num_labels, 
        dropout=0.1
    ).to(device, dtype=torch.bfloat16)

    batch_size, seq_len = 64, 512
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = {k: torch.randint(0, v, (batch_size,), device=device) for k, v in num_labels.items()}

    print('Warmup...')
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), 
        lr=1e-5
    )
    # Note: GradScaler not needed for bf16 on modern GPUs
    
    for _ in range(5):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = encoder(input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :].clone()
            logits = head(cls)
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Test 1: Raw forward+backward only
    print('\n[1] Raw forward+backward (no optimizer step)...')
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
        encoder.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f'  Throughput: {num_iters / elapsed:.1f} it/s ({batch_size * num_iters / elapsed:.0f} samples/s)')

    # Test 2: With optimizer.step()
    print('\n[2] With optimizer.step()...')
    torch.cuda.synchronize()
    start = time.perf_counter()
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
    print(f'  Throughput: {num_iters / elapsed:.1f} it/s ({batch_size * num_iters / elapsed:.0f} samples/s)')

    # Test 3: Simulating DataLoader overhead - creating new tensors each iter
    print('\n[3] Creating new tensors each iteration (simulating DataLoader)...')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        # Simulate DataLoader producing new tensors
        input_ids_batch = torch.randint(0, 30522, (batch_size, seq_len), device=device)
        attention_mask_batch = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        labels_batch = {k: torch.randint(0, v, (batch_size,), device=device) for k, v in num_labels.items()}
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = encoder(input_ids_batch, attention_mask=attention_mask_batch)
            cls = out.last_hidden_state[:, 0, :].clone()
            logits = head(cls)
            loss = head.compute_loss(logits, labels_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f'  Throughput: {num_iters / elapsed:.1f} it/s ({batch_size * num_iters / elapsed:.0f} samples/s)')

    # Test 4: With tqdm updates
    print('\n[4] With tqdm progress bar (update every iter)...')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in tqdm(range(num_iters), ncols=80):
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
    print(f'  Throughput: {num_iters / elapsed:.1f} it/s ({batch_size * num_iters / elapsed:.0f} samples/s)')

    print('\nDone!')

if __name__ == '__main__':
    main()
