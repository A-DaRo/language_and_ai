#!/usr/bin/env python3
"""
Profile Manual CUDA Graph Training performance.

Compares:
1. Standard Eager Mode (Baseline)
2. torch.compile with mode="default" (Kernel Fusion only)
3. Manual CUDA Graphs (Full graph capture)
"""

import sys
import time
import torch
import torch.nn as nn
from contextlib import nullcontext
from tqdm import tqdm
from pathlib import Path

# Adjust path to import from src
sys.path.append(str(Path.cwd() / "src"))

from neuro_stylometry.stylometry_net.transformer import AffineGuardTransformer
from neuro_stylometry.stylometry_net.classification_head import MultiTaskHead
from neuro_stylometry.hardware_ops.cuda_graphs import (
    GraphAwareTraining,
    GraphCache,
    GraphCacheConfig,
)

BATCH_SIZE = 64
SEQ_LEN = 512
WARMUP = 1
STEPS = 2

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 768) 
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = input_ids.to(dtype=self.proj.weight.dtype).unsqueeze(-1)
        x = self.proj(x)
        return {"cls_embedding": x.mean(1)}

    def forward_cls(self, input_ids, attention_mask=None):
         x = input_ids.to(dtype=self.proj.weight.dtype).unsqueeze(-1)
         x = self.proj(x)
         return x.mean(1)

class SimpleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.randn(1))
        self.task_order = ("task_1",)
        
    def forward(self, x):
        return self.forward_compiled(x)
        
    def forward_compiled(self, x):
        # x is (B, H)
        return (x,)

    def compute_loss(self, logits, labels):
        return self.compute_loss_compiled(logits, labels)[0]
        
    def compute_loss_compiled(self, logits, labels, ignore_index=-1):
        # logits is (x,)
        # labels is tuple of (B,)
        # Return scalar loss and scalar valid flag
        # Use existing tensor to create valid flag to avoid allocation issues?
        loss = logits[0].sum()
        valid = logits[0].new_ones(())
        return loss, valid

def setup_model():
    model = AffineGuardTransformer(
        model_name="roberta-base"
    ).cuda().to(dtype=torch.bfloat16)
    # model = SimpleModel().cuda().to(dtype=torch.bfloat16)

    # Tiny head for testing
    head = MultiTaskHead(
        hidden_dim=768,
        num_labels_per_task={"task_1": 2, "task_2": 5},
    ).cuda().to(dtype=torch.bfloat16)

    # head = SimpleHead().cuda().to(dtype=torch.bfloat16)
    
    return model, head

def benchmark_eager(input_ids, mask, labels):
    print("\nBenchmarking Eager Mode...")
    model, head = setup_model()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-5)
    
    # Warmup
    for _ in range(WARMUP):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=mask)
            logits = head(outputs["cls_embedding"])
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(STEPS):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=mask)
            logits = head(outputs["cls_embedding"])
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    throughput = STEPS * BATCH_SIZE / elapsed
    print(f"Eager: {throughput:.2f} samples/s ({STEPS/elapsed:.2f} it/s)")
    return throughput

def benchmark_raw_graphs(input_ids, mask, labels):
    print("\nBenchmarking RAW Manual CUDA Graphs (Full RoBERTa)...")
    model, head = setup_model()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-5)
    
    # Capture Stream
    capture_stream = torch.cuda.Stream()
    
    # Input Buffers (clones)
    input_buffers = {
        "input_ids": input_ids.detach().clone(),
        "attention_mask": mask.detach().clone(),
    }
    
# Labels (clones)
    raw_labels = {
        k: v.detach().clone() for k,v in labels.items()
    }
    
    # Sync
    torch.cuda.synchronize()
    
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    
    # Warmup
    print("Warmup...")
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                buf_ids = input_buffers["input_ids"]
                buf_mask = input_buffers["attention_mask"]
                
                # RoBERTa forward
                out = model(input_ids=buf_ids, attention_mask=buf_mask)
                
                # Head forward (MultiTaskHead)
                logits = head(out["cls_embedding"])
                
                # Compute Loss
                # Use raw_labels dict directly (fine inside graph if tensors are captured)
                loss = head.compute_loss(logits, raw_labels)
                loss.backward()
                
    torch.cuda.synchronize()
    
    print("Capturing...")
    optimizer.zero_grad(set_to_none=False)
    
    with torch.cuda.graph(g, stream=capture_stream, pool=pool):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            buf_ids = input_buffers["input_ids"]
            buf_mask = input_buffers["attention_mask"]
            
            out = model(input_ids=buf_ids, attention_mask=buf_mask)
            logits = head(out["cls_embedding"])
            loss = head.compute_loss(logits, raw_labels)
            loss.backward()

    print("RAW RoBERTa Capture Successful!")
    return 0

def benchmark_compile(input_ids, mask, labels):
    print("\nBenchmarking torch.compile(mode='default')...")
    
    # Reset model
    model, head = setup_model()
    model = torch.compile(model, mode="default")
    # head needs to compile submethods or entire module? 
    # Usually compile module works.
    head = torch.compile(head, mode="default")
    
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-5)
    
    # Warmup
    print("Warmup...")
    for _ in range(WARMUP):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=mask)
            logits = head(outputs["cls_embedding"])
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in tqdm(range(STEPS), desc="Compiled"):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=mask)
            logits = head(outputs["cls_embedding"])
            loss = head.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    throughput = STEPS * BATCH_SIZE / elapsed
    print(f"Compiled: {throughput:.2f} samples/s ({STEPS/elapsed:.2f} it/s)")
    return throughput

def benchmark_manual_graphs(input_ids, mask, labels):
    print("\nBenchmarking Manual CUDA Graphs...")
    
    # Reset model
    model, head = setup_model()
    
    # Setup Graph Trainer
    config = GraphCacheConfig(
        enabled=True,
        warmup_iterations=3,
        max_cached_graphs=16,
        use_cuda_graph_memory_pool=True
    )
    cache = GraphCache(config)
    trainer = GraphAwareTraining(cache, pad_token_id=1, ignore_index=-1)
    
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-5)
    
    # Prepare tuple labels (sorted keys)
    labels_tuple = tuple(labels[k] for k in sorted(labels.keys()))
    head.task_order = tuple(sorted(labels.keys())) # Ensure head order matches
    
    # Warmup/Capture
    print("Warmup/Capture...")
    for _ in range(5):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _, _ = trainer.run(
                model=model, head=head,
                input_ids=input_ids, attention_mask=mask,
                labels=labels_tuple, accum_steps=1
            )
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(STEPS):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _, _ = trainer.run(
                model=model, head=head,
                input_ids=input_ids, attention_mask=mask,
                labels=labels_tuple, accum_steps=1
            )
        optimizer.step()
        optimizer.zero_grad()
        
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    throughput = STEPS * BATCH_SIZE / elapsed
    print(f"Manual Graphs: {throughput:.2f} samples/s ({STEPS/elapsed:.2f} it/s)")
    return throughput

def main():
    torch.set_float32_matmul_precision("high")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        sys.exit(1)
        
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Seq Len: {SEQ_LEN}")
    
    input_ids = torch.randint(0, 50000, (BATCH_SIZE, SEQ_LEN), device="cuda")
    mask = torch.ones((BATCH_SIZE, SEQ_LEN), device="cuda")
    labels = {
        "task_1": torch.randint(0, 2, (BATCH_SIZE,), device="cuda"),
        "task_2": torch.randint(0, 5, (BATCH_SIZE,), device="cuda")
    }
    
    t_eager = benchmark_eager(input_ids, mask, labels)
    
    # try:
    #     t_compile = benchmark_compile(input_ids, mask, labels)
    # except BaseException as e:
    #     print(f"Compile failed: {e}")
    #     t_compile = 0.0
    t_compile = 0.0
    
    benchmark_raw_graphs(input_ids, mask, labels)
    return
        
    t_manual = benchmark_manual_graphs(input_ids, mask, labels)
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Eager Mode:    {t_eager:.2f} samples/s")
    print(f"torch.compile: {t_compile:.2f} samples/s")
    print(f"Manual Graphs: {t_manual:.2f} samples/s")
    
    if t_eager > 0:
        print(f"Speedup vs Eager:    {t_manual/t_eager:.2f}x")
    if t_compile > 0:
        print(f"Speedup vs Compiled: {t_manual/t_compile:.2f}x")

if __name__ == "__main__":
    main()
