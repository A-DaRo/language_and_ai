#!/usr/bin/env python3
"""Diagnose Phase D training throughput bottlenecks."""

import sys
import time
import torch
import yaml


def main():
    print("=" * 60)
    print("PHASE D THROUGHPUT DIAGNOSTICS")
    print("=" * 60)

    # 1. CUDA Check
    print("\n[1] CUDA STATUS")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Device: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"    SM count: {props.multi_processor_count}")
    else:
        print("    WARNING: Running on CPU - expect slow performance!")
        
    # 2. Config Check
    print("\n[2] HPC CONFIG")
    try:
        with open("conf/hpc/phase_d.yaml") as f:
            cfg = yaml.safe_load(f)
        opt = cfg.get("optimization", {})
        exec_cfg = cfg.get("execution", {})
        
        print(f"    torch_compile: {opt.get('use_torch_compile')}")
        print(f"    compile_mode: {opt.get('torch_compile_mode')}")
        print(f"    disable_cudagraphs: {opt.get('torch_compile_disable_cudagraphs')}")
        print(f"    device_prefetch: {opt.get('use_device_prefetch')}")
        print(f"    fused_optimizer: {opt.get('use_fused_optimizer')}")
        print(f"    telemetry_stride: {exec_cfg.get('telemetry', {}).get('stride')}")
    except Exception as e:
        print(f"    Error loading config: {e}")

    # 3. Raw Model Throughput
    print("\n[3] RAW MODEL THROUGHPUT (no training overhead)")
    if torch.cuda.is_available():
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained("roberta-base", attn_implementation="sdpa")
        model = model.cuda().eval()
        
        # Test different batch sizes
        for bs in [8, 16, 32, 64]:
            x = torch.randint(0, 1000, (bs, 512), device="cuda")
            mask = torch.ones_like(x)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(x, attention_mask=mask)
                torch.cuda.synchronize()
            
            # Measure
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(20):
                    _ = model(x, attention_mask=mask)
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            samples_per_sec = 20 * bs / elapsed
            print(f"    bs={bs:3d}: {samples_per_sec:7.1f} samples/s ({20/elapsed:.1f} it/s)")
        
        del model
        torch.cuda.empty_cache()

    # 4. torch.compile overhead
    print("\n[4] TORCH.COMPILE OVERHEAD")
    if torch.cuda.is_available():
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained("roberta-base", attn_implementation="sdpa")
        model = model.cuda().eval()
        
        x = torch.randint(0, 1000, (32, 512), device="cuda")
        mask = torch.ones_like(x)
        
        # Without compile
        with torch.no_grad():
            for _ in range(3):
                _ = model(x, attention_mask=mask)
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                _ = model(x, attention_mask=mask)
            torch.cuda.synchronize()
        elapsed_eager = time.perf_counter() - start
        print(f"    Eager mode: {20/elapsed_eager:.1f} it/s")
        
        # With compile (default mode)
        model_compiled = torch.compile(model, mode="default")
        
        # Warmup (includes compilation)
        print("    Compiling (this takes a moment)...")
        with torch.no_grad():
            for _ in range(3):
                _ = model_compiled(x, attention_mask=mask)
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                _ = model_compiled(x, attention_mask=mask)
            torch.cuda.synchronize()
        elapsed_compiled = time.perf_counter() - start
        print(f"    Compiled (default): {20/elapsed_compiled:.1f} it/s")
        print(f"    Speedup: {elapsed_eager/elapsed_compiled:.2f}x")
        
        del model, model_compiled
        torch.cuda.empty_cache()

    # 5. DataLoader check
    print("\n[5] DATALOADER / COLLATION CHECK")
    try:
        import pyarrow as pa
        
        # Check if tokenized dataset exists
        ds_path = "artifacts/phase_d/tokenized_dataset.arrow"
        try:
            table = pa.ipc.open_file(ds_path).read_all()
            print(f"    Dataset: {ds_path}")
            print(f"    Rows: {table.num_rows}")
            print(f"    Columns: {table.column_names}")
            
            # Check if pre-tokenized
            has_input_ids = "input_ids" in table.column_names
            print(f"    Pre-tokenized (AOT): {has_input_ids}")
            
            if has_input_ids:
                # Sample a row to check format
                sample = table.column("input_ids")[0].as_py()
                print(f"    Sample input_ids len: {len(sample)}")
        except FileNotFoundError:
            print(f"    Dataset not found: {ds_path}")
    except Exception as e:
        print(f"    Error: {e}")

    # 6. Training step simulation
    print("\n[6] FULL TRAINING STEP SIMULATION")
    if torch.cuda.is_available():
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained("roberta-base", attn_implementation="sdpa")
        model = model.cuda().train()
        model = torch.compile(model, mode="default")
        
        # Simple head
        head = torch.nn.Linear(768, 10).cuda()
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=1e-5,
            fused=True,
        )
        
        x = torch.randint(0, 1000, (32, 512), device="cuda")
        mask = torch.ones_like(x)
        labels = torch.randint(0, 10, (32,), device="cuda")
        
        # Warmup
        print("    Warming up...")
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, attention_mask=mask)
            logits = head(out.last_hidden_state[:, 0])
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        
        # Measure
        num_iters = 50
        start = time.perf_counter()
        for _ in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, attention_mask=mask)
            logits = head(out.last_hidden_state[:, 0])
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        print(f"    {num_iters} training steps: {elapsed:.2f}s")
        print(f"    Throughput: {num_iters/elapsed:.1f} it/s, {num_iters*32/elapsed:.0f} samples/s")
        
        # Now test with GPU sync every step (simulating the bug)
        print("\n    With GPU sync every step (BAD):")
        start = time.perf_counter()
        for _ in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            out = model(x, attention_mask=mask)
            logits = head(out.last_hidden_state[:, 0])
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            _ = loss.item()  # <-- THIS KILLS THROUGHPUT
        torch.cuda.synchronize()
        elapsed_sync = time.perf_counter() - start
        
        print(f"    {num_iters} training steps: {elapsed_sync:.2f}s")
        print(f"    Throughput: {num_iters/elapsed_sync:.1f} it/s")
        print(f"    Slowdown from sync: {elapsed_sync/elapsed:.2f}x")

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
