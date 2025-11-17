# SLURM Job Scripts for Notion Scraper

## Overview

This document explains the optimized SLURM job scripts for the Notion scraper application, which uses **puppeteer-cluster** for parallel web scraping with dynamic concurrency based on available system memory.

## Key Optimizations

### 1. **Single Task with Multiple CPUs**
```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # or 32 for production
```

**Why?** The original script used `--ntasks=16`, which allocates 16 separate MPI-style tasks. However, this Node.js application is **not an MPI application**—it's a single-process application that spawns multiple threads/workers internally via puppeteer-cluster. Using multiple tasks wastes resources.

### 2. **Explicit Memory Allocation**
```bash
#SBATCH --mem=32G  # or 64G for production
```

**Why?** The application dynamically calculates concurrency using this formula:
```javascript
const BYTES_PER_INSTANCE = 1 * 1024 * 1024 * 1024;  // 1GB per browser instance
const OS_RAM_BUFFER = 2 * 1024 * 1024 * 1024;       // 2GB buffer for OS
const freeMemory = os.freemem();
const availableMemoryForWorkers = Math.max(0, freeMemory - OS_RAM_BUFFER);
const maxConcurrency = Math.floor(availableMemoryForWorkers / BYTES_PER_INSTANCE);
```

With explicit memory allocation:
- **32GB allocation** → ~30 concurrent browser instances
- **64GB allocation** → ~62 concurrent browser instances

Without explicit allocation, the job might get unpredictable memory limits.

### 3. **Node.js Memory Limits**
```bash
export NODE_OPTIONS="--max-old-space-size=28672"  # ~28GB in MB
```

**Why?** Node.js defaults to ~2-4GB heap size. This increases it to match the allocated memory, preventing out-of-memory crashes when managing many browser instances.

### 4. **UV Thread Pool Size**
```bash
export UV_THREADPOOL_SIZE=$SLURM_CPUS_PER_TASK
```

**Why?** Node.js uses libuv for async I/O operations. The default thread pool size is 4, which becomes a bottleneck with many concurrent operations. Setting it to the CPU count (16 or 32) allows parallel file I/O and DNS operations.

### 5. **Proper Logging**
```bash
#SBATCH --output=notion-scraper-%j.out
#SBATCH --error=notion-scraper-%j.err
```

**Why?** Separates stdout and stderr with job ID in filename for easier debugging.

## Available Scripts

### `job.sh` - Dry Run (Testing)
- **Resources:** 16 CPUs, 32GB RAM, 30 minutes
- **Purpose:** Plan discovery phase only (`--dry-run`)
- **Expected concurrency:** ~30 parallel browser instances
- **Use for:** Testing site structure before full scraping

**Submit:**
```bash
sbatch job.sh
```

### `job-production.sh` - Full Production Run
- **Resources:** 32 CPUs, 64GB RAM, 2 hours
- **Purpose:** Complete scraping with auto-confirm (`--yes`)
- **Expected concurrency:** ~62 parallel browser instances
- **Use for:** Unattended full site scraping

**Submit:**
```bash
sbatch job-production.sh
```

## Performance Comparison

| Configuration | CPUs | Memory | Expected Concurrency | Discovery Speed |
|--------------|------|--------|---------------------|----------------|
| Original (incorrect) | 16 tasks × 1 CPU | Default (~2GB) | 1-2 instances | Very slow |
| Optimized (dry-run) | 1 task × 16 CPUs | 32GB | ~30 instances | **15-30x faster** |
| Optimized (production) | 1 task × 32 CPUs | 64GB | ~62 instances | **30-60x faster** |

## How the Application Uses Resources

1. **Discovery Phase** (parallel via puppeteer-cluster):
   - Spawns `maxConcurrency` browser instances
   - Each instance navigates to a page, extracts links, and closes
   - Level-by-level traversal (breadth-first search)
   - **Bottleneck:** Memory (each browser instance ~1GB)

2. **Execution Phase** (sequential):
   - Single browser instance scrapes each page
   - Downloads assets, rewrites links, saves HTML
   - **Bottleneck:** Network I/O and disk I/O

3. **Resource Utilization:**
   - **Discovery Phase:** High CPU + High Memory (parallel browsers)
   - **Execution Phase:** Low CPU + High I/O (single browser, many downloads)

## Monitoring Your Job

Check job status:
```bash
squeue -u $USER
```

View live output:
```bash
tail -f notion-scraper-<jobid>.out
```

Check memory usage:
```bash
sstat --format=JobID,MaxRSS,AveCPU -j <jobid>
```

## Troubleshooting

### Issue: "Out of memory" errors
**Solution:** Increase `--mem` allocation or reduce concurrency manually:
```javascript
// In Config.js, add:
this.FORCE_MAX_CONCURRENCY = 20;  // Override auto-calculation
```

### Issue: "Too many open files" errors
**Solution:** Increase file descriptor limit:
```bash
ulimit -n 65536
```

### Issue: Cluster workers timing out
**Solution:** Increase timeout in `Config.js`:
```javascript
this.CLUSTER_TASK_TIMEOUT = 180000;  // 3 minutes instead of 90s
```

## Advanced: Custom Concurrency

To manually override the auto-calculated concurrency, set an environment variable:
```bash
export MAX_CONCURRENCY=25
node main.js --dry-run
```

Then modify `NotionScraper.js`:
```javascript
_calculateClusterConcurrency() {
  // Check for manual override
  const envConcurrency = parseInt(process.env.MAX_CONCURRENCY, 10);
  if (envConcurrency && envConcurrency > 0) {
    return { maxConcurrency: envConcurrency, freeMemory: os.freemem() };
  }
  
  // Original auto-calculation
  const BYTES_PER_INSTANCE = 1 * 1024 * 1024 * 1024;
  // ... rest of function
}
```

## Summary

The key insight is that this Node.js application with puppeteer-cluster is **not an MPI application** and doesn't benefit from `--ntasks > 1`. Instead, it needs:

1. **One task** with many CPUs (`--cpus-per-task`)
2. **Large memory allocation** (`--mem`) for many parallel browser instances
3. **Proper Node.js configuration** (`NODE_OPTIONS`, `UV_THREADPOOL_SIZE`)
4. **Adequate runtime** for large site scraping

This approach maximizes the application's built-in parallelization capabilities while efficiently using SLURM resources.
