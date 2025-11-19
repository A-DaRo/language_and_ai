# Micro-Kernel Architecture - Usage Guide

## Overview

The Notion scraper now supports two execution modes:

1. **Legacy Mode** (`main.js`) - Original monolithic implementation
2. **Cluster Mode** (`main-cluster.js`) - New distributed micro-kernel architecture

## Cluster Mode (Recommended)

### Quick Start

```bash
# Run with default settings
npm run start:cluster

# Run with custom depth
node main-cluster.js --max-depth 3

# Show help
node main-cluster.js --help
```

### How It Works

The cluster mode uses a distributed architecture with multiple worker processes:

```
┌─────────────────────────────────────────────────────────┐
│                    Master Process                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │         ClusterOrchestrator (State Machine)      │  │
│  │  ┌──────────┐  ┌─────────────┐  ┌─────────────┐ │  │
│  │  │Bootstrap │→ │ Discovery   │→ │ Conflict    │ │  │
│  │  └──────────┘  └─────────────┘  │ Resolution  │ │  │
│  │                                  └─────────────┘ │  │
│  │  ┌──────────┐  ┌─────────────┐                  │  │
│  │  │Download  │→ │ Complete    │                  │  │
│  │  └──────────┘  └─────────────┘                  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ BrowserManager   │  │ GlobalQueueManager       │   │
│  │ - Worker Pool    │  │ - Discovery Queue        │   │
│  │ - Task Allocation│  │ - Download Queue         │   │
│  └──────────────────┘  └──────────────────────────┘   │
│            │                                             │
│            │ IPC (process.send/on('message'))           │
└────────────┼─────────────────────────────────────────────┘
             │
    ┌────────┴─────────┬───────────┬──────────┐
    ▼                  ▼           ▼          ▼
┌─────────┐       ┌─────────┐ ┌─────────┐ ┌─────────┐
│Worker 1 │       │Worker 2 │ │Worker 3 │ │Worker N │
│Puppeteer│       │Puppeteer│ │Puppeteer│ │Puppeteer│
└─────────┘       └─────────┘ └─────────┘ └─────────┘
```

### Workflow Phases

#### Phase 1: Bootstrap
- Spawns initial worker for cookie capture
- Prevents race condition with authentication
- Spawns remaining workers based on system capacity (~1GB RAM per worker)
- Broadcasts cookies to all workers

#### Phase 2: Discovery
- Parallel discovery of all pages (metadata only)
- Extracts links, titles, and hierarchical structure
- No heavy asset downloads in this phase
- Builds complete PageContext tree

#### Phase 3: Conflict Resolution
- Detects duplicate pages (same URL referenced multiple times)
- Selects canonical version for each unique page
- Calculates target file paths based on hierarchy
- Generates `linkRewriteMap` for download phase

#### Phase 4: Download
- Parallel download of unique pages
- Full scraping with assets (images, CSS, files)
- Link rewriting using the `linkRewriteMap`
- Workers write files directly to disk (no HTML over IPC)

#### Phase 5: Complete
- Statistics reporting
- Resource cleanup
- All workers gracefully terminated

### Performance Characteristics

**Capacity Planning:**
- Each worker requires ~1GB RAM (Puppeteer + Chrome)
- Workers auto-scaled based on available system memory
- CPU-bound: Won't exceed CPU core count - 1
- Default limits: 2-8 workers

**Advantages over Legacy Mode:**
- **Scalability**: Multiple workers process pages in parallel
- **Fault Tolerance**: Worker crashes don't affect Master or other workers
- **Resource Efficiency**: Better memory management with isolated processes
- **No puppeteer-cluster**: Uses native Node.js `child_process` for better control

### Architecture Components

#### Master Process
- **ClusterOrchestrator**: Main state machine
- **BrowserManager**: Worker pool management
- **GlobalQueueManager**: Task queue coordination
- **ConflictResolver**: Duplicate detection
- **SystemEventBus**: Event-driven communication

#### Worker Process
- **WorkerEntrypoint**: Isolated process entry point
- **TaskRunner**: IPC command router
- **PageProcessor**: Stateless scraping logic (reuses existing logic)

### Error Handling

- Worker crashes are detected and logged
- Failed tasks are tracked in statistics
- Graceful shutdown on SIGINT/SIGTERM
- All workers terminated before Master exits

### Debugging

```bash
# Enable debug logging (if implemented)
DEBUG=1 node main-cluster.js

# Test worker initialization only
node test-worker.js

# Test cluster layer
node test-cluster.js
```

## Legacy Mode

```bash
# Run legacy mode
npm run start:legacy
# or
node main.js --dry-run --max-depth 3
```

The legacy mode remains available for backwards compatibility and will continue to work as before.

## Migration Notes

### Key Differences

| Feature | Legacy Mode | Cluster Mode |
|---------|-------------|--------------|
| Workers | puppeteer-cluster | native child_process |
| Concurrency | Shared browser contexts | Isolated processes |
| State | In-memory, shared | Distributed via IPC |
| Discovery | Sequential with parallelism | Fully distributed |
| Downloads | Sequential | Fully distributed |
| Memory | Single process | Multi-process |
| Fault Tolerance | Single point of failure | Worker isolation |

### When to Use Each Mode

**Use Cluster Mode when:**
- Scraping large Notion workspaces (>50 pages)
- System has sufficient RAM (4GB+ recommended)
- Want better fault tolerance
- Need maximum performance

**Use Legacy Mode when:**
- System is resource-constrained
- Scraping small workspaces (<20 pages)
- Debugging issues
- Need stable, tested behavior

## Troubleshooting

### Workers Not Spawning
```
Error: Worker did not send READY within 30 seconds
```
**Solution**: Check system resources. Each worker needs ~1GB RAM.

### Out of Memory
```
Error: Cannot allocate memory
```
**Solution**: Reduce worker count or use legacy mode.

### No Cookies Captured
```
Warning: Timeout waiting for cookies, proceeding without authentication
```
**Solution**: Check if Notion page requires authentication. Bootstrap phase may need adjustment.

## Future Enhancements

- [ ] Worker replacement on crash (auto-respawn)
- [ ] Dynamic worker scaling during runtime
- [ ] Retry logic for failed tasks
- [ ] Progress bars and real-time statistics
- [ ] Checkpoint/resume functionality
- [ ] Metrics collection (task duration, memory usage)
- [ ] Web UI for monitoring

## Performance Tips

1. **Adjust max depth**: Start with `--max-depth 2` for large sites
2. **Monitor resources**: Watch memory usage with Task Manager
3. **Network speed**: Faster network = more workers beneficial
4. **SSD vs HDD**: SSD recommended for better I/O performance
