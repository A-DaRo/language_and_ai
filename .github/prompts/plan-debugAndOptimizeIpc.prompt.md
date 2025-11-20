## Plan: Debug Fatal Error and Optimize IPC Performance

**TL;DR**: The fatal error is caused by a missing `getMaxDepth()` method in `GlobalQueueManager` that was referenced but never implemented. Additionally, the current IPC protocol sends large serialized objects (Maps) through message passing, which creates JSON serialization overhead and memory pressure. The plan addresses the immediate bug fix and proposes strategic performance optimizations while maintaining clean separation of concerns.

### Steps

1. **Fix Missing `getMaxDepth()` Method** in [`GlobalQueueManager.js`](src/orchestration/GlobalQueueManager.js)
   - Add method to calculate maximum depth from discovered contexts
   - Return 0 if no contexts exist (edge case)
   - Track max depth incrementally during discovery for O(1) lookup

2. **Analyze Current IPC Overhead** - Performance Bottleneck Assessment
   - Measure serialization cost of `titleRegistry` (Object.fromEntries on every message)
   - Identify message frequency: every DISCOVER/DOWNLOAD task = 24+ serializations
   - Calculate memory duplication: registry sent 24 times vs. sent once
   - Profile actual vs. theoretical performance degradation

3. **Implement Lazy Title Registry Strategy** in [`ClusterOrchestrator.js`](src/orchestration/ClusterOrchestrator.js) and [`WorkerProxy.js`](src/cluster/WorkerProxy.js)
   - Send titleRegistry **once during worker initialization** via `IPC_INIT` payload
   - Workers maintain local copy in `TaskRunner` constructor
   - Send **delta updates** only when new titles discovered (delta = {newId: newTitle})
   - Eliminates redundant full-registry serialization on every task

4. **Replace Map with Efficient Alternatives** for High-Frequency IPC Data
   - Keep `linkRewriteMap` as plain object (already serialized correctly)
   - Remove titleRegistry from `DiscoverPayload`/`DownloadPayload` (use initialization + deltas)
   - Consider protocol buffer alternative for truly massive scale (future optimization)

5. **Validate Separation of Concerns** - Architectural Integrity Check
   - Verify: Master owns state, Workers remain stateless (except cached registry)
   - Ensure: Workers don't persist registry to disk (memory-only cache)
   - Confirm: Worker crash = registry rebuilt from Master on respawn
   - Test: Graceful degradation if registry desync occurs

### Further Considerations

1. **Performance Baseline**: Establish current vs. optimized benchmarks (time 24-page scrape before/after). Expected improvement: 20-30% reduction in IPC overhead for large sites (100+ pages).

2. **Error Handling Enhancement**: Add registry checksum validation in worker to detect desynchronization. Fall back to requesting full registry refresh if mismatch detected.
