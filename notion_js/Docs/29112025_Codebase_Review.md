# Notion Scraper - Comprehensive Codebase Review
## 29 November 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Issue Analysis: Worker Deadlock in Hidden File Downloads](#2-issue-analysis-worker-deadlock-in-hidden-file-downloads)
3. [Issue Analysis: Path Resolution Architecture](#3-issue-analysis-path-resolution-architecture)
4. [Proposed Refactoring: Path Strategy Pattern](#4-proposed-refactoring-path-strategy-pattern)
5. [Issue Analysis: ContentExpander Limitations](#5-issue-analysis-contentexpander-limitations)
6. [Proposed Design: HtmlFacade Interface](#6-proposed-design-htmlfacade-interface)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Appendix: Code References](#8-appendix-code-references)

---

## 1. Executive Summary

This document presents a comprehensive technical review of the Notion Recursive Scraper codebase, addressing critical issues identified during the run on 29 November 2025. The analysis reveals three interconnected architectural concerns:

1. **Worker Deadlock Scenario**: A resource starvation condition caused by blocking file download operations within worker processes.

2. **Path Resolution Fragmentation**: Insufficient separation between intra-page (anchor) and inter-page (navigation) path resolution strategies.

3. **Content Expansion Gaps**: Incomplete handling of interactive toggle elements, leading to content loss in offline replicas.

Each issue is analyzed with full codebase context, and concrete refactoring proposals are provided with UML diagrams, code examples, and integration strategies.

### Key Solutions Proposed

| Issue | Primary Solution | Approach |
|-------|------------------|----------|
| Worker Deadlock | Leaf-First Download + Global Hidden File Registry | Reorder download queue to process leaf pages first; deduplicate hidden files across all pages |
| Path Resolution | Path Strategy Pattern | Separate IntraPathStrategy (anchors) and InterPathStrategy (navigation) |
| Content Expansion | Toggle State Capture | Dual-state HTML capture with offline JavaScript controller |
| HTML Manipulation | HtmlFacade Interface | Unified abstraction over Puppeteer and JSDOM operations |

### Incident Context

The following error was observed during a production run:

```
5:37:57 PM [ERROR] [ORCHESTRATOR] Fatal error during orchestration
  └─ {
       "message": "Timeout waiting for available worker",
       "stack": "Error: Timeout waiting for available worker\n    at BrowserManager._allocateWorker..."
     }
```

The system crashed after 3 workers became indefinitely blocked while processing hidden file downloads on the "Material" page, which contained multiple PDF attachments with `display: contents` styling.

---

## 2. Issue Analysis: Worker Deadlock in Hidden File Downloads

### 2.1 Problem Statement

The system experiences a **deadlock condition** when all available workers become blocked on synchronous wait operations during hidden file processing. This manifests as a `Timeout waiting for available worker` error when the Master process attempts to allocate workers for new tasks.

### 2.2 Root Cause Analysis

#### 2.2.1 The Hidden File Processing Pipeline

The `FileDownloader._processHiddenFiles()` method implements a **blocking click-and-wait strategy** for extracting file URLs from interactive Notion elements:

```javascript
// FileDownloader.js:218-230
page.browser().on('targetcreated', targetHandler);
await page.setRequestInterception(true);
page.on('request', requestHandler);

try {
    const clickSuccess = await this._clickElementOrParent(page, element);
    if (!clickSuccess) continue;
    
    // CRITICAL: 3-second synchronous wait per element
    await new Promise(resolve => setTimeout(resolve, 3000));
} finally {
    page.browser().off('targetcreated', targetHandler);
    page.off('request', requestHandler);
    await page.setRequestInterception(false);
}
```

**Key Observations:**

1. **Per-element blocking wait**: Each hidden file candidate triggers a **3-second synchronous wait** to capture download URL reactions.

2. **Sequential processing**: Elements are processed one-at-a-time within a single worker's task execution.

3. **No early termination**: The wait continues even if a URL is captured immediately.

#### 2.2.2 Worker State Machine Failure

The worker state lifecycle follows this sequence:

```
IDLE → BUSY → (task completion) → IDLE
```

However, when `_processHiddenFiles()` encounters multiple elements (the log shows **118 potential hidden file elements**), the worker remains in `BUSY` state for:

```
118 elements × 3 seconds = 354 seconds (5.9 minutes) minimum
```

During this time, the worker **cannot be released** back to the idle pool, even though it's primarily waiting rather than performing computation.

#### 2.2.3 The Deadlock Cascade

With 3 workers in the pool:

1. **Worker-1** starts processing "Material" page (118 hidden files)
2. **Worker-2** starts processing another page (43 hidden files)  
3. **Worker-3** starts processing another page
4. All workers become blocked on hidden file waits
5. Master's `_allocateWorker()` enters timeout loop:

```javascript
// BrowserManager.js:155-167
async _allocateWorker() {
    const timeout = 60000; // 60 seconds
    const startTime = Date.now();
    
    while (this.idleWorkers.length === 0) {
        if (Date.now() - startTime > timeout) {
            throw new Error('Timeout waiting for available worker');
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    // ...
}
```

The 60-second timeout is insufficient for workers processing 100+ hidden file candidates.

### 2.3 Architectural Impact Assessment

| Component | Impact | Severity |
|-----------|--------|----------|
| BrowserManager | Worker pool exhaustion | Critical |
| DownloadPhase | Task backlog accumulation | High |
| FileDownloader | Blocking execution path | High |
| SystemEventBus | Stalled event propagation | Medium |

### 2.4 Proposed Solutions

#### Solution A: Asynchronous Download Queue (Recommended)

Decouple hidden file detection from URL capture by introducing an **asynchronous download queue**:

```javascript
class HiddenFileDownloadQueue {
    constructor(logger) {
        this.pending = [];
        this.completed = new Map();
        this.maxConcurrent = 5;
        this.activeCount = 0;
    }
    
    /**
     * Queue a hidden file for background processing
     * Returns immediately, allowing worker to continue
     */
    enqueue(element, page, filesDir) {
        this.pending.push({ element, page, filesDir, status: 'pending' });
        this._processNext();
    }
    
    async _processNext() {
        if (this.activeCount >= this.maxConcurrent) return;
        
        const item = this.pending.shift();
        if (!item) return;
        
        this.activeCount++;
        try {
            const url = await this._captureUrl(item.element, item.page);
            if (url) {
                this.completed.set(url, item.filesDir);
            }
        } finally {
            this.activeCount--;
            this._processNext();
        }
    }
}
```

**Benefits:**
- Worker releases immediately after queuing candidates
- Background processing doesn't block main task pipeline
- Natural rate limiting prevents resource exhaustion

#### Solution B: Adaptive Wait Strategy

Replace fixed 3-second wait with event-driven completion detection:

```javascript
async _captureUrlWithEarlyExit(element, page, timeout = 5000) {
    return new Promise((resolve, reject) => {
        let downloadUrl = null;
        let timeoutId;
        
        const targetHandler = async (target) => {
            // ... capture logic
            downloadUrl = newPage.url();
            clearTimeout(timeoutId);
            resolve(downloadUrl);
        };
        
        const requestHandler = (request) => {
            if (this.typeDetector.isDownloadableFile(request.url())) {
                downloadUrl = request.url();
                clearTimeout(timeoutId);
                request.abort();
                resolve(downloadUrl);
            } else {
                request.continue();
            }
        };
        
        // Timeout fallback
        timeoutId = setTimeout(() => {
            resolve(downloadUrl); // Return whatever we captured (or null)
        }, timeout);
        
        // Setup listeners and click...
    });
}
```

**Benefits:**
- Exits immediately upon successful capture
- Configurable timeout per-element
- Maintains fallback behavior for slow responses

#### Solution C: Worker Pool Elasticity

Implement dynamic worker scaling based on load:

```javascript
// BrowserManager.js - Enhanced allocation
async _allocateWorker() {
    const timeout = 60000;
    const startTime = Date.now();
    
    while (this.idleWorkers.length === 0) {
        if (Date.now() - startTime > timeout) {
            // Instead of throwing, attempt to spawn emergency worker
            if (this.workers.size < this.maxWorkers) {
                const emergencyWorker = await this._spawnEmergencyWorker();
                if (emergencyWorker) {
                    return emergencyWorker.workerId;
                }
            }
            throw new Error('Timeout waiting for available worker');
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    return this.idleWorkers.pop();
}
```

#### Solution D: Leaf-First Download Strategy (Recommended)

Reorder the download queue to process **leaf pages first**, minimizing hidden file explosion through natural deduplication.

**Insight**: Pages that have no children in the discovered tree ("leaves") are exactly those whose outgoing links create **BACK edges** in the PageGraph. If we download leaves first, any hidden files they contain are registered globally before their parent pages are processed.

```javascript
// ExecutionQueue.js - Enhanced with leaf-first ordering
class ExecutionQueue {
    /**
     * Build queue with leaf-first (reverse BFS) ordering.
     * @param {Array<PageContext>} contexts - Discovered pages
     * @param {PageGraph} pageGraph - Graph with edge classifications
     */
    build(contexts, pageGraph) {
        this.queue = [];
        this.pendingDownloads.clear();
        this.completedDownloads.clear();
        
        // Sort by depth descending (deepest/leaves first)
        const sortedContexts = [...contexts].sort((a, b) => {
            // Primary: depth descending (leaves first)
            if (b.depth !== a.depth) {
                return b.depth - a.depth;
            }
            // Secondary: pages with fewer children first
            return a.children.length - b.children.length;
        });
        
        for (const context of sortedContexts) {
            this.queue.push(context);
            this.pendingDownloads.set(context.id, {
                context,
                childrenCount: context.children.length,
                completedChildren: 0
            });
        }
        
        this.eventBus.emit('QUEUE:DOWNLOAD_READY', {
            count: this.queue.length,
            strategy: 'leaf-first'
        });
    }
    
    /**
     * Alternative: Topological sort using PageGraph edges.
     * Ensures children always processed before parents.
     */
    buildTopological(contexts, pageGraph) {
        const inDegree = new Map();
        const adjacency = new Map();
        
        // Build reverse adjacency (child → parent)
        for (const ctx of contexts) {
            inDegree.set(ctx.id, 0);
            adjacency.set(ctx.id, []);
        }
        
        for (const ctx of contexts) {
            if (ctx.parentId && inDegree.has(ctx.parentId)) {
                adjacency.get(ctx.id).push(ctx.parentId);
                inDegree.set(ctx.parentId, inDegree.get(ctx.parentId) + 1);
            }
        }
        
        // Kahn's algorithm - nodes with in-degree 0 are leaves
        const queue = [];
        for (const [id, degree] of inDegree) {
            if (degree === 0) {
                queue.push(contexts.find(c => c.id === id));
            }
        }
        
        this.queue = [];
        while (queue.length > 0) {
            const ctx = queue.shift();
            this.queue.push(ctx);
            
            for (const parentId of adjacency.get(ctx.id)) {
                inDegree.set(parentId, inDegree.get(parentId) - 1);
                if (inDegree.get(parentId) === 0) {
                    queue.push(contexts.find(c => c.id === parentId));
                }
            }
        }
    }
}
```

**Benefits:**
- Leaf pages have fewer hidden files (typically none, as they're end nodes)
- Hidden files discovered on leaves are deduplicated before parent processing
- No code changes to FileDownloader required - just queue ordering
- Works with existing PageGraph edge classification infrastructure

**Integration with PageGraph:**

```javascript
// GlobalQueueManager.js - Enhanced buildDownloadQueue
buildDownloadQueue(contexts) {
    // Pass the pageGraph for edge-aware ordering
    this.executionQueue.build(contexts, this.pageGraph);
}
```

#### Solution E: Global Hidden File Registry

Introduce a **global deduplication registry** that spans all pages, ensuring each hidden file URL is processed exactly once across the entire scraping session.

```javascript
/**
 * @class GlobalHiddenFileRegistry
 * @description Cross-page deduplication for hidden file downloads.
 * Shared across all worker tasks to prevent duplicate processing.
 */
class GlobalHiddenFileRegistry {
    constructor() {
        // URL → { status, savedPath, discoveredOn }
        this.registry = new Map();
        this.pending = new Set(); // URLs currently being processed
    }
    
    /**
     * Check if URL should be processed.
     * @param {string} url - Hidden file URL
     * @returns {boolean} True if not yet seen or pending
     */
    shouldProcess(url) {
        return !this.registry.has(url) && !this.pending.has(url);
    }
    
    /**
     * Mark URL as being processed.
     * @param {string} url - Hidden file URL
     * @param {string} pageId - Page that discovered this URL
     */
    markPending(url, pageId) {
        this.pending.add(url);
    }
    
    /**
     * Record successful download.
     * @param {string} url - Hidden file URL
     * @param {string} savedPath - Local path where file was saved
     * @param {string} pageId - Page that processed this URL
     */
    recordDownload(url, savedPath, pageId) {
        this.pending.delete(url);
        this.registry.set(url, {
            status: 'downloaded',
            savedPath,
            discoveredOn: pageId,
            downloadedAt: Date.now()
        });
    }
    
    /**
     * Get saved path for URL if already downloaded.
     * @param {string} url - Hidden file URL
     * @returns {string|null} Saved path or null
     */
    getSavedPath(url) {
        const entry = this.registry.get(url);
        return entry?.status === 'downloaded' ? entry.savedPath : null;
    }
    
    /**
     * Get statistics.
     */
    getStats() {
        return {
            totalSeen: this.registry.size,
            pending: this.pending.size,
            downloaded: Array.from(this.registry.values())
                .filter(e => e.status === 'downloaded').length
        };
    }
}
```

**Integration with FileDownloader:**

```javascript
// FileDownloader.js - Use global registry
async _processHiddenFiles(page, filesDir, globalRegistry) {
    const hiddenElements = await page.$$('div[data-popup-origin="true"]');
    
    for (const element of hiddenElements) {
        const downloadUrl = await this._captureUrlWithEarlyExit(element, page);
        
        if (!downloadUrl) continue;
        
        // Check global registry first
        const existingPath = globalRegistry.getSavedPath(downloadUrl);
        if (existingPath) {
            this.logger.debug('HIDDEN-FILE', 
                `Skipping duplicate: ${downloadUrl} (already at ${existingPath})`);
            // Reuse existing path for link rewriting
            this.downloadStrategy.recordDownload(downloadUrl, existingPath);
            continue;
        }
        
        // Check if another worker is processing this URL
        if (!globalRegistry.shouldProcess(downloadUrl)) {
            this.logger.debug('HIDDEN-FILE',
                `Skipping in-progress: ${downloadUrl}`);
            continue;
        }
        
        // Mark as pending and process
        globalRegistry.markPending(downloadUrl, this.currentPageId);
        
        const savedPath = await this._downloadFile(downloadUrl, filesDir);
        if (savedPath) {
            globalRegistry.recordDownload(downloadUrl, savedPath, this.currentPageId);
            this.downloadStrategy.recordDownload(downloadUrl, savedPath);
        }
    }
}
```

**Master-Worker IPC for Registry Synchronization:**

```javascript
// Protocol extension for hidden file registry
const HiddenFileProtocol = {
    QUERY_HIDDEN_FILE: 'hidden-file:query',
    HIDDEN_FILE_STATUS: 'hidden-file:status',
    REGISTER_HIDDEN_FILE: 'hidden-file:register',
    HIDDEN_FILE_REGISTERED: 'hidden-file:registered'
};

// Worker queries master before processing
async queryMasterForHiddenFile(url) {
    return new Promise((resolve) => {
        process.send({
            type: HiddenFileProtocol.QUERY_HIDDEN_FILE,
            url
        });
        
        const handler = (msg) => {
            if (msg.type === HiddenFileProtocol.HIDDEN_FILE_STATUS && 
                msg.url === url) {
                process.off('message', handler);
                resolve(msg);
            }
        };
        process.on('message', handler);
    });
}
```

### 2.5 Recommended Implementation Priority

1. **Immediate**: Implement Solution B (Adaptive Wait) - minimal code change, immediate impact
2. **Short-term**: Implement Solution D (Leaf-First Download) - leverages existing PageGraph, no FileDownloader changes
3. **Medium-term**: Implement Solution E (Global Registry) + Solution A (Async Queue) - proper deduplication architecture
4. **Long-term**: Implement Solution C (Elasticity) - infrastructure enhancement

### 2.6 Deadlock Prevention Summary

| Solution | Implementation Effort | Deadlock Prevention | Hidden File Dedup | Requires IPC Changes |
|----------|----------------------|---------------------|-------------------|---------------------|
| A. Async Queue | Medium | ✅ High | ❌ No | No |
| B. Adaptive Wait | Low | ✅ Medium | ❌ No | No |
| C. Worker Elasticity | High | ✅ Medium | ❌ No | Yes |
| **D. Leaf-First** | **Low** | **✅ High** | **✅ Indirect** | **No** |
| **E. Global Registry** | **Medium** | **✅ Medium** | **✅ Direct** | **Yes** |

The **combination of D + E** provides comprehensive deadlock prevention through both queue ordering and global deduplication.

---

## 3. Issue Analysis: Path Resolution Architecture

### 3.1 Problem Statement

The current path calculation system conflates two distinct concerns:

1. **Inter-page navigation**: Computing relative filesystem paths between different HTML files
2. **Intra-page anchors**: Resolving block ID references within the same page

This conflation leads to:
- Incorrect anchor links when source and target are the same page
- Overly complex logic in `PathCalculator.calculateRelativePathBetween()`
- Difficulty maintaining and extending path resolution strategies

### 3.2 Current Architecture Analysis

#### 3.2.1 PathCalculator Responsibility Overload

The `PathCalculator` class currently handles all path-related operations:

```javascript
// PathCalculator.js
class PathCalculator {
    calculateRelativePath(context) { ... }           // Hierarchy path
    calculateDirectoryPath(baseDir, context) { ... } // Filesystem path
    calculateFilePath(baseDir, context) { ... }      // index.html path
    calculateRelativePathBetween(source, target) { } // Navigation path
}
```

The `calculateRelativePathBetween()` method is particularly problematic:

```javascript
calculateRelativePathBetween(sourceContext, targetContext) {
    const sourcePath = this.calculateRelativePath(sourceContext);
    const targetPath = this.calculateRelativePath(targetContext);
    
    // Complex segment comparison logic
    const sourceSegments = sourcePath ? sourcePath.split(path.posix.sep).filter(s => s) : [];
    const targetSegments = targetPath ? targetPath.split(path.posix.sep).filter(s => s) : [];
    
    // Find common ancestor
    let commonDepth = 0;
    while (commonDepth < sourceSegments.length && 
           commonDepth < targetSegments.length && 
           sourceSegments[commonDepth] === targetSegments[commonDepth]) {
        commonDepth++;
    }
    
    // Calculate up-levels and down-segments
    const upLevels = sourceSegments.length - commonDepth;
    const downSegments = targetSegments.slice(commonDepth);
    
    // Build path with '../' prefixes
    // ...
}
```

**Issues:**

1. No distinction between same-page links (anchors) and cross-page links (navigation)
2. Anchor hash handling is delegated to callers (`LinkRewriter`)
3. Root page edge cases require special handling scattered across methods

#### 3.2.2 LinkRewriter's Compensation Logic

`LinkRewriter` must compensate for `PathCalculator`'s limitations:

```javascript
// LinkRewriter.js:136-155
if (targetContext) {
    let newHref;
    
    // Special case: same-page anchor
    if (targetContext.id === pageContext.id) {
        newHref = this._buildAnchorHash(blockIdRaw, targetContext, blockMapCache);
        if (!newHref) {
            newHref = 'index.html';
        }
    } else {
        // Cross-page navigation
        const relativePath = pageContext.getRelativePathTo(targetContext);
        const hash = this._buildAnchorHash(blockIdRaw, targetContext, blockMapCache);
        newHref = relativePath + hash;
    }
    
    link.setAttribute('href', newHref);
}
```

This demonstrates that `LinkRewriter` already understands the intra/inter distinction—it just can't delegate properly because `PathCalculator` doesn't support it.

### 3.3 Observed Symptoms

From production logs and code analysis:

| Scenario | Current Behavior | Expected Behavior |
|----------|-----------------|-------------------|
| Same page, with anchor | Returns `#block-id` or `index.html` | `#formatted-block-id` |
| Same page, no anchor | Returns `index.html` | Empty string or `#` |
| Child → Root | Returns `../index.html` | `../index.html` ✓ |
| Sibling → Sibling | Complex calculation | `../SiblingDir/index.html` ✓ |
| Any page → Block on other page | Path + raw anchor | Path + formatted anchor |

### 3.4 Data Flow Analysis

Current flow for link resolution:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ LinkRewriter │────▶│ PageContext  │────▶│ PathCalculator│
│             │     │ getRelPathTo │     │ calcRelBetween│
└─────────────┘     └──────────────┘     └───────────────┘
       │                                         │
       │ (same-page check)                       │
       ▼                                         ▼
┌─────────────┐                         ┌───────────────┐
│_buildAnchor │                         │ Returns path  │
│    Hash     │                         │ without hash  │
└─────────────┘                         └───────────────┘
       │                                         │
       └──────────────┬──────────────────────────┘
                      ▼
              Final href assembly
```

The current design forces `LinkRewriter` to:
1. Detect same-page scenarios
2. Call different methods for each scenario
3. Manually assemble final href

---

## 4. Proposed Refactoring: Path Strategy Pattern

### 4.1 Design Goals

1. **Single Responsibility**: Each class handles one type of path resolution
2. **Open/Closed Principle**: New path strategies can be added without modifying existing code
3. **Dependency Inversion**: High-level modules depend on abstractions, not concretions
4. **Factory Encapsulation**: Path type detection is centralized and testable

### 4.2 Proposed Class Hierarchy

```
┌──────────────────────────────────────────────────────────────┐
│                    «interface»                               │
│                    PathStrategy                              │
├──────────────────────────────────────────────────────────────┤
│ + resolve(sourceContext, targetContext, options): string     │
│ + supports(sourceContext, targetContext): boolean            │
│ + getType(): PathType                                        │
└──────────────────────────────────────────────────────────────┘
                              △
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────┴─────────┐ ┌───────┴───────┐ ┌────────┴────────┐
│  IntraPathStrategy│ │InterPathStrategy│ │ExternalPathStrategy│
├───────────────────┤ ├───────────────┤ ├─────────────────┤
│- blockIdMapper    │ │- pathCalculator│ │                 │
├───────────────────┤ ├───────────────┤ ├─────────────────┤
│+ resolve()        │ │+ resolve()    │ │+ resolve()      │
│+ supports()       │ │+ supports()   │ │+ supports()     │
└───────────────────┘ └───────────────┘ └─────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    PathStrategyFactory                       │
├──────────────────────────────────────────────────────────────┤
│ - strategies: PathStrategy[]                                 │
│ - config: Config                                             │
├──────────────────────────────────────────────────────────────┤
│ + createStrategy(source, target): PathStrategy               │
│ + resolvePath(source, target, options): string               │
│ + registerStrategy(strategy): void                           │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Detailed Design

#### 4.3.1 PathStrategy Interface

```javascript
/**
 * @interface PathStrategy
 * @description Abstract strategy for resolving paths between page contexts.
 */
class PathStrategy {
    /**
     * Path type enumeration
     */
    static PathType = {
        INTRA: 'intra',      // Same page anchor
        INTER: 'inter',      // Cross-page navigation
        EXTERNAL: 'external' // External URL (unchanged)
    };
    
    /**
     * Resolve path from source to target context.
     * @abstract
     * @param {PageContext} sourceContext - Origin page
     * @param {PageContext} targetContext - Destination page
     * @param {Object} options - Resolution options
     * @param {string} [options.blockId] - Target block ID (for anchors)
     * @param {Map} [options.blockMapCache] - Block ID mapping cache
     * @returns {string} Resolved path/href
     */
    resolve(sourceContext, targetContext, options = {}) {
        throw new Error('PathStrategy.resolve() must be implemented');
    }
    
    /**
     * Check if this strategy handles the given source-target pair.
     * @abstract
     * @param {PageContext} sourceContext - Origin page
     * @param {PageContext} targetContext - Destination page
     * @returns {boolean} True if this strategy applies
     */
    supports(sourceContext, targetContext) {
        throw new Error('PathStrategy.supports() must be implemented');
    }
    
    /**
     * Get the path type this strategy handles.
     * @abstract
     * @returns {string} PathType value
     */
    getType() {
        throw new Error('PathStrategy.getType() must be implemented');
    }
}
```

#### 4.3.2 IntraPathStrategy Implementation

```javascript
/**
 * @class IntraPathStrategy
 * @extends PathStrategy
 * @description Handles same-page anchor resolution.
 * 
 * Responsible for:
 * - Detecting same-page links (sourceContext.id === targetContext.id)
 * - Formatting block IDs as anchor hashes
 * - Block ID mapping lookup via cache
 */
class IntraPathStrategy extends PathStrategy {
    constructor(blockIdMapper) {
        super();
        this.blockIdMapper = blockIdMapper || new BlockIDMapper();
    }
    
    /**
     * @override
     */
    supports(sourceContext, targetContext) {
        if (!sourceContext || !targetContext) return false;
        return sourceContext.id === targetContext.id;
    }
    
    /**
     * @override
     */
    getType() {
        return PathStrategy.PathType.INTRA;
    }
    
    /**
     * @override
     * @description For same-page links, returns only the anchor hash.
     * If no block ID provided, returns empty string (current location).
     */
    resolve(sourceContext, targetContext, options = {}) {
        const { blockId, blockMapCache } = options;
        
        if (!blockId) {
            // Same page, no anchor - link to self
            return '';
        }
        
        // Format block ID using cache or fallback
        const formattedId = this._formatBlockId(blockId, targetContext, blockMapCache);
        return `#${formattedId}`;
    }
    
    /**
     * Format raw block ID to anchor-compatible format.
     * @private
     */
    _formatBlockId(rawBlockId, targetContext, blockMapCache) {
        if (blockMapCache && targetContext.id) {
            const pageBlockMap = blockMapCache.get(targetContext.id);
            if (pageBlockMap) {
                return this.blockIdMapper.getFormattedId(rawBlockId, pageBlockMap);
            }
        }
        
        // Fallback to direct formatting
        return this.blockIdMapper.getFormattedId(rawBlockId, null);
    }
}
```

#### 4.3.3 InterPathStrategy Implementation

```javascript
/**
 * @class InterPathStrategy
 * @extends PathStrategy
 * @description Handles cross-page navigation path resolution.
 * 
 * Responsible for:
 * - Computing relative filesystem paths between pages
 * - Handling hierarchical depth calculations
 * - Appending optional anchor hashes for block ID targets
 */
class InterPathStrategy extends PathStrategy {
    constructor(blockIdMapper) {
        super();
        this.blockIdMapper = blockIdMapper || new BlockIDMapper();
    }
    
    /**
     * @override
     */
    supports(sourceContext, targetContext) {
        if (!sourceContext || !targetContext) return false;
        
        // Must be different pages with valid IDs
        return sourceContext.id !== targetContext.id &&
               this._isInternalPage(sourceContext) &&
               this._isInternalPage(targetContext);
    }
    
    /**
     * @override
     */
    getType() {
        return PathStrategy.PathType.INTER;
    }
    
    /**
     * @override
     * @description Computes relative path from source to target.
     * Algorithm:
     * 1. Build path segments from hierarchy for both contexts
     * 2. Find common ancestor depth
     * 3. Calculate '../' prefix count
     * 4. Append target path segments
     * 5. Add index.html and optional block hash
     */
    resolve(sourceContext, targetContext, options = {}) {
        const { blockId, blockMapCache } = options;
        
        // Calculate relative navigation path
        const relativePath = this._calculateRelativeNavigation(sourceContext, targetContext);
        
        // Append anchor hash if block ID provided
        const anchorHash = blockId 
            ? this._buildAnchorHash(blockId, targetContext, blockMapCache)
            : '';
        
        return relativePath + anchorHash;
    }
    
    /**
     * Calculate filesystem-relative path from source to target.
     * @private
     */
    _calculateRelativeNavigation(sourceContext, targetContext) {
        const sourceSegments = this._getPathSegments(sourceContext);
        const targetSegments = this._getPathSegments(targetContext);
        
        // Find common ancestor
        let commonDepth = 0;
        while (commonDepth < sourceSegments.length &&
               commonDepth < targetSegments.length &&
               sourceSegments[commonDepth] === targetSegments[commonDepth]) {
            commonDepth++;
        }
        
        // Calculate up-levels (../) needed
        const upLevels = sourceSegments.length - commonDepth;
        
        // Get down-path segments
        const downSegments = targetSegments.slice(commonDepth);
        
        // Build path
        let result = '';
        
        for (let i = 0; i < upLevels; i++) {
            result += '../';
        }
        
        if (downSegments.length > 0) {
            result += downSegments.join('/') + '/';
        }
        
        result += 'index.html';
        
        return result;
    }
    
    /**
     * Get path segments for a context (excluding root).
     * @private
     */
    _getPathSegments(context) {
        const segments = [];
        let current = context;
        
        while (current) {
            if (current.depth > 0 && current.title && current.title !== 'untitled') {
                const safeName = FileSystemUtils.sanitizeFilename(current.title);
                segments.unshift(safeName);
            }
            current = current.parentContext;
        }
        
        return segments;
    }
    
    /**
     * Build anchor hash for block ID.
     * @private
     */
    _buildAnchorHash(rawBlockId, targetContext, blockMapCache) {
        if (!rawBlockId) return '';
        
        let formattedId;
        
        if (blockMapCache && targetContext.id) {
            const pageBlockMap = blockMapCache.get(targetContext.id);
            if (pageBlockMap) {
                formattedId = this.blockIdMapper.getFormattedId(rawBlockId, pageBlockMap);
            }
        }
        
        if (!formattedId) {
            formattedId = this.blockIdMapper.getFormattedId(rawBlockId, null);
        }
        
        return `#${formattedId}`;
    }
    
    /**
     * Check if context represents an internal scraped page.
     * @private
     */
    _isInternalPage(context) {
        return context && context.id && typeof context.id === 'string';
    }
}
```

#### 4.3.4 PathStrategyFactory Implementation

```javascript
/**
 * @class PathStrategyFactory
 * @description Factory for creating and invoking path resolution strategies.
 * 
 * Responsibilities:
 * - Strategy registration and lookup
 * - Automatic strategy selection based on context analysis
 * - Centralized path resolution interface
 */
class PathStrategyFactory {
    constructor(config, logger) {
        this.config = config;
        this.logger = logger;
        this.strategies = [];
        
        // Register default strategies (order matters - first match wins)
        this._registerDefaultStrategies();
    }
    
    /**
     * Register default strategy implementations.
     * @private
     */
    _registerDefaultStrategies() {
        const blockIdMapper = new BlockIDMapper();
        
        this.registerStrategy(new IntraPathStrategy(blockIdMapper));
        this.registerStrategy(new InterPathStrategy(blockIdMapper));
        this.registerStrategy(new ExternalPathStrategy());
    }
    
    /**
     * Register a path strategy.
     * @param {PathStrategy} strategy - Strategy to register
     */
    registerStrategy(strategy) {
        if (!(strategy instanceof PathStrategy)) {
            throw new Error('Strategy must extend PathStrategy');
        }
        this.strategies.push(strategy);
    }
    
    /**
     * Find appropriate strategy for given contexts.
     * @param {PageContext} sourceContext - Origin page
     * @param {PageContext} targetContext - Destination page
     * @returns {PathStrategy|null} Matching strategy or null
     */
    findStrategy(sourceContext, targetContext) {
        for (const strategy of this.strategies) {
            if (strategy.supports(sourceContext, targetContext)) {
                return strategy;
            }
        }
        return null;
    }
    
    /**
     * Resolve path using appropriate strategy.
     * @param {PageContext} sourceContext - Origin page
     * @param {PageContext} targetContext - Destination page
     * @param {Object} options - Resolution options
     * @returns {string} Resolved path/href
     * @throws {Error} If no strategy supports the context pair
     */
    resolvePath(sourceContext, targetContext, options = {}) {
        const strategy = this.findStrategy(sourceContext, targetContext);
        
        if (!strategy) {
            this.logger.warn('PathFactory', 
                `No strategy found for ${sourceContext?.id} -> ${targetContext?.id}`);
            return null;
        }
        
        this.logger.debug('PathFactory',
            `Using ${strategy.getType()} strategy for path resolution`);
        
        return strategy.resolve(sourceContext, targetContext, options);
    }
    
    /**
     * Determine path type without full resolution.
     * Useful for logging and debugging.
     * @param {PageContext} sourceContext - Origin page
     * @param {PageContext} targetContext - Destination page
     * @returns {string} PathType value
     */
    getPathType(sourceContext, targetContext) {
        const strategy = this.findStrategy(sourceContext, targetContext);
        return strategy ? strategy.getType() : 'unknown';
    }
}
```

### 4.4 Integration with Existing Components

#### 4.4.1 PageContext Updates

```javascript
// PageContext.js - Updated to use factory
class PageContext {
    constructor(url, rawTitle, depth = 0, parentContext = null, parentId = null) {
        // ... existing initialization
        
        // Replace PathCalculator with factory reference
        this.pathFactory = null; // Injected by orchestrator
    }
    
    /**
     * Set path factory (dependency injection from orchestrator)
     */
    setPathFactory(factory) {
        this.pathFactory = factory;
    }
    
    /**
     * Get relative path to another page.
     * @param {PageContext} targetContext - Target page
     * @param {Object} options - Resolution options (blockId, blockMapCache)
     * @returns {string} Resolved path
     */
    getRelativePathTo(targetContext, options = {}) {
        if (this.pathFactory) {
            return this.pathFactory.resolvePath(this, targetContext, options);
        }
        
        // Fallback to legacy PathCalculator for backward compatibility
        return this.pathCalculator.calculateRelativePathBetween(this, targetContext);
    }
}
```

#### 4.4.2 LinkRewriter Simplification

```javascript
// LinkRewriter.js - Simplified with factory
async rewriteLinksInFile(pageContext, urlToContextMap, pageGraph = null, blockMapCache = null) {
    // ... initialization code
    
    for (const link of links) {
        const href = link.getAttribute('href');
        if (!href) continue;
        
        try {
            const { urlPart, blockIdRaw } = this._parseHref(absoluteUrl);
            const targetContext = this._findTargetContext(urlPart, urlToContextMap, idToContextMap);
            
            if (targetContext) {
                // NEW: Single unified call to path factory
                const newHref = pageContext.getRelativePathTo(targetContext, {
                    blockId: blockIdRaw,
                    blockMapCache: blockMapCache
                });
                
                if (newHref !== null) {
                    link.setAttribute('href', newHref);
                    rewriteCount++;
                }
            }
        } catch (error) {
            continue;
        }
    }
    
    // ... save HTML
}
```

### 4.5 File Structure

```
src/
└── domain/
    └── path/
        ├── PathStrategy.js           # Abstract base class
        ├── IntraPathStrategy.js      # Same-page anchor handling
        ├── InterPathStrategy.js      # Cross-page navigation
        ├── ExternalPathStrategy.js   # External URL passthrough
        ├── PathStrategyFactory.js    # Strategy creation and selection
        └── PathCalculator.js         # Legacy (deprecated)
```

---

## 5. Issue Analysis: ContentExpander Limitations

> **STATUS: ✅ RESOLVED** (30 November 2025)
> 
> This issue has been fully implemented with the dual-state capture solution.
> See implementation in:
> - `src/processing/ToggleStateCapture.js` - Dual-state content capture
> - `src/processing/OfflineToggleController.js` - Runtime JavaScript generator
> - `src/worker/pipeline/steps/ToggleCaptureStep.js` - Pipeline integration
> - `Docs/ARCHITECTURE.md` - Toggle Capture Architecture section

### 5.1 Problem Statement

The `ContentExpander` class fails to adequately handle **interactive toggle elements** in Notion pages. The current implementation either:

1. **Expands all toggles** (aggressive mode) - destroying the original interactive structure
2. **Expands nothing** (passive mode) - losing hidden content in offline replicas

Neither approach preserves both the **content** and the **interactivity** of toggle elements.

### 5.2 Notion Toggle Element Anatomy

The toggle element referenced in the incident report:

```html
<div role="button" 
     tabindex="0" 
     aria-describedby=":r22:" 
     aria-expanded="false" 
     aria-label="Open" 
     style="user-select: none; transition: background 20ms ease-in; cursor: pointer; 
            position: relative; display: flex; align-items: center; 
            justify-content: center; width: 24px; height: 24px; border-radius: 4px;">
    <svg aria-hidden="true" 
         role="graphics-symbol" 
         viewBox="0 0 16 16" 
         class="arrowCaretDownFillSmall" 
         style="width: 0.8em; height: 0.8em; display: block; fill: inherit; 
                flex-shrink: 0; color: inherit; transition: transform 200ms ease-out; 
                transform: rotateZ(-90deg); opacity: 1;">
        <path d="M2.835 3.25a.8.8 0 0 0-.69 1.203l5.164 8.854a.8.8 0 0 0 1.382 0
                 l5.165-8.854a.8.8 0 0 0-.691-1.203z"></path>
    </svg>
</div>
```

**Key Attributes:**

| Attribute | Purpose | Offline Requirement |
|-----------|---------|---------------------|
| `role="button"` | Accessibility - clickable element | Must remain clickable |
| `aria-expanded="false"` | Accessibility - toggle state | Must toggle on click |
| `tabindex="0"` | Keyboard navigation | Preserved as-is |
| `transform: rotateZ(-90deg)` | Visual indicator (collapsed) | Must animate on toggle |

### 5.3 Current Implementation Analysis

#### 5.3.1 ContentExpander.expandAll() (Disabled)

```javascript
// ContentExpander.js:36-37
async expandAll(page) {
    await this._scrollToBottom(page);
    // Aggressive expansion disabled
    // await this._expandToggles(page);
}
```

The aggressive expansion was **intentionally disabled** in production due to:
- Clicking unintended interactive elements
- Triggering destructive actions (delete, share)
- Breaking page layout with simultaneous expansions

#### 5.3.2 ContentExpander.preparePage() (Current)

```javascript
// ContentExpander.js:45-58
async preparePage(page) {
    await this._scrollToBottom(page);
    await this._closeOverlays(page);
}
```

This passive approach:
- ✅ Triggers lazy-loading via scroll
- ✅ Closes interfering overlays
- ❌ Does not expand toggles
- ❌ Does not capture hidden content

### 5.4 The Fundamental Tradeoff

| Approach | Content Captured | Interactivity Preserved | Offline Behavior |
|----------|-----------------|------------------------|------------------|
| Expand All | ✅ Complete | ❌ Lost | Static expanded view |
| Expand None | ❌ Partial | ✅ Preserved | Toggles non-functional |
| **Desired** | ✅ Complete | ✅ Preserved | Interactive toggles |

### 5.5 Proposed Solution: Dual-State Capture

Capture **both states** of each toggle element and inject JavaScript to enable offline interactivity.

#### 5.5.1 Algorithm Overview

```
For each toggle element on page:
    1. Record collapsed state (HTML snapshot)
    2. Click to expand
    3. Record expanded state (HTML snapshot)  
    4. Click to collapse (restore original)
    5. Inject both states into saved HTML
    6. Add JavaScript controller for runtime toggling
```

#### 5.5.2 Toggle State Model

```javascript
/**
 * @typedef {Object} ToggleState
 * @property {string} toggleId - Unique identifier (data-block-id)
 * @property {string} collapsedHtml - HTML when aria-expanded="false"
 * @property {string} expandedHtml - HTML when aria-expanded="true"
 * @property {string} triggerSelector - CSS selector for click target
 */
```

#### 5.5.3 Enhanced ContentExpander

```javascript
/**
 * @class ToggleStateCapture
 * @description Captures dual-state toggle content for offline interactivity.
 */
class ToggleStateCapture {
    constructor(logger) {
        this.logger = logger;
        this.capturedToggles = new Map();
    }
    
    /**
     * Capture all toggle states on page.
     * @param {Page} page - Puppeteer page
     * @returns {Promise<Map<string, ToggleState>>} Toggle ID -> state mapping
     */
    async captureAllToggleStates(page) {
        const toggles = await this._findToggles(page);
        this.logger.info('TOGGLE-CAPTURE', `Found ${toggles.length} toggle elements`);
        
        for (const toggle of toggles) {
            try {
                const state = await this._captureToggleState(page, toggle);
                if (state) {
                    this.capturedToggles.set(state.toggleId, state);
                }
            } catch (error) {
                this.logger.debug('TOGGLE-CAPTURE', `Failed to capture toggle: ${error.message}`);
            }
        }
        
        return this.capturedToggles;
    }
    
    /**
     * Find all toggle elements on page.
     * @private
     */
    async _findToggles(page) {
        return await page.$$('[role="button"][aria-expanded]');
    }
    
    /**
     * Capture collapsed and expanded states of a single toggle.
     * @private
     */
    async _captureToggleState(page, toggleHandle) {
        const toggleId = await page.evaluate(el => {
            return el.closest('[data-block-id]')?.getAttribute('data-block-id') || 
                   `toggle-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        }, toggleHandle);
        
        // Get current state
        const isExpanded = await page.evaluate(
            el => el.getAttribute('aria-expanded') === 'true',
            toggleHandle
        );
        
        // Capture current state HTML
        const currentHtml = await this._getToggleContentHtml(page, toggleHandle);
        
        // Click to toggle state
        await toggleHandle.click();
        await page.waitForTimeout(300); // Wait for animation
        
        // Capture opposite state HTML
        const oppositeHtml = await this._getToggleContentHtml(page, toggleHandle);
        
        // Restore original state
        await toggleHandle.click();
        await page.waitForTimeout(300);
        
        return {
            toggleId,
            collapsedHtml: isExpanded ? oppositeHtml : currentHtml,
            expandedHtml: isExpanded ? currentHtml : oppositeHtml,
            triggerSelector: `[data-block-id="${toggleId}"] [role="button"][aria-expanded]`
        };
    }
    
    /**
     * Get HTML content associated with toggle (the hidden content area).
     * @private
     */
    async _getToggleContentHtml(page, toggleHandle) {
        return await page.evaluate(el => {
            // Find the toggle's content container (sibling or child)
            const block = el.closest('[data-block-id]');
            if (!block) return '';
            
            // Get the content area (typically a div following the trigger)
            const contentArea = block.querySelector('.notion-toggle-content') ||
                               block.querySelector('[style*="display: none"]') ||
                               block.lastElementChild;
            
            return contentArea ? contentArea.outerHTML : '';
        }, toggleHandle);
    }
}
```

### 5.6 Offline Toggle Controller

To enable interactivity in the offline replica, inject a JavaScript controller:

```javascript
/**
 * Offline Toggle Controller
 * Injected into saved HTML to enable toggle interactivity
 */
(function() {
    'use strict';
    
    // Toggle state data (injected during save)
    const TOGGLE_STATES = __TOGGLE_STATES_PLACEHOLDER__;
    
    document.addEventListener('DOMContentLoaded', function() {
        initializeToggles();
    });
    
    function initializeToggles() {
        Object.keys(TOGGLE_STATES).forEach(toggleId => {
            const state = TOGGLE_STATES[toggleId];
            const trigger = document.querySelector(state.triggerSelector);
            
            if (trigger) {
                trigger.addEventListener('click', function(e) {
                    e.preventDefault();
                    toggleState(toggleId);
                });
                
                // Initialize with collapsed state
                setToggleContent(toggleId, false);
            }
        });
    }
    
    function toggleState(toggleId) {
        const state = TOGGLE_STATES[toggleId];
        const trigger = document.querySelector(state.triggerSelector);
        const isExpanded = trigger.getAttribute('aria-expanded') === 'true';
        
        // Update ARIA state
        trigger.setAttribute('aria-expanded', !isExpanded);
        
        // Update content
        setToggleContent(toggleId, !isExpanded);
        
        // Animate icon
        const icon = trigger.querySelector('svg');
        if (icon) {
            icon.style.transform = isExpanded ? 'rotateZ(-90deg)' : 'rotateZ(0deg)';
        }
    }
    
    function setToggleContent(toggleId, expanded) {
        const state = TOGGLE_STATES[toggleId];
        const block = document.querySelector(`[data-block-id="${toggleId}"]`);
        
        if (block) {
            const contentContainer = block.querySelector('.toggle-content-container') ||
                                    createContentContainer(block);
            
            contentContainer.innerHTML = expanded ? state.expandedHtml : state.collapsedHtml;
            contentContainer.style.display = expanded ? 'block' : 'none';
        }
    }
    
    function createContentContainer(block) {
        const container = document.createElement('div');
        container.className = 'toggle-content-container';
        block.appendChild(container);
        return container;
    }
})();
```

### 5.7 Implementation Summary (30 November 2025)

The proposed solution has been fully implemented with the following enhancements:

**Files Created:**
- `src/processing/ToggleStateCapture.js` - Dual-state capture with HtmlFacade integration
- `src/processing/OfflineToggleController.js` - Self-contained JavaScript generator
- `src/worker/pipeline/steps/ToggleCaptureStep.js` - Pipeline integration step

**Files Updated:**
- `src/processing/ContentExpander.js` - Removed deprecated `_expandToggles()` method
- `src/worker/handlers/DownloadHandler.js` - Added ToggleCaptureStep to pipeline
- `Docs/ARCHITECTURE.md` - Added Toggle Capture Architecture section

**Key Implementation Decisions:**
1. Used HtmlFacade abstraction for all DOM operations (not direct page.evaluate)
2. Added configurable skip patterns for destructive elements
3. Controller generates both JS and CSS for consistent styling
4. Added keyboard accessibility support (Enter/Space)
5. Exposed `window.__notionOfflineToggle` for debugging

---

## 6. Proposed Design: HtmlFacade Interface

### 6.1 Motivation

The codebase currently performs HTML manipulation through various disparate mechanisms:

1. **Puppeteer `page.evaluate()`**: Direct DOM manipulation in browser context
2. **JSDOM**: Server-side DOM parsing for link rewriting
3. **String manipulation**: Direct HTML string editing for injections
4. **`fs` writes**: Raw file operations for HTML persistence

This fragmentation leads to:
- Inconsistent HTML modification patterns
- Duplicated DOM traversal logic
- Difficult testing of HTML transformations
- Risk of conflicting modifications

### 6.2 Current HTML Manipulation Patterns (HtmlFacade Beneficiaries)

A comprehensive audit of the codebase reveals the following components that perform HTML manipulation and would benefit from the HtmlFacade abstraction:

#### 6.2.1 Puppeteer-Based Manipulation (Browser Context)

| Component | File | Operations | Lines |
|-----------|------|------------|-------|
| **LinkExtractor** | `src/extraction/LinkExtractor.js` | `page.evaluate()`, `querySelectorAll('a[href]')`, `getAttribute('href')` | 50-75 |
| **FileDownloader** | `src/download/FileDownloader.js` | `page.evaluate()`, `page.$$()`, `querySelectorAll()`, `getAttribute()`, `setAttribute()` | 70-343 |
| **AssetDownloader** | `src/download/AssetDownloader.js` | `page.evaluate()`, `querySelectorAll('img')`, `querySelectorAll('[style*="background"]')` | 62-147 |
| **ContentExpander** | `src/processing/ContentExpander.js` | `page.evaluate()`, `querySelector()`, `querySelectorAll()`, `getAttribute('aria-expanded')` | 73-197 |
| **CookieHandler** | `src/processing/CookieHandler.js` | `page.evaluate()`, `page.waitForSelector()`, `querySelectorAll('div[role="button"]')` | 111-176 |
| **PageProcessor** | `src/scraping/PageProcessor.js` | `page.waitForSelector()`, `page.$eval()`, `page.content()` | 211-239 |
| **LinkRewriterStep** | `src/worker/pipeline/steps/LinkRewriterStep.js` | `page.evaluate()`, `querySelectorAll('a[href]')`, `getAttribute()`, `setAttribute()` | 51-63 |
| **HtmlWriteStep** | `src/worker/pipeline/steps/HtmlWriteStep.js` | `page.content()` | 43 |
| **PuppeteerCssStrategy** | `src/download/css/strategies/PuppeteerCssStrategy.js` | `page.evaluate()` | 47 |

#### 6.2.2 JSDOM-Based Manipulation (Server Context)

| Component | File | Operations | Lines |
|-----------|------|------------|-------|
| **LinkRewriter** | `src/processing/LinkRewriter.js` | `new JSDOM()`, `querySelectorAll('a[href]')`, `getAttribute()`, `setAttribute()`, `dom.serialize()` | 3-155 |
| **DownloadHandler** | `src/worker/handlers/DownloadHandler.js` | `new JSDOM()`, `document.querySelectorAll()` | 25, 145 |
| **BlockIDExtractor** | `src/extraction/BlockIDExtractor.js` | `document.querySelectorAll('[data-block-id]')`, `getAttribute('data-block-id')` | 26-38 |

#### 6.2.3 Pattern Analysis

**Common Operations Across Components:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HTML Operation Frequency Analysis                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Operation                          │ Puppeteer │ JSDOM │ Total Uses    │
├────────────────────────────────────┼───────────┼───────┼───────────────┤
│ querySelectorAll()                 │     12    │   4   │     16        │
│ querySelector()                    │      5    │   1   │      6        │
│ getAttribute()                     │      8    │   3   │     11        │
│ setAttribute()                     │      4    │   2   │      6        │
│ page.evaluate() / DOM access       │     14    │   -   │     14        │
│ innerHTML / outerHTML              │      2    │   1   │      3        │
│ page.content() / serialize()       │      3    │   2   │      5        │
└────────────────────────────────────┴───────────┴───────┴───────────────┘
```

**Duplication Hotspots:**

1. **Link Extraction Pattern** - Repeated in 4 components:
   ```javascript
   // Duplicated in: LinkExtractor, FileDownloader, LinkRewriter, LinkRewriterStep
   const links = document.querySelectorAll('a[href]');
   for (const link of links) {
       const href = link.getAttribute('href');
       // ... processing
   }
   ```

2. **Image/Asset Extraction Pattern** - Repeated in 2 components:
   ```javascript
   // Duplicated in: AssetDownloader, (potential CssDownloader)
   document.querySelectorAll('img').forEach(img => { ... });
   document.querySelectorAll('[style*="background"]').forEach(elem => { ... });
   ```

3. **Block ID Extraction Pattern** - Repeated in 2 components:
   ```javascript
   // Duplicated in: BlockIDExtractor, DownloadHandler
   const blocks = document.querySelectorAll('[data-block-id]');
   const blockId = block.getAttribute('data-block-id');
   ```

#### 6.2.4 Refactoring Opportunities

| Priority | Component | Benefit | Effort |
|----------|-----------|---------|--------|
| **P0** | LinkRewriter | Eliminate Puppeteer/JSDOM mode split | High |
| **P0** | LinkRewriterStep | Consolidate with LinkRewriter | Medium |
| **P1** | FileDownloader | Simplify DOM traversal, testability | High |
| **P1** | AssetDownloader | Consistent API, easier mocking | Medium |
| **P1** | ContentExpander | Enable toggle capture integration | Medium |
| **P2** | BlockIDExtractor | Context-agnostic extraction | Low |
| **P2** | CookieHandler | Simplify selector logic | Low |

**Migration Strategy:**

```
Phase 1: Foundation
├── Create HtmlFacade interface
├── Implement PuppeteerHtmlFacade  
└── Implement JsdomHtmlFacade

Phase 2: High-Impact Migrations
├── Migrate LinkRewriter (eliminates dual-mode complexity)
├── Migrate LinkRewriterStep (consolidate with LinkRewriter)
└── Migrate FileDownloader (largest DOM manipulation surface)

Phase 3: Complete Migration
├── Migrate AssetDownloader
├── Migrate ContentExpander
├── Migrate BlockIDExtractor
└── Migrate CookieHandler

Phase 4: Deprecation
└── Remove direct JSDOM/page.evaluate() calls
```

### 6.3 Design Goals

1. **Unified Interface**: Single entry point for all HTML operations
2. **Context Agnostic**: Works with both Puppeteer pages and JSDOM documents
3. **Chainable Operations**: Fluent API for sequential modifications
4. **Testable**: Can be mocked for unit testing
5. **Extensible**: Easy to add new HTML operations

### 6.4 Interface Definition

```javascript
/**
 * @interface HtmlFacade
 * @description Unified interface for HTML document manipulation.
 * Abstracts the underlying DOM implementation (Puppeteer vs JSDOM).
 */
class HtmlFacade {
    /**
     * Execution context enumeration
     */
    static Context = {
        BROWSER: 'browser',  // Puppeteer page.evaluate()
        SERVER: 'server'     // JSDOM or Cheerio
    };
    
    /**
     * Query elements by CSS selector.
     * @abstract
     * @param {string} selector - CSS selector
     * @returns {Promise<HtmlElement[]>} Matching elements
     */
    async query(selector) { throw new Error('Not implemented'); }
    
    /**
     * Query single element by CSS selector.
     * @abstract
     * @param {string} selector - CSS selector
     * @returns {Promise<HtmlElement|null>} First matching element
     */
    async queryOne(selector) { throw new Error('Not implemented'); }
    
    /**
     * Get attribute value from element.
     * @abstract
     * @param {HtmlElement} element - Target element
     * @param {string} name - Attribute name
     * @returns {Promise<string|null>} Attribute value
     */
    async getAttribute(element, name) { throw new Error('Not implemented'); }
    
    /**
     * Set attribute on element.
     * @abstract
     * @param {HtmlElement} element - Target element
     * @param {string} name - Attribute name
     * @param {string} value - Attribute value
     * @returns {Promise<void>}
     */
    async setAttribute(element, name, value) { throw new Error('Not implemented'); }
    
    /**
     * Get inner HTML of element.
     * @abstract
     * @param {HtmlElement} element - Target element
     * @returns {Promise<string>} Inner HTML
     */
    async getInnerHtml(element) { throw new Error('Not implemented'); }
    
    /**
     * Set inner HTML of element.
     * @abstract
     * @param {HtmlElement} element - Target element
     * @param {string} html - HTML content
     * @returns {Promise<void>}
     */
    async setInnerHtml(element, html) { throw new Error('Not implemented'); }
    
    /**
     * Insert element before reference.
     * @abstract
     * @param {HtmlElement} newElement - Element to insert
     * @param {HtmlElement} reference - Reference element
     * @returns {Promise<void>}
     */
    async insertBefore(newElement, reference) { throw new Error('Not implemented'); }
    
    /**
     * Append child to parent.
     * @abstract
     * @param {HtmlElement} parent - Parent element
     * @param {HtmlElement} child - Child element
     * @returns {Promise<void>}
     */
    async appendChild(parent, child) { throw new Error('Not implemented'); }
    
    /**
     * Create new element.
     * @abstract
     * @param {string} tagName - Element tag name
     * @returns {Promise<HtmlElement>} New element
     */
    async createElement(tagName) { throw new Error('Not implemented'); }
    
    /**
     * Serialize document to HTML string.
     * @abstract
     * @returns {Promise<string>} Full HTML document
     */
    async serialize() { throw new Error('Not implemented'); }
    
    /**
     * Inject script into document.
     * @abstract
     * @param {string} scriptContent - JavaScript code
     * @param {Object} options - Injection options
     * @returns {Promise<void>}
     */
    async injectScript(scriptContent, options = {}) { throw new Error('Not implemented'); }
    
    /**
     * Inject stylesheet into document.
     * @abstract
     * @param {string} cssContent - CSS rules
     * @param {Object} options - Injection options
     * @returns {Promise<void>}
     */
    async injectStyle(cssContent, options = {}) { throw new Error('Not implemented'); }
    
    /**
     * Get execution context.
     * @returns {string} Context type (BROWSER or SERVER)
     */
    getContext() { throw new Error('Not implemented'); }
}
```

### 6.5 Puppeteer Implementation

```javascript
/**
 * @class PuppeteerHtmlFacade
 * @extends HtmlFacade
 * @description HtmlFacade implementation using Puppeteer page context.
 */
class PuppeteerHtmlFacade extends HtmlFacade {
    /**
     * @param {import('puppeteer').Page} page - Puppeteer page instance
     */
    constructor(page) {
        super();
        this.page = page;
    }
    
    getContext() {
        return HtmlFacade.Context.BROWSER;
    }
    
    async query(selector) {
        const handles = await this.page.$$(selector);
        return handles.map(h => new PuppeteerHtmlElement(h, this.page));
    }
    
    async queryOne(selector) {
        const handle = await this.page.$(selector);
        return handle ? new PuppeteerHtmlElement(handle, this.page) : null;
    }
    
    async getAttribute(element, name) {
        return await this.page.evaluate(
            (el, attr) => el.getAttribute(attr),
            element.handle,
            name
        );
    }
    
    async setAttribute(element, name, value) {
        await this.page.evaluate(
            (el, attr, val) => el.setAttribute(attr, val),
            element.handle,
            name,
            value
        );
    }
    
    async getInnerHtml(element) {
        return await this.page.evaluate(
            el => el.innerHTML,
            element.handle
        );
    }
    
    async setInnerHtml(element, html) {
        await this.page.evaluate(
            (el, content) => { el.innerHTML = content; },
            element.handle,
            html
        );
    }
    
    async createElement(tagName) {
        const handle = await this.page.evaluateHandle(
            tag => document.createElement(tag),
            tagName
        );
        return new PuppeteerHtmlElement(handle, this.page);
    }
    
    async appendChild(parent, child) {
        await this.page.evaluate(
            (p, c) => p.appendChild(c),
            parent.handle,
            child.handle
        );
    }
    
    async serialize() {
        return await this.page.content();
    }
    
    async injectScript(scriptContent, options = {}) {
        const { placement = 'body-end', id = null } = options;
        
        await this.page.evaluate((content, place, scriptId) => {
            const script = document.createElement('script');
            script.textContent = content;
            if (scriptId) script.id = scriptId;
            
            const target = place === 'head' 
                ? document.head 
                : document.body;
            target.appendChild(script);
        }, scriptContent, placement, id);
    }
    
    async injectStyle(cssContent, options = {}) {
        const { id = null } = options;
        
        await this.page.evaluate((content, styleId) => {
            const style = document.createElement('style');
            style.textContent = content;
            if (styleId) style.id = styleId;
            document.head.appendChild(style);
        }, cssContent, id);
    }
}

/**
 * @class PuppeteerHtmlElement
 * @description Wrapper for Puppeteer ElementHandle
 */
class PuppeteerHtmlElement {
    constructor(handle, page) {
        this.handle = handle;
        this.page = page;
    }
    
    async click() {
        await this.handle.click();
    }
    
    async focus() {
        await this.handle.focus();
    }
    
    async type(text) {
        await this.handle.type(text);
    }
}
```

### 6.6 JSDOM Implementation

```javascript
/**
 * @class JsdomHtmlFacade
 * @extends HtmlFacade
 * @description HtmlFacade implementation using JSDOM for server-side processing.
 */
class JsdomHtmlFacade extends HtmlFacade {
    /**
     * @param {JSDOM} dom - JSDOM instance
     */
    constructor(dom) {
        super();
        this.dom = dom;
        this.document = dom.window.document;
    }
    
    /**
     * Create facade from HTML string.
     * @static
     * @param {string} html - HTML content
     * @returns {JsdomHtmlFacade} New facade instance
     */
    static fromHtml(html) {
        const { JSDOM } = require('jsdom');
        const dom = new JSDOM(html);
        return new JsdomHtmlFacade(dom);
    }
    
    /**
     * Create facade from file path.
     * @static
     * @async
     * @param {string} filePath - Path to HTML file
     * @returns {Promise<JsdomHtmlFacade>} New facade instance
     */
    static async fromFile(filePath) {
        const fs = require('fs/promises');
        const html = await fs.readFile(filePath, 'utf-8');
        return JsdomHtmlFacade.fromHtml(html);
    }
    
    getContext() {
        return HtmlFacade.Context.SERVER;
    }
    
    async query(selector) {
        const elements = this.document.querySelectorAll(selector);
        return Array.from(elements).map(el => new JsdomHtmlElement(el));
    }
    
    async queryOne(selector) {
        const element = this.document.querySelector(selector);
        return element ? new JsdomHtmlElement(element) : null;
    }
    
    async getAttribute(element, name) {
        return element.element.getAttribute(name);
    }
    
    async setAttribute(element, name, value) {
        element.element.setAttribute(name, value);
    }
    
    async getInnerHtml(element) {
        return element.element.innerHTML;
    }
    
    async setInnerHtml(element, html) {
        element.element.innerHTML = html;
    }
    
    async createElement(tagName) {
        const element = this.document.createElement(tagName);
        return new JsdomHtmlElement(element);
    }
    
    async appendChild(parent, child) {
        parent.element.appendChild(child.element);
    }
    
    async insertBefore(newElement, reference) {
        reference.element.parentNode.insertBefore(
            newElement.element,
            reference.element
        );
    }
    
    async serialize() {
        return this.dom.serialize();
    }
    
    async injectScript(scriptContent, options = {}) {
        const { placement = 'body-end', id = null } = options;
        
        const script = this.document.createElement('script');
        script.textContent = scriptContent;
        if (id) script.id = id;
        
        const target = placement === 'head'
            ? this.document.head
            : this.document.body;
        target.appendChild(script);
    }
    
    async injectStyle(cssContent, options = {}) {
        const { id = null } = options;
        
        const style = this.document.createElement('style');
        style.textContent = cssContent;
        if (id) style.id = id;
        this.document.head.appendChild(style);
    }
    
    /**
     * Save serialized HTML to file.
     * @async
     * @param {string} filePath - Output file path
     * @returns {Promise<void>}
     */
    async saveToFile(filePath) {
        const fs = require('fs/promises');
        const html = await this.serialize();
        await fs.writeFile(filePath, html, 'utf-8');
    }
}

/**
 * @class JsdomHtmlElement
 * @description Wrapper for JSDOM Element
 */
class JsdomHtmlElement {
    constructor(element) {
        this.element = element;
    }
    
    // Note: click(), focus(), type() are no-ops in JSDOM
    // as there's no real browser interaction
    async click() {
        // Dispatch click event for event listeners
        const event = new this.element.ownerDocument.defaultView.MouseEvent('click', {
            bubbles: true,
            cancelable: true
        });
        this.element.dispatchEvent(event);
    }
}
```

### 6.7 Factory and Integration

```javascript
/**
 * @class HtmlFacadeFactory
 * @description Factory for creating appropriate HtmlFacade instances.
 */
class HtmlFacadeFactory {
    /**
     * Create facade for Puppeteer page.
     * @param {import('puppeteer').Page} page - Puppeteer page
     * @returns {PuppeteerHtmlFacade} Browser-context facade
     */
    static forPage(page) {
        return new PuppeteerHtmlFacade(page);
    }
    
    /**
     * Create facade from HTML string.
     * @param {string} html - HTML content
     * @returns {JsdomHtmlFacade} Server-context facade
     */
    static fromHtml(html) {
        return JsdomHtmlFacade.fromHtml(html);
    }
    
    /**
     * Create facade from file.
     * @async
     * @param {string} filePath - HTML file path
     * @returns {Promise<JsdomHtmlFacade>} Server-context facade
     */
    static async fromFile(filePath) {
        return await JsdomHtmlFacade.fromFile(filePath);
    }
}
```

### 6.8 Usage Example: LinkRewriter Integration

```javascript
// Before: Direct JSDOM manipulation
const dom = new JSDOM(html);
const document = dom.window.document;
const links = document.querySelectorAll('a[href]');
for (const link of links) {
    const href = link.getAttribute('href');
    // ... processing
    link.setAttribute('href', newHref);
}
const modifiedHtml = dom.serialize();
await fs.writeFile(htmlFilePath, modifiedHtml);

// After: Using HtmlFacade
const facade = await HtmlFacadeFactory.fromFile(htmlFilePath);
const links = await facade.query('a[href]');
for (const link of links) {
    const href = await facade.getAttribute(link, 'href');
    // ... processing
    await facade.setAttribute(link, 'href', newHref);
}
await facade.saveToFile(htmlFilePath);
```

### 6.9 Usage Example: Toggle Interactivity Injection

```javascript
async function injectToggleController(facade, toggleStates) {
    // Serialize toggle states to JSON
    const statesJson = JSON.stringify(Object.fromEntries(toggleStates));
    
    // Generate controller script with embedded state
    const controllerScript = TOGGLE_CONTROLLER_TEMPLATE
        .replace('__TOGGLE_STATES_PLACEHOLDER__', statesJson);
    
    // Inject into document
    await facade.injectScript(controllerScript, {
        placement: 'body-end',
        id: 'offline-toggle-controller'
    });
    
    // Inject supporting styles
    await facade.injectStyle(`
        .toggle-content-container {
            transition: max-height 200ms ease-out;
            overflow: hidden;
        }
        [aria-expanded="false"] + .toggle-content-container {
            max-height: 0;
        }
        [aria-expanded="true"] + .toggle-content-container {
            max-height: none;
        }
    `, { id: 'offline-toggle-styles' });
}
```

### 6.10 File Structure

```
src/
├── html/
│   ├── HtmlFacade.js              # Abstract interface
│   ├── PuppeteerHtmlFacade.js     # Browser implementation
│   ├── JsdomHtmlFacade.js         # Server implementation
│   ├── HtmlFacadeFactory.js       # Factory
│   └── elements/
│       ├── PuppeteerHtmlElement.js
│       └── JsdomHtmlElement.js
└── processing/
    └── interactivity/
        ├── ToggleStateCapture.js   # State extraction
        └── ToggleController.js     # Runtime injection
```

---

## 7. Implementation Roadmap

### 7.1 Phase 1: Critical Fixes (Week 1)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Implement adaptive wait in FileDownloader | P0 | 4h | Resolves deadlock |
| Add worker timeout extension for heavy pages | P0 | 2h | Prevents crashes |
| Add emergency worker spawning | P1 | 4h | Improves resilience |

### 7.2 Phase 2: Path Refactoring (Week 2)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Create PathStrategy interface | P1 | 2h | Foundation |
| Implement IntraPathStrategy | P1 | 4h | Same-page links |
| Implement InterPathStrategy | P1 | 4h | Cross-page links |
| Create PathStrategyFactory | P1 | 2h | Integration |
| Update LinkRewriter | P1 | 4h | Simplification |
| Deprecate PathCalculator | P2 | 2h | Cleanup |

### 7.3 Phase 3: HtmlFacade (Week 3)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Define HtmlFacade interface | P1 | 2h | Foundation |
| Implement PuppeteerHtmlFacade | P1 | 6h | Browser support |
| Implement JsdomHtmlFacade | P1 | 4h | Server support |
| Migrate LinkRewriter | P2 | 4h | Validation |
| Migrate AssetDownloader | P2 | 4h | Consistency |

### 7.4 Phase 4: Toggle Interactivity (Week 4)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Implement ToggleStateCapture | P1 | 6h | Content capture |
| Create offline toggle controller | P1 | 4h | Interactivity |
| Integrate with HtmlFacade | P1 | 4h | Injection |
| End-to-end testing | P1 | 4h | Validation |

### 7.5 Success Criteria

1. **Deadlock Resolution**: Zero `Timeout waiting for available worker` errors in 10 consecutive runs
2. **Path Accuracy**: 100% correct internal links in generated offline replicas
3. **Toggle Functionality**: Interactive toggles work in offline mode without JavaScript errors
4. **Test Coverage**: >80% unit test coverage for new components

---

## 8. Appendix: Code References

### 8.1 Key Files Modified

| File | Changes | Section |
|------|---------|---------|
| `src/download/FileDownloader.js` | Adaptive wait, async queue | §2 |
| `src/cluster/BrowserManager.js` | Emergency worker spawning | §2 |
| `src/domain/path/PathStrategy.js` | New interface | §4 |
| `src/domain/path/IntraPathStrategy.js` | New implementation | §4 |
| `src/domain/path/InterPathStrategy.js` | New implementation | §4 |
| `src/domain/path/PathStrategyFactory.js` | New factory | §4 |
| `src/domain/PageContext.js` | Factory integration | §4 |
| `src/processing/LinkRewriter.js` | Simplified API | §4 |
| `src/processing/ContentExpander.js` | Toggle capture | §5 |
| `src/html/HtmlFacade.js` | New interface | §6 |
| `src/html/PuppeteerHtmlFacade.js` | New implementation | §6 |
| `src/html/JsdomHtmlFacade.js` | New implementation | §6 |

### 8.2 Dependency Graph Update

```
┌────────────────────────────────────────────────────────────────┐
│                      ClusterOrchestrator                       │
└────────────────────────┬───────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌───────────────┐
│BrowserManager│  │QueueManager  │  │PathStrategy   │
│             │  │              │  │Factory        │
└──────┬──────┘  └──────────────┘  └───────┬───────┘
       │                                    │
       │                           ┌────────┴────────┐
       │                           │                 │
       ▼                           ▼                 ▼
┌─────────────┐            ┌─────────────┐  ┌──────────────┐
│WorkerProxy  │            │IntraPath    │  │InterPath     │
│             │            │Strategy     │  │Strategy      │
└──────┬──────┘            └─────────────┘  └──────────────┘
       │
       │ IPC
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      Worker Process                          │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │DownloadHandler│  │HtmlFacade   │  │ToggleCapture │       │
│  │             │──▶│Factory       │──▶│             │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                            │                                 │
│                    ┌───────┴───────┐                        │
│                    ▼               ▼                        │
│            ┌─────────────┐ ┌──────────────┐                 │
│            │Puppeteer    │ │JSDOM         │                 │
│            │HtmlFacade   │ │HtmlFacade    │                 │
│            └─────────────┘ └──────────────┘                 │
└──────────────────────────────────────────────────────────────┘
```

### 8.3 Test Strategy

| Component | Test Type | Coverage Target |
|-----------|-----------|-----------------|
| PathStrategy | Unit | 95% |
| PathStrategyFactory | Unit + Integration | 90% |
| HtmlFacade | Unit | 85% |
| ToggleStateCapture | Integration | 80% |
| FileDownloader (async queue) | Unit + Integration | 85% |

---

*Document prepared by: GitHub Copilot*
*Date: 29 November 2025*
*Version: 1.1*

**Revision History:**
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 29 Nov 2025 | Initial document with deadlock analysis, path strategy, ContentExpander, and HtmlFacade design |
| 1.1 | 29 Nov 2025 | Added leaf-first download strategy (§2.4 Solution D), global hidden file registry (§2.4 Solution E), comprehensive HtmlFacade beneficiaries analysis (§6.2) |

