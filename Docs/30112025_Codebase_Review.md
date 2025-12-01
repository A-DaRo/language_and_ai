# Notion Scraper - Codebase Review & Architecture Simplification
## 30 November 2025 (Revised)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Package Analysis](#3-package-analysis)
   - 3.1 [Core (`src/core/`)](#31-core-srccore)
   - 3.2 [Domain (`src/domain/`)](#32-domain-srcdomain)
   - 3.3 [Orchestration (`src/orchestration/`)](#33-orchestration-srcorchestration)
   - 3.4 [Cluster (`src/cluster/`)](#34-cluster-srccluster)
   - 3.5 [Worker (`src/worker/`)](#35-worker-srcworker)
   - 3.6 [Processing (`src/processing/`)](#36-processing-srcprocessing)
   - 3.7 [Extraction (`src/extraction/`)](#37-extraction-srcextraction)
   - 3.8 [Download (`src/download/`)](#38-download-srcdownload)
   - 3.9 [HTML Abstraction (`src/html/`)](#39-html-abstraction-srchtml)
4. [Critical Bugs Analysis](#4-critical-bugs-analysis)
   - 4.1 [Bug #1: Inter-Page Path Resolution](#41-bug-1-inter-page-path-resolution)
   - 4.2 [Bug #2: Intra-Page Anchor Resolution](#42-bug-2-intra-page-anchor-resolution)
   - 4.3 [Bug #3: Toggle Capture Failure](#43-bug-3-toggle-capture-failure)
5. [Architectural Issues & Redundancy](#5-architectural-issues--redundancy)
6. [Simplification Recommendations](#6-simplification-recommendations)
7. [Implementation Priorities](#7-implementation-priorities)

---

## 1. Executive Summary

### Current State

The codebase exhibits **three persistent bugs** and **significant architectural redundancy**:

| Category | Issue | Root Cause |
|----------|-------|------------|
| **Bug** | Inter-page links broken (`./index.html` vs `../index.html`) | `LinkRewriterStep` bypasses `PathStrategyFactory` |
| **Bug** | Intra-page anchors broken (ToC → file path, not `#anchor`) | Same root cause |
| **Bug** | Toggle content empty | DOM heuristics fail against Notion's structure |
| **Architecture** | Dead code paths | `PathStrategyFactory` exists but unused in download pipeline |
| **Architecture** | Duplicated concerns | `LinkRewriter.js` vs `LinkRewriterStep.js` do overlapping work |
| **Architecture** | Scattered path logic | Path computation in 5+ locations |

### Key Finding

**The single root cause of both path-related bugs:**

```
ConflictResolver._calculateFilePath()  →  Returns root-relative paths ("Syllabus/index.html")
                    ↓
LinkRewriterStep.process()             →  Uses these directly WITHOUT computing source-relative paths
                    ↓
Result                                 →  All links become root-relative, breaking from nested pages
```

The `PathStrategyFactory` and its strategies (`IntraPathStrategy`, `InterPathStrategy`) were designed to solve this—but `LinkRewriterStep` **doesn't use them**.

---

## 2. Architecture Overview

### Process Boundary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MASTER PROCESS                                  │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────────────┐   │
│  │ SystemEvent │   │ ClusterOrches-   │   │ GlobalQueueManager         │   │
│  │ Bus         │◄──│ trator           │──►│ (DiscoveryQueue,           │   │
│  │ (routing)   │   │ (state machine)  │   │  ExecutionQueue,           │   │
│  └─────────────┘   └────────┬─────────┘   │  TitleRegistry)            │   │
│        ▲                    │             └────────────────────────────┘   │
│        │                    │                                               │
│        │            ┌───────▼─────────┐                                    │
│        │            │ BrowserManager  │──────► WorkerProxy[] ──────────┐   │
│        │            │ (spawns workers)│                                │   │
│        │            └─────────────────┘                                │   │
└────────┼───────────────────────────────────────────────────────────────┼───┘
         │                                                               │
         │                    IPC (ProtocolDefinitions)                  │
         │                                                               │
┌────────┼───────────────────────────────────────────────────────────────┼───┐
│        │                     WORKER PROCESSES                          │   │
│        │            ┌─────────────────┐                                │   │
│        └────────────│ WorkerEntrypoint│◄───────────────────────────────┘   │
│                     └────────┬────────┘                                    │
│                              │                                              │
│                     ┌────────▼────────┐                                    │
│                     │   TaskRunner    │                                    │
│                     │ (routes tasks)  │                                    │
│                     └────────┬────────┘                                    │
│                              │                                              │
│              ┌───────────────┴───────────────┐                             │
│              ▼                               ▼                              │
│     ┌─────────────────┐            ┌─────────────────┐                     │
│     │DiscoveryHandler │            │DownloadHandler  │                     │
│     │ → LinkExtractor │            │ → ScrapingPipe- │                     │
│     │ → BlockIDExtract│            │   line (steps)  │                     │
│     └─────────────────┘            └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Communication Contracts

| Boundary | Mechanism | Defined In |
|----------|-----------|------------|
| Master ↔ Worker | IPC Messages | `ProtocolDefinitions.js` |
| Master Components | EventBus Events | `SystemEventBus.js` |
| Worker Pipeline | Direct Method Calls | `PipelineStep.js` |

---

## 3. Package Analysis

### 3.1 Core (`src/core/`)

**Purpose:** Shared infrastructure used by both Master and Worker processes.

| Module | Concern | Process |
|--------|---------|---------|
| `Config.js` | Configuration constants, URL patterns, timeouts | Both |
| `Logger.js` | Logging facade (strategy pattern) | Both |
| `logger/*.js` | Console, File, Dashboard, IPC strategies | Both |
| `SystemEventBus.js` | Event routing (Master-only singleton) | **Master** |
| `ProtocolDefinitions.js` | IPC message types, serialization helpers | Both |

**Boundaries:**
- ✅ `SystemEventBus` is Master-only (documented, enforced by singleton)
- ✅ `ProtocolDefinitions` defines the IPC contract clearly
- ⚠️ `Logger` has cross-cutting concern (acceptable for logging)

**No Issues.**

---

### 3.2 Domain (`src/domain/`)

**Purpose:** Domain models representing scraped pages and path computation.

| Module | Concern |
|--------|---------|
| `PageContext.js` | Page identity, hierarchy, metadata, serialization |
| `path/PathStrategy.js` | Abstract base for path strategies |
| `path/IntraPathStrategy.js` | Same-page anchor resolution (`#block-id`) |
| `path/InterPathStrategy.js` | Cross-page path resolution (`../sibling/`) |
| `path/ExternalPathStrategy.js` | External URLs (pass-through) |
| `path/PathStrategyFactory.js` | Strategy selection based on context |
| `path/PathCalculator.js` | Low-level path math |

**Boundaries:**
- ✅ Clear strategy pattern implementation
- ✅ `PageContext` is serialization-aware (`toJSON`/`fromJSON`)
- ⚠️ `pathSegments` pre-computed at construction (good for IPC)

**Critical Issue:** `PathStrategyFactory` is **not used** where it matters most—`LinkRewriterStep`.

---

### 3.3 Orchestration (`src/orchestration/`)

**Purpose:** Master-side state management and workflow coordination.

| Module | Concern |
|--------|---------|
| `ClusterOrchestrator.js` | State machine (phases), workflow entry point |
| `GlobalQueueManager.js` | Facade for queues + title registry |
| `PageGraph.js` | Page hierarchy graph representation |
| `GlobalHiddenFileRegistry.js` | Cross-worker file deduplication |
| `phases/*.js` | Phase strategies (Bootstrap, Discovery, Download, etc.) |
| `queues/DiscoveryQueue.js` | BFS frontier for discovery |
| `queues/ExecutionQueue.js` | Download task queue (leaf-first) |
| `queues/TitleRegistry.js` | ID → human-readable title mapping |
| `analysis/ConflictResolver.js` | Deduplication, generates `linkRewriteMap` |
| `analysis/EdgeClassifier.js` | Classifies parent/child/sibling edges |
| `analysis/GraphAnalyzer.js` | Graph analysis utilities |

**Boundaries:**
- ✅ `ClusterOrchestrator` properly delegates to phase strategies
- ✅ `GlobalQueueManager` is a clean facade
- ✅ Phases are isolated strategies

**Critical Issue in `ConflictResolver.js`:**
```javascript
// Line 85: Returns ROOT-relative path
static _calculateFilePath(context) {
    const relativePath = context.getRelativePath();
    return `${relativePath}/index.html`;  // e.g., "Syllabus/index.html"
}
```

This path is **correct for filesystem saving** but **wrong for link rewriting** (which needs source-relative paths).

---

### 3.4 Cluster (`src/cluster/`)

**Purpose:** Worker lifecycle management (Master-side).

| Module | Concern |
|--------|---------|
| `BrowserManager.js` | Spawns/terminates worker processes |
| `BrowserInitializer.js` | Worker-side browser setup |
| `WorkerProxy.js` | Master-side handle for a worker |
| `proxy/WorkerStateManager.js` | Worker state tracking (idle/busy/crashed) |
| `proxy/WorkerMessageHandler.js` | IPC message routing |
| `proxy/WorkerLifecycleManager.js` | Process lifecycle events |

**Boundaries:**
- ✅ Clear separation: `WorkerProxy` (Master) vs `WorkerEntrypoint` (Worker)
- ✅ State management delegated to sub-components

**No Issues.**

---

### 3.5 Worker (`src/worker/`)

**Purpose:** Task execution inside worker processes.

| Module | Concern |
|--------|---------|
| `WorkerEntrypoint.js` | IPC listener, initializes browser |
| `TaskRunner.js` | Routes tasks to handlers |
| `handlers/DiscoveryHandler.js` | Executes discovery tasks |
| `handlers/DownloadHandler.js` | Executes download tasks |
| `pipeline/ScrapingPipeline.js` | Orchestrates download steps |
| `pipeline/PipelineStep.js` | Abstract base for steps |
| `pipeline/steps/*.js` | Concrete steps (see below) |
| `io/HtmlWriter.js` | Writes HTML to disk |

**Pipeline Steps (execution order):**

| Step | Concern |
|------|---------|
| `NavigationStep` | Navigate to URL, wait for load |
| `CookieConsentStep` | Dismiss cookie banners |
| `ExpansionStep` | Scroll to load lazy content |
| `ToggleCaptureStep` | Capture toggle states |
| `LinkRewriterStep` | **Rewrite internal links** ← BUG HERE |
| `AssetDownloadStep` | Download images, CSS |
| `HtmlWriteStep` | Save final HTML |

**Critical Issue in `LinkRewriterStep.js`:**
```javascript
// Lines 55-65: Direct string replacement, ignores PathStrategyFactory
for (const [key, value] of Object.entries(linkMap)) {
    if (href.includes(key)) {
        await facade.setAttribute(link, 'href', value);  // Root-relative!
        break;
    }
}
```

This **should** use `PathStrategyFactory` to compute source-relative paths.

---

### 3.6 Processing (`src/processing/`)

**Purpose:** Content transformation utilities.

| Module | Concern |
|--------|---------|
| `LinkRewriter.js` | **Post-hoc** link rewriting (uses `PathStrategyFactory`) |
| `BlockIDMapper.js` | Block ID formatting (raw → UUID) |
| `ToggleStateCapture.js` | Dual-state toggle capture |
| `OfflineToggleController.js` | Generates runtime JS for toggles |
| `ContentExpander.js` | Expands toggles during scraping |
| `CookieHandler.js` | Cookie consent handling |

**Redundancy Issue:**
- `LinkRewriter.js` uses `PathStrategyFactory` correctly
- `LinkRewriterStep.js` does NOT use it

These two modules do **overlapping work** with **different implementations**:

| Module | When Used | Uses PathStrategyFactory? |
|--------|-----------|---------------------------|
| `LinkRewriter.js` | Post-processing (file-based) | ✅ Yes |
| `LinkRewriterStep.js` | Pipeline (live Puppeteer page) | ❌ No |

**This is the architectural bug.**

---

### 3.7 Extraction (`src/extraction/`)

**Purpose:** Content extraction from Puppeteer pages.

| Module | Concern |
|--------|---------|
| `LinkExtractor.js` | Extracts all `<a>` links from page |
| `BlockIDExtractor.js` | Extracts block IDs for anchor mapping |

**Boundaries:**
- ✅ Single responsibility per module
- ✅ Used correctly by `DiscoveryHandler`

**No Issues.**

---

### 3.8 Download (`src/download/`)

**Purpose:** Asset downloading utilities.

| Module | Concern |
|--------|---------|
| `AssetDownloader.js` | Coordinates CSS + image downloads |
| `CssDownloader.js` | Downloads and rewrites CSS URLs |
| `FileDownloader.js` | Downloads individual files |
| `css/CssParser.js` | Parses CSS for URL extraction |
| `file/DownloadTracker.js` | Tracks download state |

**Boundaries:**
- ✅ Clean delegation chain
- ✅ No cross-cutting concerns

**No Issues.**

---

### 3.9 HTML Abstraction (`src/html/`)

**Purpose:** Context-agnostic DOM manipulation.

| Module | Concern |
|--------|---------|
| `HtmlFacade.js` | Abstract interface |
| `PuppeteerHtmlFacade.js` | Live browser implementation |
| `JsdomHtmlFacade.js` | File-based implementation |
| `HtmlFacadeFactory.js` | Creates appropriate facade |

**Boundaries:**
- ✅ Excellent abstraction pattern
- ✅ Enables code reuse between live and offline processing

**No Issues.**

---

## 4. Critical Bugs Analysis

### 4.1 Bug #1: Inter-Page Path Resolution

**Symptom:** Links like `./index.html` instead of `../index.html`

**Data Flow:**

```
ConflictResolver.resolve()
    │
    ├─► _calculateFilePath(context) → "Syllabus/index.html" (ROOT-relative)
    │
    └─► linkRewriteMap.set(id, "Syllabus/index.html")
           │
           ▼
LinkRewriterStep.process()
    │
    ├─► For each <a href="...29d979ee...">
    │       if (href.includes("29d979ee"))
    │           link.href = "Syllabus/index.html"  ← WRONG!
    │
    └─► Should be: "../Syllabus/index.html" (from Lab_Session_1/)
```

**Root Cause:** `linkRewriteMap` values are root-relative, but are used directly without computing the relative path from the SOURCE page.

**Fix:** `LinkRewriterStep` must either:
1. Use `PathStrategyFactory.resolvePath(sourceContext, targetContext)`, OR
2. Receive a `Map<sourceId:targetId, relativePath>` (pre-computed on Master)

---

### 4.2 Bug #2: Intra-Page Anchor Resolution

**Symptom:** ToC links go to file paths instead of `#block-id`

**Data Flow:**

```
ToC Link: href="/29d979ee-ca9f-81cf#heading-block"
    │
    ▼
LinkRewriterStep.process()
    │
    ├─► href.includes("29d979ee") → TRUE
    │
    └─► link.href = "Lab_Session_1/index.html"  ← WRONG!
           │
           └─► Should be: "#heading-block" (anchor-only)
```

**Root Cause:** Same as Bug #1. `LinkRewriterStep` doesn't distinguish between:
- Intra-page links (same page, anchor only)
- Inter-page links (different page, relative path)

**Fix:** Must use `PathStrategyFactory` which has `IntraPathStrategy.supports()`:
```javascript
// IntraPathStrategy correctly detects anchor-only links
if (targetHref.startsWith('#')) {
    return true;  // Use IntraPathStrategy → returns "#block-id"
}
```

---

### 4.3 Bug #3: Toggle Capture Failure

**Symptom:** Empty `collapsedHtml`/`expandedHtml` in captured toggles

**Root Cause:** `ToggleStateCapture._getToggleContentHtml()` uses 6 heuristics that don't match Notion's actual DOM structure:

| Heuristic | Assumption | Reality |
|-----------|------------|---------|
| `.notion-toggle-content` | Class exists | Doesn't exist in modern Notion |
| `[style*="display: none"]` | Inline styles | Notion uses React conditional rendering |
| `aria-controls` | Points to content | Often not present |
| Sibling navigation | Content is sibling | Content is dynamically inserted |

**Fix:** Wait for content to appear after click:
```javascript
await toggleElement.click();
await this._waitForContentChange(page, previousHtml, timeout);
const expandedHtml = await this._getToggleContentHtml(...);
```

---

## 5. Architectural Issues & Redundancy

### Issue A: Parallel Implementations

| Concern | Module 1 | Module 2 | Problem |
|---------|----------|----------|---------|
| Link Rewriting | `LinkRewriter.js` | `LinkRewriterStep.js` | Different algorithms |
| Path Calculation | `PathStrategyFactory` | `ConflictResolver._calculateFilePath` | Different contexts |
| Block ID Formatting | `BlockIDMapper` | `BlockIDExtractor` | Overlapping |

### Issue B: Unused Code

`PathStrategyFactory` and its strategies are **well-designed but unused** in the critical download path:

```
✅ Used by: LinkRewriter.js (post-processing)
❌ NOT used by: LinkRewriterStep.js (download pipeline)
```

### Issue C: Scattered Path Logic

Path-related code exists in **5+ locations**:

1. `PageContext.getRelativePath()` - hierarchy path
2. `PageContext.getRelativePathTo()` - navigation path
3. `PathCalculator.calculateRelativePath()` - low-level math
4. `InterPathStrategy._calculateRelativeNavigation()` - cross-page
5. `ConflictResolver._calculateFilePath()` - output path

---

## 6. Simplification Recommendations

### Recommendation 1: Unify Link Rewriting

**Delete** `LinkRewriterStep.js` and use `LinkRewriter.js` directly in the pipeline:

```javascript
// DownloadHandler.js - Use LinkRewriter instead of LinkRewriterStep
const linkRewriter = new LinkRewriter(config, logger, cssDownloader);
await linkRewriter.rewriteLinksInPage(page, pageContext, contextMap);
```

Or **fix** `LinkRewriterStep` to use `PathStrategyFactory`:

```javascript
// LinkRewriterStep.js - FIXED
async process(context) {
    const factory = new PathStrategyFactory(this.config, this.logger);
    
    for (const link of links) {
        const href = await facade.getAttribute(link, 'href');
        const targetContext = this._resolveTarget(href, contextMap);
        
        // Use strategy pattern for correct path type
        const resolvedPath = factory.resolvePath(
            context.pageContext,  // source
            targetContext,        // target
            { targetHref: href }  // options (for anchor detection)
        );
        
        await facade.setAttribute(link, 'href', resolvedPath);
    }
}
```

### Recommendation 2: Pre-Compute All Paths on Master

Since Master has complete context, compute paths before download phase:

```javascript
// ConflictResolver.js - Compute per-source paths
static buildSourceRelativeMap(allContexts) {
    const map = new Map();  // "sourceId:targetId" → relativePath
    
    for (const source of allContexts) {
        for (const target of allContexts) {
            const key = `${source.id}:${target.id}`;
            const path = source.getRelativePathTo(target);
            map.set(key, path);
        }
    }
    
    return map;
}
```

### Recommendation 3: Consolidate Path Logic

Create single source of truth for path computation:

```
src/domain/path/
├── PathResolver.js        ← NEW: Single entry point
│   ├── resolveForFilesystem(context) → "Syllabus/index.html"
│   ├── resolveForNavigation(source, target) → "../Syllabus/index.html"
│   └── resolveForAnchor(href) → "#block-id"
└── (delete other files or make internal)
```

### Recommendation 4: Improve Toggle Capture

Add content-wait loop:

```javascript
// ToggleStateCapture.js
async _waitForContentChange(page, previousHtml, timeout = 2000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
        const current = await this._getToggleContentHtml(...);
        if (current !== previousHtml && current !== '') {
            return current;
        }
        await page.waitForTimeout(100);
    }
    return '';  // Timeout
}
```

---

## 7. Implementation Priorities

### Phase 1: Fix Critical Bugs (Immediate)

| Task | File | Effort |
|------|------|--------|
| Make `LinkRewriterStep` use `PathStrategyFactory` | `LinkRewriterStep.js` | 2h |
| Add anchor-only detection | Already in `IntraPathStrategy` | 0h |
| Pass `pageContext` to pipeline | `DownloadHandler.js` | 1h |
| Add content-wait to toggle capture | `ToggleStateCapture.js` | 1h |

### Phase 2: Reduce Redundancy (Short-term)

| Task | Files | Effort |
|------|-------|--------|
| Consolidate `LinkRewriter` + `LinkRewriterStep` | Both | 4h |
| Create unified `PathResolver` | `src/domain/path/` | 4h |
| Remove unused path code | Various | 2h |

### Phase 3: Architecture Cleanup (Medium-term)

| Task | Effort |
|------|--------|
| Document package boundaries in code | 2h |
| Add integration tests for path resolution | 4h |
| Add output validation step | 4h |

---

## File Reference

| Issue | Primary File | Fix Location |
|-------|-------------|--------------|
| Inter-page paths | `LinkRewriterStep.js:55-65` | Use `PathStrategyFactory` |
| Intra-page anchors | `LinkRewriterStep.js:55-65` | Same fix |
| Toggle capture | `ToggleStateCapture.js:180-200` | Add content wait |
| Redundant rewriters | `LinkRewriter.js`, `LinkRewriterStep.js` | Consolidate |

---

*Document Version: 2.0*  
*Date: November 30, 2025*  
*Focus: Bug analysis + Architecture simplification*
