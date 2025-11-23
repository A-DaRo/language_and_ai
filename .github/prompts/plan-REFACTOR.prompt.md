# Codebase Refactoring & Audit Report

## Executive Summary

This document outlines the results of a comprehensive codebase audit against the `Docs/ARCHITECTURE.md` specification and the strict requirement that **no `.js` file shall exceed 100 lines of code (LOC) and 200 total lines**.

The audit reveals that while the system architecture is sound and aligned with the documentation, several core components have grown into "God Classes" that violate the Single Responsibility Principle and the size constraints.

## Audit Results: Size Violations

The following files exceed the strict 200-line limit and require immediate refactoring:

| File | Lines | Primary Responsibility | Refactoring Strategy |
|------|-------|------------------------|----------------------|
| `src/orchestration/ClusterOrchestrator.js` | 670 | Workflow Coordination | Split into Phase Strategies |
| `src/download/CssDownloader.js` | 465 | CSS Processing | Split into Puppeteer/JSDOM Strategies |
| `src/orchestration/GlobalQueueManager.js` | 400 | Queue Management | Split into Discovery/Execution Queues |
| `src/worker/TaskRunner.js` | 322 | Task Routing | Split into Task Handlers |
| `src/cluster/WorkerProxy.js` | 312 | IPC Management | Split into IPC/Lifecycle components |
| `src/download/FileDownloader.js` | 280 | File Downloads | Extract FileType Strategies |
| `src/domain/PageContext.js` | 260 | Data Model | Extract Path Calculation Logic |
| `src/cluster/BrowserManager.js` | 259 | Resource Management | Split Allocation/Monitoring |
| `src/scraping/PageProcessor.js` | 248 | Scraping Logic | Extract Navigation/Extraction Logic |
| `src/processing/ContentExpander.js` | 238 | DOM Manipulation | Extract Overlay/Scroll Logic |
| `src/worker/WorkerEntrypoint.js` | 221 | Process Bootstrap | Extract Error Handling |
| `src/core/ProtocolDefinitions.js` | 215 | Type Definitions | Split into Message/Payload/Error Defs |
| `src/download/AssetDownloader.js` | 209 | Image Downloads | Extract Hashing/Retry Logic |
| `src/core/Logger.js` | 208 | Logging | Extract Transports to separate files |

## Refactoring Strategy

We will apply the **Strategy Pattern** and **Composition** to decompose these monolithic classes into smaller, focused components.

### 1. ClusterOrchestrator Refactoring

**Current State**: A single class managing 6 distinct phases of the application lifecycle.
**Target State**: A lightweight `ClusterOrchestrator` that delegates to specialized `PhaseStrategy` implementations.

**New Structure**:
```
src/orchestration/
├── ClusterOrchestrator.js (Main Coordinator)
└── phases/
    ├── BootstrapPhase.js
    ├── DiscoveryPhase.js
    ├── UserConfirmationPhase.js
    ├── ConflictResolutionPhase.js
    ├── DownloadPhase.js
    └── CompletionPhase.js
```

### 2. CssDownloader Refactoring

**Current State**: A class handling both Puppeteer (active) and JSDOM (passive) CSS processing, plus asset downloading.
**Target State**: Separate strategies for different runtime contexts.

**New Structure**:
```
src/download/css/
├── CssDownloader.js (Facade)
├── strategies/
│   ├── PuppeteerCssStrategy.js (Active scraping)
│   └── JsdomCssStrategy.js (Post-processing)
└── CssAssetManager.js (Shared asset logic)
```

### 3. GlobalQueueManager Refactoring

**Current State**: Manages both BFS discovery queue and dependency-aware execution queue, plus the title registry.
**Target State**: Distinct queue implementations for different phases.

**New Structure**:
```
src/orchestration/queues/
├── GlobalQueueManager.js (Facade)
├── DiscoveryQueue.js (BFS Logic)
├── ExecutionQueue.js (Dependency Logic)
└── TitleRegistry.js (ID-to-Title Map)
```

### 4. TaskRunner Refactoring

**Current State**: A switch statement handling all IPC message types and their execution logic.
**Target State**: A router delegating to specialized handlers.

**New Structure**:
```
src/worker/
├── TaskRunner.js (Router)
└── handlers/
    ├── DiscoveryHandler.js
    ├── DownloadHandler.js
    ├── CookieHandler.js
    └── ShutdownHandler.js
```

## Architectural Alignment

The refactoring will strictly adhere to the **Micro-Kernel** architecture defined in `Docs/ARCHITECTURE.md`.

1.  **Master Process**: Remains the "Brain", coordinating phases via the new Phase Strategies.
2.  **Worker Process**: Remains the "Muscle", executing tasks via the new Task Handlers.
3.  **Event Bus**: Continues to be the primary communication mechanism between decoupled components.

## Deprecated Methods

The following methods identified in the audit are candidates for removal or marking as deprecated:

*   `CookieHandler.handle()`: Replaced by `ensureConsent()`.
*   `CssDownloader.downloadAndRewriteCss()`: Should be clearly marked for JSDOM post-processing only, distinct from `downloadFromPuppeteer()`.

## Implementation Plan

1.  **Phase 1**: Refactor `ClusterOrchestrator` into `src/orchestration/phases/`.
2.  **Phase 2**: Refactor `CssDownloader` into `src/download/css/strategies/`.
3.  **Phase 3**: Refactor `GlobalQueueManager` into `src/orchestration/queues/`.
4.  **Phase 4**: Refactor `TaskRunner` into `src/worker/handlers/`.
5.  **Phase 5**: Verify all file sizes are < 200 lines.
6.  **Phase 6**: Update `Docs/ARCHITECTURE.md` to reflect the new component structure.

This plan ensures we meet the strict readability requirements while maintaining the robust distributed architecture of the system.

## Detailed JSDoc Specifications

The following specifications define the contracts for the new components.

### 1. Orchestration Phases (`src/orchestration/phases/`)

#### `PhaseStrategy` (Interface)
```javascript
/**
 * @interface PhaseStrategy
 * @classdesc Abstract base class defining the contract for orchestration phases.
 * Each phase encapsulates a distinct stage of the scraping workflow.
 */
class PhaseStrategy {
  /**
   * @constructor
   * @param {ClusterOrchestrator} orchestrator - Reference to the main orchestrator context.
   */
  constructor(orchestrator) {}

  /**
   * @abstract
   * @async
   * @method execute
   * @summary Executes the logic for this specific phase.
   * @description Must be implemented by concrete strategies.
   * @returns {Promise<void>} Resolves when the phase is complete.
   * @throws {Error} If the phase fails critically.
   */
  async execute() { throw new Error('Not implemented'); }

  /**
   * @protected
   * @method transitionTo
   * @summary Helper to transition the orchestrator to the next phase.
   * @param {string} phaseName - The name of the next phase.
   */
  transitionTo(phaseName) {}
}
```

#### `BootstrapPhase`
```javascript
/**
 * @class BootstrapPhase
 * @implements {PhaseStrategy}
 * @classdesc Handles the initialization of the worker cluster and initial cookie capture.
 */
class BootstrapPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Spawns workers and captures initial session state.
   * @description
   * 1. Calculates optimal worker count based on system resources.
   * 2. Spawns the initial worker pool.
   * 3. Navigates to the root URL to capture authentication cookies.
   * 4. Broadcasts cookies to all workers.
   * 5. Initializes workers with the TitleRegistry.
   * @returns {Promise<void>}
   */
  async execute() {}
}
```

#### `DiscoveryPhase`
```javascript
/**
 * @class DiscoveryPhase
 * @implements {PhaseStrategy}
 * @classdesc Manages the BFS discovery of the site structure.
 */
class DiscoveryPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Performs parallel BFS traversal to map the site.
   * @description
   * 1. Consumes tasks from `DiscoveryQueue`.
   * 2. Dispatches `IPC_DISCOVER` tasks to idle workers.
   * 3. Processes results and enqueues child pages.
   * 4. Updates `TitleRegistry` with discovered page titles.
   * 5. Continues until the discovery frontier is empty.
   * @returns {Promise<void>}
   */
  async execute() {}
}
```

#### `UserConfirmationPhase`
```javascript
/**
 * @class UserConfirmationPhase
 * @implements {PhaseStrategy}
 * @classdesc Handles user interaction to confirm the scraping plan.
 */
class UserConfirmationPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Displays the site tree and prompts for confirmation.
   * @description
   * 1. Generates a visual representation of the discovered hierarchy.
   * 2. Displays summary statistics (total pages, depth).
   * 3. Prompts the user (Y/N) via `UserPrompt`.
   * 4. Aborts the workflow if declined or timed out.
   * @returns {Promise<void>}
   */
  async execute() {}
}
```

#### `ConflictResolutionPhase`
```javascript
/**
 * @class ConflictResolutionPhase
 * @implements {PhaseStrategy}
 * @classdesc Analyzes the graph to resolve duplicates and generate file paths.
 */
class ConflictResolutionPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Prunes the graph and generates the link rewrite map.
   * @description
   * 1. Identifies duplicate pages (same URL, different paths).
   * 2. Selects canonical instances based on depth and discovery order.
   * 3. Generates the `linkRewriteMap` (URL -> Local Path).
   * 4. Prepares the `ExecutionQueue` with canonical pages.
   * @returns {Promise<void>}
   */
  async execute() {}
}
```

#### `DownloadPhase`
```javascript
/**
 * @class DownloadPhase
 * @implements {PhaseStrategy}
 * @classdesc Manages the full-fidelity download of pages and assets.
 */
class DownloadPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Orchestrates parallel page downloading.
   * @description
   * 1. Consumes tasks from `ExecutionQueue` (respecting dependencies).
   * 2. Dispatches `IPC_DOWNLOAD` tasks to workers.
   * 3. Tracks progress via `DashboardController`.
   * 4. Handles retries for failed downloads.
   * @returns {Promise<void>}
   */
  async execute() {}
}
```

### 2. CSS Strategies (`src/download/css/strategies/`)

#### `CssProcessingStrategy` (Interface)
```javascript
/**
 * @interface CssProcessingStrategy
 * @classdesc Defines the contract for CSS processing in different runtime contexts.
 */
class CssProcessingStrategy {
  /**
   * @abstract
   * @async
   * @method process
   * @summary Downloads and rewrites CSS for a specific context.
   * @param {Object} target - The target to process (Puppeteer Page or JSDOM Document).
   * @param {string} outputDir - The directory to save assets to.
   * @returns {Promise<number>} The number of stylesheets processed.
   */
  async process(target, outputDir) { throw new Error('Not implemented'); }
}
```

#### `PuppeteerCssStrategy`
```javascript
/**
 * @class PuppeteerCssStrategy
 * @implements {CssProcessingStrategy}
 * @classdesc Handles CSS extraction from a live Puppeteer page.
 */
class PuppeteerCssStrategy extends CssProcessingStrategy {
  /**
   * @async
   * @method process
   * @summary Extracts CSS from a live browser session.
   * @description
   * 1. Evaluates the page to find `<link rel="stylesheet">` tags.
   * 2. Downloads CSS content via Node.js `axios`.
   * 3. Processes `@import` and `url()` references.
   * 4. Injects local paths back into the live DOM via `page.evaluate`.
   * @param {import('puppeteer').Page} page - The active Puppeteer page.
   * @param {string} outputDir - The output directory.
   * @returns {Promise<number>}
   */
  async process(page, outputDir) {}
}
```

#### `JsdomCssStrategy`
```javascript
/**
 * @class JsdomCssStrategy
 * @implements {CssProcessingStrategy}
 * @classdesc Handles CSS rewriting for static HTML files using JSDOM.
 */
class JsdomCssStrategy extends CssProcessingStrategy {
  /**
   * @async
   * @method process
   * @summary Rewrites CSS links in a static DOM.
   * @description
   * 1. Scans the JSDOM document for `<link>` and `<style>` tags.
   * 2. Downloads referenced CSS files.
   * 3. Localizes assets.
   * 4. Updates the DOM attributes to point to local files.
   * @param {import('jsdom').JSDOM} dom - The parsed DOM.
   * @param {string} outputDir - The output directory.
   * @returns {Promise<number>}
   */
  async process(dom, outputDir) {}
}
```

### 3. Queue Components (`src/orchestration/queues/`)

#### `DiscoveryQueue`
```javascript
/**
 * @class DiscoveryQueue
 * @classdesc Manages the BFS frontier for the discovery phase.
 */
class DiscoveryQueue {
  /**
   * @method enqueue
   * @summary Adds a page to the discovery queue.
   * @param {PageContext} context - The page context.
   * @param {boolean} [isRoot=false] - Whether this is the root page.
   */
  enqueue(context, isRoot = false) {}

  /**
   * @method next
   * @summary Retrieves the next available discovery task.
   * @returns {Object|null} The task payload or null if empty.
   */
  next() {}

  /**
   * @method isComplete
   * @summary Checks if discovery is finished.
   * @returns {boolean} True if queues are empty and no pending tasks.
   */
  isComplete() {}
}
```

#### `ExecutionQueue`
```javascript
/**
 * @class ExecutionQueue
 * @classdesc Manages the download queue with dependency constraints.
 * Ensures parents are downloaded only after their children (bottom-up).
 */
class ExecutionQueue {
  /**
   * @method build
   * @summary Constructs the queue from the canonical page list.
   * @param {Array<PageContext>} contexts - The list of pages to download.
   * @param {Map} dependencyGraph - Parent-child relationships.
   */
  build(contexts, dependencyGraph) {}

  /**
   * @method next
   * @summary Retrieves the next ready download task.
   * @description Returns pages whose children have all been processed.
   * @returns {Object|null} The task payload.
   */
  next() {}

  /**
   * @method markComplete
   * @summary Marks a page as downloaded and updates parent dependencies.
   * @param {string} pageId - The ID of the completed page.
   */
  markComplete(pageId) {}
}
```

#### `TitleRegistry`
```javascript
/**
 * @class TitleRegistry
 * @classdesc Centralized store for page titles.
 * Decouples title storage from PageContext to reduce IPC overhead.
 */
class TitleRegistry {
  /**
   * @method register
   * @summary Associates a human-readable title with a page ID.
   * @param {string} id - The page ID.
   * @param {string} title - The resolved title.
   */
  register(id, title) {}

  /**
   * @method get
   * @summary Retrieves the title for a given ID.
   * @param {string} id - The page ID.
   * @returns {string} The title or a fallback.
   */
  get(id) {}

  /**
   * @method serialize
   * @summary Returns the plain object representation for IPC transmission.
   * @returns {Object} ID-to-Title map.
   */
  serialize() {}
}
```

### 4. Task Handlers (`src/worker/handlers/`)

#### `TaskHandler` (Interface)
```javascript
/**
 * @interface TaskHandler
 * @classdesc Abstract base class for worker task handlers.
 */
class TaskHandler {
  /**
   * @constructor
   * @param {Object} context - Worker context (browser, page, config, logger).
   */
  constructor(context) {}

  /**
   * @abstract
   * @async
   * @method handle
   * @summary Executes the specific task logic.
   * @param {Object} payload - The task payload from IPC.
   * @returns {Promise<Object>} The result to send back to Master.
   */
  async handle(payload) { throw new Error('Not implemented'); }
}
```

#### `DiscoveryHandler`
```javascript
/**
 * @class DiscoveryHandler
 * @implements {TaskHandler}
 * @classdesc Handles `IPC_DISCOVER` messages.
 * Performs lightweight page analysis to extract title and links.
 */
class DiscoveryHandler extends TaskHandler {
  /**
   * @async
   * @method handle
   * @summary Navigates to the URL and extracts metadata.
   * @description
   * 1. Navigates to `payload.url` (waitUntil: domcontentloaded).
   * 2. Extracts page title.
   * 3. Extracts internal links via `LinkExtractor`.
   * 4. Captures cookies if `isFirstPage` is true.
   * @param {Object} payload - Discovery payload.
   * @returns {Promise<Object>} Discovery result.
   */
  async handle(payload) {}
}
```

#### `DownloadHandler`
```javascript
/**
 * @class DownloadHandler
 * @implements {TaskHandler}
 * @classdesc Handles `IPC_DOWNLOAD` messages.
 * Executes the full scraping pipeline.
 */
class DownloadHandler extends TaskHandler {
  /**
   * @async
   * @method handle
   * @summary Executes the `ScrapingPipeline`.
   * @description
   * 1. Validates absolute paths in payload.
   * 2. Configures the pipeline steps (Navigation, Expansion, Assets, Links, Write).
   * 3. Runs the pipeline.
   * 4. Returns execution statistics.
   * @param {Object} payload - Download payload.
   * @returns {Promise<Object>} Download result.
   */
  async handle(payload) {}
}
```
