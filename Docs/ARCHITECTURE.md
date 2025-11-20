# Architecture Documentation

## Executive Summary

The Notion Scraper is a **Reactive Event-Driven Micro-Kernel** architecture built on Node.js, designed to recursively scrape Notion pages through distributed multi-process execution. The system has evolved from a monolithic sequential executor to a sophisticated distributed system that coordinates worker processes via a central event bus, achieving scalability, fault tolerance, and optimal resource utilization.

## High-Level Architecture Overview

### From Monolith to Micro-Kernel

The architecture represents a fundamental shift from direct method invocation to asynchronous message-passing between isolated processes:

**Previous Architecture (Monolithic)**:
- Single-process execution with sequential page processing
- Direct method calls between components
- Single Puppeteer browser instance
- In-memory state management

**Current Architecture (Micro-Kernel)**:
- Multi-process distributed system with parallel execution
- Event-driven communication via SystemEventBus
- Worker pool with isolated Puppeteer instances
- Centralized orchestration with stateless workers

### Core Architectural Principles

1. **Micro-Kernel Pattern**: A minimal core (Master Process) manages resources while delegating heavy computation to isolated workers
2. **Separation of Concerns**: Clear boundaries between decision-making (Master) and execution (Workers)
3. **Single Responsibility**: Each class has one well-defined purpose
4. **Dependency Injection**: Components receive dependencies through constructors
5. **Event-Driven Communication**: Components coordinate via events, not direct method calls
6. **Stateless Workers**: Worker processes maintain no global state, enabling fault tolerance
7. **Strict BFS Traversal**: Breadth-first search ensures proper hierarchy and prevents infinite loops
8. **Two-Phase Execution**: Discovery (lightweight) followed by Execution (full scraping)
9. **Centralized Title Registry**: Single source of truth for ID-to-title mapping maintained by GlobalQueueManager, enabling efficient internal operations with raw IDs while providing instant human-readable title lookups for display and logging
9. **Centralized Title Registry**: Single source of truth for ID-to-title mapping in GlobalQueueManager, enabling efficient internal operations with raw IDs while providing instant human-readable title lookups

### Runtime Contexts: Master vs Worker

The system operates in **two distinct runtime contexts**, each with specific responsibilities and constraints:

#### Master Process (The Brain)
**Responsibilities:**
- Decision-making and state management
- Resource allocation and worker lifecycle management
- BFS queue coordination and dependency tracking
- Edge classification and graph analysis
- Plan generation and conflict resolution

**Constraints:**
- No heavy computation or HTML parsing
- No Puppeteer instances (browser management only)
- Lightweight, event-driven coordination only

**Key Components**: `ClusterOrchestrator`, `GlobalQueueManager`, `BrowserManager`, `ConflictResolver`

#### Worker Process (The Muscle)
**Responsibilities:**
- Page navigation and scraping
- HTML parsing and content extraction
- Asset downloading and file I/O
- Link rewriting and CSS processing

**Constraints:**
- Stateless execution (no knowledge of global queue)
- No communication with other workers
- Receives commands via IPC, returns results
- Isolated Puppeteer browser instance per worker

**Key Components**: `WorkerEntrypoint`, `TaskRunner`, `PageProcessor`, `AssetDownloader`

### Communication Architecture

The system uses a layered communication model:

1. **Event Bus Layer** (`SystemEventBus`): Master-side observer pattern for component coordination
2. **IPC Protocol Layer** (`ProtocolDefinitions`): Strict message contracts for Master-Worker communication
3. **Proxy Layer** (`WorkerProxy`): Adapters that bridge IPC gap, translating between event bus and process messages

**Message Flow Example**:
```
Master: ClusterOrchestrator emits EVENT
        ↓
Master: SystemEventBus routes to BrowserManager
        ↓
Master: BrowserManager selects idle WorkerProxy
        ↓
Master: WorkerProxy.sendCommand() → child.send(IPC_MESSAGE)
        ↓
Worker: process.on('message') receives IPC_MESSAGE
        ↓
Worker: TaskRunner routes to appropriate handler
        ↓
Worker: Execute task (PageProcessor.scrapePage)
        ↓
Worker: process.send(IPC_RESULT)
        ↓
Master: WorkerProxy.on('message') receives IPC_RESULT
        ↓
Master: WorkerProxy emits EVENT on SystemEventBus
        ↓
Master: ClusterOrchestrator handles completion
```



## System Architecture

### Package Structure and Responsibilities

The application is divided into the following packages, organized by runtime context and responsibility:

#### Master Process Packages

*   **`src/core`**: Fundamental infrastructure (Configuration, Logging, Event Bus, Protocol Definitions)
*   **`src/cluster`**: Worker process lifecycle management (Initialization, Proxy, Pool Management)
*   **`src/orchestration`**: High-level workflow coordination and state machines
    *   **`src/orchestration/ui`**: User interface components for CLI interaction
    *   **`src/orchestration/analysis`**: Graph analysis and conflict resolution
*   **`src/domain`**: Serializable domain entities (PageContext)
*   **`src/utils`**: Shared utility functions (FileSystem, Integrity)

#### Worker Process Packages

*   **`src/worker`**: Worker entrypoint and task routing
*   **`src/scraping`**: Puppeteer interaction for page navigation and saving
*   **`src/processing`**: Content manipulation (expansion, cookies, link rewriting)
*   **`src/extraction`**: Data extraction from DOM (Links)
*   **`src/download`**: Resource downloading (Assets, CSS, Files)
    *   **`src/download/css`**: Specialized CSS processing components

#### Shared Packages

*   **`src/domain`**: Domain model definitions (used by both Master and Worker)
*   **`src/core`**: Configuration and logging (instantiated in both contexts)

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                        MASTER PROCESS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  SystemEventBus (Core)                    │ │
│  │           [Central Event Coordination Layer]              │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │ Events                            │
│  ┌──────────────────────────┼────────────────────────────────┐ │
│  │                          │                                 │ │
│  │  ┌───────────────────────▼─────────┐  ┌─────────────────┐ │ │
│  │  │   ClusterOrchestrator           │  │ GlobalQueueMgr  │ │ │
│  │  │   [State Machine]               │◄─┤ [BFS Frontier]  │ │ │
│  │  └───────────────────┬─────────────┘  └─────────────────┘ │ │
│  │                      │                                     │ │
│  │  ┌───────────────────▼─────────────┐  ┌─────────────────┐ │ │
│  │  │   BrowserManager                │  │ ConflictResolver│ │ │
│  │  │   [Worker Pool]                 │  │ [Graph Pruning] │ │ │
│  │  └───────────────────┬─────────────┘  └─────────────────┘ │ │
│  │                      │                                     │ │
│  │  ┌───────────────────▼─────────────┐                      │ │
│  │  │   WorkerProxy (x N)             │                      │ │
│  │  │   [IPC Adapter]                 │                      │ │
│  │  └───────────────────┬─────────────┘                      │ │
│  └──────────────────────┼──────────────────────────────────────┘
│                         │ IPC Messages                          │
└─────────────────────────┼───────────────────────────────────────┘
                          │
     ════════════════════════════════════════
              Inter-Process Communication
     ════════════════════════════════════════
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      WORKER PROCESS (x N)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              WorkerEntrypoint (Bootstrapper)              │ │
│  │              [IPC Listener + Browser Init]                │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                     │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │              TaskRunner (Command Router)                  │ │
│  │              [Message Type Dispatcher]                    │ │
│  └─────────────┬─────────────────────┬───────────────────────┘ │
│                │                     │                         │
│  ┌─────────────▼────────┐  ┌─────────▼──────────┐            │ │
│  │   PageProcessor      │  │  LinkRewriter      │            │ │
│  │   [Scraping Logic]   │  │  [Offline Links]   │            │ │
│  └──────────┬───────────┘  └────────────────────┘            │ │
│             │                                                 │ │
│  ┌──────────▼──────────┐  ┌────────────────────┐            │ │
│  │  AssetDownloader    │  │  CssDownloader     │            │ │
│  │  [Images/Files]     │  │  [Stylesheets]     │            │ │
│  └─────────────────────┘  └────────────────────┘            │ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```



---

## Centralized Title Registry Architecture

### Overview

The system employs a **centralized ID-to-title mapping** strategy where `GlobalQueueManager` maintains the single source of truth for page titles. This architectural decision provides several key benefits:

1. **Efficient Internal Operations**: Components use raw Notion IDs (32-char hex) for all internal operations (deduplication, queuing, mapping)
2. **Instant Title Lookups**: Human-readable titles available on-demand via `getTitleById(id)` or `getTitleRegistry()`
3. **Simplified Data Model**: PageContext stores only sanitized filesystem-safe titles; no redundant title fields
4. **Optimized IPC**: Title registry sent once during worker initialization, eliminating 98% of serialization overhead
5. **Worker-Side Caching**: Workers maintain ephemeral memory-only copy for display/logging purposes
6. **Immutability**: Titles set once during discovery phase and never modified

### Title Flow Through System Phases

**Phase 1: Discovery**
1. Worker navigates to page, extracts raw Notion ID from URL
2. Worker resolves human-readable title from page URL or `page.title()`
3. Worker returns `{ pageId, resolvedTitle, links }` to Master
4. Master stores in registry: `queueManager.idToTitleMap.set(pageId, resolvedTitle)`
5. Title is now immutable for this page ID

**Phase 2: Conflict Resolution**
- ConflictResolver receives `titleRegistry` parameter for logging
- Uses `titleRegistry[context.id]` for human-readable duplicate detection messages
- Operates primarily on raw IDs for efficiency

**Phase 3: Download**
- Workers use cached registry from initialization (no serialization per task)
- Workers query local cache: `this.titleRegistry[pageId]` for display/logging
- Master-side registry remains authoritative, worker cache is ephemeral

### Key Components Integration

**GlobalQueueManager** (Master Process):
```javascript
class GlobalQueueManager {
  constructor() {
    this.idToTitleMap = new Map(); // pageId -> human-readable title
    // ...
  }
  
  completeDiscovery(pageId, links, metadata, resolvedTitle) {
    // Store title once, immutable thereafter
    if (resolvedTitle && !this.idToTitleMap.has(pageId)) {
      this.idToTitleMap.set(pageId, resolvedTitle);
    }
  }
  
  getTitleRegistry() {
    return Object.fromEntries(this.idToTitleMap);
  }
  
  getTitleById(pageId) {
    return this.idToTitleMap.get(pageId) || null;
  }
  
  getMaxDepth() {
    return this.maxDepth; // O(1) lookup, tracked incrementally
  }
}
```

**PageContext** (Both Master and Worker):
```javascript
class PageContext {
  constructor(url, rawTitle, depth, parentContext, parentId) {
    this.id = this._extractNotionId(url);
    this.title = FileSystemUtils.sanitizeFilename(rawTitle); // Filesystem-safe only
    // No originalTitle, no displayTitle
  }
  
  // Human-readable title accessed via: queueManager.getTitleById(context.id)
}
```

**ClusterOrchestrator** (Master Process):
```javascript
_displayPageTree(rootContext) {
  const titleRegistry = this.queueManager.getTitleRegistry();
  const rootLabel = titleRegistry[rootContext.id] || rootContext.title;
  // ...
}

_handleTaskComplete(workerId, taskType, result) {
  const titleRegistry = this.queueManager.getTitleRegistry();
  const displayTitle = result.resolvedTitle || titleRegistry[result.pageId];
  this.logger.info('TASK', `✓ Discovered: ${displayTitle}`);
}
```

### IPC Protocol Integration

**Lazy Initialization Strategy** (Performance Optimization):

The system uses a **lazy title registry pattern** to minimize IPC overhead:

1. **Initialization Phase**: Master sends full registry once via `IPC_INIT` message during worker bootstrap
2. **Task Execution**: Discovery and Download payloads no longer include registry
3. **Worker Cache**: Workers maintain local copy in `TaskRunner.titleRegistry`
4. **Performance**: Eliminates 98% of redundant serialization (1 send vs. N×2 sends for N pages)

**ProtocolDefinitions.js**:
```javascript
/**
 * @typedef {Object} InitPayload
 * @property {Object} config - Worker configuration
 * @property {Object} [titleRegistry] - Initial ID-to-title map (sent once)
 */

/**
 * @typedef {Object} DiscoverPayload
 * @property {string} pageId - Notion ID
 * // NOTE: No titleRegistry field (workers use cached copy)
 */

/**
 * @typedef {Object} DownloadPayload
 * @property {string} pageId - Notion ID
 * @property {Object} linkRewriteMap - ID-to-path mapping
 * // NOTE: No titleRegistry field (workers use cached copy)
 */

function serializeTitleMap(map) {
  return Object.fromEntries(map);
}

function deserializeTitleMap(obj) {
  return new Map(Object.entries(obj));
}
```

**TaskRunner** (Worker Process):
```javascript
class TaskRunner {
  constructor(browser) {
    this.titleRegistry = {}; // Cached from IPC_INIT, updated with deltas
  }
  
  setTitleRegistry(titleRegistry, isDelta = false) {
    if (isDelta) {
      this.titleRegistry = { ...this.titleRegistry, ...titleRegistry };
    } else {
      this.titleRegistry = titleRegistry || {};
    }
  }
}

async _executeDownload(payload) {
  const displayTitle = this.titleRegistry[payload.pageId] || 'Unknown';
  console.log(`[TaskRunner] Downloading: ${displayTitle}`);
  // ...
  return { pageId, resolvedTitle, links };
}
```

### Benefits vs Previous Architecture

**Before v1** (Scattered Title Tracking):
- `PageContext` had 3 title fields: `originalTitle`, `displayTitle`, `title`
- `setDisplayTitle()` method for updates
- Title resolution logic duplicated across components
- Inconsistent fallback patterns: `context.displayTitle || context.title || 'Untitled'`
- Complex serialization with redundant fields

**After v1** (Centralized Registry):
- `PageContext` has 1 title field: `title` (sanitized for filesystem)
- No title mutation methods
- Single source of truth in `GlobalQueueManager.idToTitleMap`
- Consistent lookups: `titleRegistry[id]` or `getTitleById(id)`
- Clean serialization without title redundancy

**After v2** (Lazy Initialization Optimization - **Current**):
- Registry sent **once** via `BrowserManager.initializeWorkers()` during bootstrap phase
- Workers cache registry in `TaskRunner.titleRegistry` (ephemeral, memory-only)
- DISCOVER/DOWNLOAD payloads no longer include registry (**98% IPC reduction**)
- Worker crash = cache lost, respawn = fresh initialization from Master
- Performance: **O(1)** serialization instead of **O(N×2)** for N pages
- Scalability: For 100-page sites, 201 serializations → 1 (99.5% reduction)

### Performance Optimization: Lazy Title Registry

**Problem**: In v1, the title registry was serialized and transmitted with every DISCOVER and DOWNLOAD task, creating significant IPC overhead:
- For N pages: 1 (bootstrap) + N (discovery) + N (download) = **2N+1 serializations**
- Each serialization: `Object.fromEntries(Map)` + JSON encoding
- Memory duplication: Registry sent N×2 times over IPC channels

**Solution**: Lazy initialization pattern sends registry once during worker bootstrap:

**Implementation Flow**:
1. **Bootstrap Phase**: `ClusterOrchestrator._phaseBootstrap()` calls `BrowserManager.initializeWorkers(titleRegistry)`
2. **Worker Initialization**: Each `WorkerProxy.sendInitialization()` sends `IPC_INIT` with registry
3. **Worker Cache**: `TaskRunner.setTitleRegistry(registry, false)` stores in `this.titleRegistry`
4. **Task Execution**: Workers use cached `this.titleRegistry[pageId]` for display/logging
5. **Fault Tolerance**: Worker crash → respawn → Master resends registry via initialization

**Performance Impact**:
- **24-page site**: 49 → 1 serializations (98% reduction, ~20-30ms saved)
- **100-page site**: 201 → 1 serializations (99.5% reduction, ~100ms saved)
- **Memory**: Eliminates N×2 redundant IPC copies (1KB × 200 = 200KB saved for 100 pages)

**Architectural Guarantees Maintained**:
- ✅ Master owns authoritative state (`GlobalQueueManager.idToTitleMap`)
- ✅ Workers remain stateless (cache is ephemeral, not persisted)
- ✅ Separation of concerns preserved (workers cache read-only data)
- ✅ No synchronization issues (registry is append-only during scraping)

---

## Detailed Package Documentation

### 1. Core Package (`src/core`) - Infrastructure Foundation

The core package provides foundational services used throughout both Master and Worker contexts. These components are instantiated independently in each runtime context.

#### 1.1 `Config.js` - Configuration Management

**Runtime Context**: Both Master and Worker  
**Design Pattern**: Configuration Object  
**Thread-Safety**: Immutable after initialization

**Class Description**:
Centralized configuration management singleton that provides read-only access to application settings. Initialized once at startup with command-line arguments and environment variables.

**Constructor Signature**:
```javascript
/**
 * @constructor
 * @param {Object} options - Configuration options
 * @param {string} options.url - Root Notion URL to scrape
 * @param {string} options.outputDir - Base output directory
 * @param {number} [options.maxDepth=Infinity] - Maximum recursion depth
 * @param {boolean} [options.headless=true] - Run browser in headless mode
 * @param {number} [options.maxRetries=3] - Maximum retry attempts
 * @param {number} [options.timeout=30000] - Page load timeout (ms)
 */
```

**Public Methods**:

```javascript
/**
 * @method getBaseUrl
 * @summary Returns the protocol and host of the target Notion page
 * @description Extracts and normalizes the base URL from the configured root URL.
 *              Used for determining if discovered links are internal or external.
 * @returns {string} Base URL in format "https://domain.com"
 * @example
 * // Config initialized with "https://notion.so/page-abc123"
 * config.getBaseUrl() // Returns "https://notion.so"
 */
getBaseUrl()

/**
 * @method isNotionUrl
 * @summary Checks if a URL belongs to the Notion domain
 * @description Validates URLs against known Notion domain patterns including
 *              notion.so, notion.site, and custom Notion domains.
 * @param {string} url - URL to validate
 * @returns {boolean} True if URL is a Notion URL
 * @throws {TypeError} If url is not a string
 */
isNotionUrl(url)

/**
 * @method extractPageNameFromUrl
 * @summary Parses a Notion URL to extract a human-readable page name
 * @description Extracts the readable page title from Notion's URL format which
 *              typically follows the pattern: /Page-Title-123abc456def
 *              Falls back to "page" if extraction fails.
 * @param {string} url - Notion URL to parse
 * @returns {string} Sanitized page name suitable for directory naming
 * @example
 * extractPageNameFromUrl("https://notion.so/My-Project-abc123")
 * // Returns "My-Project"
 */
extractPageNameFromUrl(url)

/**
 * @method getOutputDir
 * @summary Returns the base output directory for scraped content
 * @returns {string} Absolute path to output directory
 */
getOutputDir()

/**
 * @method getMaxDepth
 * @summary Returns the configured maximum recursion depth
 * @returns {number} Maximum depth (Infinity if unlimited)
 */
getMaxDepth()

/**
 * @method isHeadless
 * @summary Returns whether browser should run in headless mode
 * @returns {boolean} True if headless mode enabled
 */
isHeadless()

/**
 * @method getTimeout
 * @summary Returns the configured page load timeout
 * @returns {number} Timeout in milliseconds
 */
getTimeout()

/**
 * @method getMaxRetries
 * @summary Returns maximum retry attempts for failed operations
 * @returns {number} Maximum retry count
 */
getMaxRetries()
```

**Dependencies**: None  
**Used By**: All components requiring configuration data

---

#### 1.2 `Logger.js` - Structured Logging

**Runtime Context**: Both Master and Worker  
**Design Pattern**: Facade + Strategy  
**Thread-Safety**: Safe for multi-process (each process has own instance)

**Class Description**:
Standardized logging utility with category-based filtering, color-coded output, and timestamp support. Provides consistent formatting across the application and enables filtering by category for debugging.

**Constructor Signature**:
```javascript
/**
 * @constructor
 * @param {Object} [options={}] - Logger configuration
 * @param {string} [options.level='info'] - Minimum log level (debug|info|warn|error)
 * @param {boolean} [options.colors=true] - Enable colored output
 * @param {boolean} [options.timestamps=true] - Include timestamps
 * @param {string} [options.prefix=''] - Optional prefix for all messages
 */
```

**Public Methods**:

```javascript
/**
 * @method info
 * @summary Logs informational messages
 * @description Outputs informational messages with category tagging for filtering.
 *              Used for normal application flow and progress updates.
 * @param {string} category - Message category (e.g., "SCRAPER", "DOWNLOAD")
 * @param {string} message - Log message
 * @example
 * logger.info("SCRAPER", "Processing page: Home")
 * // Output: [12:34:56] [SCRAPER] Processing page: Home
 */
info(category, message)

/**
 * @method success
 * @summary Logs success messages with green coloring
 * @description Special informational message for completed operations.
 *              Automatically colored green in TTY environments.
 * @param {string} category - Message category
 * @param {string} message - Success message
 */
success(category, message)

/**
 * @method warn
 * @summary Logs warning messages
 * @description Outputs warnings for non-fatal issues that may require attention.
 *              Colored yellow in TTY environments.
 * @param {string} category - Message category
 * @param {string} message - Warning message
 */
warn(category, message)

/**
 * @method error
 * @summary Logs error messages with optional stack traces
 * @description Outputs error messages with full context. If an Error object is
 *              provided, includes stack trace for debugging.
 * @param {string} category - Message category
 * @param {string} message - Error message
 * @param {Error} [error] - Optional Error object with stack trace
 * @example
 * try {
 *   await downloadFile(url)
 * } catch (err) {
 *   logger.error("DOWNLOAD", "Failed to download file", err)
 * }
 */
error(category, message, error)

/**
 * @method debug
 * @summary Logs debug messages (only if debug level enabled)
 * @description Outputs verbose debugging information. Only displayed if logger
 *              level is set to 'debug'. Useful for development troubleshooting.
 * @param {string} category - Message category
 * @param {string} message - Debug message
 */
debug(category, message)

/**
 * @method separator
 * @summary Prints a visual separator line
 * @description Outputs a decorative separator for visual organization of logs.
 *              Useful for delimiting major phases or sections.
 * @param {string} [message=''] - Optional message to display in separator
 * @example
 * logger.separator("Starting Discovery Phase")
 * // Output: ═══════════════ Starting Discovery Phase ═══════════════
 */
separator(message)

/**
 * @method getElapsedTime
 * @summary Returns time since logger initialization
 * @description Calculates elapsed time from logger creation to current moment.
 *              Useful for tracking total execution duration.
 * @returns {number} Elapsed time in milliseconds
 */
getElapsedTime()
```

**Dependencies**: None  
**Used By**: All components for logging

---

#### 1.3 `SystemEventBus.js` - Event Coordination (NEW)

**Runtime Context**: Master Process Only  
**Design Pattern**: Singleton + Observer + Mediator  
**Thread-Safety**: Single-threaded (Master only)

**Class Description**:
The central nervous system of the Master Process. Decouples component implementation from coordination logic through publish-subscribe event model. All Master-side components communicate through this bus rather than direct method calls.

**Extends**: `EventEmitter` (Node.js)

**Constructor Signature**:
```javascript
/**
 * @constructor
 * @private
 * @description Private constructor - use SystemEventBus.getInstance()
 */
```

**Singleton Access**:
```javascript
/**
 * @static
 * @method getInstance
 * @summary Returns the singleton SystemEventBus instance
 * @description Lazy initialization of the global event bus. Ensures only one
 *              event bus exists in the Master Process.
 * @returns {SystemEventBus} The singleton instance
 * @example
 * const bus = SystemEventBus.getInstance()
 * bus.emit('WORKER:AVAILABLE', { workerId: 1 })
 */
static getInstance()
```

**Public Methods**:

```javascript
/**
 * @method emit
 * @summary Emits an event to all registered listeners
 * @description Synchronous event emission with optional payload. Events are
 *              processed immediately in the order listeners were registered.
 * @param {string} event - Event name (use constants from EVENT_TYPES)
 * @param {*} [payload] - Optional event data
 * @returns {boolean} True if event had listeners
 * @see {@link ProtocolDefinitions.EVENT_TYPES}
 * @example
 * bus.emit('JOB:COMPLETED', {
 *   taskId: 'task-123',
 *   result: { success: true, data: {...} }
 * })
 */
emit(event, payload)

/**
 * @method on
 * @summary Registers an event listener
 * @description Subscribes a callback to be invoked when the specified event
 *              is emitted. Listener remains active until explicitly removed.
 * @param {string} event - Event name to listen for
 * @param {Function} callback - Handler function
 * @returns {SystemEventBus} This instance (for chaining)
 * @example
 * bus.on('WORKER:AVAILABLE', ({ workerId }) => {
 *   console.log(`Worker ${workerId} is ready`)
 * })
 */
on(event, callback)

/**
 * @method once
 * @summary Registers a one-time event listener
 * @description Subscribes a callback that will be automatically removed after
 *              first invocation. Useful for initialization sequences.
 * @param {string} event - Event name to listen for
 * @param {Function} callback - Handler function
 * @returns {SystemEventBus} This instance (for chaining)
 */
once(event, callback)

/**
 * @method off
 * @summary Removes an event listener
 * @description Unsubscribes a previously registered callback.
 * @param {string} event - Event name
 * @param {Function} callback - Handler function to remove
 * @returns {SystemEventBus} This instance (for chaining)
 */
off(event, callback)

/**
 * @method removeAllListeners
 * @summary Removes all listeners for an event
 * @description Clears all subscriptions for the specified event. If no event
 *              specified, clears all listeners for all events.
 * @param {string} [event] - Optional event name
 * @returns {SystemEventBus} This instance (for chaining)
 */
removeAllListeners(event)
```

**Event Types** (defined in `ProtocolDefinitions.js`):

```javascript
/**
 * System Lifecycle Events
 */

/**
 * @event SYSTEM:INIT
 * @summary System initialization started
 * @description Emitted at application start, before worker spawning.
 *              Components use this to perform initialization.
 * @payload {Object} config - Application configuration
 * @example
 * bus.on('SYSTEM:INIT', ({ config }) => {
 *   this.initialize(config)
 * })
 */

/**
 * @event SYSTEM:SHUTDOWN
 * @summary Graceful shutdown requested
 * @description Emitted on SIGINT or SIGTERM. Components should cleanup resources.
 * @payload {Object} { reason: string } - Shutdown reason
 */

/**
 * Worker Lifecycle Events
 */

/**
 * @event WORKER:REGISTERED
 * @summary Worker process successfully spawned and ready
 * @description Emitted when a worker completes handshake and is ready for tasks.
 * @payload {Object} { workerId: string, proxy: WorkerProxy }
 */

/**
 * @event WORKER:AVAILABLE
 * @summary Worker is idle and ready for new tasks
 * @description Emitted when worker completes a task or on initial registration.
 * @payload {Object} { workerId: string }
 */

/**
 * @event WORKER:BUSY
 * @summary Worker is currently executing a task
 * @description Emitted when worker receives a task assignment.
 * @payload {Object} { workerId: string, taskId: string }
 */

/**
 * @event WORKER:ERROR
 * @summary Worker encountered an error
 * @description Emitted when worker task fails or worker process crashes.
 * @payload {Object} { workerId: string, error: SerializedError, taskId: string }
 */

/**
 * @event WORKER:DIED
 * @summary Worker process exited unexpectedly
 * @description Emitted when worker process terminates. BrowserManager handles respawn.
 * @payload {Object} { workerId: string, exitCode: number, signal: string }
 */

/**
 * Task Execution Events
 */

/**
 * @event JOB:SUBMIT
 * @summary New task submitted for execution
 * @description Emitted by Orchestrator when task should be dispatched.
 * @payload {Object} { taskId: string, type: string, payload: Object }
 */

/**
 * @event JOB:COMPLETED
 * @summary Task completed successfully
 * @description Emitted when worker returns successful result.
 * @payload {Object} { taskId: string, workerId: string, result: Object }
 */

/**
 * @event JOB:FAILED
 * @summary Task failed with error
 * @description Emitted when worker returns error result.
 * @payload {Object} { taskId: string, workerId: string, error: SerializedError }
 */

/**
 * Discovery Phase Events
 */

/**
 * @event DISCOVERY:START
 * @summary Discovery phase initiated
 * @payload {Object} { rootUrl: string, maxDepth: number }
 */

/**
 * @event DISCOVERY:LEVEL_COMPLETE
 * @summary BFS level completed
 * @payload {Object} { level: number, pagesDiscovered: number }
 */

/**
 * @event DISCOVERY:COMPLETE
 * @summary Discovery phase finished
 * @payload {Object} { totalPages: number, rootContext: PageContext }
 */

/**
 * Execution Phase Events
 */

/**
 * @event EXECUTION:START
 * @summary Execution phase initiated
 * @payload {Object} { totalPages: number }
 */

/**
 * @event EXECUTION:COMPLETE
 * @summary Execution phase finished
 * @payload {Object} { totalPages: number, linksRewritten: number }
 */
```

**Dependencies**: None  
**Used By**: `ClusterOrchestrator`, `BrowserManager`, `GlobalQueueManager`, `WorkerProxy`

---

#### 1.4 `ProtocolDefinitions.js` - IPC Contract (NEW)

**Runtime Context**: Both Master and Worker  
**Design Pattern**: Data Transfer Object (DTO)  
**Thread-Safety**: Immutable constant definitions

**File Description**:
Defines the strict contract for Inter-Process Communication between Master and Worker. Uses TypeScript-style JSDoc annotations to ensure type safety. Prevents "magic string" errors and provides single source of truth for message formats.

**Message Type Constants**:

```javascript
/**
 * @typedef {Object} MessageType
 * @property {string} INIT - Initialize worker with configuration
 * @property {string} DISCOVER - Lightweight page discovery
 * @property {string} DOWNLOAD - Full page scraping and download
 * @property {string} SET_COOKIES - Broadcast cookies to worker
 * @property {string} SHUTDOWN - Graceful worker termination
 * @property {string} RESULT - Task completion (success or failure)
 * @property {string} READY - Worker handshake acknowledgment
 */

/**
 * @constant {MessageType}
 * @description Message type identifiers for IPC communication
 */
export const MESSAGE_TYPES = {
  INIT: 'IPC_INIT',
  DISCOVER: 'IPC_DISCOVER',
  DOWNLOAD: 'IPC_DOWNLOAD',
  SET_COOKIES: 'IPC_SET_COOKIES',
  SHUTDOWN: 'IPC_SHUTDOWN',
  RESULT: 'IPC_RESULT',
  READY: 'IPC_READY'
}
```

**Message Payload Definitions**:

```javascript
/**
 * @typedef {Object} InitPayload
 * @description Sent from Master to Worker on startup
 * @property {Object} config - Serialized configuration object
 * @property {string} workerId - Unique worker identifier
 * @property {string} outputDir - Base output directory
 */

/**
 * @typedef {Object} DiscoverPayload
 * @description Sent from Master to Worker for lightweight discovery
 * @property {string} taskId - Unique task identifier for tracking
 * @property {string} url - Target URL to discover
 * @property {number} depth - Current BFS depth level
 * @property {boolean} isFirstPage - Whether this is the root page
 * @property {Array<Object>} cookies - Optional cookies from previous discovery
 * @property {string} parentUrl - URL of parent page (for hierarchy)
 */

/**
 * @typedef {Object} DownloadPayload
 * @description Sent from Master to Worker for full scraping
 * @property {string} taskId - Unique task identifier
 * @property {string} url - Target URL to scrape
 * @property {string} savePath - Absolute path where HTML should be saved
 * @property {Array<Object>} cookies - Session cookies to apply
 * @property {Map<string, string>} linkMap - URL → Local path mapping for rewriting
 * @property {Object} context - Serialized PageContext for metadata
 */

/**
 * @typedef {Object} SetCookiesPayload
 * @description Broadcast cookies to all workers after first successful page
 * @property {Array<Object>} cookies - Puppeteer cookie objects
 * @property {string} domain - Cookie domain
 */

/**
 * @typedef {Object} DiscoverResult
 * @description Worker response for discovery task
 * @property {string} taskId - Original task identifier
 * @property {boolean} success - Whether operation succeeded
 * @property {string} title - Extracted page title
 * @property {Array<Object>} links - Discovered links with metadata
 * @property {Array<Object>} [cookies] - Session cookies (if isFirstPage)
 * @property {Object} [error] - Serialized error (if success=false)
 */

/**
 * @typedef {Object} DownloadResult
 * @description Worker response for download task
 * @property {string} taskId - Original task identifier
 * @property {boolean} success - Whether operation succeeded
 * @property {string} savedPath - Path where HTML was saved
 * @property {number} assetCount - Number of assets downloaded
 * @property {number} linkCount - Number of links found
 * @property {Object} [error] - Serialized error (if success=false)
 */

/**
 * @typedef {Object} SerializedError
 * @description JSON-safe error representation
 * @property {string} name - Error name (e.g., "TypeError")
 * @property {string} message - Error message
 * @property {string} stack - Stack trace
 * @property {string} [code] - Error code (if applicable)
 */

/**
 * @typedef {Object} WorkerMessage
 * @description Generic IPC message envelope
 * @property {string} type - Message type (from MESSAGE_TYPES)
 * @property {Object} payload - Message-specific payload
 * @property {number} timestamp - Message creation timestamp
 */
```

**Serialization Utilities**:

```javascript
/**
 * @function serializeError
 * @summary Converts Error object to JSON-safe representation
 * @description Native Error objects don't serialize properly with JSON.stringify.
 *              This function extracts all relevant properties into plain object.
 * @param {Error} error - Error to serialize
 * @returns {SerializedError} JSON-safe error object
 * @example
 * try {
 *   throw new TypeError("Invalid URL")
 * } catch (err) {
 *   process.send({
 *     type: 'IPC_RESULT',
 *     payload: { success: false, error: serializeError(err) }
 *   })
 * }
 */
export function serializeError(error)

/**
 * @function deserializeError
 * @summary Reconstructs Error object from serialized form
 * @description Creates a new Error instance with properties from serialized form.
 *              Useful for displaying errors in Master Process.
 * @param {SerializedError} serialized - Serialized error object
 * @returns {Error} Reconstructed error
 */
export function deserializeError(serialized)
```

**Dependencies**: None  
**Used By**: `WorkerProxy`, `TaskRunner`, all IPC communication

---



---

### 2. Domain Package (`src/domain`) - Data Model

The domain package defines the core data structures that represent the scraped content. These objects must be JSON-serializable to cross IPC boundaries.

#### 2.1 `PageContext.js` - Page Metadata Model

**Runtime Context**: Both Master and Worker  
**Design Pattern**: Data Transfer Object + Value Object  
**Serialization**: JSON-compatible (no circular references)

**Class Description**:
Represents a single Notion page within the hierarchical structure. It is the **Single Source of Truth** for file paths, relationships, and metadata. Must be fully serializable for IPC transmission between Master and Worker processes.

**Constructor Signature**:
```javascript
/**
 * @constructor
 * @param {string} url - Canonical URL of the page
 * @param {string} title - Page title
 * @param {number} depth - BFS depth level (0 = root)
 * @param {string|null} parentId - ID of parent PageContext (not reference)
 */
constructor(url, title, depth, parentId = null)
```

**Properties**:
```javascript
/**
 * @property {string} id - Unique identifier (generated from URL hash)
 * @property {string} url - Canonical page URL
 * @property {string} title - Page title
 * @property {string} displayTitle - Sanitized title for display
 * @property {number} depth - BFS depth level
 * @property {string|null} parentId - Parent context ID (serializable)
 * @property {Array<string>} childIds - Array of child context IDs
 * @property {string|null} section - Hierarchical section metadata
 * @property {string|null} subsection - Hierarchical subsection metadata
 * @property {string} relativePath - Path relative to output root
 * @property {number} timestamp - Creation timestamp
 */
```

**Public Methods**:

```javascript
/**
 * @method setSection
 * @summary Sets the hierarchical section metadata
 * @description Assigns section information extracted from page structure.
 *              Used for organizing pages into logical groupings.
 * @param {string} section - Section identifier
 * @returns {void}
 * @note Human-readable titles stored in GlobalQueueManager.idToTitleMap
 */
setSection(section)

/**
 * @method setSubsection
 * @summary Sets the hierarchical subsection metadata
 * @description Assigns subsection information for deeper hierarchy.
 * @param {string} subsection - Subsection identifier
 * @returns {void}
 */
setSubsection(subsection)

/**
 * @method addChild
 * @summary Adds a child page to the hierarchy
 * @description Registers a child PageContext by storing its ID (not reference).
 *              Maintains serialization compatibility.
 * @param {PageContext} childContext - Child context to add
 * @returns {void}
 * @throws {TypeError} If childContext is not a PageContext instance
 */
addChild(childContext)

/**
 * @method getRelativePath
 * @summary Calculates the relative path from the root
 * @description Traverses parent chain to build hierarchical path.
 *              Used for determining directory structure.
 * @returns {string} Relative path (e.g., "Parent/Child/Grandchild")
 * @example
 * // Given hierarchy: Root > Projects > WebApp
 * context.getRelativePath() // Returns "Projects/WebApp"
 */
getRelativePath()

/**
 * @method getDirectoryPath
 * @summary Returns the absolute directory path for saving the page
 * @description Combines base directory with relative path to create full filesystem path.
 * @param {string} baseDir - Base output directory
 * @returns {string} Absolute directory path
 * @example
 * context.getDirectoryPath("/output")
 * // Returns "/output/Projects/WebApp"
 */
getDirectoryPath(baseDir)

/**
 * @method getFilePath
 * @summary Returns the absolute path to the index.html file
 * @description Convenience method that combines directory path with index.html filename.
 * @param {string} baseDir - Base output directory
 * @returns {string} Absolute file path
 * @example
 * context.getFilePath("/output")
 * // Returns "/output/Projects/WebApp/index.html"
 */
getFilePath(baseDir)

/**
 * @method getRelativePathTo
 * @summary Calculates the relative file path from this page to another page
 * @description Used for link rewriting to convert absolute URLs to relative filesystem paths.
 *              Handles different directory depths correctly.
 * @param {PageContext} targetContext - Target page context
 * @returns {string} Relative path (e.g., "../../Other/Page/index.html")
 * @example
 * // From: /output/Projects/WebApp/index.html
 * // To: /output/Docs/API/index.html
 * context.getRelativePathTo(docsContext)
 * // Returns "../../Docs/API/index.html"
 */
getRelativePathTo(targetContext)

/**
 * @method isRoot
 * @summary Checks if this is the root page
 * @description Convenience method for determining if page is at depth 0.
 * @returns {boolean} True if root page
 */
isRoot()

/**
 * @method getDepthLabel
 * @summary Returns a human-readable depth indicator
 * @description Generates visual depth representation for logging.
 * @returns {string} Depth label (e.g., "  L1" for depth 1)
 */
getDepthLabel()

/**
 * @method toJSON
 * @summary Serializes context to JSON-compatible object
 * @description Converts PageContext to plain object for IPC transmission.
 *              Removes circular references and non-serializable properties.
 * @returns {Object} Serialized representation
 * @example
 * process.send({
 *   type: 'IPC_RESULT',
 *   payload: { context: pageContext.toJSON() }
 * })
 */
toJSON()

/**
 * @static
 * @method fromJSON
 * @summary Reconstructs PageContext from serialized form
 * @description Creates a new PageContext instance from plain object.
 *              Used when receiving contexts from Worker processes.
 * @param {Object} json - Serialized context
 * @returns {PageContext} Reconstructed context instance
 * @example
 * const context = PageContext.fromJSON(message.payload.context)
 */
static fromJSON(json)
```

**Serialization Requirements**:

The PageContext must be fully JSON-serializable to cross IPC boundaries. Key constraints:

1. **No Circular References**: `parent` property replaced with `parentId` string
2. **No Function References**: Methods are not serialized, only data
3. **Reconstructable**: `fromJSON()` static method rebuilds instance with methods
4. **Child Storage**: Children stored as array of IDs, not object references

**Example Serialization Flow**:
```javascript
// Master Process
const context = new PageContext("https://notion.so/page", "Page Title", 1, "parent-id")
const proxy = getWorkerProxy()
proxy.sendCommand('DOWNLOAD', {
  context: context.toJSON() // Serializes to plain object
})

// Worker Process receives message
process.on('message', (msg) => {
  const context = PageContext.fromJSON(msg.payload.context)
  // Now has full PageContext instance with methods
  const path = context.getFilePath("/output")
})
```

**Dependencies**: `FileSystemUtils` (for path sanitization)  
**Used By**: All components that handle page metadata  
**Source**: [`src/domain/PageContext.js`](../src/domain/PageContext.js)

---

### 3. Cluster Package (`src/cluster`) - Worker Lifecycle Management (NEW)

The cluster package is the **Micro-Kernel** of the system, responsible for managing the physical worker processes. This layer abstracts away the complexity of process management, providing a clean interface for the orchestration layer.

#### 3.1 `BrowserInitializer.js` - Worker Pool Bootstrap (NEW)

**Runtime Context**: Master Process Only  
**Design Pattern**: Factory Method + Capacity Planning  
**Responsibility**: Physical creation and initialization of worker processes

**Behavioral Description**:

The BrowserInitializer is responsible for the **cold start** of the worker pool. It analyzes system resources (available RAM, CPU cores) and determines the optimal number of worker processes to spawn. Each worker is a complete Node.js child process running its own V8 instance and isolated Puppeteer browser.

**Capacity Planning Algorithm**:
1. Query system memory via `os.totalmem()` and `os.freemem()`
2. Calculate available memory considering OS overhead
3. Assume ~1GB per worker (Chromium + Node.js + page content)
4. Determine `MAX_CONCURRENCY = Math.floor(freeMemory / 1GB)`
5. Apply configurable bounds (min: 1, max: 8 or `os.cpus().length`)

**Process Spawning**:
- Uses native `child_process.fork()` to spawn workers
- Entry point: `src/worker/WorkerEntrypoint.js`
- IPC channel automatically established by Node.js
- Each worker receives unique ID: `worker-${timestamp}-${index}`

**Handshake Protocol**:
1. Master spawns child process
2. Child initializes Puppeteer browser
3. Child sends `IPC_READY` message with worker ID
4. Master receives handshake, creates `WorkerProxy`
5. Master emits `WORKER:REGISTERED` event on SystemEventBus

**Failure Handling**:
- If worker fails to send `IPC_READY` within timeout (30s), consider spawn failed
- Log error and reduce target concurrency
- Never block initialization - proceed with fewer workers

**Key Contracts**:
- **Input**: System resources (implicit)
- **Output**: Array of `WorkerProxy` instances
- **Side Effects**: Spawns N child processes, increases system memory usage
- **Guarantees**: At least 1 worker spawned (or throws initialization error)

**Source**: [`src/cluster/BrowserInitializer.js`](../src/cluster/BrowserInitializer.js)

---

#### 3.2 `WorkerProxy.js` - IPC Adapter (NEW)

**Runtime Context**: Master Process Only  
**Design Pattern**: Proxy + Adapter + Facade  
**Responsibility**: Master-side representation of remote worker process

**Behavioral Description**:

WorkerProxy is the Master's **handle** for a Worker process. It bridges the gap between the Master's event-driven architecture (SystemEventBus) and Node.js IPC message passing. Each WorkerProxy wraps exactly one `ChildProcess` instance.

**Dual Communication Direction**:

**Outbound (Master → Worker)**:
The proxy exposes a clean command interface that hides IPC complexity:
```javascript
proxy.sendCommand('DISCOVER', {
  taskId: 'task-123',
  url: 'https://notion.so/page',
  depth: 1,
  isFirstPage: false
})
```
Internally, this:
1. Validates message format against `ProtocolDefinitions`
2. Adds envelope metadata (timestamp, workerId)
3. Serializes to JSON
4. Calls `childProcess.send(message)`
5. Handles serialization errors gracefully

**Inbound (Worker → Master)**:
The proxy listens to the child process's message stream:
```javascript
childProcess.on('message', (msg) => {
  // Deserialize and validate
  // Convert to SystemEventBus events
})
```

**Message Translation**:
- `IPC_RESULT` (success) → `JOB:COMPLETED` event
- `IPC_RESULT` (error) → `JOB:FAILED` event
- Worker crash (`exit` event) → `WORKER:DIED` event

**State Tracking**:
- **Idle**: Ready for new tasks, available in BrowserManager pool
- **Busy**: Currently executing task, tracked with taskId
- **Dead**: Process exited, proxy marked for garbage collection

**Error Handling**:
- Serialization errors: Log and emit `WORKER:ERROR` event
- Process exit: Capture exit code and signal, emit `WORKER:DIED`
- Timeout: If no response within configured timeout, consider task failed

**Process Lifecycle Management**:
```javascript
// Worker initialization (v2 optimization - sends registry once)
proxy.sendInitialization(titleRegistry) // Sends IPC_INIT with cached data

// Graceful termination
proxy.shutdown() // Sends SHUTDOWN message, waits for ack
proxy.terminate() // Force kill if shutdown times out
```

**Key Contracts**:
- **Input**: `ChildProcess` instance, worker ID
- **Output**: SystemEventBus events representing worker state changes
- **Guarantees**: All IPC errors are caught and converted to events
- **Thread Safety**: Single-threaded (Master only), no race conditions

**Source**: [`src/cluster/WorkerProxy.js`](../src/cluster/WorkerProxy.js)

---

#### 3.3 `BrowserManager.js` - Worker Pool State Machine (NEW)

**Runtime Context**: Master Process Only  
**Design Pattern**: Object Pool + State Machine  
**Responsibility**: Worker pool resource allocation and lifecycle management

**Behavioral Description**:

BrowserManager is the **gatekeeper** that controls access to worker resources. It maintains the pool state and ensures workers are efficiently allocated to tasks without conflicts.

**Pool State Management**:

The manager maintains two critical data structures:
```javascript
{
  idleStack: ['worker-1', 'worker-3'],        // LIFO stack for hot workers
  busyMap: Map {                               // Currently allocated workers
    'worker-2' => { taskId: 'task-456', startTime: 123456 }
  },
  workerRegistry: Map {                        // All worker proxies
    'worker-1' => WorkerProxy,
    'worker-2' => WorkerProxy,
    'worker-3' => WorkerProxy
  }
}
```

**Resource Allocation Algorithm**:

When `BrowserManager.execute(taskType, payload)` is called:
1. Check if `idleStack` has available workers
2. If empty, queue task (wait for `WORKER:AVAILABLE` event)
3. Pop worker from `idleStack` (LIFO = cache locality)
4. Add to `busyMap` with task metadata
5. Emit `WORKER:BUSY` event
6. Call `WorkerProxy.sendCommand(taskType, payload)`

**Worker Initialization** (Performance Optimization):

When `BrowserManager.initializeWorkers(titleRegistry)` is called (during bootstrap):
1. Iterate over all registered workers in parallel
2. Call `WorkerProxy.sendInitialization(titleRegistry)` for each
3. Workers receive `IPC_INIT` message with full registry
4. Workers cache registry in `TaskRunner.titleRegistry`
5. Eliminates need to send registry with every task (98% IPC reduction)

**Completion Flow**:

When `JOB:COMPLETED` or `JOB:FAILED` event received:
1. Look up worker in `busyMap`
2. Remove from `busyMap`
3. Push back to `idleStack`
4. Emit `WORKER:AVAILABLE` event
5. Process queued tasks if any

**Supervisor Responsibilities**:

The BrowserManager acts as a supervisor for worker health:

**Crash Detection and Recovery**:
When `WORKER:DIED` event received:
1. Remove from all tracking structures
2. Log crash with exit code/signal
3. Spawn replacement worker (if under target concurrency)
4. Re-queue failed task (with retry limit)

**Deadlock Prevention**:
- Track task start times
- If task exceeds maximum duration (configurable, default 5 minutes):
  - Log timeout warning
  - Terminate worker forcefully
  - Spawn replacement
  - Mark task as failed

**Graceful Shutdown**:
```javascript
async shutdown() {
  // 1. Stop accepting new tasks
  this.acceptingTasks = false
  
  // 2. Wait for busy workers to complete (with timeout)
  await this.waitForAllIdle(timeout = 60000)
  
  // 3. Send SHUTDOWN to all workers
  for (const proxy of this.workerRegistry.values()) {
    await proxy.shutdown()
  }
  
  // 4. Force terminate any remaining after grace period
  await Promise.race([
    this.waitForAllTerminated(),
    sleep(10000).then(() => this.terminateAll())
  ])
}
```

**Metrics and Monitoring**:
The manager exposes real-time statistics:
- `getAvailableCount()`: Number of idle workers
- `getBusyCount()`: Number of allocated workers
- `getTotalCount()`: Total workers in pool
- `getAverageTaskDuration()`: Performance metric
- `getTaskQueueLength()`: Backlog size

**Key Contracts**:
- **Input**: Array of `WorkerProxy` instances from `BrowserInitializer`
- **Guarantees**: 
  - No worker allocated to multiple tasks simultaneously
  - All completed workers return to pool
  - Crashed workers are replaced
  - Graceful shutdown completes or times out deterministically

**Source**: [`src/cluster/BrowserManager.js`](../src/cluster/BrowserManager.js)

---

### 4. Worker Package (`src/worker`) - Execution Environment (NEW)

The worker package defines the code that runs **inside child processes**. This code has zero knowledge of the Master's state, orchestration logic, or other workers. It is purely reactive, responding to commands over IPC.

#### 4.1 `WorkerEntrypoint.js` - Process Bootstrap (NEW)

**Runtime Context**: Worker Process Only  
**Design Pattern**: Entry Point + Bootstrapper  
**Responsibility**: Worker process initialization and message loop

**Behavioral Description**:

This is the `main()` function for worker processes. When the Master calls `child_process.fork('src/worker/WorkerEntrypoint.js')`, this file executes as a standalone Node.js process.

**Initialization Sequence**:

```javascript
// 1. Setup environment
process.title = `notion-scraper-worker-${process.pid}`
process.on('uncaughtException', handleFatalError)
process.on('unhandledRejection', handleFatalError)

// 2. Wait for INIT message from parent
const initPromise = new Promise((resolve) => {
  process.once('message', (msg) => {
    if (msg.type === 'IPC_INIT') resolve(msg.payload)
  })
})

// 3. Initialize Puppeteer browser
const config = await initPromise
const browser = await puppeteer.launch({
  headless: config.headless,
  args: ['--no-sandbox', '--disable-setuid-sandbox']
})

// 4. Create page for reuse
const page = await browser.newPage()

// 5. Instantiate TaskRunner
const taskRunner = new TaskRunner(browser, page, config, logger)

// 6. Send READY handshake
process.send({
  type: 'IPC_READY',
  payload: { workerId: process.env.WORKER_ID }
})

// 7. Enter message loop
process.on('message', async (msg) => {
  await taskRunner.handleMessage(msg)
})
```

**Isolation Guarantees**:

The worker is completely isolated:
- **Process Isolation**: Separate memory space, cannot access Master's memory
- **Browser Isolation**: Each worker has own Chromium instance
- **State Isolation**: No shared state between workers
- **Crash Isolation**: Worker crash doesn't affect Master or siblings

**Error Handling Philosophy**:

```javascript
function handleFatalError(error) {
  logger.error('WORKER', 'Fatal error in worker', error)
  
  // Send error to parent if possible
  try {
    process.send({
      type: 'IPC_RESULT',
      payload: { success: false, error: serializeError(error) }
    })
  } catch (ipcError) {
    // IPC channel broken, just log
  }
  
  // Exit with error code
  process.exit(1)
}
```

**Key Design Decisions**:

1. **Single Page Reuse**: Worker creates one Puppeteer page and reuses it for all tasks
   - Rationale: Creating new pages is expensive (~500ms overhead)
   - Trade-off: Must ensure proper cleanup between tasks

2. **Synchronous Message Handling**: Only one task processed at a time
   - Rationale: Puppeteer is not designed for concurrent page operations
   - Implementation: Task loop waits for completion before accepting next message

3. **No Timeout Logic**: Worker never times out tasks
   - Rationale: Timeout enforcement is Master's responsibility
   - Worker focuses on execution, Master handles policy

**Source**: [`src/worker/WorkerEntrypoint.js`](../src/worker/WorkerEntrypoint.js)

---

#### 4.2 `TaskRunner.js` - Command Router (REFACTORED)

**Runtime Context**: Worker Process Only  
**Design Pattern**: Command Pattern + Strategy Pattern + Pipeline Coordinator  
**Responsibility**: Message routing, task execution coordination, and pipeline orchestration

**Behavioral Description**:

TaskRunner is the **switchboard** inside the worker. It receives IPC messages, routes them to appropriate handlers, and returns results.

**Message Routing**:

```javascript
async handleMessage(message) {
  const { type, payload } = message
  
  try {
    let result
    
    switch (type) {
      case 'IPC_DISCOVER':
        result = await this.handleDiscover(payload)
        break
      
      case 'IPC_DOWNLOAD':
        result = await this.handleDownload(payload)
        break
      
      case 'IPC_SET_COOKIES':
        result = await this.handleSetCookies(payload)
        break
      
      case 'IPC_SHUTDOWN':
        await this.handleShutdown()
        return // Don't send result, just exit
      
      default:
        throw new Error(`Unknown message type: ${type}`)
    }
    
    // Send success result
    process.send({
      type: 'IPC_RESULT',
      payload: { success: true, taskId: payload.taskId, ...result }
    })
    
  } catch (error) {
    // Send error result
    process.send({
      type: 'IPC_RESULT',
      payload: { 
        success: false, 
        taskId: payload.taskId, 
        error: serializeError(error) 
      }
    })
  }
}
```

**Task Handlers**:

**Discovery Handler** (`handleDiscover`):
- Purpose: Lightweight page metadata extraction
- Behavior:
  1. Navigate to URL with `domcontentloaded` wait (fast)
  2. Handle cookie consent if `isFirstPage`
  3. Extract page title from DOM
  4. Extract all internal links (no external resources)
  5. Return cookies if first page (for broadcast)
- Output: `{ title, links, cookies? }`
- Performance: ~1-2 seconds per page

**Download Handler** (`handleDownload`):
- Purpose: Full page scraping with all resources using Pipeline Pattern
- Behavior:
  1. Validate payload structure (enforce absolute paths via `_validateDownloadPayload()`)
  2. Create or reuse Puppeteer page
  3. Initialize pipeline context with browser, page, config, logger, payload, fileSystem
  4. Construct ScrapingPipeline with 5 sequential steps:
     - **NavigationStep**: Apply cookies, navigate to URL with `networkidle0`
     - **ExpansionStep**: Expand all collapsible content via ContentExpander
     - **AssetDownloadStep**: Download images, CSS (via `downloadFromPuppeteer`), files
     - **LinkRewriterStep**: Rewrite internal links to relative paths
     - **HtmlWriteStep**: Save HTML to absolute path using WorkerFileSystem
  5. Execute pipeline sequentially with error isolation per step
  6. Return truthful statistics from actual operations (not fake success)
- Output: `{ success: true, pageId, url, savedPath, assetsDownloaded, linksRewritten }`
- Performance: ~10-30 seconds per page (depends on assets)
- Critical: Replaces 27-line stub that caused "Ghost Execution" failure

**Cookie Broadcast Handler** (`handleSetCookies`):
- Purpose: Synchronize authentication state across all workers
- Behavior:
  1. Receive cookies array from Master
  2. Apply cookies to worker's browser context
  3. Persist cookies for all future navigations
- Output: `{ success: true }`
- Critical: Ensures all workers share same session

**Payload Validation** (`_validateDownloadPayload`):
- Purpose: Prevent "Ghost Execution" by enforcing absolute paths
- Validation Rules:
  1. Required fields: `url`, `pageId`, `savePath`
  2. **Critical**: `savePath` must be absolute (via `path.isAbsolute()`)
  3. Throws descriptive error if validation fails
- Rationale: Workers in child processes have different working directories than Master
- Error Example: `Invalid download payload: savePath must be absolute. Received: course_material/Page/index.html`
- Guarantees: All file writes use Master-calculated absolute paths

**Shutdown Handler** (`handleShutdown`):
- Purpose: Graceful worker termination
- Behavior:
  1. Close all Puppeteer pages
  2. Close browser instance
  3. Flush logs
  4. Exit process with code 0
- Output: None (process terminates)

**Error Serialization**:

Worker errors must be JSON-serializable for IPC transmission:
```javascript
function serializeError(error) {
  return {
    name: error.name,
    message: error.message,
    stack: error.stack,
    code: error.code,
    // Include Puppeteer-specific properties
    isTimeout: error.name === 'TimeoutError',
    isNavigation: error.message?.includes('navigation')
  }
}
```

**Page State Management**:

Between tasks, the worker must reset page state:
```javascript
async resetPage() {
  // Clear existing cookies (except those set by SET_COOKIES)
  const currentCookies = await this.page.cookies()
  await this.page.deleteCookie(...currentCookies.filter(c => !c.persistent))
  
  // Clear cache
  await this.page._client.send('Network.clearBrowserCache')
  
  // Reset viewport
  await this.page.setViewport({ width: 1920, height: 1080 })
  
  // Close any popup windows
  const pages = await this.browser.pages()
  for (const p of pages.slice(1)) {
    await p.close()
  }
}
```

**Source**: [`src/worker/TaskRunner.js`](../src/worker/TaskRunner.js)

---

### 5. Orchestration Package (`src/orchestration`) - Workflow Coordination

The orchestration package contains the **brain** of the Master Process. It implements the high-level workflow, state machines, and decision-making logic.

#### 5.1 `ClusterOrchestrator.js` - Master State Machine (NEW/REFACTORED)

**Runtime Context**: Master Process Only  
**Design Pattern**: State Machine + Mediator + Coordinator  
**Responsibility**: Top-level workflow coordination, connecting queue to workers

**Behavioral Description**:

ClusterOrchestrator is the **conductor** of the distributed scraping symphony. It coordinates between the GlobalQueueManager (what to do), BrowserManager (who does it), and ConflictResolver (how to optimize it).

**State Machine**:

The orchestrator operates in three distinct phases:

```
INIT → DISCOVERY → USER_CONFIRMATION → PRUNING → EXECUTION → COMPLETE
```

**Phase 1: INIT**

**Phase 1: INIT**
- Emit `SYSTEM:INIT` event
- BrowserInitializer spawns worker pool
- BrowserManager registers workers
- GlobalQueueManager initializes with root URL
- **Initialize workers with titleRegistry** (sent once via IPC_INIT)
- Transition to DISCOVERY

**Phase 2: DISCOVERY** (Lightweight BFS)
- Event Loop:
  ```javascript
  on('WORKER:AVAILABLE') → {
    if (queueManager.hasDiscoveryTasks()) {
      task = queueManager.nextDiscovery()
      browserManager.execute('DISCOVER', task)
    }
  }
  
  on('JOB:COMPLETED') → {
    result = event.payload
    if (result.links) {
      queueManager.enqueueDiscoveredLinks(result.links)
    }
    if (result.cookies && !cookiesCaptured) {
      this.broadcastCookies(result.cookies)
      cookiesCaptured = true
    }
  }
  ```
- Termination: When `queueManager.isDiscoveryComplete()`
- Output: Complete PageContext tree with all URLs discovered
- Transition to USER_CONFIRMATION

**Phase 3: USER_CONFIRMATION** (Interactive Prompt)
- Display discovered site structure via `_displayPageTree()`
- Show summary statistics (total pages, max depth, conflicts)
- Prompt user with yes/no confirmation (60-second timeout)
- Handle user input:
  - 'y/yes' → Proceed to PRUNING phase
  - 'n/no' → Abort and transition to COMPLETE
  - Timeout → Abort and transition to COMPLETE
  - Ctrl+C (SIGINT) → Abort and transition to COMPLETE
- Non-interactive mode detection (CI/CD environments)
- Edge cases:
  - Zero pages discovered → Auto-abort
  - Very large trees (>100 pages) → Display truncated tree with summary
- Transition to PRUNING (if confirmed) or COMPLETE (if aborted)

**Phase 4: PRUNING** (Graph Optimization)
- Pause all workers (stop dispatching tasks)
- Invoke `ConflictResolver.prune(rootContext)`
- Build canonical URL → Local Path mapping
- Generate download plan (only canonical instances)
- Transition to EXECUTION

**Phase 5: EXECUTION** (Full Scraping)
- Reset worker states (clear partial data)
- Event Loop (similar to discovery):
  ```javascript
  on('WORKER:AVAILABLE') → {
    if (queueManager.hasDownloadTasks()) {
      task = queueManager.nextDownload()
      task.linkMap = this.canonicalLinkMap
      browserManager.execute('DOWNLOAD', task)
    }
  }
  
  on('JOB:COMPLETED') → {
    queueManager.markDownloadComplete(task.url)
    // If this page had parent, check if parent now ready
    if (task.parentUrl) {
      queueManager.decrementPendingChildren(task.parentUrl)
    }
  }
  ```
- Termination: When `queueManager.isDownloadComplete()`
- Transition to COMPLETE

**Phase 6: COMPLETE**
- Accepts optional `aborted` parameter (true if user declined or dry run)
- If aborted:
  - Log "Scraping process aborted. Cleaning up..."
  - Display discovered pages (no download statistics)
  - Show page tree structure
- If successful:
  - Emit `EXECUTION:COMPLETE` event
  - Trigger `IntegrityAuditor.audit()`
  - Display full statistics (discovered/downloaded/failed)
- `BrowserManager.shutdown()`
- Display statistics

**Cookie Management Strategy**:

Critical issue: Workers need shared authentication state, but only first page captures cookies.

**Solution - Cookie Broadcast**:
1. First `JOB:COMPLETED` in DISCOVERY phase includes cookies
2. Orchestrator captures these cookies
3. Broadcasts to ALL workers via `SET_COOKIES` message
4. All subsequent tasks operate with authenticated session

**Race Condition Prevention**:
```javascript
let cookiesCaptured = false
let cookieBroadcastPromise = null

on('JOB:COMPLETED') → {
  if (result.cookies && !cookiesCaptured) {
    cookiesCaptured = true
    cookieBroadcastPromise = this.broadcastCookiesToAllWorkers(result.cookies)
  }
}

// Before dispatching any worker task:
if (cookieBroadcastPromise) {
  await cookieBroadcastPromise
}
```

**Dependency Tracking** (Bottom-Up Execution):

Pages with children must be downloaded AFTER their children (for proper link rewriting).

**Implementation**:
```javascript
// During PRUNING phase
const dependencyGraph = new Map() // parent → [children]
for (const context of allContexts) {
  if (context.parentId) {
    dependencyGraph.get(context.parentId).push(context.id)
  }
}

// During EXECUTION phase
class GlobalQueueManager {
  canDownload(pageContext) {
    const pendingChildren = this.pendingChildrenCount.get(pageContext.id)
    return pendingChildren === 0 // All children completed
  }
  
  markDownloadComplete(pageId) {
    const parentId = this.getParentId(pageId)
    if (parentId) {
      this.pendingChildrenCount.set(
        parentId,
        this.pendingChildrenCount.get(parentId) - 1
      )
      // Parent might now be ready for download
      if (this.canDownload(parentId)) {
        this.downloadQueue.push(parentId)
      }
    }
  }
}
```

**Error Handling and Retry Strategy**:

```javascript
const maxRetries = 3
const retryMap = new Map() // taskId → attemptCount

on('JOB:FAILED') → {
  const { taskId, error } = event.payload
  const attempts = retryMap.get(taskId) || 0
  
  if (attempts < maxRetries && isRetryableError(error)) {
    retryMap.set(taskId, attempts + 1)
    logger.warn('ORCHESTRATOR', `Retrying task ${taskId} (attempt ${attempts + 1})`)
    // Re-queue task
    queueManager.enqueue(originalTask)
  } else {
    logger.error('ORCHESTRATOR', `Task ${taskId} failed permanently`, error)
    // Mark as failed, continue with other pages
    failedPages.add(taskId)
  }
}

function isRetryableError(error) {
  // Network errors, timeouts - yes
  // Syntax errors, missing pages - no
  return error.isTimeout || 
         error.isNavigation || 
         error.code === 'ECONNRESET'
}
```

**Source**: [`src/orchestration/ClusterOrchestrator.js`](../src/orchestration/ClusterOrchestrator.js)

---

#### 5.2 `GlobalQueueManager.js` - BFS Frontier Management (NEW)

**Runtime Context**: Master Process Only  
**Design Pattern**: Queue + Monitor + Dependency Tracker  
**Responsibility**: BFS queue management with dependency-aware scheduling

**Behavioral Description**:

GlobalQueueManager maintains the **frontier** of the BFS traversal and ensures proper ordering constraints are met. It operates in two modes corresponding to the Discovery and Execution phases.

**Discovery Mode** (Simple BFS):

```javascript
class GlobalQueueManager {
  constructor() {
    this.currentLevel = []      // Current BFS level queue
    this.nextLevel = []          // Next BFS level queue
    this.visitedUrls = new Set() // Deduplication
    this.urlToContext = new Map() // URL → PageContext mapping
    this.currentDepth = 0
  }
  
  enqueueDiscovery(url, depth, parentUrl, linkMetadata) {
    // Deduplication
    if (this.visitedUrls.has(url)) return
    this.visitedUrls.add(url)
    
    // Create PageContext
    const context = new PageContext(url, linkMetadata.title, depth, parentUrl)
    this.urlToContext.set(url, context)
    
    // Add to appropriate level
    if (depth === this.currentDepth) {
      this.currentLevel.push({ url, context, depth, parentUrl })
    } else {
      this.nextLevel.push({ url, context, depth, parentUrl })
    }
  }
  
  nextDiscovery() {
    if (this.currentLevel.length === 0) {
      // Level complete, advance
      this.currentLevel = this.nextLevel
      this.nextLevel = []
      this.currentDepth++
      emit('DISCOVERY:LEVEL_COMPLETE', { level: this.currentDepth - 1 })
    }
    
    return this.currentLevel.shift() // FIFO
  }
  
  isDiscoveryComplete() {
    return this.currentLevel.length === 0 && this.nextLevel.length === 0
  }
}
```

**Key Invariants**:
- All URLs in `currentLevel` have same depth
- No URL appears in both `currentLevel` and `nextLevel`
- `visitedUrls` is superset of all queued URLs
- Level transition happens atomically

**Execution Mode** (Dependency-Aware):

After pruning, the queue switches to execution mode where ordering constraints matter:

```javascript
buildDownloadQueue(canonicalContexts, dependencyGraph) {
  // canonicalContexts = output of ConflictResolver
  // dependencyGraph = Map<parentId, childIds[]>
  
  this.downloadQueue = []
  this.pendingChildrenCount = new Map()
  
  // Initialize dependency counts
  for (const [parentId, childIds] of dependencyGraph) {
    this.pendingChildrenCount.set(parentId, childIds.length)
  }
  
  // Seed queue with leaf nodes (no children)
  const leafNodes = canonicalContexts.filter(ctx => {
    const childCount = this.pendingChildrenCount.get(ctx.id) || 0
    return childCount === 0
  })
  
  this.downloadQueue.push(...leafNodes)
}

nextDownload() {
  const context = this.downloadQueue.shift()
  if (!context) return null
  
  // Calculate absolute path in Master process (prevents Ghost Execution)
  const relativePath = context.getRelativePath()
  const absolutePath = this._calculateAbsolutePath(relativePath)
  
  return {
    url: context.url,
    pageId: context.id,
    savePath: absolutePath,  // Absolute path sent to worker
    depth: context.depth,
    parentId: context.parentId
  }
}

_calculateAbsolutePath(relativePath) {
  // Resolve relative path against output directory
  const absoluteOutputDir = path.resolve(process.cwd(), this.config.outputDir)
  const absolutePath = path.resolve(absoluteOutputDir, relativePath)
  return absolutePath
}

markDownloadComplete(pageId) {
  const context = this.urlToContext.get(pageId)
  const parentId = context.parentId
  
  if (parentId) {
    // Decrement parent's pending count
    const currentCount = this.pendingChildrenCount.get(parentId)
    const newCount = currentCount - 1
    this.pendingChildrenCount.set(parentId, newCount)
    
    // If parent now ready, add to queue
    if (newCount === 0) {
      const parentContext = this.urlToContext.get(parentId)
      this.downloadQueue.push(parentContext)
    }
  }
}

isDownloadComplete() {
  return this.downloadQueue.length === 0 && 
         Array.from(this.pendingChildrenCount.values()).every(count => count === 0)
}
```

**Scheduling Guarantees**:
1. **Leaf-First**: Pages with no children download first
2. **Bottom-Up**: Parent only downloads after ALL children complete
3. **No Deadlock**: Acyclic graph (enforced by ConflictResolver) prevents circular dependencies
4. **Progress**: At least one page is always ready (unless complete)

**Statistics and Monitoring**:

```javascript
getStatistics() {
  return {
    discovered: this.visitedUrls.size,
    currentLevelRemaining: this.currentLevel.length,
    nextLevelQueued: this.nextLevel.length,
    downloadRemaining: this.downloadQueue.length,
    pendingParents: Array.from(this.pendingChildrenCount.values())
                         .filter(count => count > 0).length
  }
}
```

**Source**: [`src/orchestration/GlobalQueueManager.js`](../src/orchestration/GlobalQueueManager.js)

---

#### 5.3 `ConflictResolver.js` - Graph Pruning (NEW)

**Runtime Context**: Master Process Only (runs between DISCOVERY and EXECUTION)  
**Design Pattern**: Strategy + Filter  
**Responsibility**: Duplicate detection and canonical instance selection

**Behavioral Description**:

ConflictResolver solves the critical problem: **What if the same Notion page is linked from multiple parents?**

During BFS discovery, the same page may be discovered multiple times through different paths:
```
Root
├── Projects
│   └── WebApp  (first discovery - depth 2)
└── Archive
    └── WebApp  (second discovery - depth 3, SAME page!)
```

**The Problem**:
- Naive approach: Download WebApp twice, waste resources
- Graph issue: WebApp has TWO parents in the discovery tree
- Link rewriting ambiguity: Which path should internal links use?

**The Solution - Canonical Instance Selection**:

```javascript
class ConflictResolver {
  static prune(rootContext) {
    const urlToContexts = new Map() // url → [contexts that have this URL]
    
    // 1. Collect all contexts
    this.collectAllContexts(rootContext, urlToContexts)
    
    // 2. For each unique URL, select canonical instance
    const canonicalMap = new Map() // url → canonical PageContext
    const duplicates = []
    
    for (const [url, contexts] of urlToContexts) {
      if (contexts.length === 1) {
        canonicalMap.set(url, contexts[0])
      } else {
        // Multiple instances - apply selection criteria
        const canonical = this.selectCanonical(contexts)
        canonicalMap.set(url, canonical)
        duplicates.push(...contexts.filter(c => c !== canonical))
      }
    }
    
    // 3. Build URL → Local Path mapping
    const linkMap = new Map()
    for (const [url, context] of canonicalMap) {
      linkMap.set(url, context.getRelativePath())
    }
    
    return {
      canonicalContexts: Array.from(canonicalMap.values()),
      linkMap: linkMap,
      duplicatesRemoved: duplicates.length
    }
  }
}
```

**Canonical Selection Criteria** (in priority order):

```javascript
selectCanonical(contexts) {
  // 1. Prefer shallower depth (closer to root)
  const minDepth = Math.min(...contexts.map(c => c.depth))
  let candidates = contexts.filter(c => c.depth === minDepth)
  
  if (candidates.length === 1) return candidates[0]
  
  // 2. Prefer context discovered first (BFS order)
  candidates.sort((a, b) => a.timestamp - b.timestamp)
  let preferred = candidates[0]
  
  // 3. Prefer better hierarchical context (has section/subsection)
  const withMetadata = candidates.filter(c => c.section || c.subsection)
  if (withMetadata.length > 0) {
    preferred = withMetadata[0]
  }
  
  return preferred
}
```

**Rationale**:
- **Depth**: Shallower = more prominent in hierarchy = better canonical choice
- **Discovery Order**: BFS guarantees breadth-first, so earlier = more direct path
- **Metadata**: Rich context (sections) indicates better placement

**Link Map Generation**:

The link map is critical for the Execution phase:

```javascript
// Discovery found WebApp at two locations:
// /Projects/WebApp (depth 2, canonical)
// /Archive/WebApp (depth 3, duplicate)

linkMap = {
  'https://notion.so/webapp-abc123': 'Projects/WebApp/index.html'
}

// During Execution, when ANY page links to WebApp:
// <a href="https://notion.so/webapp-abc123"> 
// is rewritten to:
// <a href="../../Projects/WebApp/index.html">
// regardless of which discovery instance triggered the link
```

**Dependency Graph Adjustment**:

After pruning, update the dependency graph to reflect only canonical instances:

```javascript
adjustDependencyGraph(canonicalContexts) {
  const graph = new Map()
  
  for (const context of canonicalContexts) {
    // Only include children that are also canonical
    const canonicalChildren = context.childIds
      .map(id => findContextById(id, canonicalContexts))
      .filter(child => child !== null)
    
    if (canonicalChildren.length > 0) {
      graph.set(context.id, canonicalChildren.map(c => c.id))
    }
  }
  
  return graph
}
```

**Source**: [`src/orchestration/analysis/ConflictResolver.js`](../src/orchestration/analysis/ConflictResolver.js)

---

#### 5.4 `GraphAnalyzer.js` - Edge Classification

**Runtime Context**: Master Process (during DISCOVERY)  
**Design Pattern**: Visitor + Classifier  
**Responsibility**: BFS edge classification for cycle detection and graph analysis

**Behavioral Description**:

GraphAnalyzer classifies every edge discovered during BFS traversal into one of four categories, providing insights into the page graph structure.

**Edge Types and Semantics**:

1. **Tree Edge** (Green): First discovery, becomes parent-child relationship
   - Condition: `targetUrl` not in `visitedUrls`
   - Action: Create new PageContext, add to BFS queue
   - Meaning: Legitimate hierarchical relationship

2. **Back Edge** (Red): Points to ancestor (creates cycle)
   - Condition: `targetUrl` in `visitedUrls` AND is ancestor of `fromContext`
   - Action: Log warning, do NOT queue (would create infinite loop)
   - Meaning: Child linking back to parent/grandparent

3. **Forward Edge** (Blue): Skips levels to descendant
   - Condition: `targetUrl` in `visitedUrls` AND is descendant of `fromContext`
   - Action: Log info, do NOT queue (already discovered)
   - Meaning: Shortcut link (e.g., grandparent → grandchild)

4. **Cross Edge** (Yellow): Between sibling branches or same level
   - Condition: `targetUrl` in `visitedUrls` AND neither ancestor nor descendant
   - Action: Log info, do NOT queue (already discovered)
   - Meaning: Inter-branch reference

**Classification Algorithm**:

```javascript
classifyEdge(fromContext, toUrl, toContext) {
  // Tree edge: first discovery
  if (!toContext) {
    this.treeEdges.push({ from: fromContext.url, to: toUrl })
    return 'tree'
  }
  
  // Check ancestral relationship
  if (this.isAncestor(fromContext, toUrl)) {
    this.backEdges.push({ from: fromContext.url, to: toUrl })
    return 'back' // Cycle detected!
  }
  
  if (this.isDescendant(fromContext, toUrl)) {
    this.forwardEdges.push({ from: fromContext.url, to: toUrl })
    return 'forward' // Shortcut
  }
  
  // Neither ancestor nor descendant = cross edge
  this.crossEdges.push({ from: fromContext.url, to: toUrl })
  return 'cross'
}

isAncestor(context, targetUrl) {
  let current = context
  while (current) {
    if (current.url === targetUrl) return true
    current = this.getParentContext(current)
  }
  return false
}
```

**Cycle Detection**:

Back edges indicate cycles in the page graph:
```
A → B → C → A  (back edge C → A creates cycle)
```

**Logging and Analysis**:
```javascript
logFinalStatistics() {
  logger.info('GRAPH', `Edge Classification:`)
  logger.info('GRAPH', `  Tree edges: ${this.treeEdges.length} (parent-child)`)
  logger.info('GRAPH', `  Back edges: ${this.backEdges.length} (cycles)`)
  logger.info('GRAPH', `  Forward edges: ${this.forwardEdges.length} (shortcuts)`)
  logger.info('GRAPH', `  Cross edges: ${this.crossEdges.length} (inter-branch)`)
  
  if (this.backEdges.length > 0) {
    logger.warn('GRAPH', `⚠️  Detected ${this.backEdges.length} cycles in page graph`)
    this.backEdges.forEach(edge => {
      logger.warn('GRAPH', `  ${edge.from} → ${edge.to}`)
    })
  }
}
```

**Source**: [`src/orchestration/analysis/GraphAnalyzer.js`](../src/orchestration/analysis/GraphAnalyzer.js)

---
**Source**: [`src/orchestration/analysis/GraphAnalyzer.js`](../src/orchestration/analysis/GraphAnalyzer.js)

---

### 6. Scraping, Processing & Extraction Packages - Content Handling

These packages run in the **Worker Process** context and handle the actual interaction with Notion pages, content manipulation, and data extraction.

#### 6.1 `PageProcessor.js` - Scraping Coordination (REFACTORED)

**Runtime Context**: Worker Process Only  
**Design Pattern**: Facade + Coordinator  
**Responsibility**: Orchestrate page scraping workflow, coordinate multiple processing components

**Behavioral Description**:

PageProcessor is the **central coordinator** within a worker. It sequences the scraping workflow: navigate → handle UI → expand content → extract data → download assets → save.

**Key Refactoring for Statelessness**:

**Before** (Monolithic):
```javascript
class PageProcessor {
  constructor() {
    this.visitedUrls = new Set() // ❌ Global state
    this.urlToContextMap = new Map() // ❌ Shared state
  }
}
```

**After** (Stateless):
```javascript
class PageProcessor {
  constructor(config, logger, cookieHandler, contentExpander, 
              linkExtractor, assetDownloader, fileDownloader) {
    // ✅ Only dependencies, no state
    // All context passed via method parameters
  }
  
  async scrapePage(page, pageContext, cookies, isFirstPage) {
    // ✅ Everything needed is in parameters
  }
}
```

**Discovery Workflow** (Lightweight):

```javascript
async discoverPageInfo(page, url, isFirstPage, cookies) {
  // 1. Apply cookies (if provided)
  if (cookies && cookies.length > 0) {
    await page.setCookie(...cookies)
  }
  
  // 2. Navigate (fast - domcontentloaded only)
  await page.goto(url, { 
    waitUntil: 'domcontentloaded',  // Don't wait for images/CSS
    timeout: 30000 
  })
  
  // 3. Handle cookie consent (if first page)
  if (isFirstPage) {
    await this.cookieHandler.ensureConsent(page, 'DISCOVERY')
  }
  
  // 4. Extract title
  const title = await page.evaluate(() => {
    // Try multiple selectors
    const h1 = document.querySelector('h1')
    const title = document.querySelector('[data-content-editable-root="true"]')
    return h1?.textContent || title?.textContent || document.title
  })
  
  // 5. Extract links (no expansion, surface-level only)
  const links = await this.linkExtractor.extractLinks(page, url)
  
  // 6. Capture cookies (if first page)
  let sessionCookies = null
  if (isFirstPage) {
    sessionCookies = await page.cookies()
  }
  
  return { title, links, cookies: sessionCookies }
}
```

**Performance**: ~1-2 seconds per page

**Download Workflow** (Full Scraping):

```javascript
async scrapePage(page, pageContext, cookies, linkMap) {
  // 1. Apply cookies
  await page.setCookie(...cookies)
  
  // 2. Navigate (full load - networkidle0)
  await page.goto(pageContext.url, {
    waitUntil: 'networkidle0',  // Wait for all network activity
    timeout: 60000
  })
  
  // 3. Handle cookie consent (idempotent)
  await this.cookieHandler.ensureConsent(page, pageContext.title)
  
  // 4. Expand all content
  await this.contentExpander.expandAll(page)
  
  // 5. Download images and rewrite img tags
  const imageCount = await this.assetDownloader.downloadAndRewriteImages(
    page, 
    pageContext.getDirectoryPath(config.outputDir)
  )
  
  // 6. Download embedded files
  const fileCount = await this.fileDownloader.downloadAndRewriteFiles(
    page,
    pageContext.getDirectoryPath(config.outputDir)
  )
  
  // 7. Save HTML
  const htmlPath = await this.savePageHtml(page, pageContext)
  
  // 8. Rewrite links (post-processing)
  const linksRewritten = await this.rewriteLinksInSavedHtml(
    htmlPath,
    pageContext,
    linkMap
  )
  
  return {
    savedPath: htmlPath,
    assetCount: imageCount + fileCount,
    linksRewritten: linksRewritten
  }
}
```

**Performance**: ~10-30 seconds per page (depends on content volume)

**HTML Saving Strategy**:

```javascript
async savePageHtml(page, pageContext) {
  // Get full HTML (after all manipulations)
  const html = await page.content()
  
  // Ensure directory exists
  const dir = pageContext.getDirectoryPath(config.outputDir)
  await fs.mkdir(dir, { recursive: true })
  
  // Save to disk
  const filePath = pageContext.getFilePath(config.outputDir)
  await fs.writeFile(filePath, html, 'utf-8')
  
  return filePath
}
```

**Critical HTML Over IPC Issue** (Avoided):

The refactoring plan identified a major anti-pattern: Workers should NOT send HTML content to Master via IPC.

**Why**: JSON serialization of 2MB HTML strings blocks event loop and bloats Master memory.

**Solution**: Workers write directly to disk, only return file path to Master.

**Source**: [`src/scraping/PageProcessor.js`](../src/scraping/PageProcessor.js)

---

#### 6.2 `ContentExpander.js` - Interactive Content Expansion

**Runtime Context**: Worker Process Only  
**Design Pattern**: Strategy + Iterator  
**Responsibility**: Reveal all hidden/collapsible content on Notion pages

**Behavioral Description**:

Notion pages heavily use progressive disclosure (toggles, accordions, lazy-loaded databases). ContentExpander ensures ALL content is visible before scraping.

**Expansion Strategy** (Multi-Pass Iterative):

```javascript
async expandAll(page) {
  // 1. Scroll to bottom (triggers lazy-loading)
  await this.scrollToBottom(page)
  await page.waitForTimeout(1000) // Allow lazy content to load
  
  // 2. Iterative expansion (nested toggles require multiple passes)
  let previousHeight = 0
  let iterationCount = 0
  const maxIterations = 10
  
  while (iterationCount < maxIterations) {
    // Expand all expandable elements
    const expandedCount = await this.expandToggles(page)
    
    // Check if page height changed (new content revealed)
    const currentHeight = await page.evaluate(() => document.body.scrollHeight)
    
    if (expandedCount === 0 || currentHeight === previousHeight) {
      break // No more content to expand
    }
    
    previousHeight = currentHeight
    iterationCount++
    await page.waitForTimeout(500) // Allow animations/rendering
  }
  
  logger.debug('EXPAND', `Expansion complete after ${iterationCount} iterations`)
}
```

**Toggle Expansion Logic**:

```javascript
async expandToggles(page) {
  return await page.evaluate(() => {
    let expandedCount = 0
    
    // Selector priority (most specific to most general)
    const selectors = [
      '[aria-expanded="false"]',           // ARIA-compliant toggles
      '.notion-toggle-block:not(.open)',   // Notion-specific toggles
      'button[aria-label*="Expand"]',      // Explicit expand buttons
      'details:not([open])',               // HTML5 details/summary
      '.accordion:not(.expanded)'          // Generic accordions
    ]
    
    for (const selector of selectors) {
      const elements = document.querySelectorAll(selector)
      
      for (const el of elements) {
        // Safety check: Don't click destructive actions
        const text = el.textContent.toLowerCase()
        if (text.includes('delete') || text.includes('remove')) {
          continue
        }
        
        // Click to expand
        try {
          el.click()
          expandedCount++
        } catch (e) {
          // Element not clickable, skip
        }
      }
    }
    
    return expandedCount
  })
}
```

**Scroll Strategy** (Lazy-Loading Trigger):

```javascript
async scrollToBottom(page) {
  await page.evaluate(async () => {
    await new Promise((resolve) => {
      let totalHeight = 0
      const distance = 100 // Pixels per scroll
      const timer = setInterval(() => {
        window.scrollBy(0, distance)
        totalHeight += distance
        
        if (totalHeight >= document.body.scrollHeight) {
          clearInterval(timer)
          resolve()
        }
      }, 100) // 100ms between scrolls
    })
  })
}
```

**Why Iterative?**

Notion uses nested toggles:
```
Toggle A (collapsed)
  └─ Content
      └─ Toggle B (collapsed)
          └─ More content
```

- First pass: Expands A, reveals B
- Second pass: Expands B, reveals more content

**Source**: [`src/processing/ContentExpander.js`](../src/processing/ContentExpander.js)

---

#### 6.3 `CookieHandler.js` - Consent Banner Management

**Runtime Context**: Worker Process Only  
**Design Pattern**: Retry + Idempotent Operation  
**Responsibility**: Handle cookie consent banners without blocking or duplicate attempts

**Behavioral Description**:

Cookie consent banners must be handled exactly once per page load, with intelligent retry and deduplication.

**Deduplication Strategy** (WeakSet):

```javascript
class CookieHandler {
  constructor(config, logger) {
    this.config = config
    this.logger = logger
    this.processedPages = new WeakSet() // ✅ Auto garbage-collected
  }
  
  async ensureConsent(page, label) {
    // Check if already processed
    if (this.processedPages.has(page)) {
      logger.debug('COOKIE', `Already handled for ${label}`)
      return false
    }
    
    // Attempt to handle
    const handled = await this.attemptHandle(page)
    
    // Mark as processed (even if banner not found)
    this.processedPages.add(page)
    
    return handled
  }
}
```

**Why WeakSet?**
- Prevents duplicate consent attempts on same Puppeteer page object
- Automatically garbage-collected when page is closed (no memory leak)
- Does NOT prevent consent on new page loads (desired behavior)

**Banner Detection and Interaction**:

```javascript
async attemptHandle(page) {
  const maxRetries = 3
  const retryDelay = 1000
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      // Wait for banner (with timeout)
      const bannerFound = await page.waitForSelector(
        'button[id*="cookie"], button[id*="consent"], #onetrust-accept-btn-handler',
        { timeout: 5000 }
      ).catch(() => null)
      
      if (!bannerFound) {
        logger.debug('COOKIE', 'No consent banner found')
        return false
      }
      
      // Click consent button
      await Promise.all([
        page.click('button[id*="cookie"], button[id*="consent"]'),
        page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 5000 })
          .catch(() => {}) // Navigation not always triggered
      ])
      
      logger.success('COOKIE', `Consent handled (attempt ${attempt})`)
      return true
      
    } catch (error) {
      if (attempt < maxRetries) {
        logger.warn('COOKIE', `Attempt ${attempt} failed, retrying...`)
        await page.waitForTimeout(retryDelay)
      } else {
        logger.warn('COOKIE', `Failed after ${maxRetries} attempts`)
        return false
      }
    }
  }
}
```

**Exponential Backoff** (Optional Enhancement):
```javascript
const delay = retryDelay * Math.pow(2, attempt - 1)
await page.waitForTimeout(delay)
```

**Source**: [`src/processing/CookieHandler.js`](../src/processing/CookieHandler.js)

---

#### 6.4 `LinkExtractor.js` - Hierarchical Link Discovery

**Runtime Context**: Worker Process Only  
**Design Pattern**: Extractor + Filter + Enricher  
**Responsibility**: Extract internal links with hierarchical context metadata

**Behavioral Description**:

LinkExtractor finds all Notion page links and enriches them with contextual information (sections, subsections) to aid hierarchy construction.

**Extraction Strategy**:

```javascript
async extractLinks(page, currentUrl, baseUrl) {
  return await page.evaluate((currentUrl, baseUrl) => {
    const links = []
    const seen = new Set()
    
    // Find all anchor tags
    const anchors = document.querySelectorAll('a[href]')
    
    for (const anchor of anchors) {
      const href = anchor.getAttribute('href')
      
      // Resolve to absolute URL
      const absoluteUrl = new URL(href, baseUrl).href
      
      // Filter: Only internal Notion links
      if (!absoluteUrl.includes('notion.so') && 
          !absoluteUrl.includes('notion.site')) {
        continue
      }
      
      // Filter: No self-references
      if (absoluteUrl === currentUrl) {
        continue
      }
      
      // Deduplicate
      if (seen.has(absoluteUrl)) {
        continue
      }
      seen.add(absoluteUrl)
      
      // Extract metadata
      const linkInfo = {
        url: absoluteUrl,
        title: anchor.textContent.trim() || 'Untitled',
        section: this.findSection(anchor),
        subsection: this.findSubsection(anchor)
      }
      
      links.push(linkInfo)
    }
    
    return links
  }, currentUrl, baseUrl)
}
```

**Hierarchical Context Extraction**:

```javascript
// Executed in browser context
findSection(anchorElement) {
  // Traverse up DOM to find section header
  let current = anchorElement.parentElement
  
  while (current) {
    // Check for Notion section markers
    if (current.classList.contains('notion-header-block') ||
        current.tagName === 'H1' || current.tagName === 'H2') {
      return current.textContent.trim()
    }
    current = current.parentElement
  }
  
  return null
}

findSubsection(anchorElement) {
  // Similar logic for H3, H4
  let current = anchorElement.parentElement
  
  while (current) {
    if (current.tagName === 'H3' || current.tagName === 'H4') {
      return current.textContent.trim()
    }
    current = current.parentElement
  }
  
  return null
}
```

**Why Hierarchical Metadata?**

Enables smarter PageContext construction:
```javascript
// Link discovered under "Documentation > API Reference"
const linkInfo = {
  url: 'https://notion.so/endpoint-abc',
  title: 'GET /users',
  section: 'Documentation',
  subsection: 'API Reference'
}

// Used to set PageContext metadata
pageContext.setSection('Documentation')
pageContext.setSubsection('API Reference')
```

**Source**: [`src/extraction/LinkExtractor.js`](../src/extraction/LinkExtractor.js)

---

#### 6.5 `LinkRewriter.js` - Offline Link Transformation (REFACTORED)

**Runtime Context**: Worker Process Only  
**Design Pattern**: Transformer + Post-Processor  
**Responsibility**: Convert online URLs to offline-compatible relative paths

**Behavioral Description**:

LinkRewriter is invoked AFTER HTML is saved to disk. It loads the HTML, rewrites links, and saves back.

**Refactored Behavior** (Hot Rewriting):

**Before**: Separate post-processing phase  
**After**: Integrated into worker's download task

```javascript
async rewriteLinksInFile(htmlPath, pageContext, linkMap) {
  // 1. Load saved HTML
  const html = await fs.readFile(htmlPath, 'utf-8')
  const dom = new JSDOM(html)
  const document = dom.window.document
  
  let rewriteCount = 0
  
  // 2. Find all internal links
  const anchors = document.querySelectorAll('a[href]')
  
  for (const anchor of anchors) {
    const href = anchor.getAttribute('href')
    const absoluteUrl = this.resolveUrl(href, pageContext.url)
    
    // Check if this URL has a local mapping
    if (linkMap.has(absoluteUrl)) {
      const targetRelativePath = linkMap.get(absoluteUrl)
      
      // Calculate relative path from current page to target
      const relativePath = pageContext.getRelativePathTo(targetRelativePath)
      
      // Rewrite
      anchor.setAttribute('href', relativePath)
      rewriteCount++
    }
  }
  
  // 3. Download and localize CSS
  await this.cssDownloader.downloadAndRewriteCss(
    dom,
    path.dirname(htmlPath),
    pageContext.url
  )
  
  // 4. Save modified HTML
  await fs.writeFile(htmlPath, dom.serialize(), 'utf-8')
  
  return rewriteCount
}
```

**Link Map Usage**:

The `linkMap` (from ConflictResolver) ensures consistent rewriting:
```javascript
linkMap = {
  'https://notion.so/page-abc': 'Projects/WebApp/index.html',
  'https://notion.so/page-xyz': 'Docs/API/index.html'
}

// From any page, links point to canonical instances
```

**Source**: [`src/processing/LinkRewriter.js`](../src/processing/LinkRewriter.js)

---

### 7. Download Package (`src/download`) - Asset Management

#### 7.1 `AssetDownloader.js` - Image and Media Download

**Runtime Context**: Worker Process Only  
**Design Pattern**: Downloader + Cache + Hasher  
**Responsibility**: Download images/media with deduplication and retry logic

**Key Features**:

1. **Content-Based Hashing**: Same image from different URLs gets same filename
   ```javascript
   const hash = crypto.createHash('md5').update(imageBuffer).digest('hex')
   const filename = `${hash}.${extension}`
   ```

2. **Download Cache**: Prevents re-downloading same asset
   ```javascript
   if (this.downloadCache.has(assetUrl)) {
     return this.downloadCache.get(assetUrl) // Return cached path
   }
   ```

3. **Retry Logic**: Exponential backoff for transient failures
   ```javascript
   for (let attempt = 1; attempt <= maxRetries; attempt++) {
     try {
       return await this.fetchAsset(url)
     } catch (error) {
       if (attempt < maxRetries && isRetryable(error)) {
         await sleep(1000 * Math.pow(2, attempt))
       }
     }
   }
   ```

**Source**: [`src/download/AssetDownloader.js`](../src/download/AssetDownloader.js)

---

#### 7.2 `CssDownloader.js` - Stylesheet Localization (REFACTORED)

**Runtime Context**: Worker Process Only  
**Design Pattern**: Facade + Adapter + Coordinator  
**Responsibility**: Coordinate CSS downloading with dual-mode operation for Puppeteer and JSDOM contexts

**Critical Refactoring**: The CssDownloader now supports two distinct operational modes to handle different scraping contexts:

**Mode 1: Active Scraping with Puppeteer** (`downloadFromPuppeteer`):
- **Use Case**: During live page scraping when a Puppeteer Page object is available
- **Three-Phase Approach**:
  1. **Extract CSS Links (Browser Context)**: Use `page.evaluate()` to get stylesheet hrefs from live DOM
  2. **Download & Process (Node.js Context)**: Fetch CSS content, parse with CssContentProcessor, handle @import chains and url() assets
  3. **Rewrite DOM (Browser Context)**: Update `<link>` elements with local paths using `page.evaluate()`
- **Method**: `async downloadFromPuppeteer(page, outputDir)`
- **Integration**: Properly integrates with CssContentProcessor using options object with baseUrl, cssDir, assetDir, referenceContext, caches, and bound callback methods

**Mode 2: Post-Processing with JSDOM** (`downloadAndRewriteCss`):
- **Use Case**: Post-scraping HTML manipulation when working with saved HTML documents
- **Context**: JSDOM-based DOM manipulation, no live browser
- **Method**: `async downloadAndRewriteCss(dom, outputDir, baseUrl)`
- **Note**: Legacy method, not used during active scraping phase

**Why Dual Modes?**
- **Interface Mismatch**: Puppeteer Page and JSDOM Document have different APIs
- **Context Separation**: DOM operations require browser context, CSS processing requires Node.js context
- **Performance**: Three-phase approach minimizes context switches between browser and Node.js

**Delegates to**:
- `CssContentProcessor`: Parse CSS, rewrite @import and url() references
- `CssAssetDownloader`: Download CSS-referenced assets (fonts, images) with retry logic

**Statistics**:
```javascript
getStats() {
  return {
    stylesheetsDownloaded: this.cssCount,
    totalCssAssets: this.assetDownloader.assetCount
  }
}
```

**Source**: [`src/download/CssDownloader.js`](../src/download/CssDownloader.js), [`src/download/css/CssContentProcessor.js`](../src/download/css/CssContentProcessor.js), [`src/download/css/CssAssetDownloader.js`](../src/download/css/CssAssetDownloader.js)

---

#### 7.3 Worker Pipeline Architecture (NEW)

**Runtime Context**: Worker Process Only  
**Design Pattern**: Pipeline + Chain of Responsibility  
**Responsibility**: Sequential execution of download steps with error isolation

**Architectural Problem Solved**:

The original `TaskRunner._executeDownload()` was a 27-line stub that returned fake success without writing files ("Ghost Execution" failure). The refactoring introduced a **Pipeline Pattern** to enforce proper sequencing and truthful result reporting.

**Pipeline Components**:

**Base Class** (`PipelineStep.js`):
```javascript
class PipelineStep {
  constructor(stepName) {
    this.stepName = stepName
  }
  
  async process(context) {
    throw new Error('Subclasses must implement process()')
  }
}
```

**Pipeline Controller** (`ScrapingPipeline.js`):
```javascript
class ScrapingPipeline {
  constructor(steps, logger) {
    this.steps = steps
    this.logger = logger
  }
  
  async execute(context) {
    const startTime = Date.now()
    
    for (const step of this.steps) {
      const stepStart = Date.now()
      try {
        await step.process(context)
        this.logger.debug('PIPELINE', `${step.stepName} completed in ${Date.now() - stepStart}ms`)
      } catch (error) {
        this.logger.error('PIPELINE', `${step.stepName} failed`, error)
        throw error
      }
    }
    
    this.logger.info('PIPELINE', `Pipeline completed in ${Date.now() - startTime}ms`)
  }
}
```

**Pipeline Steps** (Executed in Order):

1. **NavigationStep**: Navigate to page URL with cookies
   - Applies cookies from payload
   - Navigates to `payload.url` with `networkidle0` wait
   - Handles cookie consent if first page

2. **ExpansionStep**: Expand all collapsible content
   - Uses ContentExpander to reveal toggles, accordions, lazy-loaded content
   - Ensures complete content visibility before extraction

3. **AssetDownloadStep**: Download all page assets
   - Downloads images via `AssetDownloader`
   - Downloads CSS via `CssDownloader.downloadFromPuppeteer()` (Puppeteer-compatible)
   - Downloads files via `FileDownloader`
   - Soft failure: Returns 0 counts on error instead of throwing

4. **LinkRewriterStep**: Rewrite internal links to relative paths
   - Rewrites `<a>` tags to use local filesystem paths
   - Uses link map from ConflictResolver for canonical URLs

5. **HtmlWriteStep**: Save final HTML to disk
   - Extracts HTML via `page.content()`
   - Writes to absolute path from `payload.savePath`
   - Uses WorkerFileSystem for safe I/O

**Pipeline Context Structure**:
```javascript
{
  browser: Browser,              // Puppeteer browser instance
  page: Page,                    // Puppeteer page instance
  config: Config,                // Global configuration
  logger: Logger,                // Logging instance
  payload: DownloadPayload,      // IPC payload with absolute paths
  fileSystem: WorkerFileSystem,  // I/O abstraction
  stats: {                       // Accumulated statistics
    assetsDownloaded: 0,
    linksRewritten: 0
  },
  downloadedAssets: []           // Asset metadata
}
```

**Error Handling Philosophy**:
- **Critical Steps** (Navigation, HTML Write): Throw on error, abort pipeline
- **Asset Steps**: Log error but continue (soft failure)
- **Step Isolation**: Each step wrapped in try-catch, error logged with step name

**Benefits of Pipeline Pattern**:
1. **Separation of Concerns**: Each step has single responsibility
2. **Testability**: Steps can be tested independently
3. **Error Traceability**: Failures pinpointed to specific step
4. **Extensibility**: New steps added without modifying existing code
5. **Truthful Reporting**: Stats accumulated from actual operations

**Source**: [`src/worker/pipeline/PipelineStep.js`](../src/worker/pipeline/PipelineStep.js), [`src/worker/pipeline/ScrapingPipeline.js`](../src/worker/pipeline/ScrapingPipeline.js), [`src/worker/pipeline/steps/`](../src/worker/pipeline/steps/)

---

#### 7.4 WorkerFileSystem - Absolute Path Enforcement (NEW)

**Runtime Context**: Worker Process Only  
**Design Pattern**: Adapter + Validator  
**Responsibility**: Safe file I/O with absolute path validation

**Critical Problem Solved**:

The "Ghost Execution" failure occurred because workers used relative paths that resolved differently in child processes compared to the master process. This adapter enforces absolute path usage at I/O boundaries.

**Key Methods**:

```javascript
class WorkerFileSystem {
  async safeWrite(filePath, content) {
    // Validate absolute path
    if (!path.isAbsolute(filePath)) {
      throw new Error(
        `WorkerFileSystem.safeWrite requires absolute path. Received: ${filePath}`
      )
    }
    
    // Ensure directory exists
    await this.ensureDir(path.dirname(filePath))
    
    // Write with explicit logging
    await fs.writeFile(filePath, content, 'utf-8')
    this.logger.debug('WRITE', `File written: ${filePath} (${content.length} bytes)`)
  }
  
  async ensureDir(dirPath) {
    if (!path.isAbsolute(dirPath)) {
      throw new Error(
        `WorkerFileSystem.ensureDir requires absolute path. Received: ${dirPath}`
      )
    }
    
    await fs.mkdir(dirPath, { recursive: true })
    this.logger.debug('DIR', `Directory ensured: ${dirPath}`)
  }
}
```

**Usage in HtmlWriteStep**:
```javascript
class HtmlWriteStep extends PipelineStep {
  async process(context) {
    const { page, payload, fileSystem, logger } = context
    
    // Extract HTML
    const html = await page.content()
    
    // Write using safe abstraction (validates absolute path)
    await fileSystem.safeWrite(payload.savePath, html)
    
    logger.success('WRITE', `HTML saved: ${payload.savePath}`)
  }
}
```

**Guarantees**:
- All file writes use absolute paths calculated in Master process
- Workers cannot write to ambiguous relative paths
- Explicit logging for audit trail
- Early validation prevents "silent success" bugs

**Source**: [`src/worker/io/WorkerFileSystem.js`](../src/worker/io/WorkerFileSystem.js)

---

###
*   **Constructor**: `new GraphAnalyzer(logger)`
*   **Edge Types**:
    *   **Tree edges**: Parent → Child (first discovery)
    *   **Back edges**: To ancestor or self-loop (creates cycles)
    *   **Forward edges**: Skip levels to descendant (shortcuts)
    *   **Cross edges**: Between branches or same level
*   **Methods**:
    *   `classifyEdge(fromContext, toContext, toUrl, linkTitle)`: Classify an edge when encountering an already-discovered node.
        *   **Returns**: `'tree'|'back'|'forward'|'cross'` (Edge type).
    *   `addTreeEdge(fromUrl, toUrl, title)`: Add a tree edge to classification.
    *   `logEdgeStatistics(level)`: Log edge classification statistics for current BFS level.
    *   `logFinalEdgeStatistics()`: Log final edge classification summary with cycle detection.
    *   `resetEdgeClassification()`: Reset all edge classification arrays.
    *   `getStats()`: Get current edge statistics.
        *   **Returns**: `{tree: number, back: number, forward: number, cross: number}` (Edge counts).
    *   `_isAncestor(fromContext, targetUrl)`: Check if targetUrl is an ancestor of fromContext. *(Private)*

### 4. Scraping Package (`src/scraping`)

Handles the direct interaction with Puppeteer for page navigation and saving.

#### `PageProcessor.js`
Handles the scraping of individual Notion pages. Coordinates navigation, content expansion, link extraction, and asset downloads.
*   **Constructor**: `new PageProcessor(config, logger, cookieHandler, contentExpander, linkExtractor, assetDownloader, fileDownloader)`
*   **State Management**: 
    *   Maintains `visitedUrls` Set to prevent duplicate processing
    *   Manages `urlToContextMap` for link rewriting coordination
*   **Methods**:
    *   `scrapePage(page, pageContext, isFirstPage)`: Scrape a single page and return its links.
        *   `page`: Puppeteer page instance.
        *   `pageContext`: Context of the page to scrape.
        *   `isFirstPage`: Whether this is the first page (for cookie handling).
        *   **Returns**: `Promise<Array<Object>>` (List of discovered link objects).
        *   **Workflow**: Navigate → Handle cookies → Expand content → Extract links → Download assets → Save HTML.
    *   `discoverPageInfo(page, url, isFirstPage)`: Lightweight metadata fetch used during discovery.
        *   `page`: Puppeteer page instance.
        *   `url`: URL to discover.
        *   `isFirstPage`: Whether this is the first page.
        *   **Returns**: `Promise<{title: string, links: Array<Object>}>` (Page metadata).
        *   **Note**: Uses 'domcontentloaded' for speed, skips heavy downloads.
    *   `registerPageContext(url, context)`: Register a page context for link rewriting.
    *   `getContextMap()`: Get the map of registered page contexts.
        *   **Returns**: `Map<string, PageContext>` (URL → PageContext mapping).
    *   `isUrlRegistered(url)`: Check if a URL is already registered.
    *   `resetVisited()`: Reset visited state before fresh execution phase.
    *   `resetContextMap()`: Reset the page context map prior to discovery.
    *   `_savePageHtml(page, pageContext)`: Save page HTML to disk. *(Private)*

### 5. Processing Package (`src/processing`)

Manipulates page content and handles specific page interactions.

#### `LinkRewriter.js`
Handles offline link rewriting and CSS localization for scraped pages. Transforms online-dependent content to self-contained offline structure.
*   **Constructor**: `new LinkRewriter(config, logger, cssDownloader)`
*   **Methods**:
    *   `rewriteLinksInFile(pageContext, urlToContextMap)`: Rewrite all internal links in saved HTML files to point to local paths.
        *   `pageContext`: The context of the page to rewrite.
        *   `urlToContextMap`: Map of URLs to PageContexts.
        *   **Returns**: `Promise<number>` (Number of internal links rewritten).
        *   **Workflow**: Load HTML → Identify internal links → Calculate relative paths → Rewrite hrefs → Download CSS → Save.

#### `ContentExpander.js`
Expands toggles, databases, and other collapsible content on Notion pages using aggressive iterative expansion.
*   **Constructor**: `new ContentExpander(config, logger)`
*   **Expansion Strategy**: Multi-iteration click-and-wait cycles to reveal nested content.
*   **Methods**:
    *   `expandAll(page)`: Expand all content on the page.
        *   `page`: Puppeteer page instance.
        *   **Workflow**: Scroll to bottom (lazy-loading) → Iterative toggle expansion.
    *   `_scrollToBottom(page)`: Scroll to the bottom to trigger lazy-loading. *(Private)*
    *   `_expandToggles(page)`: Aggressively expand toggles, buttons, and interactive elements. *(Private)*
        *   **Target Elements**: aria-expanded="false", .notion-toggle-block, buttons, expandable divs.
        *   **Safety**: Excludes destructive actions (delete, remove, share).

#### `CookieHandler.js`
Handles cookie consent banners on Notion pages with intelligent retry and deduplication.
*   **Constructor**: `new CookieHandler(config, logger)`
*   **State Management**: Uses WeakSet to track processed pages (auto garbage-collected).
*   **Methods**:
    *   `ensureConsent(page, label)`: Public entry point that safely handles the cookie banner once per page context.
        *   `page`: Puppeteer page instance.
        *   `label`: Optional label for logging.
        *   **Returns**: `Promise<boolean>` (True if banner was found and handled).
        *   **Features**: Configurable retries, exponential backoff, per-page tracking.
    *   `handle(page, label)`: Backwards-compatibility shim. *(Deprecated)*
    *   `_attemptHandle(page)`: Attempt to handle cookie banner once. *(Private)*
    *   `_isPageUsable(page)`: Check if page is still open and usable. *(Private)*

### 6. Extraction Package (`src/extraction`)

Extracts structured data from the DOM.

#### `LinkExtractor.js`
Extracts and categorizes links from Notion pages with hierarchical context metadata.
*   **Constructor**: `new LinkExtractor(config, logger)`
*   **Extraction Features**:
    *   Filters for internal Notion links (same domain)
    *   Resolves relative URLs to absolute URLs
    *   Deduplicates discovered links
    *   Extracts hierarchical context (sections, subsections) from DOM structure
    *   Excludes self-references and invalid URLs
*   **Methods**:
    *   `extractLinks(page, currentUrl, baseUrl)`: Extract all internal Notion page links with their hierarchical context.
        *   `page`: Puppeteer page object.
        *   `currentUrl`: Current page URL to filter out (prevent self-references).
        *   `baseUrl`: Optional base URL (defaults to config base URL).
        *   **Returns**: `Promise<Array<{url: string, title: string, section?: string, subsection?: string}>>` (Link objects with metadata).

### 7. Download Package (`src/download`)

Manages the retrieval of external resources.

#### `AssetDownloader.js`
Downloads and manages assets (images, files, etc.) with retry logic and content-based hashing.
*   **Constructor**: `new AssetDownloader(config, logger)`
*   **Features**:
    *   Content-based MD5 hashing for unique filenames
    *   Download cache to prevent duplicate requests
    *   Retry logic with exponential backoff
    *   Extracts from both img tags and CSS background properties
*   **Methods**:
    *   `downloadAndRewriteImages(page, outputDir)`: Download all images and assets from a page and rewrite their paths.
        *   `page`: Puppeteer page instance.
        *   `outputDir`: Directory to save images to (images/ subdirectory created).
        *   **Workflow**: Extract URLs → Download with retries → Generate safe filenames → Rewrite DOM references.
    *   `_downloadAsset(url, outputPath)`: Download a single asset with retry logic. *(Private)*
    *   `_generateFilename(url, content)`: Generate safe filename using content hash. *(Private)*

#### `CssDownloader.js`
Coordinator for CSS localization. Downloads external CSS stylesheets and rewrites `<link>` tags. Delegates to specialized components.
*   **Constructor**: `new CssDownloader(config, logger)`
*   **Dependencies**: `CssContentProcessor`, `CssAssetDownloader`
*   **Architecture**: Facade pattern - coordinates CSS processing and asset downloads.
*   **Methods**:
    *   `downloadAndRewriteCss(dom, pageDir, pageUrl)`: Download all CSS files linked in the HTML and rewrite the DOM.
        *   `dom`: JSDOM instance.
        *   `pageDir`: Directory of the HTML file.
        *   `pageUrl`: Original URL of the page.
        *   **Returns**: `Promise<{stylesheets: number, assets: number}>` (Stats on stylesheets and assets).
        *   **Workflow**: Extract stylesheets → Download CSS → Process content → Download assets → Rewrite links.
    *   `_downloadCssFile(cssUrl, outputDir, pageUrl)`: Download and process a single CSS file. *(Private)*
    *   `_cssFileExists(filepath)`: Check if CSS file already downloaded. *(Private)*

#### `FileDownloader.js`
Downloads embedded files (PDFs, code files, documents, etc.) from Notion pages.
*   **Constructor**: `new FileDownloader(config, logger)`
*   **Supported File Types**:
    *   Documents: PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX
    *   Archives: ZIP, RAR, 7Z, TAR, GZ
    *   Code: PY, JS, TS, JAVA, CPP, C, H, IPYNB
    *   Data: JSON, XML, CSV, TXT, MD
    *   Media: MP4, AVI, MOV, MP3, WAV
*   **Methods**:
    *   `downloadAndRewriteFiles(page, outputDir)`: Download all embedded files from a page and rewrite their links.
        *   `page`: Puppeteer page instance.
        *   `outputDir`: Directory to save files to (files/ subdirectory created).
    *   `isDownloadableFile(url, linkText)`: Check if a URL is a downloadable file.
        *   **Detection**: Matches against file extensions and Notion file URL patterns.
    *   `_downloadFile(url, outputPath)`: Download a single file with retry logic. *(Private)*

#### Download CSS Subpackage (`src/download/css`)

Specialized CSS processing components extracted for clean separation.

##### `CssContentProcessor.js`
Handles CSS parsing and URL rewriting for @import rules and asset references.
*   **Constructor**: `new CssContentProcessor(config, logger)`
*   **Methods**:
    *   `processCssContent(cssContent, cssUrl, baseUrl)`: Process CSS content and rewrite URLs.
        *   `cssContent`: Raw CSS content string.
        *   `cssUrl`: URL of the CSS file.
        *   `baseUrl`: Base URL for relative resolution.
        *   **Returns**: `{processedCss: string, assetUrls: Array<string>}` (Rewritten CSS and extracted asset URLs).
        *   **Features**: Rewrites @import statements, rewrites asset URLs (url(...)), preserves data URIs.
    *   `rewriteImports(cssContent, cssUrl)`: Rewrite @import statements to local paths. *(Private)*
    *   `rewriteAssetUrls(cssContent, cssUrl, baseUrl)`: Rewrite asset URLs in CSS. *(Private)*

##### `CssAssetDownloader.js`
Downloads CSS-referenced assets (fonts, images) with retry logic and exponential backoff.
*   **Constructor**: `new CssAssetDownloader(config, logger)`
*   **Features**:
    *   Exponential backoff retry strategy (configurable max retries)
    *   Request timeout handling
    *   Safe filename generation with content hashing
*   **Methods**:
    *   `downloadCssAsset(assetUrl, outputDir)`: Download a CSS asset with retry logic.
        *   `assetUrl`: URL of the asset to download.
        *   `outputDir`: Directory to save asset to.
        *   **Returns**: `Promise<string|null>` (Local filename or null on failure).
    *   `_fetchWithRetries(url, attempt)`: Fetch URL with exponential backoff retry. *(Private)*
    *   `_generateSafeFilename(url)`: Generate safe filename from URL. *(Private)*

### 8. Utils Package (`src/utils`)

Shared helpers.

#### `UserPrompt.js`
Terminal interaction utility for user confirmation prompts.

**Key Features**:
- Readline-based interactive prompts
- Timeout handling (default 60 seconds)
- Input validation with retry logic (max 3 attempts)
- SIGINT (Ctrl+C) handling for graceful abort
- Non-interactive mode detection (returns default for CI/CD)
- Lazy readline initialization

**Primary Methods**:
- `async promptYesNo(question, defaultAnswer, timeout)`: Generic yes/no prompt
  - `question` (string): Question to display
  - `defaultAnswer` (boolean|null): Default answer if user presses Enter (null = no default)
  - `timeout` (number): Timeout in milliseconds (default: 60000)
  - **Returns**: `Promise<boolean>` - true for yes, false for no/timeout/abort
  
- `async promptConfirmDownload(stats)`: Specialized prompt for download confirmation
  - `stats` (Object): Discovery statistics { totalPages, maxDepth, conflicts? }
  - **Returns**: `Promise<boolean>` - true to proceed, false to abort
  
- `close()`: Cleanup readline interface and remove SIGINT handler

**Error Handling**:
- Timeout → returns false
- SIGINT (Ctrl+C) → returns false with "^C (Aborted by user)" message
- Invalid input → re-prompt up to 3 times, then return false
- Non-interactive environment → uses defaultAnswer or true

**Usage Example**:
```javascript
const prompt = new UserPrompt();
const proceed = await prompt.promptConfirmDownload({ totalPages: 42, maxDepth: 3 });
prompt.close(); // Important: cleanup
```

#### `FileSystemUtils.js`
Centralized logic for file naming.
*   **Methods**:
    *   `static sanitizeFilename(filename)`: Converts a string into a safe filename.

#### `IntegrityAuditor.js`
Performs a post-scrape integrity audit.
*   **Constructor**: `new IntegrityAuditor(config, logger)`
*   **Methods**:
    *   `audit(contexts)`: Audit all saved page contexts.
        *   `contexts`: Array of PageContext objects.
        *   **Returns**: `Promise<Object>` (Audit results including missing files and residual links).

---

## Workflow and Data Flow

### Two-Phase Scraping Architecture

The scraper implements a strict two-phase approach to ensure efficiency and proper hierarchy:

#### Phase 1: Discovery (Lightweight)
**Purpose**: Build the complete PageContext tree without heavy downloads.

**Process**:
1. **Initialization**: Create root PageContext, reset state
2. **BFS Traversal**: Strict level-by-level processing
   - Navigate to page (domcontentloaded only)
   - Extract page title and links
   - Classify edges (tree/back/forward/cross)
   - Register new PageContext objects
   - Build parent-child relationships
3. **Output**: Complete page hierarchy, edge statistics

**Key Characteristics**:
- Fast navigation (no networkidle wait)
- No asset downloads
- No content expansion
- Edge classification for cycle detection
- Builds urlToContextMap for link rewriting

#### Phase 2: Execution (Full Scraping)
**Purpose**: Traverse the planned tree and perform complete page capture.

**Process**:
1. **Preparation**: Reset visited state, validate contexts
2. **BFS Traversal**: Follow tree edges only (ignore back/forward/cross)
   - Navigate to page (networkidle0)
   - Handle cookie consent
   - Expand all content
   - Download assets (images, files)
   - Save HTML to disk
3. **Link Rewriting**: Post-processing to enable offline browsing
   - Load saved HTML files
   - Rewrite internal links to relative paths
   - Download and localize CSS
4. **Output**: Complete offline-ready site

**Key Characteristics**:
- Heavy operations (full page load, asset downloads)
- Follows only tree edges (prevents duplicate work)
- Preserves BFS order for proper depth tracking
- Post-processing link rewriting phase

### Data Flow Diagram

```
User Input (URL, Depth)
        ↓
NotionScraper.run()
        ↓
    Initialize
    - Create Browser
    - Create Dependencies
        ↓
    ┌─────────────────────────────────────────┐
    │         DISCOVERY PHASE                 │
    └─────────────────────────────────────────┘
        ↓
RecursiveScraper.discover()
    - PageProcessor.discoverPageInfo()
    - LinkExtractor.extractLinks()
    - GraphAnalyzer.classifyEdge()
        ↓
    PageContext Tree Built
    (urlToContextMap populated)
        ↓
    PlanDisplayer.displayTree()
    UserPrompt.promptForPlanDecision()
        ↓
    ┌─────────────────────────────────────────┐
    │         EXECUTION PHASE                 │
    └─────────────────────────────────────────┘
        ↓
RecursiveScraper.execute()
    ↓
PageProcessor.scrapePage() (for each page)
    - CookieHandler.ensureConsent()
    - ContentExpander.expandAll()
    - LinkExtractor.extractLinks()
    - AssetDownloader.downloadAndRewriteImages()
    - FileDownloader.downloadAndRewriteFiles()
    - Save HTML to disk
        ↓
LinkRewriter.rewriteLinksInFile() (for each page)
    - Load saved HTML
    - Rewrite internal links
    - CssDownloader.downloadAndRewriteCss()
        - CssContentProcessor.processCssContent()
        - CssAssetDownloader.downloadCssAsset()
    - Save modified HTML
        ↓
    ┌─────────────────────────────────────────┐
    │         COMPLETION                      │
    └─────────────────────────────────────────┘
        ↓
StatisticsDisplayer.printStatistics()
IntegrityAuditor.audit()
NotionScraper.cleanup()
```

### Edge Classification System

During discovery, the GraphAnalyzer classifies each discovered link:

- **Tree Edge** (Green): First discovery, becomes parent-child relationship
  - Action: Create new PageContext, add to queue
  
- **Back Edge** (Red): Points to ancestor or self
  - Indicates: Cycle in page graph
  - Action: Log warning, do not queue
  - Example: Child page linking back to parent
  
- **Forward Edge** (Blue): Skip levels to descendant
  - Indicates: Shortcut in hierarchy
  - Action: Log, do not queue (already discovered)
  - Example: Parent linking to grandchild
  
- **Cross Edge** (Yellow): Between branches or same level
  - Indicates: Connection between sibling branches
  - Action: Log, do not queue (already discovered)
  - Example: Sibling page references

This classification helps identify potential issues (cycles, missing pages) and provides insights into the page graph structure.

### Directory Structure Output

```
output/
├── index.html                 # Root page
├── images/                    # Root page images
├── css/                       # Root page CSS
├── files/                     # Root page files
├── Child_Page_1/
│   ├── index.html
│   ├── images/
│   ├── css/
│   ├── files/
│   └── Grandchild_Page/
│       ├── index.html
│       ├── images/
│       ├── css/
│       └── files/
└── Child_Page_2/
    ├── index.html
    ├── images/
    ├── css/
    └── files/
```

Each page gets its own directory named after its sanitized title, containing an `index.html` and subdirectories for assets. This mirrors the Notion page hierarchy and enables clean relative path resolution for offline browsing.

---

## Design Patterns

### 1. Facade Pattern
**Used in**: `NotionScraper`, `CssDownloader`
- Provides simplified interface to complex subsystems
- Coordinates multiple components
- Hides internal complexity from callers

### 2. Strategy Pattern
**Used in**: Edge classification, content expansion
- Different algorithms for different scenarios
- GraphAnalyzer classifies edges based on graph structure
- ContentExpander adapts to different page types

### 3. Dependency Injection
**Used throughout**: All class constructors
- Components receive dependencies at construction
- Enables testing and loose coupling
- Clear dependency graphs

### 4. Singleton Pattern
**Used in**: `StateManager` (legacy), WeakSet in `CookieHandler`
- Single instance where needed
- Careful use to avoid global state issues

### 5. Template Method Pattern
**Used in**: Two-phase scraping workflow
- Defines algorithm skeleton (discover → execute)
- Subclasses/phases customize specific steps
- Ensures consistent workflow

---

## Code Quality Standards

### File Size Limits
- **Maximum file size**: 200 lines of code
- **Method size**: <100 LOC (most <50 LOC)
- **Rationale**: Improves readability, maintainability, testability

### Documentation Standards
All classes and methods include:
- `@classdesc`: High-level class purpose with implementation details
- `@summary`: Brief one-line method description
- `@description`: Detailed method behavior and workflow
- `@param` with types: Typed parameter documentation
- `@returns` with types: Return value documentation
- `@throws`: Error conditions
- `@see`: Cross-references to related components
- `@private`: Internal method markers
- `@deprecated`: Backwards-compatibility indicators

### Naming Conventions
- **Classes**: PascalCase (e.g., `PageProcessor`)
- **Methods**: camelCase (e.g., `scrapePage`)
- **Private methods**: Prefixed with `_` (e.g., `_savePageHtml`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`)
- **Files**: Match class name (e.g., `PageProcessor.js`)

### Error Handling
- Try-catch blocks around I/O operations
- Graceful degradation where possible
- Detailed error logging with context
- Retry logic for network operations

---

## Performance Considerations

### Optimization Strategies
1. **Two-phase approach**: Discovery is lightweight (no asset downloads)
2. **BFS traversal**: Ensures proper depth ordering, prevents duplicate work
3. **Caching**: Asset download caches prevent duplicate requests
4. **Parallel operations**: Where safe (within same BFS level)
5. **Lazy loading**: Content expansion triggers Notion's lazy-loading

### Bottlenecks
- **Page navigation**: Notion pages can be slow to load
- **Asset downloads**: Many assets per page, network-dependent
- **CSS processing**: Complex stylesheets with many dependencies
- **Content expansion**: Nested toggles require multiple iterations

### Scaling Considerations
- **Worker Pool**: Currently configured by CPU cores (default 4 workers)
- **Parallelism**: Discovery phase processes BFS level in parallel across workers
- **Memory**: PageContext tree in Master, separate Puppeteer instances per Worker
- **Network**: Bottleneck is Notion API rate limits, not local throughput
- **Disk I/O**: Many small file writes (asset downloads), could benefit from batching
- **Horizontal scaling**: Master/Worker separation enables future distributed deployment

---

## 8. Complete System Workflow

This section describes the end-to-end execution flow across all phases, showing how Master and Worker processes coordinate through IPC and events.

### 8.1 System Startup Phase

**Master Process Bootstrap** (see [`main-cluster.js`](../main-cluster.js)):

1. **Configuration Load**: `Config.getInstance()` reads `config.json` or falls back to defaults
2. **Logger Init**: `Logger.getInstance()` sets up file/console transports with timestamps
3. **Event Bus Init**: `SystemEventBus.getInstance()` creates the singleton event coordinator
4. **Browser Pool Init**: `BrowserInitializer.createBrowserPool()` spawns Worker processes:
   ```javascript
   // main-cluster.js
   const BrowserInitializer = require('./src/cluster/BrowserInitializer');
   const workers = BrowserInitializer.createBrowserPool(config);
   // workers = [WorkerProxy, WorkerProxy, WorkerProxy, WorkerProxy]
   ```
5. **Manager Init**: `BrowserManager` wraps the pool, initializes idle/busy tracking
6. **Orchestrator Init**: `ClusterOrchestrator` creates with root URL, starts in `IDLE` state

**Worker Process Bootstrap** (see [`src/worker/WorkerEntrypoint.js`](../src/worker/WorkerEntrypoint.js)):

1. **IPC Listener**: `process.on('message', handleMasterMessage)` registered
2. **Puppeteer Launch**: `puppeteer.launch()` with headless config
3. **Task Runner Init**: `TaskRunner` instance created with browser reference
4. **Ready Signal**: Worker sends `{type: 'worker:ready', workerId: process.pid}` to Master
5. **Wait State**: Worker idles until Master sends task commands

**Event Flow**:
```
[main-cluster.js]
    ↓ spawn()
[WorkerEntrypoint.js] → {type: 'worker:ready'} → [Master]
    ↓
[BrowserManager] adds WorkerProxy to idle pool
    ↓
[ClusterOrchestrator.start()] → transition to DISCOVERY
```

---

### 8.2 Discovery Phase (BFS Breadth-First)

**Phase Goal**: Build complete page graph without downloading assets.

**Master Orchestration** (see [`src/orchestration/ClusterOrchestrator.js`](../src/orchestration/ClusterOrchestrator.js)):

```javascript
async runDiscoveryPhase() {
    this._transitionTo('DISCOVERY');
    SystemEventBus.emit('phase:discovery:start');
    
    // Initialize queue with root
    this.queue.enqueue(this.rootPageContext);
    
    while (!this.queue.isEmpty()) {
        const currentLevel = this.queue.getCurrentLevelPages();
        
        // Parallel discovery across workers
        const results = await Promise.all(
            currentLevel.map(page => this._discoverPage(page))
        );
        
        // Flatten children from all results
        const allChildren = results.flat();
        
        // Enqueue children for next BFS level
        allChildren.forEach(child => this.queue.enqueue(child));
        
        this.queue.advanceLevel();
    }
    
    SystemEventBus.emit('phase:discovery:complete', { 
        totalPages: this.queue.getAllPages().length 
    });
}
```

**Per-Page Discovery Flow**:

1. **Acquire Worker**: `BrowserManager.acquireWorker()` gets idle WorkerProxy
2. **Send Task**: Master sends IPC message:
   ```javascript
   workerProxy.send({
       type: 'command:scrape',
       taskId: 'task-123',
       pageContext: pageContext.toJSON(),
       phase: 'discovery',
       options: { downloadAssets: false, expandContent: true }
   });
   ```
3. **Worker Executes** (see [`src/worker/TaskRunner.js`](../src/worker/TaskRunner.js)):
   - Navigate to URL: `page.goto(url)`
   - Expand toggles: `ContentExpander.expandTogglesAndDatabases()`
   - Extract links: `LinkExtractor.extractAll(page)` → child URLs
   - Extract cookies: `CookieHandler.extractCookies()` → cookie state
   - Create child PageContexts: `LinkExtractor.createPageContexts()`
   - **Skip asset downloads**: No images, CSS, JS fetched
4. **Worker Responds**: IPC message back to Master:
   ```javascript
   process.send({
       type: 'result:scrape:success',
       taskId: 'task-123',
       result: {
           children: [childContext1.toJSON(), childContext2.toJSON()],
           cookies: ['__session=...', 'token_v2=...'],
           metadata: { title: 'Page Title', linksFound: 2 }
       }
   });
   ```
5. **Master Processes**:
   - Deserialize children: `PageContext.fromJSON()`
   - Emit event: `SystemEventBus.emit('page:discovered', pageContext)`
   - Release worker: `BrowserManager.releaseWorker(workerProxy)`
   - Update graph: Add edges parent→children

**Event Timeline**:
```
[ClusterOrchestrator] emit('phase:discovery:start')
    ↓ for each BFS level
    ↓   for each page in level (parallel)
[WorkerProxy] send({type: 'command:scrape', phase: 'discovery'})
[Worker] execute scraping, emit internal events
[Worker] send({type: 'result:scrape:success'})
[Master] emit('page:discovered', pageContext)
[Master] BrowserManager.releaseWorker()
    ↓ level complete
[ClusterOrchestrator] queue.advanceLevel()
    ↓ all levels complete
[ClusterOrchestrator] emit('phase:discovery:complete', {totalPages: N})
```

**Parallelism**: All pages within the same BFS level are processed concurrently across the worker pool, limited only by the number of available workers.

---

### 8.3 Graph Pruning Phase

**Phase Goal**: Remove duplicate pages from BFS traversal before expensive execution.

**Conflict Detection** (see [`src/orchestration/analysis/ConflictResolver.js`](../src/orchestration/analysis/ConflictResolver.js)):

```javascript
detectConflicts(pageGraph) {
    const urlMap = new Map(); // url → [PageContext, PageContext, ...]
    
    // Group all pages by normalized URL
    for (const page of pageGraph.getAllPages()) {
        const normalizedUrl = this._normalizeUrl(page.url);
        if (!urlMap.has(normalizedUrl)) {
            urlMap.set(normalizedUrl, []);
        }
        urlMap.get(normalizedUrl).push(page);
    }
    
    // Find conflicts (URLs with multiple PageContexts)
    const conflicts = [];
    for (const [url, pages] of urlMap.entries()) {
        if (pages.length > 1) {
            conflicts.push({ url, pages, duplicates: pages.length - 1 });
        }
    }
    
    return conflicts;
}
```

**Resolution Strategy**:
- **Keep**: PageContext discovered earliest (smallest BFS depth)
- **Remove**: All later duplicates (higher depth or later discovery order)
- **Update Parents**: Redirect parent links to the kept canonical PageContext

**Example**:
```
Before Pruning:
  Root → Page A (depth=1) → Page C (depth=2)
      ↘ Page B (depth=1) → Page C' (depth=2, duplicate of C)

After Pruning:
  Root → Page A (depth=1) → Page C (depth=2, canonical)
      ↘ Page B (depth=1) ↗ (redirect to canonical C)
  
Pages to execute: 3 (Root, A, B, C)
Pages skipped: 1 (C')
```

**Event Flow**:
```
[ClusterOrchestrator] emit('phase:pruning:start')
[ConflictResolver] detectConflicts() → conflicts[]
[ConflictResolver] resolveConflicts() → keep/remove decisions
[GraphAnalyzer] classifyEdges() → tree/forward/cross/back edges
[ClusterOrchestrator] updateGraph() → remove duplicates, update parent links
[ClusterOrchestrator] emit('phase:pruning:complete', {removed: N})
```

---

### 8.4 Execution Phase (Asset Downloads)

**Phase Goal**: For each unique page, download HTML + all assets with full fidelity.

**Master Orchestration**:

```javascript
async runExecutionPhase() {
    this._transitionTo('EXECUTION');
    SystemEventBus.emit('phase:execution:start');
    
    const canonicalPages = this.queue.getCanonicalPages(); // after pruning
    
    for (const page of canonicalPages) {
        await this._executePage(page);
    }
    
    SystemEventBus.emit('phase:execution:complete', {
        totalPages: canonicalPages.length
    });
}
```

**Per-Page Execution Flow**:

1. **Acquire Worker**: `BrowserManager.acquireWorker()`
2. **Send Execution Task**:
   ```javascript
   workerProxy.send({
       type: 'command:scrape',
       taskId: 'task-456',
       pageContext: pageContext.toJSON(),
       phase: 'execution',
       options: { 
           downloadAssets: true,  // CHANGED
           expandContent: true,
           rewriteLinks: true
       }
   });
   ```
3. **Worker Executes Full Scraping via Pipeline**:
   - **Payload Validation**: `TaskRunner._validateDownloadPayload()` enforces absolute paths
   - **Pipeline Context Creation**: Initializes context with browser, page, payload, fileSystem, stats
   - **Pipeline Construction**: Creates ScrapingPipeline with 5 steps:
     1. **NavigationStep**: Apply cookies, navigate to `payload.url` with `networkidle0`
     2. **ExpansionStep**: `ContentExpander.expandAll(page)` reveals all hidden content
     3. **AssetDownloadStep**: 
        - Images via `AssetDownloader.downloadImages(page, outputDir)`
        - CSS via `CssDownloader.downloadFromPuppeteer(page, outputDir)` (Puppeteer-compatible)
        - Files via `FileDownloader.downloadFiles(page, outputDir)`
     4. **LinkRewriterStep**: Rewrite internal links to relative paths using link map
     5. **HtmlWriteStep**: `WorkerFileSystem.safeWrite(payload.savePath, html)` with absolute path validation
   - **Pipeline Execution**: Sequential step execution with error isolation
   - **Statistics**: Accumulates truthful stats from actual operations (not fake success)
4. **Worker Returns Result**:
   ```javascript
   {
     success: true,
     pageId: payload.pageId,
     url: payload.url,
     savedPath: payload.savePath,  // Absolute path confirmed
     assetsDownloaded: context.stats.assetsDownloaded,
     linksRewritten: context.stats.linksRewritten
   }
   ```
5. **Master Updates State**: Release worker, mark page complete, update statistics
     - Preserve anchor links (fragments)
   - **Save Final HTML**: Write rewritten HTML to disk
4. **Worker Responds**:
   ```javascript
   process.send({
       type: 'result:scrape:success',
       taskId: 'task-456',
       result: {
           htmlPath: 'output/my-page/index.html',
           assetsDownloaded: 23,
           cssFilesProcessed: 5,
           metadata: { title: 'My Page', size: 45678 }
       }
   });
   ```
5. **Master Processes**:
   - Emit: `SystemEventBus.emit('page:executed', pageContext)`
   - Release worker: `BrowserManager.releaseWorker(workerProxy)`
   - Update progress: Increment completed page counter

**Asset Download Caching**: `AssetDownloader` maintains a `Set<string>` of downloaded URLs per worker to avoid duplicate downloads within the same Worker process. Cross-worker deduplication is handled by file system checks (if file exists, skip download).

**Event Timeline**:
```
[ClusterOrchestrator] emit('phase:execution:start')
    ↓ for each canonical page (sequential)
[WorkerProxy] send({type: 'command:scrape', phase: 'execution'})
[Worker] emit('page:scraping:start') (internal)
[Worker] emit('asset:download:start') (per asset)
[Worker] emit('asset:download:complete') (per asset)
[Worker] emit('page:scraping:complete') (internal)
[Worker] send({type: 'result:scrape:success'})
[Master] emit('page:executed', pageContext)
    ↓ all pages complete
[ClusterOrchestrator] emit('phase:execution:complete', {totalPages: N})
```

---

### 8.5 Completion and Shutdown

**Final Steps**:

1. **Orchestrator Completion**:
   ```javascript
   // ClusterOrchestrator.js
   async complete() {
       this._transitionTo('COMPLETED');
       SystemEventBus.emit('workflow:complete', {
           totalPages: this.queue.getAllPages().length,
           duration: Date.now() - this.startTime
       });
   }
   ```

2. **Integrity Audit** (see [`src/utils/IntegrityAuditor.js`](../src/utils/IntegrityAuditor.js)):
   - Verify all expected pages have output files
   - Check for broken links in rewritten HTML
   - Report missing assets or incomplete downloads

3. **Worker Shutdown**:
   ```javascript
   // BrowserManager.js
   async shutdownAll() {
       for (const worker of this.allWorkers) {
           worker.send({ type: 'command:shutdown' });
       }
       
       await Promise.all(
           this.allWorkers.map(w => w.waitForExit())
       );
   }
   ```

4. **Worker Cleanup** (see [`src/worker/WorkerEntrypoint.js`](../src/worker/WorkerEntrypoint.js)):
   ```javascript
   async function handleShutdown() {
       await browser.close(); // Close Puppeteer
       process.exit(0);       // Terminate Worker process
   }
   ```

5. **Master Exit**:
   ```javascript
   // main-cluster.js
   SystemEventBus.emit('system:shutdown');
   await browserManager.shutdownAll();
   logger.info('All workers terminated. Exiting.');
   process.exit(0);
   ```

**Event Flow**:
```
[ClusterOrchestrator] emit('workflow:complete')
[IntegrityAuditor] emit('audit:start')
[IntegrityAuditor] emit('audit:complete', {errors: []})
[BrowserManager] send({type: 'command:shutdown'}) to all Workers
[Workers] browser.close() → process.exit(0)
[Master] emit('system:shutdown')
[Master] process.exit(0)
```

---

## 9. Critical Architectural Decisions and Trade-offs

### Decision 1: Master/Worker Split vs. Shared-Memory Threads

**Choice**: Multi-process architecture with IPC instead of worker_threads.

**Rationale**:
- Puppeteer browser instances are heavyweight, benefit from process isolation
- Crash in one Worker doesn't affect Master or other Workers
- IPC overhead acceptable (serialization cost < benefit of isolation)
- Easier debugging: each Worker is independent process with own PID

**Trade-off**:
- ❌ Higher memory overhead (each Worker = separate V8 heap)
- ❌ Serialization cost for PageContext and messages
- ✅ Better fault tolerance and stability
- ✅ Clearer separation of concerns

---

### Decision 2: Two-Phase Workflow (Discovery → Execution)

**Choice**: Separate BFS discovery from asset downloading instead of single-pass scraping.

**Rationale**:
- Discovery is fast (no asset downloads), builds complete graph quickly
- Pruning after discovery eliminates duplicate work before expensive execution
- Enables better progress reporting (know total page count upfront)
- Simplifies dependency tracking (graph is known before execution)

**Trade-off**:
- ❌ Two navigations per page (discovery + execution) = more time
- ❌ More complex orchestration logic
- ✅ Significant savings from pruning duplicates (30-50% in typical Notion sites)
- ✅ Better UX (accurate progress bars)

---

### Decision 3: Event-Driven Coordination via SystemEventBus

**Choice**: Centralized event bus in Master for component coordination.

**Rationale**:
- Decouples components (Orchestrator, Manager, Auditor don't directly reference each other)
- Enables extensibility (new listeners can subscribe without modifying emitters)
- Simplifies logging and monitoring (centralized event stream)
- Matches Micro-Kernel philosophy (Master is thin coordinator)

**Trade-off**:
- ❌ Harder to trace control flow (implicit coupling via events)
- ❌ Potential performance cost if too many listeners
- ✅ Flexibility and maintainability win for long-term project evolution
- ✅ Easier testing (mock event bus)

---

### Decision 4: IPC Protocol Design (Typed Messages)

**Choice**: Strongly-typed message contracts in `ProtocolDefinitions.js` instead of ad-hoc JSON.

**Rationale**:
- Prevents runtime errors from malformed messages
- Self-documenting protocol (see `ProtocolDefinitions.js` for full contract)
- Enables validation (future enhancement: JSON schema validation)
- Centralized message types prevent drift across Master/Worker implementations

**Trade-off**:
- ❌ More boilerplate for new message types
- ❌ Requires discipline to update ProtocolDefinitions when adding features
- ✅ Catches bugs at "protocol level" before they reach business logic
- ✅ Easier onboarding (new developers read one file to understand IPC)

---

### Decision 5: Pipeline Pattern for Download Execution (NEW)

**Choice**: Sequential pipeline of specialized steps instead of monolithic download function.

**Rationale**:
- Solves \"Ghost Execution\" failure (stub function returning fake success)
- Each step has single responsibility (Navigation, Expansion, Assets, Links, HTML Write)
- Error isolation: Asset download failures don't abort entire pipeline
- Truthful result reporting from actual operations, not hardcoded success
- Testability: Steps can be unit tested independently

**Trade-off**:
- ❌ More classes and files (5 step classes + 2 orchestration classes)
- ❌ Slight overhead from context passing between steps
- ✅ Eliminated critical \"silent failure\" bug (files not written)
- ✅ Better error tracing (know which step failed)
- ✅ Extensibility (new steps added without modifying existing code)

**Implementation Details**:
- Base class `PipelineStep` enforces `process(context)` contract
- `ScrapingPipeline` executes steps sequentially with timing and error logging
- Context object accumulates state and statistics across steps
- Used in `TaskRunner._executeDownload()` to replace 27-line stub

---

### Decision 6: Absolute Path Enforcement via WorkerFileSystem (NEW)

**Choice**: Master calculates absolute paths, Workers validate at I/O boundaries.

**Rationale**:
- Worker processes have different working directories than Master
- Relative paths resolve ambiguously in child processes
- \"Ghost Execution\" symptom: System logged success but files written to wrong location
- WorkerFileSystem adapter validates `path.isAbsolute()` before any write operation
- Early validation prevents silent failures

**Trade-off**:
- ❌ Requires Master to compute all paths (cannot delegate to Workers)
- ❌ Additional abstraction layer for file I/O
- ✅ Guaranteed correct file locations (no ambiguous resolution)
- ✅ Explicit error messages when validation fails
- ✅ Audit trail via logging (every write logged with absolute path)

**Implementation Details**:
- `GlobalQueueManager._calculateAbsolutePath()` computes absolute paths in Master
- `DownloadPayload.savePath` sent as absolute path via IPC
- `TaskRunner._validateDownloadPayload()` enforces absolute path in payload
- `WorkerFileSystem.safeWrite()` validates path before write
- `HtmlWriteStep` uses WorkerFileSystem for guaranteed safe I/O

---

### Decision 7: Dual-Mode CSS Downloader (Puppeteer vs JSDOM) (NEW)

**Choice**: Separate methods for active scraping vs post-processing instead of unified interface.

**Rationale**:
- Puppeteer Page and JSDOM Document have incompatible APIs
- Active scraping requires three-phase approach: browser extract → Node process → browser rewrite
- Post-processing can use simpler JSDOM-based manipulation
- Interface mismatch was causing CSS downloads to fail silently

**Trade-off**:
- ❌ Two code paths to maintain (downloadFromPuppeteer, downloadAndRewriteCss)
- ❌ Callers must know which mode to use
- ✅ Eliminated interface mismatch causing CSS download failures
- ✅ Proper integration with CssContentProcessor for @import chains
- ✅ Separation of concerns: browser operations vs Node processing

**Implementation Details**:
- `CssDownloader.downloadFromPuppeteer(page, outputDir)` for active scraping
- Three phases: Extract CSS links via page.evaluate() → Process in Node.js → Rewrite DOM
- `AssetDownloadStep` uses downloadFromPuppeteer() instead of downloadAndRewriteCss()
- Proper CssContentProcessor integration with options object and bound callbacks

---

## 10. Logging and UI Architecture

### Overview

The system implements a **multi-transport logging architecture** with a **real-time terminal dashboard** for enhanced observability and user experience. The design separates concerns between log storage, display, and business logic coordination.

### 10.1 Logger Strategy Pattern

**Design Pattern**: Strategy Pattern + Singleton  
**Runtime Context**: Both Master and Worker  
**Key Benefit**: Pluggable log destinations without coupling to business logic

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Logger (Singleton)                       │
│                  [Multi-Transport Dispatcher]               │
└────────────────────┬────────────────────────────────────────┘
                     │ dispatch(level, category, message, meta)
         ┌───────────┼───────────┬────────────────────┐
         │           │           │                    │
┌────────▼──────┐ ┌─▼─────────┐ ┌▼──────────────┐ ┌─▼─────────────┐
│ ConsoleStrategy│ │FileStrategy│ │DashboardStrategy│ │CustomStrategy│
│ [stdout/stderr]│ │ [Markdown] │ │ [UI Ticker]  │ │  [External]  │
└────────────────┘ └────────────┘ └───────────────┘ └──────────────┘
```

#### Logger Strategies

**ConsoleStrategy** (`src/core/logger/ConsoleStrategy.js`):
- Outputs to `stdout`/`stderr` with ANSI color codes
- Used when dashboard is disabled or in CI environments
- Includes timestamps and category tags
- Filters debug messages based on `DEBUG` environment variable

**FileStrategy** (`src/core/logger/FileStrategy.js`):
- Writes markdown-formatted logs to timestamped files
- Location: `{outputDir}/logs/run-{timestamp}.md`
- Table format: `| Time | Level | Category | Message |`
- Includes JSON metadata blocks for errors
- Automatically closes stream on process exit

**DashboardStrategy** (`src/core/logger/DashboardStrategy.js`):
- Sends log messages to TerminalDashboard for real-time display
- Maintains circular buffer of recent logs (default: 5 lines)
- Skips debug messages to reduce UI clutter
- Includes level icons (❌ error, ⚠️ warn, ℹ️ info)

#### Logger Initialization

```javascript
// Master Process (main-cluster.js)
const logger = Logger.getInstance();

// Determine UI availability
const enableUI = !process.env.CI && process.stdout.isTTY;

// Initialize with strategies
logger.init({
  console: !enableUI,  // Console fallback if no UI
  file: true,          // Always log to file
  outputDir: config.OUTPUT_DIR,
  dashboard: false     // Dashboard strategy added later
});

// After worker pool initialization
if (enableUI) {
  const DashboardStrategy = require('./src/core/logger/DashboardStrategy');
  logger.addStrategy(new DashboardStrategy(dashboardCtrl.getDashboard()));
}
```

#### Key Benefits

1. **Separation of Concerns**: Business logic never directly touches `console.log`
2. **Flexible Output**: Add new destinations without modifying existing code
3. **Dual Logging**: Persistent markdown file + real-time UI display
4. **Graceful Degradation**: Falls back to console if UI unavailable
5. **Error Isolation**: Strategy failures don't crash logging system

### 10.2 Terminal Dashboard Architecture

**Design Pattern**: MVC (Model-View-Controller)  
**Runtime Context**: Master Process Only  
**Key Benefit**: Real-time visibility without cluttering console output

#### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              SystemEventBus (Event Source)                  │
└────────────────────┬────────────────────────────────────────┘
                     │ Events (WORKER:BUSY, JOB:COMPLETED, ...)
┌────────────────────▼────────────────────────────────────────┐
│           DashboardController (Controller)                  │
│         [Event → UI Translation Logic]                      │
│  - Maps Worker IDs to UI Slots                              │
│  - Translates events to visual updates                      │
│  - Tracks state (phase, progress, completion)               │
└────────────────────┬────────────────────────────────────────┘
                     │ updateProgress(), setWorkerStatus()
┌────────────────────▼────────────────────────────────────────┐
│           TerminalDashboard (View)                          │
│         [cli-progress MultiBar Wrapper]                     │
│  - Main Progress Bar (phase, percentage)                    │
│  - Worker Status Rows (N workers × status)                  │
│  - Recent Activity Log (last 5 log lines)                   │
└─────────────────────────────────────────────────────────────┘
```

#### TerminalDashboard (View Layer)

**File**: `src/ui/TerminalDashboard.js`  
**Dependency**: `cli-progress` npm package  
**Responsibility**: Render multi-bar progress UI in terminal

**Visual Layout**:
```
[████████████░░░░░░░░] 45% | 45/100 | EXECUTION: Downloading...
  ⚙️ Worker 1: Processing [task-abc1] "Introduction to AI"
  ⚙️ Worker 2: IDLE
  ⚙️ Worker 3: Processing [task-def2] "Neural Networks"
  ⚙️ Worker 4: IDLE
Recent: 10:32:15 ℹ️ [DOWNLOAD] Image saved | 10:32:16 ℹ️ [CSS] Stylesheet downloaded
```

**Public Methods**:
```javascript
class TerminalDashboard {
  constructor(workerCount, options = {}) { ... }
  
  // Update main progress bar
  updateProgress(current, total, phaseLabel, statusText)
  
  // Update worker status row
  setWorkerStatus(slotIndex, statusText, isError = false)
  
  // Update recent activity ticker
  updateLogs(logLines)
  
  // Mark as complete
  complete()
  
  // Stop rendering and restore terminal
  stop()
}
```

**Configuration Options**:
- `enabled`: Enable/disable UI rendering
- `maxLogLines`: Number of recent log lines to display (default: 5)

#### DashboardController (Controller Layer)

**File**: `src/ui/DashboardController.js`  
**Responsibility**: Bridge between SystemEventBus and TerminalDashboard

**Key Responsibilities**:
1. **ID Mapping**: Maps worker UUIDs to sequential UI slots (0, 1, 2, 3)
2. **Event Translation**: Converts system events to visual updates
3. **State Tracking**: Maintains phase, progress, and completion state
4. **Lifecycle Management**: Initializes after workers spawn, stops on shutdown

**Event Handlers**:
```javascript
// Discovery Phase
bus.on('DISCOVERY:START', () => {
  dashboard.updateProgress(0, 100, 'DISCOVERY', 'Crawling page hierarchy...');
});

bus.on('DISCOVERY:COMPLETE', ({ totalPages }) => {
  dashboard.updateProgress(0, totalPages, 'DISCOVERY', `Found ${totalPages} pages`);
});

// Execution Phase
bus.on('EXECUTION:START', () => {
  dashboard.updateProgress(0, this.totalPages, 'EXECUTION', 'Starting downloads...');
});

bus.on('EXECUTION:PROGRESS', ({ completed }) => {
  const remaining = this.totalPages - completed;
  dashboard.updateProgress(completed, this.totalPages, 'EXECUTION', `${remaining} remaining`);
});

// Worker Status
bus.on('WORKER:BUSY', ({ workerId, taskId, pageTitle }) => {
  const slot = this.workerSlotMap.get(workerId);
  dashboard.setWorkerStatus(slot, `Processing [${taskId.substring(0, 8)}] "${pageTitle}"`);
});

bus.on('WORKER:AVAILABLE', ({ workerId }) => {
  const slot = this.workerSlotMap.get(workerId);
  dashboard.setWorkerStatus(slot, 'IDLE');
});

// Job Completion
bus.on('JOB:COMPLETED', ({ pageTitle }) => {
  this.completedPages++;
  dashboard.updateProgress(this.completedPages, this.totalPages, 'EXECUTION', `Completed: ${pageTitle}`);
});
```

### 10.3 Integration Flow

**Initialization Sequence**:
```
1. main-cluster.js starts
2. Logger initialized with Console + File strategies
3. ClusterOrchestrator spawned
4. BrowserInitializer spawns workers
5. ClusterOrchestrator emits BOOTSTRAP:COMPLETE event
6. main-cluster.js receives event, creates DashboardController
7. DashboardController.start() initializes TerminalDashboard
8. Logger adds DashboardStrategy pointing to TerminalDashboard
9. System now logs to: File + Console (if no UI) + Dashboard (if UI enabled)
```

**Shutdown Sequence**:
```
1. ClusterOrchestrator.shutdown() called
2. DashboardController.stop() restores terminal
3. Logger.close() flushes file stream
4. Process exits cleanly
```

### 10.4 Lazy Title Registry Optimization

**Integration with Dashboard**: The dashboard leverages the lazy title registry pattern (documented in Section "Centralized Title Registry Architecture") to display human-readable page titles without IPC overhead:

**Worker Side**:
- Workers cache title registry received via `IPC_INIT` during bootstrap
- When processing tasks, workers use cached registry for display/logging
- Example: `this.titleRegistry[pageId]` → "Introduction to AI"

**Master Side**:
- Dashboard receives events with `pageId` or `taskId`
- Uses `GlobalQueueManager.getTitleById(pageId)` for human-readable display
- Example: `WORKER:BUSY` event includes both `taskId` and `pageTitle` for UI

**Performance Impact on Dashboard**:
- **Before**: Each WORKER:BUSY event required registry lookup → potential N×2 lookups for N tasks
- **After**: Event payload includes pre-resolved title → O(1) display, no lookup needed
- **Memory**: Dashboard controller maintains lightweight `workerSlotMap` (4 entries for 4 workers)

### 10.5 Design Benefits

1. **Observability**: Real-time visibility into system state without verbose console logs
2. **Dual Persistence**: Markdown files for post-mortem analysis + live UI for monitoring
3. **Graceful Degradation**: Automatic fallback to console if TTY unavailable (CI/CD)
4. **Separation of Concerns**: 
   - Business logic emits events (no UI coupling)
   - Controller handles translation (no rendering coupling)
   - View handles rendering (no business logic)
5. **Testability**: Each layer mocked independently (see tests/unit/ui/)
6. **Error Isolation**: Dashboard failures don't affect logging or scraping

### 10.6 Future Enhancements

**Planned (ROADMAP.md Phase 11)**:
- Web-based dashboard with historical data
- Export logs to structured formats (JSON, CSV)
- Integration with monitoring systems (Prometheus, Grafana)
- Real-time error filtering and alerting

**Proposed**:
- Color-coded worker statuses (green=idle, yellow=busy, red=error)
- ETA calculation based on average task duration
- Bandwidth and memory usage graphs
- Pause/resume controls for long-running scrapes

---

## 11. Known Limitations and Future Improvements

### Current Limitations

1. **Sequential Execution Phase**: Pages are executed one-at-a-time even though Workers are available for parallelism. Could parallelize within constraints (rate limits).

2. **No Distributed Deployment**: Master and Workers run on same machine. Architecture supports distribution (IPC could be replaced with network protocol) but not yet implemented.

3. **Limited Error Recovery**: Worker crashes are detected but page is marked failed without retry. Could implement retry queue with exponential backoff.

4. **Memory Growth**: PageContext tree held entirely in Master memory. For very large sites (>10,000 pages), could overflow. Need streaming or DB-backed storage.

5. **No Incremental Updates**: Full re-scrape each run. Could track page checksums and skip unchanged pages.

### Planned Enhancements (see [`ROADMAP.md`](ROADMAP.md))

**Phase 8**: Advanced Error Recovery
- Automatic retry with exponential backoff
- Partial page recovery (save assets even if HTML fails)
- Worker health checks and automatic replacement

**Phase 9**: Performance Optimization
- Parallel execution phase (respecting rate limits)
- Smarter asset caching (persistent cache across runs)
- Lazy PageContext loading (database-backed)

**Phase 10**: Distributed Scraping
- Replace IPC with network protocol (gRPC or ZeroMQ)
- Multi-machine Worker pools
- Distributed queue (Redis or RabbitMQ)

---

## 11. Conclusion

The **Reactive Event-Driven Micro-Kernel Architecture** with **Multi-Transport Logging and Real-Time Dashboard** represents a complete transformation of the scraper from a monolithic sequential script into a robust, observable, and user-friendly distributed system. Recent refactoring has further strengthened reliability through pipeline-based execution, absolute path enforcement, and comprehensive UI integration.

**Key Achievements**:
- ✅ **Process Isolation**: Master/Worker separation prevents cascading failures
- ✅ **Event-Driven Coordination**: SystemEventBus decouples components for flexibility
- ✅ **Structured Workflows**: Two-phase BFS ensures correctness and deduplication
- ✅ **Typed Communication**: IPC protocol prevents runtime message errors
- ✅ **Comprehensive Monitoring**: Event logging provides full system observability
- ✅ **Code Quality**: Sub-200-line files, extensive JSDoc, clear contracts
- ✅ **Pipeline Pattern**: Sequential step execution with error isolation eliminates "Ghost Execution" failures
- ✅ **Absolute Path Validation**: WorkerFileSystem prevents ambiguous path resolution in child processes
- ✅ **Dual-Mode CSS Processing**: Puppeteer and JSDOM compatibility resolves interface mismatches
- ✅ **Multi-Transport Logging**: Strategy pattern enables flexible log destinations
- ✅ **Real-Time Dashboard**: Terminal UI provides live visibility without console clutter
- ✅ **Lazy Title Registry**: 98% IPC reduction via one-time initialization

**Recent Enhancements (v2.2 - Logging & UI)**:
1. **Logger Strategy Pattern**: Console, File, and Dashboard strategies with pluggable architecture
2. **Terminal Dashboard**: Real-time multi-bar progress UI using cli-progress
3. **Dashboard Controller**: Event-to-UI translation layer maintaining separation of concerns
4. **Graceful Degradation**: Automatic console fallback in non-TTY environments
5. **Persistent Logs**: Markdown-formatted logs with structured metadata for debugging
6. **Enhanced Observability**: Worker status, phase tracking, and recent activity ticker

**Previous Refactoring (v2.1)**:
1. **Pipeline Architecture**: Replaced 27-line stub with 5-step pipeline (Navigation → Expansion → Assets → Links → HTML Write)
2. **Path Safety**: Introduced WorkerFileSystem adapter with `path.isAbsolute()` validation
3. **CSS Compatibility**: Added `downloadFromPuppeteer()` method with three-phase execution
4. **Truthful Reporting**: Statistics accumulated from actual operations, not hardcoded success values
5. **Error Traceability**: Step-level error logging pinpoints failure location in pipeline

**Architectural Principles Upheld**:
1. **Separation of Concerns**: Each package has clear responsibility (orchestration, execution, logging, UI)
2. **Single Responsibility**: Classes do one thing well (Logger dispatches, Dashboard renders, Controller translates)
3. **Dependency Inversion**: Components depend on abstractions (events, strategies) not concretions
4. **Open/Closed**: System is open for extension (new strategies, new UI components) but closed for modification
5. **Fail-Fast Validation**: Absolute path enforcement and payload validation prevent silent failures

**System Characteristics**:
- **Fault Tolerant**: Worker crashes don't affect Master or other Workers
- **Scalable**: Horizontal scaling via Worker pool size
- **Observable**: Rich event stream + real-time UI + persistent logs for complete visibility
- **Maintainable**: Clear structure, extensive documentation, small files, layered architecture
- **Testable**: Decoupled components, dependency injection, mockable IPC, comprehensive test coverage
- **Reliable**: Pipeline pattern with validation prevents "Ghost Execution" and silent failures
- **User-Friendly**: Live dashboard provides intuitive progress tracking without overwhelming output

This architecture provides a solid foundation for future enhancements including distributed deployment, advanced error recovery, performance optimization, and web-based monitoring dashboards, while maintaining the stability, correctness, and observability of the scraping process. The recent logging and UI refactoring has elevated the developer and user experience to production-grade quality.

---

**Document Version**: 2.2 (Multi-Transport Logging + Terminal Dashboard)  
**Last Updated**: November 2024  
**Related Documents**:
- [`REFACTORING_PLAN.md`](REFACTORING_PLAN.md) - Original transformation roadmap
- [`Advanced_Improvements.md`](Advanced_Improvements.md) - Detailed implementation notes
- [`ROADMAP.md`](ROADMAP.md) - Future enhancement plan
- [`CLUSTER_MODE.md`](CLUSTER_MODE.md) - Cluster deployment guide
- [`plan-ImprovedLoggingAndUi.prompt.md`](../.github/prompts/plan-ImprovedLoggingAndUi.prompt.md) - Logging & UI implementation plan
