# Refactoring Plan: Reactive Event-Driven Micro-Kernel Architecture

## 1. Executive Summary

This document outlines the comprehensive plan to refactor the current `notion-recursive-scraper` from a monolithic, sequential execution model to a **Reactive Event-Driven Micro-Kernel** architecture. This shift is designed to improve scalability, fault tolerance, and resource utilization by distributing the scraping workload across multiple isolated worker processes coordinated by a central event bus.

## 2. Current State vs. Proposed Architecture

### 2.1 Current State (Monolithic)
*   **Execution Model:** Sequential or single-instance recursive discovery and scraping.
*   **Resource Management:** Single `puppeteer` browser instance managed by the main process.
*   **Communication:** Direct method calls between classes (`NotionScraper` -> `RecursiveScraper` -> `PageProcessor`).
*   **State:** In-memory state (visited sets, context maps) held within the main process.
*   **Dependencies:** Includes `puppeteer-cluster` but does not effectively utilize it for distributed processing in the core logic.

### 2.2 Proposed Architecture (Micro-Kernel)
*   **Execution Model:** Distributed, multi-process system.
*   **Resource Management:** Pool of Node.js child processes (`Worker`), each managing its own Puppeteer instance, controlled by a `BrowserManager`.
*   **Communication:** Asynchronous Event Bus (`SystemEventBus`) and Inter-Process Communication (IPC) via strict protocols.
*   **State:** Centralized orchestration state (`GlobalQueueManager`) with stateless workers.
*   **Dependencies:** Native `child_process` for worker management, removing the need for `puppeteer-cluster`.

### 2.3 Runtime Contexts
The system will operate in two distinct runtime contexts. It is critical that developers understand which code runs where.

*   **Master Process (The Brain):**
    *   **Role:** Decision making, state management, resource allocation.
    *   **Key Components:** `SystemEventBus`, `ClusterOrchestrator`, `GlobalQueueManager`, `BrowserManager`, `WorkerProxy`.
    *   **Constraints:** No heavy computation, no HTML parsing, no Puppeteer instances.
*   **Worker Process (The Muscle):**
    *   **Role:** Execution, parsing, downloading.
    *   **Key Components:** `WorkerEntrypoint`, `TaskRunner`, `PageProcessor`, `Puppeteer`.
    *   **Constraints:** Stateless. No knowledge of the global queue or other workers.

## 3. Detailed Implementation Guidelines by Phase

### Phase 1: The Nervous System (Infrastructure)

**Goal:** Define the language and communication channels of the distributed system.

#### Step 1.1: `src/core/ProtocolDefinitions.js`
*   **Objective:** Define the strict contract for IPC (Inter-Process Communication).
*   **Guidelines:**
    *   Export constant strings for message types (e.g., `MSG_INIT`, `MSG_DISCOVER`, `MSG_DOWNLOAD`).
    *   Define payload structures (e.g., `InitPayload = { config: Config }`).
    *   **Critical:** Define the `WorkerResult` shape. It must handle both Success (with data) and Failure (with error details).
    *   *Pattern:* Use a TypeScript-like definition approach (even in JS) using JSDoc types to ensure clarity.

#### Step 1.2: `src/core/SystemEventBus.js`
*   **Objective:** Create the central observer for the Master Process.
*   **Guidelines:**
    *   Extend Node.js `EventEmitter`.
    *   Implement as a **Singleton**.
    *   Define strictly typed events:
        *   `SYSTEM:INIT`: Triggered by `main.js`.
        *   `WORKER:REGISTERED`: A new process is online. Payload: `{ workerId }`.
        *   `WORKER:AVAILABLE`: A worker is idle. Payload: `{ workerId }`.
        *   `JOB:COMPLETED`: A task finished. Payload: `{ workerId, result }`.
        *   `JOB:FAILED`: A task errored. Payload: `{ workerId, error }`.
        *   `CMD:SHUTDOWN`: Graceful exit signal.

---

### Phase 2: The Domain Layer (Data Serialization)

**Goal:** Ensure `PageContext` can survive the IPC trip (JSON serialization/deserialization).

#### Step 2.1: `src/domain/PageContext.js` (Refactor)
*   **Objective:** Make the domain model transferrable.
*   **Guidelines:**
    *   **Remove Circular References:** The `parent` property cannot hold a direct reference to another `PageContext` object. Replace it with `parentId` (string).
    *   **Method Separation:** Separate data holding (properties) from logic (methods).
    *   **Serialization:** Add `toJSON()` method to strip non-transferrable data.
    *   **Rehydration:** Add `static fromJSON(json)` to reconstruct a valid `PageContext` instance (with methods) from the raw data received from a worker.

---

### Phase 3: The Worker Layer (Execution Environment)

**Goal:** Build the standalone script that runs inside the child process.

#### Step 3.1: `src/worker/WorkerEntrypoint.js`
*   **Objective:** The `main()` for the child process.
*   **Guidelines:**
    *   **Isolation:** This file must *not* import any Master-side orchestration logic.
    *   **Lifecycle:**
        1.  Import `puppeteer`.
        2.  Listen for `process.on('message')`.
        3.  Handle `MSG_INIT`: Launch `puppeteer.launch()`. Store the browser instance globally (within the process).
        4.  Emit `process.send({ type: 'READY' })` on success.
    *   **Error Handling:** Wrap the top-level logic in `try/catch`. If the browser crashes, `process.exit(1)` to let the Master spawn a replacement.

#### Step 3.2: `src/worker/TaskRunner.js`
*   **Objective:** The router that dispatches commands to specific logic.
*   **Guidelines:**
    *   **Routing:** Switch on message type (`MSG_DISCOVER` vs `MSG_DOWNLOAD`).
    *   **Instantiation:** Create fresh instances of `PageProcessor` or `AssetDownloader` for each task (or reuse if stateless).
    *   **Execution:** `await processor.execute(payload)`.
    *   **Response:** Send `process.send({ type: 'RESULT', data: ... })` back to parent.

#### Step 3.3: `src/scraping/PageProcessor.js` (Stateless Refactor)
*   **Objective:** Ensure scraping logic works without shared state.
*   **Guidelines:**
    *   **Input:** All necessary data (Cookies, Config, Target URL) must be passed in the arguments.
    *   **Output:** Return a plain object `{ title, links, assets }`. Do *not* try to update a global `ContextMap` directly (that lives in the Master).
    *   **Dependencies:** Ensure `LinkRewriter` and `AssetDownloader` are initialized with the process-local configuration.

---

### Phase 4: The Cluster Layer (Resource Management)

**Goal:** Manage the physical processes from the Master.

#### Step 4.1: `src/cluster/WorkerProxy.js`
*   **Objective:** The Master's handle for a Worker.
*   **Guidelines:**
    *   **Constructor:** Accept `childProcess` instance and `id`.
    *   **Event Translation:**
        *   `child.on('message', msg => ...)`
        *   If msg is `READY` -> Emit `WORKER:REGISTERED` to SystemBus.
        *   If msg is `RESULT` -> Emit `JOB:COMPLETED` to SystemBus.
    *   **Command Interface:** Implement `sendCommand(type, payload)` using `child.send()`.

#### Step 4.2: `src/cluster/BrowserInitializer.js`
*   **Objective:** Physical creation of resources.
*   **Guidelines:**
    *   **Capacity Planning:** Use `os.totalmem()` and `os.freemem()`. Assume ~1GB per worker.
    *   **Spawning:** Loop `MAX_CONCURRENCY` times. Call `child_process.fork('src/worker/WorkerEntrypoint.js')`.
    *   **Handoff:** Immediately create a `WorkerProxy` for the new process.

#### Step 4.3: `src/cluster/BrowserManager.js`
*   **Objective:** Pool management (Idle/Busy).
*   **Guidelines:**
    *   **State:** `idleStack` (Array of IDs) and `busyMap` (Map<ID, Task>).
    *   **Allocation:**
        *   On `CMD:EXECUTE` (from Orchestrator): Pop ID from `idleStack`. If empty, throw error (Orchestrator shouldn't have asked).
        *   Retrieve `WorkerProxy`. Call `proxy.sendCommand()`. Move to `busyMap`.
    *   **Reclamation:**
        *   On `JOB:COMPLETED` (from Bus): Look up ID in `busyMap`. Move back to `idleStack`. Emit `WORKER:AVAILABLE`.

---

### Phase 5: The Orchestration Layer (Workflow Logic)

**Goal:** Connect the "Brain" (Queue) to the "Muscle" (Cluster).

#### Step 5.1: `src/orchestration/GlobalQueueManager.js`
*   **Objective:** Manage the frontier.
*   **Guidelines:**
    *   **Discovery Queue:** Standard FIFO. Use a `Set` to track `visitedUrls` (uniqueness check).
    *   **Download Queue:** Priority Queue based on dependency resolution.
    *   **Dependency Tracking:** Maintain a map `pendingChildrenCount`. Only release a Parent URL when count reaches 0.

#### Step 5.2: `src/orchestration/analysis/ConflictResolver.js`
*   **Objective:** Prune the graph.
*   **Guidelines:**
    *   **Input:** The full `PageContext` tree from Discovery.
    *   **Logic:** Traverse BFS. The first time a Page ID is seen, mark it as **Canonical**. Every subsequent time, mark as **Reference**.
    *   **Output:** A "Download Plan" containing only Canonical nodes.

#### Step 5.3: `src/orchestration/ClusterOrchestrator.js`
*   **Objective:** The main State Machine.
*   **Guidelines:**
    *   **Init:** Trigger `BrowserInitializer`.
    *   **State - Discovery:**
        *   Listen for `WORKER:AVAILABLE`.
        *   `const job = queue.next()`.
        *   `BrowserManager.execute(workerId, MSG_DISCOVER, job)`.
    *   **Transition:** When Queue empty & Workers idle -> Trigger `ConflictResolver`. Show UI prompt.
    *   **State - Download:**
        *   Listen for `WORKER:AVAILABLE`.
        *   `const leaf = downloadQueue.nextLeaf()`.
        *   `BrowserManager.execute(workerId, MSG_DOWNLOAD, leaf)`.
    *   **Cookie Broadcast:** On first successful scrape, define `IPC_SET_COOKIES` and broadcast to all workers.

---

### Phase 6: Integration & Verification

**Goal:** Bring it all together.

#### Step 6.1: `main.js` (The Bootstrapper)
*   **Objective:** Initialize the singletons.
*   **Guidelines:**
    1.  Instantiate `SystemEventBus`.
    2.  Instantiate `BrowserManager` (starts listening).
    3.  Instantiate `ClusterOrchestrator` (starts listening).
    4.  Call `BrowserInitializer.initialize()`.
    5.  Emit `SYSTEM:INIT`.

#### Step 6.2: Cleanup
*   **Guidelines:**
    *   Deprecate `RecursiveScraper.js` (its logic is now split between Orchestrator and Queue).
    *   Verify no old "direct method calls" remain between Master and Worker logic.

---

## 4. Technical Constraints & Standards

*   **No `puppeteer-cluster`:** We are replacing this dependency with native `child_process`.
*   **Strict Types:** Use JSDoc or TypeScript interfaces for all IPC messages.
*   **Error Propagation:** If a worker fails a task, the error must propagate back to the `GlobalQueueManager` to decide on a retry or failure.
*   **Graceful Shutdown:** On `SIGINT` (Ctrl+C), the Master must send `MSG_SHUTDOWN` to all workers before exiting.

## 5. Migration Strategy

To avoid breaking the current functionality immediately, we will build the new architecture alongside the existing one.

1.  **Scaffold:** Create the new directory structure (`src/cluster`, `src/worker`).
2.  **Implement Core:** Build the Event Bus and Protocol definitions.
3.  **Implement Worker:** Build the Worker logic and test it in isolation (e.g., run a single worker script manually).
4.  **Implement Master:** Build the Cluster and Orchestration layers.
5.  **Switch:** Change `main.js` to use the new entry point.
6.  **Verify:** Run the scraper against the test Notion page.

## 6. JSDoc Specifications

This section defines the complete JSDoc documentation for all new and refactored classes. All documentation must follow these standards strictly.

### 6.1 Core Layer

#### `src/core/ProtocolDefinitions.js`

```javascript
/**
 * @fileoverview IPC Protocol Definitions for Master-Worker Communication
 * @module core/ProtocolDefinitions
 * @description Defines the strict contract for inter-process communication between
 * the Master orchestrator and Worker processes. All messages must conform to these
 * type definitions to ensure type safety across process boundaries.
 */

/**
 * @typedef {Object} MessageType
 * @property {string} MSG_INIT - Initialize worker with configuration
 * @property {string} MSG_DISCOVER - Command worker to discover links on a page
 * @property {string} MSG_DOWNLOAD - Command worker to download page assets
 * @property {string} MSG_SET_COOKIES - Broadcast authentication cookies to worker
 * @property {string} MSG_SHUTDOWN - Gracefully terminate worker process
 * @property {string} MSG_READY - Worker signals successful initialization
 * @property {string} MSG_RESULT - Worker returns task execution result
 */

/**
 * @typedef {Object} InitPayload
 * @property {Config} config - Application configuration object
 * @property {string} workerId - Unique identifier for this worker
 */

/**
 * @typedef {Object} DiscoverPayload
 * @property {string} url - Target URL to scrape for links
 * @property {string} parentId - ID of the parent PageContext
 * @property {number} depth - Current depth in the scraping tree
 * @property {Array<Object>} cookies - Authentication cookies for the session
 */

/**
 * @typedef {Object} DownloadPayload
 * @property {PageContext} context
 * @property {Array<Object>} cookies
 * @property {string} savePath // Master dictates the path
 * @property {Object<string, string>} relativeLinkMap // Map of ID -> RelativePath for rewriting
 */

/**
 * @typedef {Object} WorkerError
 * @property {string} message
 * @property {string} name
 * @property {string} stack
 */

/**
 * @typedef {Object} WorkerResult
 * @property {boolean} success
 * @property {Object} [data]
 * @property {WorkerError} [error] // Changed from Error to WorkerError
 */

/**
 * @constant {MessageType}
 * @description Enumeration of all valid IPC message types
 */
export const MESSAGE_TYPES = { /* ... */ };
```

#### `src/core/SystemEventBus.js`

```javascript
/**
 * @fileoverview Central Event Bus for Master Process Coordination
 * @module core/SystemEventBus
 * @description Implements the Observer pattern as a singleton EventEmitter.
 * All Master-side components communicate through this bus to maintain loose coupling.
 */

/**
 * @class SystemEventBus
 * @extends EventEmitter
 * @implements {Singleton}
 * @description Central nervous system of the Master process. Coordinates all
 * asynchronous events between orchestration, cluster management, and queue systems.
 * 
 * @example
 * const bus = SystemEventBus.getInstance();
 * bus.on('WORKER:AVAILABLE', ({ workerId }) => {
 *   console.log(`Worker ${workerId} is ready for tasks`);
 * });
 */

/**
 * @private
 * @static
 * @type {SystemEventBus}
 * @description Singleton instance holder
 */

/**
 * @private
 * @constructor
 * @description Private constructor to enforce singleton pattern.
 * Use getInstance() instead.
 * @throws {Error} If called directly instead of through getInstance()
 */

/**
 * @static
 * @method getInstance
 * @returns {SystemEventBus} The singleton instance
 * @description Lazy-initializes and returns the global event bus instance.
 * Thread-safe for Node.js single-threaded event loop.
 */

/**
 * @method emit
 * @param {string} event - Event name (use SYSTEM/WORKER/JOB/CMD namespace)
 * @param {Object} payload - Event data
 * @returns {boolean} True if event had listeners
 * @fires SystemEventBus#SYSTEM:INIT
 * @fires SystemEventBus#WORKER:REGISTERED
 * @fires SystemEventBus#WORKER:AVAILABLE
 * @fires SystemEventBus#JOB:COMPLETED
 * @fires SystemEventBus#JOB:FAILED
 * @fires SystemEventBus#CMD:SHUTDOWN
 */

/**
 * System initialization event
 * @event SystemEventBus#SYSTEM:INIT
 * @type {Object}
 * @property {Config} config - Application configuration
 */

/**
 * Worker process registered successfully
 * @event SystemEventBus#WORKER:REGISTERED
 * @type {Object}
 * @property {string} workerId - Unique worker identifier
 */

/**
 * Worker is idle and ready for tasks
 * @event SystemEventBus#WORKER:AVAILABLE
 * @type {Object}
 * @property {string} workerId - Unique worker identifier
 */

/**
 * Task completed successfully
 * @event SystemEventBus#JOB:COMPLETED
 * @type {Object}
 * @property {string} workerId - Worker that completed the task
 * @property {WorkerResult} result - Task execution result
 */

/**
 * Task failed with error
 * @event SystemEventBus#JOB:FAILED
 * @type {Object}
 * @property {string} workerId - Worker where failure occurred
 * @property {Error} error - Error object with details
 */

/**
 * Graceful shutdown command
 * @event SystemEventBus#CMD:SHUTDOWN
 * @type {Object}
 * @property {string} [reason] - Optional shutdown reason
 */
```

### 6.2 Domain Layer

#### `src/domain/PageContext.js`

```javascript
/**
 * @fileoverview Domain model for scraped page metadata
 * @module domain/PageContext
 * @description Represents a single page in the Notion workspace. This class is
 * designed for IPC serialization - all properties are JSON-compatible.
 */

/**
 * @class PageContext
 * @description Immutable value object representing a discovered page with its
 * metadata and relationships. Supports serialization for cross-process transfer.
 * 
 * @property {string} id - Unique Notion page identifier (UUID format)
 * @property {string} url - Canonical URL of the page
 * @property {string} title - Page title extracted from DOM
 * @property {number} depth - Distance from root page (0 = entry point)
 * @property {string|null} parentId - ID of parent page (null for root)
 * @property {Array<string>} childIds - Array of child page IDs
 * @property {Array<string>} assetUrls - URLs of CSS/JS/image assets
 * @property {Date} discoveredAt - Timestamp of discovery
 */

/**
 * @constructor
 * @param {Object} props - Page properties
 * @param {string} props.id - Unique page identifier
 * @param {string} props.url - Page URL
 * @param {string} props.title - Page title
 * @param {number} props.depth - Tree depth
 * @param {string} [props.parentId=null] - Parent page ID
 * @throws {TypeError} If required properties are missing
 */

/**
 * @method toJSON
 * @returns {Object} Plain object suitable for JSON.stringify
 * @description Serializes the PageContext for IPC transfer. Strips any
 * non-serializable properties (functions, circular references).
 * 
 * @example
 * const json = context.toJSON();
 * process.send({ type: 'RESULT', data: json });
 */

/**
 * @static
 * @method fromJSON
 * @param {Object} json - Plain object from JSON.parse
 * @returns {PageContext} Fully reconstructed PageContext instance with methods
 * @description Rehydrates a PageContext from serialized data. Converts ISO
 * date strings back to Date objects.
 * 
 * @example
 * const context = PageContext.fromJSON(message.data);
 * console.log(context.getDepthLabel()); // Method works after rehydration
 */

/**
 * @method addChild
 * @param {string} childId - ID of child page to add
 * @returns {void}
 * @description Registers a child page relationship. Prevents duplicates.
 */

/**
 * @method getDepthLabel
 * @returns {string} Human-readable depth indicator (e.g., "L2")
 * @description Formats the depth as a level label for UI display.
 */

/**
 * @method isRoot
 * @returns {boolean} True if this is the entry point page
 * @description Checks if this page has no parent.
 */
```

### 6.3 Worker Layer

#### `src/worker/WorkerEntrypoint.js`

```javascript
/**
 * @fileoverview Worker Process Entry Point
 * @module worker/WorkerEntrypoint
 * @description The main() function for child processes. Initializes Puppeteer,
 * establishes IPC channel, and delegates tasks to TaskRunner.
 * 
 * **CRITICAL**: This file runs in a SEPARATE Node.js process. It has NO access
 * to Master-side state, singletons, or in-memory queues.
 */

/**
 * @private
 * @type {Browser}
 * @description Puppeteer browser instance (process-scoped singleton)
 */

/**
 * @async
 * @function initializeBrowser
 * @param {Object} config - Configuration from Master
 * @returns {Promise<Browser>} Launched Puppeteer browser instance
 * @throws {Error} If browser fails to launch after retry attempts
 * @description Launches a headless Chromium instance with stealth plugins.
 * Uses exponential backoff for retry logic.
 */

/**
 * @function setupIPCListener
 * @returns {void}
 * @description Attaches listener to process.on('message'). Routes incoming
 * commands to TaskRunner based on message type.
 * 
 * @listens process#message
 */

/**
 * @async
 * @function handleShutdown
 * @returns {Promise<void>}
 * @description Gracefully closes browser and exits process. Waits for pending
 * page operations to complete (max 5s timeout).
 */

/**
 * @function main
 * @returns {void}
 * @description Entry point. Waits for MSG_INIT, launches browser, signals READY.
 * 
 * @example
 * // Spawned by Master:
 * // child_process.fork('src/worker/WorkerEntrypoint.js')
 */
```

#### `src/worker/TaskRunner.js`

```javascript
/**
 * @fileoverview Task Execution Router for Worker Process
 * @module worker/TaskRunner
 * @description Receives IPC commands and dispatches to appropriate executors
 * (PageProcessor, AssetDownloader). Handles result/error serialization.
 */

/**
 * @class TaskRunner
 * @description Stateless command dispatcher. Each task execution creates fresh
 * instances of processors to avoid state leakage between tasks.
 */

/**
 * @constructor
 * @param {Browser} browser - Puppeteer browser instance
 * @param {Config} config - Configuration object from Master
 */

/**
 * @async
 * @method execute
 * @param {string} messageType - Command type from MESSAGE_TYPES
 * @param {Object} payload - Task-specific data
 * @returns {Promise<WorkerResult>} Execution result with success/error
 * @description Main dispatcher. Routes to specialized handlers based on type.
 * 
 * @example
 * const result = await runner.execute('MSG_DISCOVER', { url: '...' });
 * if (result.success) {
 *   process.send({ type: 'RESULT', data: result.data });
 * }
 */

/**
 * @private
 * @async
 * @method _handleDiscover
 * @param {DiscoverPayload} payload - Discovery task parameters
 * @returns {Promise<Object>} Discovered links and metadata
 * @throws {Error} If page navigation fails or timeout occurs
 * @description Instantiates PageProcessor and extracts links from target page.
 */

/**
 * @private
 * @async
 * @method _handleDownload
 * @param {DownloadPayload} payload - Download task parameters
 * @returns {Promise<Object>} Downloaded file paths and status
 * @description Instantiates AssetDownloader and saves HTML/CSS/JS to disk.
 */

/**
 * @private
 * @method _serializeError
 * @param {Error} error - Error object to serialize
 * @returns {Object} JSON-compatible error representation
 * @description Converts Error to plain object (message, stack, name) for IPC.
 */
```

### 6.4 Cluster Layer

#### `src/cluster/WorkerProxy.js`

```javascript
/**
 * @fileoverview Master-side Handle for Worker Process
 * @module cluster/WorkerProxy
 * @description Wraps a child_process instance with a clean API. Translates
 * Node.js IPC events into SystemEventBus events.
 */

/**
 * @class WorkerProxy
 * @description Facade for a Worker process. Abstracts IPC details from
 * higher-level orchestration logic.
 * 
 * @property {string} id - Unique worker identifier (UUID)
 * @property {ChildProcess} process - Underlying Node.js child process
 * @property {string} state - Current state: 'INITIALIZING' | 'IDLE' | 'BUSY' | 'CRASHED'
 */

/**
 * @constructor
 * @param {ChildProcess} childProcess - Spawned process from child_process.fork()
 * @param {string} id - Unique identifier for this worker
 * @param {SystemEventBus} eventBus - Global event bus instance
 */

/**
 * @method sendCommand
 * @param {string} type - Message type from MESSAGE_TYPES
 * @param {Object} payload - Command-specific data
 * @returns {void}
 * @throws {Error} If worker is in CRASHED state
 * @description Sends an IPC message to the worker process. Non-blocking.
 * 
 * @example
 * proxy.sendCommand('MSG_DISCOVER', { url: 'https://...' });
 */

/**
 * @private
 * @method _attachListeners
 * @returns {void}
 * @description Sets up IPC and process event handlers (message, exit, error).
 * Translates worker events into SystemEventBus emissions.
 * 
 * @fires SystemEventBus#WORKER:REGISTERED
 * @fires SystemEventBus#JOB:COMPLETED
 * @fires SystemEventBus#JOB:FAILED
 */

/**
 * @method terminate
 * @param {number} [timeout=5000] - Grace period before force kill (ms)
 * @returns {Promise<void>}
 * @description Gracefully shuts down worker. Sends MSG_SHUTDOWN, waits for exit,
 * then force-kills if timeout exceeded.
 */

/**
 * @method isAvailable
 * @returns {boolean} True if worker is in IDLE state
 * @description Checks if worker can accept new tasks.
 */
```

#### `src/cluster/BrowserInitializer.js`

```javascript
/**
 * @fileoverview Worker Pool Bootstrap Logic
 * @module cluster/BrowserInitializer
 * @description Calculates optimal concurrency and spawns worker processes.
 * Implements capacity planning based on system resources.
 */

/**
 * @class BrowserInitializer
 * @static
 * @description Utility class for pool initialization. All methods are static.
 */

/**
 * @static
 * @method calculateOptimalConcurrency
 * @returns {number} Recommended number of workers
 * @description Uses os.totalmem(), os.freemem(), and os.cpus() to determine
 * safe concurrency level. Assumes 1GB RAM and 1 CPU core per worker.
 * Leaves 2GB RAM buffer for Master process.
 * 
 * @example
 * const workers = BrowserInitializer.calculateOptimalConcurrency();
 * // On 16GB machine with 8 cores: returns 6 (safe for 14GB / 1GB per worker)
 */

/**
 * @static
 * @async
 * @method initialize
 * @param {number} concurrency - Number of workers to spawn
 * @param {Config} config - Application configuration
 * @param {SystemEventBus} eventBus - Global event bus
 * @returns {Promise<Array<WorkerProxy>>} Array of initialized worker proxies
 * @throws {Error} If any worker fails initialization after max retries
 * @description Spawns N worker processes using child_process.fork(). Waits
 * for all workers to emit READY before resolving.
 * 
 * @example
 * const proxies = await BrowserInitializer.initialize(4, config, bus);
 * console.log(`Spawned ${proxies.length} workers`);
 */

/**
 * @private
 * @static
 * @method _spawnWorker
 * @param {string} id - Worker ID
 * @param {Config} config - Configuration to send via MSG_INIT
 * @param {SystemEventBus} eventBus - Event bus reference
 * @returns {Promise<WorkerProxy>} Proxy for the spawned worker
 * @description Forks WorkerEntrypoint.js, sends MSG_INIT, waits for READY.
 */
```

#### `src/cluster/BrowserManager.js`

```javascript
/**
 * @fileoverview Worker Pool State Manager
 * @module cluster/BrowserManager
 * @description Maintains the Idle/Busy dichotomy. Allocates workers to tasks
 * and reclaims them upon completion. Does NOT decide WHICH task to run (that's
 * the Orchestrator's job).
 */

/**
 * @class BrowserManager
 * @description Resource allocator for the worker pool. Ensures no worker is
 * double-booked and tracks task assignments.
 * 
 * @property {Array<string>} idleStack - Stack of available worker IDs (LIFO)
 * @property {Map<string, Object>} busyMap - Map of workerId -> {taskId, startTime}
 * @property {Map<string, WorkerProxy>} proxyMap - Map of workerId -> WorkerProxy
 */

/**
 * @constructor
 * @param {SystemEventBus} eventBus - Global event bus
 */

/**
 * @method registerWorkers
 * @param {Array<WorkerProxy>} proxies - Worker proxies from BrowserInitializer
 * @returns {void}
 * @description Adds workers to the pool and marks them as idle.
 * 
 * @fires SystemEventBus#WORKER:AVAILABLE (for each worker)
 */

/**
 * @method execute
 * @param {string} workerId - ID of idle worker to use
 * @param {string} messageType - Command type
 * @param {Object} payload - Task data
 * @returns {void}
 * @throws {Error} If worker is not in idle stack
 * @description Allocates a worker to a task. Moves worker from idle to busy.
 * Delegates command sending to WorkerProxy.
 */

/**
 * @private
 * @method _onJobCompleted
 * @param {Object} event - JOB:COMPLETED event payload
 * @returns {void}
 * @description Reclaims worker after successful task. Moves from busy to idle.
 * 
 * @fires SystemEventBus#WORKER:AVAILABLE
 * @listens SystemEventBus#JOB:COMPLETED
 */

/**
 * @private
 * @method _onJobFailed
 * @param {Object} event - JOB:FAILED event payload
 * @returns {void}
 * @description Handles worker failure. If recoverable, reclaims worker. If
 * crashed, removes from pool and spawns replacement.
 * 
 * @listens SystemEventBus#JOB:FAILED
 */

/**
 * @method getAvailableCount
 * @returns {number} Number of idle workers
 * @description Used by Orchestrator to decide whether to dispatch more tasks.
 */

/**
 * @method getAllocatedCount
 * @returns {number} Number of busy workers
 * @description Used for monitoring and statistics.
 */
```

### 6.5 Orchestration Layer

#### `src/orchestration/GlobalQueueManager.js`

```javascript
/**
 * @fileoverview Centralized Task Queue Manager
 * @module orchestration/GlobalQueueManager
 * @description Manages both Discovery and Download queues. Implements uniqueness
 * checks, dependency tracking, and priority ordering.
 */

/**
 * @class GlobalQueueManager
 * @description The "Frontier" of the scraping process. Determines WHAT to scrape
 * next, while the Orchestrator determines WHEN and WHO does it.
 * 
 * @property {Array<DiscoverPayload>} discoveryQueue - FIFO queue of pages to scrape
 * @property {Set<string>} visitedUrls - Set of already-discovered URLs (dedup)
 * @property {Array<PageContext>} downloadQueue - Priority queue of pages to download
 * @property {Map<string, number>} dependencyMap - pageId -> pendingChildCount
 */

/**
 * @constructor
 * @description Initializes empty queues and tracking structures.
 */

/**
 * @method enqueueDiscovery
 * @param {string} url - URL to discover
 * @param {string} parentId - Parent page ID
 * @param {number} depth - Current tree depth
 * @returns {boolean} True if URL was added (not duplicate)
 * @description Adds a URL to the discovery queue if not already visited.
 * Thread-safe for single-threaded event loop.
 */

/**
 * @method nextDiscovery
 * @returns {DiscoverPayload|null} Next discovery task, or null if queue empty
 * @description Pops the next URL from the discovery queue. FIFO order.
 */

/**
 * @method buildDownloadQueue
 * @param {Array<PageContext>} canonicalPages - Pages from ConflictResolver
 * @returns {void}
 * @description Initializes the download queue after discovery phase. Builds
 * dependency map to ensure parent pages are downloaded after all children.
 */

/**
 * @method nextDownload
 * @returns {PageContext|null} Next leaf page to download, or null if none ready
 * @description Returns a page with zero pending children. Updates dependency
 * counts when a page is dequeued.
 */

/**
 * @method markDownloadComplete
 * @param {string} pageId - ID of completed page
 * @returns {void}
 * @description Decrements the dependency counter for the parent page. May
 * unlock the parent for download if all children are complete.
 */

/**
 * @method isDiscoveryComplete
 * @returns {boolean} True if discovery queue is empty and no workers are busy
 * @description Used by Orchestrator to detect phase transition.
 */

/**
 * @method isDownloadComplete
 * @returns {boolean} True if all downloads finished
 * @description Signals end of scraping process.
 */

/**
 * @method getStatistics
 * @returns {Object} Queue stats for UI display
 * @property {number} discovered - Total discovered URLs
 * @property {number} pendingDiscovery - Remaining discovery tasks
 * @property {number} pendingDownload - Remaining download tasks
 * @description Aggregates queue state for StatisticsDisplayer.
 */
```

#### `src/orchestration/analysis/ConflictResolver.js`

```javascript
/**
 * @fileoverview Duplicate Page Detector
 * @module orchestration/analysis/ConflictResolver
 * @description Identifies canonical vs. reference instances of pages that
 * appear multiple times in the tree (e.g., linked from multiple parents).
 */

/**
 * @class ConflictResolver
 * @static
 * @description Utility class with pure functions. No instance state.
 */

/**
 * @static
 * @method resolve
 * @param {Map<string, PageContext>} contextMap - All discovered pages
 * @returns {Array<PageContext>} Canonical pages only (first occurrence of each ID)
 * @description Performs BFS traversal from root. The first time a page ID is
 * encountered, it's marked as canonical. Subsequent occurrences are pruned.
 * 
 * @example
 * const canonical = ConflictResolver.resolve(globalContextMap);
 * console.log(`Reduced ${contextMap.size} to ${canonical.length} unique pages`);
 */

/**
 * @private
 * @static
 * @method _bfsTraversal
 * @param {PageContext} root - Entry point page
 * @param {Map<string, PageContext>} contextMap - Full context map
 * @returns {Set<string>} Set of canonical page IDs
 * @description Breadth-first search to determine canonical order.
 */

/**
 * @static
 * @method getDuplicateReport
 * @param {Map<string, PageContext>} contextMap - All discovered pages
 * @returns {Map<string, Array<string>>} Map of pageId -> [parentIds who link to it]
 * @description Identifies pages with multiple parents for UI reporting.
 */
```

#### `src/orchestration/ClusterOrchestrator.js`

```javascript
/**
 * @fileoverview Master State Machine
 * @module orchestration/ClusterOrchestrator
 * @description The "Brain" that coordinates the entire scraping workflow.
 * Listens to SystemEventBus and issues commands to BrowserManager and
 * GlobalQueueManager. Implements the two-phase state machine (Discovery -> Download).
 */

/**
 * @class ClusterOrchestrator
 * @description Central coordinator. Maintains workflow state, triggers phase
 * transitions, and handles cookie propagation.
 * 
 * @property {string} phase - Current phase: 'INIT' | 'DISCOVERY' | 'ANALYSIS' | 'DOWNLOAD' | 'COMPLETE'
 * @property {BrowserManager} browserManager - Worker pool manager
 * @property {GlobalQueueManager} queueManager - Task queue manager
 * @property {SystemEventBus} eventBus - Global event bus
 * @property {Array<Object>} cookies - Authenticated session cookies (shared across workers)
 */

/**
 * @constructor
 * @param {BrowserManager} browserManager - Initialized browser manager
 * @param {GlobalQueueManager} queueManager - Initialized queue manager
 * @param {SystemEventBus} eventBus - Global event bus
 */

/**
 * @async
 * @method start
 * @param {string} entryUrl - Root Notion page URL
 * @returns {Promise<void>}
 * @description Main entry point. Initializes workers, enqueues root URL, and
 * starts the state machine loop.
 * 
 * @fires SystemEventBus#SYSTEM:INIT
 */

/**
 * @private
 * @method _onWorkerAvailable
 * @param {Object} event - WORKER:AVAILABLE event payload
 * @returns {void}
 * @description Core dispatch logic. Checks current phase and queue state, then
 * allocates tasks to available workers.
 * 
 * @listens SystemEventBus#WORKER:AVAILABLE
 */

/**
 * @private
 * @async
 * @method _onJobCompleted
 * @param {Object} event - JOB:COMPLETED event payload
 * @returns {Promise<void>}
 * @description Handles task results. In Discovery phase, extracts cookies and
 * enqueues child URLs. In Download phase, marks dependencies complete.
 * 
 * @listens SystemEventBus#JOB:COMPLETED
 */

/**
 * @private
 * @async
 * @method _transitionToDownload
 * @returns {Promise<void>}
 * @description Phase transition handler. Runs ConflictResolver, displays plan
 * via PlanDisplayer, prompts user for confirmation, builds download queue.
 * 
 * @fires SystemEventBus#PHASE:CHANGED
 */

/**
 * @private
 * @method _bootstrapSession
 * @description Executes the first request on a single worker to capture
 * auth cookies before enabling full concurrency.
 */

/**
 * @private
 * @method _broadcastCookies
 * @param {Array<Object>} cookies - Session cookies from first successful scrape
 * @returns {void}
 * @description Sends MSG_SET_COOKIES to all workers. Ensures authentication
 * state is shared across the pool.
 */

/**
 * @method shutdown
 * @returns {Promise<void>}
 * @description Gracefully terminates all workers and closes the event bus.
 * Called on SIGINT or after download completion.
 * 
 * @fires SystemEventBus#CMD:SHUTDOWN
 */

/**
 * @method getStatus
 * @returns {Object} Current orchestration state for monitoring
 * @property {string} phase - Current workflow phase
 * @property {Object} queueStats - Statistics from GlobalQueueManager
 * @property {Object} clusterStats - Worker pool statistics
 * @description Provides snapshot for StatisticsDisplayer.
 */
```

### 6.6 Refactored Existing Classes

#### `src/scraping/PageProcessor.js` (Stateless Refactor)

```javascript
/**
 * @fileoverview Stateless Page Scraping Logic
 * @module scraping/PageProcessor
 * @description Extracts links and metadata from a single page. Runs in Worker
 * process. Does NOT maintain global state or context maps.
 * 
 * **BEFORE REFACTOR**: Held reference to global StateManager and ContextMap.
 * **AFTER REFACTOR**: Pure function - all inputs via parameters, all outputs
 * via return value.
 */

/**
 * @class PageProcessor
 * @description Encapsulates the page scraping algorithm. Stateless - can be
 * instantiated fresh for each task without side effects.
 */

/**
 * @constructor
 * @param {Browser} browser - Puppeteer browser instance (from WorkerEntrypoint)
 * @param {Config} config - Configuration object (clone, not shared reference)
 */

/**
 * @async
 * @method execute
 * @param {string} url - Target page URL
 * @param {Array<Object>} cookies - Authentication cookies to inject
 * @returns {Promise<Object>} Scraping result
 * @property {string} result.title - Page title
 * @property {Array<string>} result.childUrls - Discovered links (absolute URLs)
 * @property {Array<string>} result.assetUrls - CSS/JS/image URLs
 * @property {string} [result.savedPath] -  Worker confirms where it saved
 * @throws {Error} If navigation fails or timeout exceeded
 * @description Main execution method. Navigates to URL, injects cookies,
 * extracts data, returns plain object (no PageContext creation - that's the
 * Master's job).
 */

/**
 * @private
 * @async
 * @method _navigate
 * @param {Page} page - Puppeteer page instance
 * @param {string} url - Target URL
 * @returns {Promise<void>}
 * @description Handles navigation with retry logic and timeout.
 */

/**
 * @private
 * @async
 * @method _extractLinks
 * @param {Page} page - Puppeteer page instance
 * @returns {Promise<Array<string>>} Absolute URLs of links
 * @description Evaluates JavaScript in page context to extract href attributes.
 */

/**
 * @private
 * @async
 * @method _extractAssets
 * @param {Page} page - Puppeteer page instance
 * @returns {Promise<Array<string>>} Absolute URLs of assets
 * @description Extracts <link>, <script>, and <img> src/href attributes.
 */
```


### 8. Critical Architectural Risks (Possible Bugs) Pay attention to when developing

#### A. The "HTML over IPC" Bottleneck (`PageProcessor.js`)
**The Flaw:**
The `PageProcessor.execute` JSDoc states:
> `@property {string} result.html - Raw HTML content`

**Why it fails:**
Passing raw HTML strings from the Worker to the Master via Node.js IPC is a major performance anti-pattern.
1.  **Serialization Cost:** JSON stringifying/parsing a 2MB HTML string blocks the Event Loop on both processes.
2.  **Memory Bloat:** The Master process (which is supposed to be lightweight) will spike in memory usage as it buffers these strings, only to likely write them to disk or discard them.

**The Fix:**
The **Worker** must write the file to disk.
*   **Discovery Phase:** Return `links`, `title`, and `metadata` only. Do *not* return HTML.
*   **Download Phase:** The Worker performs the download, rewriting, and saving. It returns `{ success: true, savedPath: '/...' }`.

#### B. Error Object Serialization (`ProtocolDefinitions.js`)
**The Flaw:**
> `@property {Error} [error] - Error object (present on failure)`

**Why it fails:**
Native JavaScript `Error` objects are **not JSON serializable**.
```javascript
JSON.stringify({ e: new Error('fail') }) // returns "{}"
```
If a worker crashes or fails, the Master will receive an empty object, making debugging impossible.

**The Fix:**
The `WorkerResult` typedef must define a **SerializedError** shape:
```javascript
@property {Object} [error]
@property {string} error.message
@property {string} error.name
@property {string} error.stack
```
The `TaskRunner._serializeError` method mentioned in the plan handles this, but the *Protocol Definition* JSDoc must reflect the *serialized* output, not the `Error` type.

#### C. The "Cookie Capture" Race Condition (`ClusterOrchestrator.js`)
**The Flaw:**
> `@description Handles task results. In Discovery phase, extracts cookies...`

**Why it fails:**
If `MAX_CONCURRENCY` is 8, the system might spawn 8 workers and send 8 discovery tasks simultaneously.
1.  Worker 1 hits the login page.
2.  Worker 2 hits a protected page (and fails/redirects because no cookies yet).
3.  Worker 1 finishes, returns cookies.
4.  Orchestrator broadcasts cookies.
Workers 2-8 have already likely failed or scraped a "Please Log In" page.

**The Fix:**
The `ClusterOrchestrator` needs a specific **Bootstrap Phase**.
1.  Spawn **only** 1 Worker initially (or ensure only 1 task is queued).
2.  Execute Root URL.
3.  Wait for `IPC_RESULT` containing cookies.
4.  Broadcast cookies.
5.  *Then* ramp up to `MAX_CONCURRENCY`.

#### D. Path Resolution Logic Gap (`GlobalQueueManager` vs. `Worker`)
**The Flaw:**
In the old architecture, `PageContext` was a tree. A node could call `this.parent.title` to calculate `Material/Week1/Intro`.
In the new architecture, `PageContext` is a flattened DTO (JSON). It only has `parentId` (string).
**The Worker cannot calculate the directory structure** because it doesn't know the titles of the parent, grandparent, etc.

**The Fix:**
The **Master** (specifically `GlobalQueueManager` or `ConflictResolver`) must calculate the target file path.
The `DownloadPayload` in `ProtocolDefinitions` needs an extra field:
```javascript
/**
 * @typedef {Object} DownloadPayload
 * @property {PageContext} context
 * @property {string} targetDirectory // Master tells Worker WHERE to save
 * @property {Map<string, string>} internalLinkMap // Master tells Worker how to rewrite
 */
```

---

### 9. Implementation & Logic Gaps

#### A. The `WorkerProxy` Ownership Ambiguity
**The Issue:**
`BrowserInitializer` spawns workers and returns proxies. `BrowserManager` registers them.
However, `BrowserInitializer` waits for `READY` *before* returning.
If the `WorkerEntrypoint` sends `READY`, but the `BrowserManager` hasn't registered the listeners yet, the event might be missed (though unlikely with `await`).

**Refinement:**
The `BrowserManager` should be the factory. `BrowserInitializer` should be a logic utility, not the owner of the spawning process.
*   *Better Flow:* `BrowserManager` calls `BrowserInitializer.spawnOne()`. `BrowserManager` attaches listeners immediately.

#### B. `ConflictResolver` JSDoc Scope
**The Issue:**
> `@method resolve ... @returns {Array<PageContext>}`

The logic for "Pruning" is tricky. The JSDoc implies it just returns a list of unique pages.
However, for the *Worker* to rewrite links correctly during the Download Phase, it needs a **Global Map** of `NotionID -> RelativeFilePath`.
The `ConflictResolver` must generate this map so it can be passed (or queried) during the download phase. If we don't generate this map *before* downloading starts, we can't rewrite links in a single pass.

## 7. Documentation Standards

All code must adhere to these standards:

1. **File Headers**: Every file must have `@fileoverview`, `@module`, and `@description` tags.
2. **Class Documentation**: Include `@class`, `@description`, `@property` for public properties, and `@example` where helpful.
3. **Method Documentation**: Include `@param` (with types), `@returns`, `@throws`, `@async`, and `@description`.
4. **Event Documentation**: Use `@fires` and `@listens` to document event-driven interactions.
5. **Type Safety**: Define `@typedef` for all IPC message payloads and results.
6. **Visibility**: Use `@private` for internal methods, `@public` for API surface.
7. **Examples**: Provide `@example` blocks for non-obvious usage patterns.
8. **Context Warnings**: Add `@description` notes warning about Master vs. Worker context where critical.

## 8. Conclusion

This refactoring will result in a robust, scalable system capable of handling large Notion workspaces efficiently. The separation of concerns (Master vs. Worker, Logic vs. Resource) will make the codebase easier to maintain and extend in the future. The comprehensive JSDoc specifications ensure that every class, method, and interaction is clearly documented for future maintainability.
