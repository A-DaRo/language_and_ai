### Critique to proposed plan

The proposed plan is a **solid tactical response** to the immediate symptoms ("ghost execution" and missing titles). It correctly identifies that the `TaskRunner` contains stub logic and introduces the **Pipeline Pattern** to organize the complex scraping workflow.

However, from a **System Architecture** perspective, the plan has three significant gaps that, if left unaddressed, will lead to `v2` bugs (broken offline styles, path permission errors, and race conditions):

1.  **Path Resolution Responsibility Misplacement**:
    *   *proposed Plan:* Suggests the Worker should calculate paths using `process.cwd()`.
    *   *Architecture Violation:* The Worker is "dumb." It should not know about the project root or configuration flags like `OUTPUT_DIR`.
    *   *Correction:* The **Master (GlobalQueueManager)** must calculate the **Absolute Path** and send it in the `IPC_DOWNLOAD` payload. The Worker should simply write to the path it is told.

2.  **Incomplete Content Strategy (The CSS Gap)**:
    *   *proposed Plan:* Suggests "deferring complex CSS... to post-processing."
    *   *Risk:* This violates the "1:1 Offline Replica" requirement. If the Worker saves HTML without downloading and rewriting CSS links *immediately*, the downloaded page is broken until some undefined later stage. The Worker must handle CSS during the download task.

3.  **Component Lifecycle Ambiguity**:
    *   *proposed Plan:* "Instantiate missing components in constructor."
    *   *Architecture Alignment:* `AssetDownloader` and `CssDownloader` have internal caches. If they are re-instantiated per task, we lose caching efficiency. If they are instantiated once per Worker, we need to ensure they are stateless regarding the *current page* (e.g., clearing visited sets between tasks).

---

### Enhanced & Complete Refactoring Plan"

#### Phase 1: Protocol & Data Integrity (Master Side)

**Goal**: Ensure the Worker receives complete, executable instructions without needing to calculate state.

1.  **Absolute Path Generation (`GlobalQueueManager`)**:
    *   Update `GlobalQueueManager.nextDownload()` to calculate the **absolute filesystem path** for the target page.
    *   Use `path.resolve(config.outputDir, context.relativeDir, 'index.html')`.
    *   **Update `ProtocolDefinitions.js`**: Ensure `DownloadPayload` strictly types `savePath` as a string.

2.  **Title Registry Synchronization (`ClusterOrchestrator`)**:
    *   Implement the `_broadcastRegistryUpdate()` method.
    *   **Trigger**: Immediately after the **Discovery Phase** ends and **Pruning** is complete.
    *   **Action**: Send a new `IPC_UPDATE_REGISTRY` message to all idle workers *before* starting the Execution Phase. This ensures workers have the full title map before the first download task.

3.  **Master-Side Payload Validation**:
    *   The `BrowserManager` should validate that the `savePath` is absolute and the `linkMap` is populated *before* sending the IPC message. Fail early on the Master side to prevent Worker crashes.

#### Phase 2: The Scraping Pipeline (Worker Side)

**Goal**: Replace the `TaskRunner` stub with a sequential, error-tolerant processing chain.

**New Module**: `src/worker/pipeline/`

1.  **`ScrapingPipeline.js` (The Controller)**:
    *   Accepts an array of `PipelineStep` instances.
    *   Maintains a `PipelineContext` object (shared state for *this specific task only*).
    *   Executes steps in order. If a step fails, it captures the error, logs the stage, and aborts.

2.  **The Steps (Implementation)**:
    *   **`NavigationStep`**: Wraps `page.goto` with `networkidle0`. Handles 404s.
    *   **`AuthCheckStep`**: Verifies cookies are working (checks for "Login" text).
    *   **`ExpansionStep`**: Uses `ContentExpander` to open toggles/databases.
    *   **`AssetCollectionStep`**: Scans DOM for `<img>`, `<video>`, `<link rel="stylesheet">`, and script tags. Populates `context.assets`.
    *   **`AssetDownloadStep`**:
        *   Iterates `context.assets`.
        *   Uses `AssetDownloader` (for media) and `CssDownloader` (for styles).
        *   **Crucial**: Downloads CSS *and* parses it to download fonts/images referenced inside CSS.
    *   **`LinkRewriterStep`**:
        *   Uses `payload.linkMap` to rewrite `<a>` tags to relative local paths.
        *   Rewrites `<img>` and `<link>` tags to point to downloaded assets.
    *   **`HtmlWriteStep`**:
        *   Uses `WorkerFileSystem` to save the final DOM state to `payload.savePath`.

#### Phase 3: Infrastructure Hardening (Cluster Layer)

**Goal**: Prevent "Ghost Workers" and handle crashes gracefully.

1.  **Worker Initialization Fix (`BrowserManager`)**:
    *   Modify `BrowserManager.spawnWorker()`.
    *   When a worker sends `IPC_READY`, immediately send `IPC_INIT` containing the **Configuration** and the **Current Title Registry**.
    *   This ensures that respawned workers (after a crash) are identical to original workers.

2.  **The `WorkerFileSystem` Adapter (`src/worker/io/`)**:
    *   Implement `safeWrite(absolutePath, content)`.
    *   **Verification**: Before writing, check `path.isAbsolute(absolutePath)`. If false, throw critical error.
    *   **Logging**: Log every write operation with the file size and destination. This is the "Anti-Ghost" measure.

### Implementation Order

1.  **Master**: Update Path Generation & Title Broadcasting. (Fixes data input).
2.  **Worker IO**: Implement `WorkerFileSystem`. (Fixes data output visibility).
3.  **Worker Logic**: Implement `ScrapingPipeline` & Steps. (Fixes the "stub" logic).
4.  **Cluster**: Update Respawn/Init logic. (Fixes stability).


## Proposed JSDocs

### 1. Core Protocol Definitions (Refactored)

**File:** `src/core/ProtocolDefinitions.js`

```javascript
/**
 * @fileoverview IPC Protocol Definitions
 * @module core/ProtocolDefinitions
 * @description Defines the strict data contracts for Master-Worker communication,
 * updated to support absolute paths and lazy title registry synchronization.
 */

/**
 * @typedef {Object} DownloadPayload
 * @description Command payload for the Execution Phase (Asset Download).
 * @property {string} taskId - Unique tracking ID.
 * @property {string} pageId - Notion Page ID (32-char hex).
 * @property {string} url - The canonical URL to scrape.
 * @property {string} savePath - **ABSOLUTE** filesystem path where index.html must be saved.
 * @property {Object<string, string>} linkMap - Map of [PageID -> RelativePath] for link rewriting.
 * @property {Array<Object>} cookies - Session cookies to apply before navigation.
 * @property {Object} [context] - Serialized PageContext (optional metadata).
 */

/**
 * @typedef {Object} InitPayload
 * @description Command payload for Worker initialization.
 * @property {Object} config - General system configuration.
 * @property {string} workerId - Unique ID assigned to this worker.
 * @property {Object<string, string>} [titleRegistry] - Initial ID-to-Title map.
 */

/**
 * @typedef {Object} UpdateRegistryPayload
 * @description Command to synchronize the Title Registry after Discovery/Pruning.
 * @property {Object<string, string>} titleRegistry - The authoritative ID-to-Title map.
 */
```

### 2. Global Queue Manager (Refactored)

**File:** `src/orchestration/GlobalQueueManager.js`

```javascript
/**
 * @fileoverview Central Task Queue & Path Calculator
 * @module orchestration/GlobalQueueManager
 */

/**
 * @class GlobalQueueManager
 * @description Manages the BFS frontier and calculates absolute filesystem paths
 * to prevent "ghost writes" in workers.
 */

/**
 * @method nextDownload
 * @summary Retrieves the next available leaf node for downloading.
 * @description Pops a page from the ready queue. Calculates the **absolute**
 * output path based on the configured output directory and the page's relative structure.
 * 
 * @param {string} outputDir - The system's root output directory (absolute).
 * @returns {DownloadPayload|null} The fully constructed payload or null if queue empty.
 */

/**
 * @method _calculateAbsolutePath
 * @private
 * @summary Resolves the target filesystem path.
 * @param {string} outputDir - Root output directory.
 * @param {PageContext} context - The page context containing hierarchy info.
 * @returns {string} The absolute path ending in 'index.html'.
 */
```

### 3. Browser Manager (Refactored)

**File:** `src/cluster/BrowserManager.js`

```javascript
/**
 * @fileoverview Worker Pool Lifecycle Manager
 * @module cluster/BrowserManager
 */

/**
 * @class BrowserManager
 * @description Manages the pool of WorkerProxies. Handles spawning, allocation,
 * and critically, the re-initialization of workers after crashes.
 */

/**
 * @method spawnWorker
 * @summary Creates a new physical worker process.
 * @description Forks a new process. Upon receiving the `IPC_READY` handshake,
 * immediately sends the `IPC_INIT` message containing the cached Config and Title Registry.
 * 
 * @param {string} workerId - Unique ID for the new worker.
 * @returns {Promise<WorkerProxy>} The initialized proxy.
 */

/**
 * @method _handleWorkerExit
 * @private
 * @summary Handles unexpected worker termination.
 * @description Cleans up the dead proxy. Spawns a replacement worker and
 * **guarantees** it receives the current Title Registry payload so it can
 * resume work immediately.
 * 
 * @param {string} workerId - ID of the crashed worker.
 * @fires SystemEventBus#WORKER:DIED
 */
```

### 4. Worker I/O Adapter (New)

**File:** `src/worker/io/WorkerFileSystem.js`

```javascript
/**
 * @fileoverview I/O Abstraction Layer
 * @module worker/io/WorkerFileSystem
 */

/**
 * @class WorkerFileSystem
 * @description A facade for Node.js `fs` operations that enforces safety checks
 * and provides granular logging to prevent "ghost execution".
 */

/**
 * @constructor
 * @description Initializes the file system adapter.
 * @param {Logger} logger - Worker-context logger instance.
 */

/**
 * @method safeWrite
 * @async
 * @summary Writes content to a specific absolute path.
 * @description Ensures the target directory exists (mkdir -p). Verifies the path
 * is absolute. Writes the file and logs the operation size.
 * 
 * @param {string} absolutePath - Target file path. MUST be absolute.
 * @param {string|Buffer} content - Content to write.
 * @returns {Promise<void>}
 * @throws {Error} If path is relative or write fails.
 */

/**
 * @method ensureDir
 * @async
 * @summary Idempotent directory creation.
 * @param {string} dirPath - Absolute directory path.
 * @returns {Promise<void>}
 */
```

### 5. Scraping Pipeline (New)

**File:** `src/worker/pipeline/ScrapingPipeline.js`

```javascript
/**
 * @fileoverview Linear Execution Controller
 * @module worker/pipeline/ScrapingPipeline
 */

/**
 * @class ScrapingPipeline
 * @description Orchestrates the sequential execution of scraping steps.
 * Implements the "Pipe and Filter" pattern to isolate failures.
 */

/**
 * @constructor
 * @param {Array<PipelineStep>} steps - Ordered list of steps to execute.
 * @param {Logger} logger - Logger instance.
 */

/**
 * @method execute
 * @async
 * @summary Runs the pipeline for a specific task.
 * @description Iterates through steps. Passes the shared `PipelineContext` to each.
 * Stops execution immediately if a step throws an error.
 * 
 * @param {PipelineContext} context - Shared state object (Browser, Page, Config, Payload).
 * @returns {Promise<void>}
 * @throws {Error} Propagates error from failing step.
 */
```

### 6. Pipeline Steps (New)

**File:** `src/worker/pipeline/PipelineStep.js` (Base)

```javascript
/**
 * @class PipelineStep
 * @abstract
 * @description Interface for a single unit of work in the scraping process.
 */

/**
 * @method process
 * @abstract
 * @async
 * @param {PipelineContext} context - Mutable context object.
 * @returns {Promise<void>}
 */
```

**File:** `src/worker/pipeline/steps/NavigationStep.js`

```javascript
/**
 * @class NavigationStep
 * @extends PipelineStep
 * @description Handles page navigation with strict network idle checks.
 */

/**
 * @method process
 * @summary Navigates Puppeteer page to target URL.
 * @description Sets cookies. Calls `page.goto` with `networkidle0`.
 * Throws if 404 or timeout.
 */
```

**File:** `src/worker/pipeline/steps/AssetDownloadStep.js`

```javascript
/**
 * @class AssetDownloadStep
 * @extends PipelineStep
 * @description Identifies and downloads external resources.
 */

/**
 * @method process
 * @summary Downloads CSS, Images, and embedded Files.
 * @description
 * 1. Scans DOM for assets.
 * 2. Invokes `AssetDownloader` and `CssDownloader`.
 * 3. Rewrites DOM attributes (src, href) to point to local filenames.
 * 4. Updates `context.stats` with download counts.
 */
```

**File:** `src/worker/pipeline/steps/HtmlWriteStep.js`

```javascript
/**
 * @class HtmlWriteStep
 * @extends PipelineStep
 * @description Finalizes the scraping process by saving the DOM.
 */

/**
 * @method process
 * @summary Serializes and saves the current page state.
 * @description
 * 1. serializes `page.content()`.
 * 2. Invokes `WorkerFileSystem.safeWrite()` using the `savePath` from payload.
 */
```

### 7. Task Runner (Refactored)

**File:** `src/worker/TaskRunner.js`

```javascript
/**
 * @fileoverview Worker Command Router
 * @module worker/TaskRunner
 */

/**
 * @method handleDownload
 * @async
 * @summary Orchestrates the download task using the ScrapingPipeline.
 * @description
 * 1. Validates the payload (must have `savePath`).
 * 2. Constructs the `PipelineContext`.
 * 3. Instantiates the `ScrapingPipeline` with `[Navigation, Expansion, AssetDownload, LinkRewriter, HtmlWrite]`.
 * 4. Executes the pipeline.
 * 5. Returns success result with stats.
 * 
 * @param {DownloadPayload} payload - The download instruction.
 * @returns {Promise<WorkerResult>}
 */
```