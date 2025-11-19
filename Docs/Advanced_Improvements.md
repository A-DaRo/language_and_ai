This is a comprehensive architectural breakdown for the **Reactive Event-Driven Micro-Kernel** architecture. This design moves the system from a monolithic sequential executor to a distributed, multi-process system coordinated by a central event bus.

The following classes represent the complete blueprint. They are categorized by their **Runtime Context** (Master Process vs. Worker Process) and their **Package**.

---

# 1. The Nervous System (Shared Infrastructure)

These classes provide the communication backbone. They exist primarily in the **Master Process** but define the protocols used by the whole system.

### `SystemEventBus` (New)
*   **Package:** `src/core`
*   **Design Pattern:** **Singleton / Observer / Mediator**
*   **Description:** The central nervous system. It decouples the implementation of resources (Browsers) from the logic of execution (Orchestrator). It extends Node.js `EventEmitter`.
*   **Functioning:**
    *   All components subscribe to relevant events here.
    *   Components emit events rather than calling methods on other components.
    *   **Key Events:** `SYSTEM:INIT`, `WORKER:REGISTERED`, `WORKER:AVAILABLE`, `WORKER:BUSY`, `JOB:SUBMIT`, `JOB:COMPLETED`, `CMD:SHUTDOWN`.

### `ProtocolDefinitions` (New)
*   **Package:** `src/core`
*   **Design Pattern:** **Data Transfer Object (DTO)**
*   **Description:** A collection of static definitions or Typescript interfaces ensuring strict typing for Inter-Process Communication (IPC).
*   **Functioning:**
    *   Defines the shape of messages like `IPC_INIT`, `IPC_DISCOVER_REQUEST`, `IPC_DOWNLOAD_REQUEST`, `IPC_RESULT_SUCCESS`.
    *   Prevents "magic string" errors between Master and Worker.

---

# 2. The Cluster Layer (Master Process)

These classes manage the physical resources (Node.js Child Processes). They act as the "Micro-Kernel," managing the hardware abstraction.

### `BrowserInitializer` (New)
*   **Package:** `src/cluster`
*   **Design Pattern:** **Factory Method**
*   **Description:** Responsible for the initial physical creation of resources based on system capacity.
*   **Functioning:**
    *   On `SYSTEM:INIT`, it utilizes the `os` module to calculate available RAM.
    *   Determines `MAX_CONCURRENCY` (e.g., Free RAM / 1GB).
    *   Loops to `child_process.fork()` the `src/worker/entrypoint.js`.
    *   Does **not** manage the lifecycle after spawning; simply hands the process reference to the `BrowserManager`.

### `BrowserManager` (New)
*   **Package:** `src/cluster`
*   **Design Pattern:** **Object Pool**
*   **Description:** The Gatekeeper. It manages the pool of worker resources, tracking who is idle and who is busy.
*   **Functioning:**
    *   Listens for `WORKER:REGISTERED`. Wraps the raw process in a `WorkerProxy`.
    *   Maintains two collections: `idleWorkers` (Stack) and `busyWorkers` (Map).
    *   Listens for `CMD:EXECUTE`. Pops an idle worker, marks it busy, and delegates the command.
    *   Listens for `JOB:COMPLETED`. Moves the worker back to the `idleWorkers` stack and emits `WORKER:AVAILABLE`.
    *   Handles process death/restarts (Supervisor logic).

### `WorkerProxy` (New)
*   **Package:** `src/cluster`
*   **Design Pattern:** **Proxy / Adapter**
*   **Description:** A Master-side representation of a remote Worker Process. It bridges the IPC gap.
*   **Functioning:**
    *   Wraps the `ChildProcess` instance.
    *   **Inbound:** Listens to `process.on('message')`. Converts raw JSON from the child into typed `SystemEventBus` events (e.g., receiving `IPC_RESULT` emits `JOB:COMPLETED`).
    *   **Outbound:** Exposes methods like `sendCommand(type, payload)` which uses `process.send()` to serialize instructions to the child.

---

# 3. The Orchestration Layer (Master Process)

These classes represent the "Brain." They contain the business logic, state, and decision-making algorithms. They do not touch Puppeteer directly.

### `ClusterOrchestrator` (Refactored from `NotionScraper`)
*   **Package:** `src/orchestration`
*   **Design Pattern:** **Mediator / State Machine**
*   **Description:** The high-level coordinator. It connects the `GlobalQueueManager` (Logic) to the `BrowserManager` (Resources).
*   **Functioning:**
    *   **State 1 (Discovery):** Listens for `WORKER:AVAILABLE`. Asks Queue for next URL. Dispatches `DISCOVER` command.
    *   **State 2 (Pruning):** Pauses workers. Triggers `ConflictResolver`.
    *   **State 3 (Download):** Listens for `WORKER:AVAILABLE`. Asks Queue for next Leaf Node. Dispatches `DOWNLOAD` command.
    *   **Cookie Management:** Captures cookies from the first worker and broadcasts `IPC_SET_COOKIES` to the pool.

### `GlobalQueueManager` (New/Refactored)
*   **Package:** `src/orchestration`
*   **Design Pattern:** **Priority Queue / Monitor**
*   **Description:** Manages the BFS frontier and dependency blocking.
*   **Functioning:**
    *   **Discovery Phase:** Standard FIFO Queue for BFS. Ensures URL uniqueness.
    *   **Download Phase:** A specialized "Ready Queue."
        *   Stores the dependency graph (Parent -> Children).
        *   Only releases a Parent Node when all Children have emitted `JOB:COMPLETED`.
        *   Prioritizes Leaf Nodes (bottom-up).

### `ConflictResolver` (New)
*   **Package:** `src/orchestration/analysis`
*   **Design Pattern:** **Strategy / Filter**
*   **Description:** Implements the "Maximal-Micro-Step" logic to prune the graph into a strict tree for downloading.
*   **Functioning:**
    *   Runs between Discovery and Download phases.
    *   Analyzes the full graph. Identifies the "Canonical" instance of every page (closest to root).
    *   Marks all other instances as "Internal References" (skips download, marks for link rewriting).
    *   Generates the `urlToLocalPathMap` used for global link resolution.

### `PlanDisplayer` & `UserPrompt` (Refactored)
*   **Package:** `src/orchestration/ui`
*   **Description:** Updated to work with the asynchronous nature of the new architecture. They now listen to `SystemEventBus` events (e.g., `DISCOVERY:FINISHED`) to trigger the UI, rather than being called sequentially.

---

# 4. The Worker Layer (Worker Process)

These classes run inside the **Child Process**. They are isolated, stateless, and focus purely on execution.

### `WorkerEntrypoint` (New)
*   **Package:** `src/worker`
*   **Design Pattern:** **Entry Point / Bootstrapper**
*   **Description:** The `main` file executed by `child_process.fork()`.
*   **Functioning:**
    *   Initializes the `puppeteer` instance.
    *   Sends the `IPC_READY` handshake to the parent.
    *   Sets up the `process.on('message')` listener to route payloads to the `TaskRunner`.

### `TaskRunner` (New)
*   **Package:** `src/worker`
*   **Design Pattern:** **Command / Switchboard**
*   **Description:** The logic controller inside the worker.
*   **Functioning:**
    *   Receives parsed commands (`INIT`, `DISCOVER`, `DOWNLOAD`).
    *   Instantiates the necessary processing classes (`PageProcessor`, `AssetDownloader`) on demand.
    *   executes the task.
    *   Catches errors and sends `IPC_ERROR` back to parent.
    *   On success, sends `IPC_RESULT` back to parent.

### `PageProcessor` (Refactored)
*   **Package:** `src/scraping`
*   **Design Pattern:** **Service**
*   **Description:** Refactored to be **Stateless**.
*   **Changes:**
    *   Previously, it might have held state about the BFS tree. Now, it acts purely on the inputs provided in the `DOWNLOAD` command.
    *   It no longer calculates where to save files. The Master provides the `savePath` in the command payload.
    *   It no longer calculates relative links. The Master provides a `linkMap` or the worker requests resolution via IPC.

### `LinkRewriter` (Refactored)
*   **Package:** `src/processing`
*   **Description:** Moves logic to the worker.
*   **Changes:**
    *   Now performs "Hot Rewriting" during the download phase.
    *   Uses the `internalLinkMap` passed by the Master to rewrite `href` attributes in the DOM before saving to disk.

### `AssetDownloader` & `ContentExpander` (Refactored)
*   **Package:** `src/download` & `src/processing`
*   **Description:** These largely remain the same logic-wise but are instantiated inside the `WorkerProcess` context. They share the worker's single Puppeteer `Page` instance.

---

# 5. The Domain Layer (Shared)

### `PageContext` (Refactored)
*   **Package:** `src/domain`
*   **Design Pattern:** **Data Transfer Object (Serializable)**
*   **Description:** The data model for a page.
*   **Changes:**
    *   Must be fully **JSON-Serializable**. Circular references (parents holding children) must be handled carefully (e.g., using IDs instead of object references) so they can be passed over IPC between Master and Worker.

---

# Summary of Flow

1.  **Master:** `BrowserInitializer` spawns processes.
2.  **Master:** `BrowserManager` registers them via `WorkerProxy`.
3.  **Master:** `GlobalQueueManager` holds Root URL.
4.  **Master:** `ClusterOrchestrator` sees `WORKER:AVAILABLE` + Queue Item -> Emits `CMD:EXECUTE` (Discover).
5.  **IPC:** `WorkerProxy` sends JSON to Child.
6.  **Worker:** `WorkerEntrypoint` -> `TaskRunner` -> `PageProcessor`.
7.  **Worker:** Scrapes -> sends `IPC_RESULT` (Links).
8.  **IPC:** `WorkerProxy` receives JSON -> Emits `JOB:COMPLETED`.
9.  **Master:** `GlobalQueueManager` adds new links to BFS queue.
10. **Repeat** until Queue empty.