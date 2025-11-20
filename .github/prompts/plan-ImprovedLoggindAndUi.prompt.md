# **Plan: High-Fidelity Terminal Dashboard Implementation**

**Objective:**
Refactor the system's logging and UI to implement a persistent, multi-line "System Monitor" dashboard. This will replace all verbose console output during execution, routing it to a file instead. The implementation must strictly adhere to the Observer and Strategy design patterns to maintain separation of concerns.

---

### **Developer Guidelines & Best Practices for UI/Logging Implementation**

#### **1. Guiding Principle: "The Worker is Blind and Mute"**

*   **Core Concept:** A Worker process should have zero awareness of the UI. It should never call `console.log` for status updates.
*   **Implementation Rule:**
    *   **NEVER** instantiate `ConsoleStrategy` inside a Worker. The *only* logging strategy a worker should ever have is `IpcStrategy`.
    *   **DO** use the logger (`logger.info`, `logger.debug`) liberally within worker modules (`TaskRunner`, `Pipeline Steps`). These calls will be converted into IPC messages automatically.
    *   **TEST:** After implementation, search the entire `src/worker` directory for `console.log`. Any instance found is a bug and must be replaced with `logger.info`.

#### **2. Guiding Principle: "The Master is the Single Source of Truth"**

*   **Core Concept:** The Master process dictates all state, including what is displayed. The UI is a passive "view" that only renders the state it is given.
*   **Implementation Rule:**
    *   The `TerminalDashboard` class should contain **NO application logic**. It should not calculate percentages, decide colors, or track progress. Its only job is to take strings and numbers and pass them to the `cli-progress` library.
    *   All stateful logic (e.g., "what is the current progress percentage?", "which worker is busy?") belongs in the `DashboardController`, which derives this state from `SystemEventBus` events.
    *   **TEST:** Review `TerminalDashboard.js`. If you see any logic like `this.completedTasks++`, it's a bug. That state belongs in the controller.

#### **3. On Event-Driven Integrity**

*   **Core Concept:** The UI must be 100% reactive to `SystemEventBus` events. Do not use timers (`setInterval`) to poll for state changes.
*   **Implementation Rule:**
    *   **DO** add a `setInterval` *only* for the "Elapsed Time" display in the header. This is a purely cosmetic timer and is acceptable.
    *   For all other dynamic data (progress, worker status), the update must be triggered by an event handler in `DashboardController`.
    *   **Instruction:** The developer must identify every piece of information displayed on the two dashboard "Looks" and ensure a corresponding event is emitted by the `ClusterOrchestrator` or `GlobalQueueManager` to drive its updates. If an event is missing (e.g., `CONFLICT:DETECTED`), it must be added.

#### **4. On Code Consistency & Style**

*   **Instruction:** All new UI-related modules must strictly adhere to the project's existing coding standards.
*   **Action Checklist:**
    *   Does every new file (`TerminalDashboard.js`, `DashboardController.js`, `IpcStrategy.js`, etc.) have a file-level JSDoc header (`@fileoverview`, `@module`)?
    *   Does every class and method have a complete JSDoc block?
    *   Are all class methods and properties `camelCase`? Are constants `UPPER_SNAKE_CASE`?
    *   Are all dependencies (`require` statements) at the top of the file?

#### **5. The "Halt and Clear" User Experience**

*   **Core Concept:** The transition from the verbose discovery phase to the clean dashboard must be seamless and deliberate.
*   **Implementation Rule:**
    *   The sequence in `main-cluster.js` is critical and must be followed precisely:
        1.  Discovery phase runs (verbose `ConsoleStrategy` active).
        2.  After discovery, the `PlanDisplayer` shows the site tree.
        3.  The `UserPrompt` waits for input.
        4.  **CRITICAL:** Before initializing the dashboard, you **must** call `console.clear()`. This is a native Node.js command that wipes the terminal screen.
        5.  Immediately after clearing, initialize the `DashboardController` and switch the logger to `DashboardStrategy`. The dashboard will appear on a fresh screen.

#### **6. IMPORTANT REMINDERS**

*   **Reminder 1: Update the Architecture Document (`ARCHITECTURE.md`)**
    *   **Instruction:** After the code is implemented and working, the developer **must** update the architecture documentation.
    *   **Specific Sections to Update:**
        1.  **`High-Level Architecture Overview`**: Add a new sub-section explaining the logging architecture (the Strategy Pattern, `IpcStrategy`, and the role of the Master logger).
        2.  **`Package Structure`**: Add the new `src/ui` package and its components (`TerminalDashboard`, `DashboardController`).
        3.  **`Core Package`**: Update the `Logger.js` section to describe the new `switchMode` method and the multi-strategy system.
        4.  **`Data Flow Diagram`**: Add a new "Logging Flow" diagram or note showing how worker logs are piped through IPC to the Master's `FileStrategy` and `DashboardStrategy`.

*   **Reminder 2: Update the Testing Suite (`tests/`)**
    *   **Instruction:** The new UI components are critical and must be unit tested.
    *   **Specific Tests to Add/Update:**
        1.  **`tests/unit/core/Logger.test.js`**: Add tests for the `switchMode` method. Verify that it correctly removes `ConsoleStrategy` and adds `DashboardStrategy` while leaving `FileStrategy` untouched.
        2.  **`tests/unit/ui/DashboardController.test.js` (New)**: This is a crucial test.
            *   Use a `MockSystemEventBus` and a mocked `TerminalDashboard`.
            *   Emit a `WORKER:BUSY` event on the mock bus.
            *   Assert that `dashboard.updateWorkerStatus` was called with the correct arguments.
            *   Test all event-to-UI mappings.
        3.  **`tests/unit/core/logger/IpcStrategy.test.js` (New)**:
            *   Mock `process.send`.
            *   Call `strategy.log()`.
            *   Assert that `process.send` was called with a correctly formatted `IPC_LOG` message.

---

## **Plan Implementation Steps**

### **Phase 1: Refactor the Core Logging System**

**Goal:** Transform the `Logger` from a simple dispatcher into a flexible system that supports different output modes without destructive re-initialization.

#### **Step 1.1: Refine `FileStrategy.js` for "As-Is" Verbose Logging**

*   **Instruction:** The `FileStrategy` should no longer format output as a Markdown table. It must produce a clean, verbose log file that mirrors what the `ConsoleStrategy` would output.
*   **File:** `src/core/logger/FileStrategy.js`
*   **Action:**
    *   Remove the Markdown table header (`| Time | Level | ...`) from the constructor.
    *   Modify the `log()` method to write a simple, formatted string:
        ```javascript
        // Inside FileStrategy.log()
        const time = new Date().toLocaleTimeString();
        const levelTag = `[${level.toUpperCase()}]`;
        const categoryTag = `[${category}]`;
        let line = `${time} ${levelTag} ${categoryTag} ${message}\n`;

        if (meta && (level === 'error' || level === 'debug')) {
            line += `  └─ ${JSON.stringify(meta, null, 2).replace(/\n/g, '\n     ')}\n`;
        }
        this.stream.write(line);
        ```

#### **Step 1.2: Update `Config.js` for Log Configuration**

*   **Instruction:** Add configuration options to control the logging behavior.
*   **File:** `src/core/Config.js`
*   **Action:**
    *   Add `this.LOG_DIR = './logs'`.
    *   Add `this.LOG_FILE_ENABLED = true`.

#### **Step 1.3: Introduce `IpcStrategy` for Worker Logging**

*   **Instruction:** Create a new logging strategy for workers that sends log data to the Master via IPC instead of writing to stdout. This is the **key** to cleaning up the terminal.
*   **File:** `src/core/logger/IpcStrategy.js` (New)
*   **Action:** Implement the class as detailed in the previous response. It should have a single `log()` method that calls `process.send()`.

#### **Step 1.4: Refactor `Logger.js` for Safe Mode Switching**

*   **Instruction:** Replace the destructive `reinit` logic with a safe `switchMode` method.
*   **File:** `src/core/Logger.js`
*   **Action:**
    *   Remove the `reinit` option from the `init()` method's JSDoc and logic.
    *   Add the new `switchMode(mode, context)` method as detailed in the previous response. This method will be responsible for removing the `ConsoleStrategy` and adding the `DashboardStrategy`.

---

### **Phase 2: Build the UI Components**

**Goal:** Create the "dumb" rendering components that will display the dashboard.

#### **Step 2.1: Implement `TerminalDashboard.js`**

*   **Instruction:** Implement the renderer for both "Discovery" and "Download" modes. This class should only manage `cli-progress` bars and have no business logic.
*   **File:** `src/ui/TerminalDashboard.js` (New)
*   **Action:** Implement the JSDoc and class structure below.

```javascript
/**
 * @fileoverview Renderer for the Multi-Bar Terminal Dashboard
 * @module ui/TerminalDashboard
 */

const cliProgress = require('cli-progress');

/**
 * @class TerminalDashboard
 * @description Encapsulates the rendering logic for the terminal UI. It is a "dumb"
 * component that receives data and updates its visual state. It does not contain
 * any application logic.
 */
class TerminalDashboard {
  /**
   * @constructor
   * @param {number} workerCount - The number of worker slots to create.
   */
  constructor(workerCount) {
    this.multibar = new cliProgress.MultiBar({
      clearOnComplete: false,
      hideCursor: true,
      format: '{bar} | {label}', // A generic format
    }, cliProgress.Presets.shades_classic);

    // Header and Footer for static text
    this.header = this.multibar.create(1, 1, { label: 'Initializing...' });
    this.header.setFormat('{label}');

    this.footer = this.multibar.create(1, 1, { label: 'System is starting...' });
    this.footer.setFormat('  └─ {label}');

    // Dynamic bars for progress and workers
    this.progressBars = new Map();
    this.workerBars = new Map();

    for (let i = 0; i < workerCount; i++) {
      const bar = this.multibar.create(1, 0, { label: `Worker ${i + 1}: [IDLE]` });
      bar.setFormat('   {label}');
      this.workerBars.set(i, bar);
    }
  }

  /**
   * @method setMode
   * @summary Reconfigures the dashboard layout for a specific phase.
   * @param {'discovery' | 'download'} mode - The phase to display.
   * @param {Object} [initialData={}] - Initial data for the mode.
   */
  setMode(mode, initialData = {}) {
    this.progressBars.forEach(bar => this.multibar.remove(bar));
    this.progressBars.clear();

    if (mode === 'discovery') {
      const bar = this.multibar.create(1, 0, { label: 'Pages Found: 0 | In Queue: 0 | Conflicts: 0' });
      bar.setFormat('  Progress: {label}');
      this.progressBars.set('discoveryStats', bar);
    } else if (mode === 'download') {
      const bar = this.multibar.create(initialData.total || 1, initialData.completed || 0, {
        label: `[Pending: ${initialData.pending || 0}] [Active: 0] [Complete: 0/${initialData.total || 0}] [Failed: 0]`
      });
      bar.setFormat('  Progress: {label}');
      this.progressBars.set('downloadStats', bar);
    }
  }

  /**
   * @method updateHeader
   * @param {string} title - The main title to display.
   */
  updateHeader(title) {
    this.header.update(1, { label: title });
  }

  /**
   * @method updateDiscoveryStats
   * @param {Object} stats - Discovery statistics.
   * @param {number} stats.pagesFound
   * @param {number} stats.inQueue
   * @param {number} stats.conflicts
   * @param {number} stats.currentDepth
   */
  updateDiscoveryStats({ pagesFound, inQueue, conflicts, currentDepth }) {
    const bar = this.progressBars.get('discoveryStats');
    if (bar) {
      bar.update(1, { label: `Pages Found: ${pagesFound} | In Queue: ${inQueue} | Conflicts: ${conflicts} | Current Depth: ${currentDepth}` });
    }
  }

  /**
   * @method updateDownloadStats
   * @param {Object} stats - Download statistics.
   * @param {number} stats.pending
   * @param {number} stats.active
   * @param {number} stats.completed
   * @param {number} stats.total
   * @param {number} stats.failed
   */
  updateDownloadStats({ pending, active, completed, total, failed }) {
    const bar = this.progressBars.get('downloadStats');
    if (bar) {
        bar.setTotal(total);
        bar.update(completed, { label: `[Pending: ${pending}] [Active: ${active}] [Complete: ${completed}/${total}] [Failed: ${failed}]` });
    }
  }

  /**
   * @method updateWorkerStatus
   * @param {number} slotIndex - The visual slot for the worker.
   * @param {string} statusText - The text to display.
   */
  updateWorkerStatus(slotIndex, statusText) {
    const bar = this.workerBars.get(slotIndex);
    if (bar) {
      bar.update(1, { label: `Worker ${slotIndex + 1}: ${statusText}` });
    }
  }

  /**
   * @method updateFooter
   * @param {string} text - The text for the log ticker.
   */
  updateFooter(text) {
    this.footer.update(1, { label: text });
  }

  /**
   * @method stop
   * @description Stops the renderer and restores the cursor.
   */
  stop() {
    this.multibar.stop();
  }
}

module.exports = TerminalDashboard;
```

#### **Step 2.2: Implement `DashboardController.js`**

*   **Instruction:** Create the controller that listens to the `SystemEventBus` and calls the appropriate rendering methods on the `TerminalDashboard`.
*   **File:** `src/ui/DashboardController.js` (New)
*   **Action:** Implement the JSDoc and class structure below.

```javascript
/**
 * @fileoverview UI Logic Coordinator
 * @module ui/DashboardController
 */

/**
 * @class DashboardController
 * @description The "brain" of the UI. It translates system events into visual
 * updates on the dashboard, keeping the rendering logic simple.
 */
class DashboardController {
  /**
   * @constructor
   * @param {SystemEventBus} eventBus - The master event bus.
   * @param {BrowserManager} browserManager - For mapping worker IDs to slots.
   */
  constructor(eventBus, browserManager) {
    this.bus = eventBus;
    this.manager = browserManager;
    this.dashboard = null;
    this.workerSlotMap = new Map();
  }

  /**
   * @method start
   * @summary Initializes the dashboard and attaches all event listeners.
   */
  start() {
    const workerIds = this.manager.getAllWorkerIds();
    workerIds.forEach((id, index) => this.workerSlotMap.set(id, index));

    this.dashboard = new TerminalDashboard(workerIds.length);
    this._attachListeners();
  }

  /**
   * @method getDashboard
   * @returns {TerminalDashboard} The raw dashboard instance for the logger.
   */
  getDashboard() {
    return this.dashboard;
  }
  
  /**
   * @method stop
   * @description Stops the dashboard renderer.
   */
  stop() {
    if (this.dashboard) this.dashboard.stop();
  }

  /**
   * @private
   * @method _attachListeners
   * @description Subscribes to all relevant SystemEventBus events.
   */
  _attachListeners() {
    this.bus.on('PHASE:CHANGED', ({ phase, data }) => {
      this.dashboard.setMode(phase, data);
      this.dashboard.updateHeader(`JBC090 Language & AI | ${phase.toUpperCase()} Phase | Elapsed: ...`); // Elapsed time needs a ticker
    });

    this.bus.on('DISCOVERY:PROGRESS', (stats) => {
        this.dashboard.updateDiscoveryStats(stats);
    });

    this.bus.on('EXECUTION:PROGRESS', (stats) => {
        this.dashboard.updateDownloadStats(stats);
    });

    this.bus.on('WORKER:BUSY', ({ workerId, task }) => {
      const slot = this.workerSlotMap.get(workerId);
      this.dashboard.updateWorkerStatus(slot, `[BUSY] ${task.description}`);
    });

    this.bus.on('WORKER:AVAILABLE', ({ workerId }) => {
      const slot = this.workerSlotMap.get(workerId);
      this.dashboard.updateWorkerStatus(slot, `[IDLE] Waiting for task...`);
    });
  }
}

module.exports = DashboardController;
```

---

### **Phase 3: Integration into the Main Application**

**Goal:** Modify the application's entry point (`main-cluster.js`) and core components to support and drive the new UI system.

#### **Step 3.1: Update `main-cluster.js`**

*   **Instruction:** Implement the new workflow: Init with Console -> Discover -> Halt -> Prompt -> Clear & Switch to Dashboard -> Execute -> Shutdown.
*   **File:** `main-cluster.js`
*   **Action:**
    1.  At startup, initialize `Logger` with **both** `FileStrategy` and `ConsoleStrategy`.
    2.  After the `orchestrator.start()` call for discovery completes, **halt**.
    3.  Call `logger.removeStrategy(consoleStrategy)` to silence console logs.
    4.  Display the site tree and use `UserPrompt`.
    5.  If user says "yes":
        *   Call `console.clear()`.
        *   Instantiate `DashboardController` and call `start()`.
        *   Call `logger.switchMode('dashboard', { dashboardInstance: dashboardCtrl.getDashboard() })`.
        *   Proceed with the execution phase (`orchestrator.execute()`).
    6.  At the very end, call `dashboardCtrl.stop()` before printing the final summary.

#### **Step 3.2: Update `ClusterOrchestrator.js` to Emit Progress Events**

*   **Instruction:** The orchestrator must emit new events for the dashboard to consume.
*   **File:** `src/orchestration/ClusterOrchestrator.js`
*   **Action:**
    *   In the discovery loop, after processing links, emit `DISCOVERY:PROGRESS` with stats from `GlobalQueueManager`.
    *   In the execution loop, after a job completes, emit `EXECUTION:PROGRESS` with stats from `GlobalQueueManager`.
    *   When dispatching a task to a worker, include a user-friendly `description` in the `WORKER:BUSY` event payload.
        ```javascript
        // When dispatching a discovery task
        this.bus.emit('WORKER:BUSY', { workerId, task: { description: `Discovering '${title}'...` } });
        // When dispatching a download task
        this.bus.emit('WORKER:BUSY', { workerId, task: { description: `Downloading '${title}'...` } });
        ```

#### **Step 3.3: Update Worker Infrastructure**

*   **Instruction:** Ensure all workers use `IpcStrategy` for logging.
*   **File:** `src/worker/WorkerEntrypoint.js`
*   **Action:** As planned, inside the `IPC_INIT` handler, initialize the worker's logger instance with **only** the `IpcStrategy`.
*   **File:** `src/cluster/WorkerProxy.js`
*   **Action:** Ensure the `IPC_LOG` message handler is implemented to forward worker logs to the Master's logger instance.