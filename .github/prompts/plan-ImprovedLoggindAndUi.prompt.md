### 1. Architectural Overview

We will introduce a new package `src/ui` and refactor `src/core/Logger.js`.

**Data Flow:**
1.  **Master/Workers** do work and emit events via `SystemEventBus`.
2.  **Master/Workers** call `Logger.info()`, `Logger.error()` for verbose details.
3.  **Logger** routes text:
    *   → **Strategy A (File)**: Appends to `logs/run-XYZ.md` (Verbose).
    *   → **Strategy B (UI)**: Updates the "Recent Logs" section of the dashboard (Brief).
4.  **Dashboard** listens to `SystemEventBus`:
    *   Updates Progress Bars (on `JOB:COMPLETED`).
    *   Updates Worker Status Rows (on `WORKER:BUSY`/`IDLE`).

---

### 2. New Modules & Classes

#### A. Refactored Logger (Strategy Pattern)
We transform `Logger` from a basic console wrapper into a multi-transport dispatcher.

**File:** `src/core/logger/LoggerStrategies.js`
*   **`LogStrategy` (Interface)**: Defines `log(level, category, message)`.
*   **`ConsoleStrategy`**: Standard `console.log` (used in debug mode or if UI disabled).
*   **`FileStrategy`**: Writes markdown-formatted logs to a stream.
*   **`DashboardStrategy`**: Sends the latest log line to the UI component for display in a "ticker".

**File:** `src/core/Logger.js`
*   Maintains an array of active strategies.
*   `Logger.init({ file: true, dashboard: true })` sets up the strategies.

#### B. The Terminal Dashboard (The UI)
**File:** `src/ui/TerminalDashboard.js`
We will use `cli-progress` (specifically `MultiBar`) to handle the rendering complexity.

*   **Visual Layout:**
    1.  **Global Status**: Phase (Discovery/Download), Elapsed Time.
    2.  **Global Progress**: [====================] 45/100 (45%)
    3.  **Worker Slots**:
        *   Worker 1: [IDLE]
        *   Worker 2: [BUSY] Downloading assets for "Intro to AI"...
        *   Worker 3: [BUSY] Processing CSS...
    4.  **Recent Activity**: (The last 3 log messages).

#### C. The Dashboard Controller
**File:** `src/ui/DashboardController.js`
*   **Responsibility**: Connects `SystemEventBus` to `TerminalDashboard`.
*   **Reasoning**: Keeps the UI rendering logic dumb. This controller interprets events like `WORKER:BUSY` and tells the UI "Update Row 2 to text 'Processing...'".

---

### 3. Implementation Steps

#### Step 1: Create Log Strategies
Move the current console logging logic into a strategy and add the file writer.

**Dependencies**: `mkdirp` (for creating /logs), `moment` (for timestamping).

```javascript
// src/core/logger/FileStrategy.js
class FileStrategy {
  constructor(baseDir) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.filepath = path.join(baseDir, 'logs', `run-${timestamp}.md`);
    this.stream = fs.createWriteStream(this.filepath, { flags: 'a' });
    this.stream.write(`# Run Log - ${timestamp}\n\n`);
  }

  log(level, category, message, meta) {
    const time = new Date().toISOString().split('T')[1].slice(0, -1);
    const icon = level === 'error' ? '❌' : (level === 'warn' ? '⚠️' : 'ℹ️');
    let line = `| ${time} | ${icon} **${category}** | ${message} |\n`;
    
    if (meta && (level === 'error' || level === 'debug')) {
      line += `\`\`\`json\n${JSON.stringify(meta, null, 2)}\n\`\`\`\n`;
    }
    this.stream.write(line);
  }
}
```

#### Step 2: Implement Terminal Dashboard
Use `cli-progress` to create a multi-bar interface.

```javascript
// src/ui/TerminalDashboard.js
const cliProgress = require('cli-progress');

class TerminalDashboard {
  constructor(totalWorkers) {
    this.multibar = new cliProgress.MultiBar({
      clearOnComplete: false,
      hideCursor: true,
      format: '{bar} {percentage}% | {value}/{total} | {phase} | {status}'
    }, cliProgress.Presets.shades_classic);

    // 1. Main Bar
    this.mainBar = this.multibar.create(100, 0, { phase: 'INIT', status: 'Starting...' });

    // 2. Worker Bars (Bars used as status text holders)
    this.workerBars = new Map();
    for(let i=0; i<totalWorkers; i++) {
      // Using a bar formatted to just show text
      const bar = this.multibar.create(1, 0, { 
        phase: `Worker ${i+1}`, 
        status: 'Waiting...' 
      });
      // Hack: Set format specifically for workers to look like rows
      bar.setFormat(`  {phase} | {status}`); 
      this.workerBars.set(i, bar);
    }
  }

  updateMain(current, total, phase, status) {
    this.mainBar.setTotal(total);
    this.mainBar.update(current, { phase, status });
  }

  updateWorker(workerIndex, status) {
    const bar = this.workerBars.get(workerIndex);
    if(bar) bar.update(0, { status });
  }
  
  stop() {
    this.multibar.stop();
  }
}
```

#### Step 3: Connect Events (The Controller)

```javascript
// src/ui/DashboardController.js
class DashboardController {
  constructor(eventBus, browserManager) {
    this.bus = eventBus;
    this.manager = browserManager;
    this.dashboard = null;
    
    // Map Worker IDs (UUIDs) to UI Slots (0, 1, 2, 3)
    this.workerSlotMap = new Map(); 
  }

  init() {
    const workerCount = this.manager.getTotalCount();
    this.dashboard = new TerminalDashboard(workerCount);
    
    // Map worker IDs to slots
    const workerIds = this.manager.getAllWorkerIds();
    workerIds.forEach((id, index) => this.workerSlotMap.set(id, index));

    this._attachListeners();
  }

  _attachListeners() {
    // Progress Updates
    this.bus.on('DISCOVERY:COMPLETE', (payload) => {
        this.dashboard.updateMain(0, payload.totalPages, 'EXECUTION', 'Starting downloads...');
    });
    
    this.bus.on('EXECUTION:PROGRESS', (payload) => {
        this.dashboard.updateMain(payload.completed, payload.total, 'EXECUTION', 'Downloading...');
    });

    // Worker Status Updates
    this.bus.on('WORKER:BUSY', ({ workerId, taskId }) => {
        const slot = this.workerSlotMap.get(workerId);
        this.dashboard.updateWorker(slot, `Processing Task ${taskId.substring(0,6)}...`);
    });

    this.bus.on('WORKER:AVAILABLE', ({ workerId }) => {
        const slot = this.workerSlotMap.get(workerId);
        this.dashboard.updateWorker(slot, 'IDLE');
    });
  }
}
```

---

### 4. JSDocs for New Modules

#### `src/core/logger/LogStrategy.js`

```javascript
/**
 * @interface LogStrategy
 * @description Interface for log output destinations.
 */

/**
 * @method log
 * @param {string} level - 'info', 'warn', 'error', 'debug'
 * @param {string} category - The source module name.
 * @param {string} message - The log text.
 * @param {Object} [meta] - Optional JSON metadata (errors, objects).
 */
```

#### `src/ui/TerminalDashboard.js`

```javascript
/**
 * @fileoverview Terminal UI Renderer
 * @module ui/TerminalDashboard
 */

/**
 * @class TerminalDashboard
 * @description Encapsulates `cli-progress` logic to render the multi-bar UI.
 * It purely handles rendering; it does not contain business logic.
 */

/**
 * @constructor
 * @param {number} workerCount - Number of worker rows to initialize.
 */

/**
 * @method updateProgress
 * @param {number} current - Current completed items.
 * @param {number} total - Total items.
 * @param {string} phaseLabel - E.g., "DISCOVERY" or "DOWNLOAD".
 */

/**
 * @method setWorkerStatus
 * @param {number} slotIndex - Visual slot index (0 to N-1).
 * @param {string} statusText - Text to display (e.g., "Downloading image.png").
 * @param {boolean} [isError=false] - If true, formats text red.
 */
```

#### `src/ui/DashboardController.js`

```javascript
/**
 * @fileoverview UI Logic Coordinator
 * @module ui/DashboardController
 */

/**
 * @class DashboardController
 * @implements {Observer}
 * @description Acts as the bridge between the SystemEventBus and the Visual Dashboard.
 * It translates system events (WORKER:BUSY, JOB:COMPLETED) into visual updates.
 */

/**
 * @constructor
 * @param {SystemEventBus} eventBus - Singleton event bus.
 * @param {BrowserManager} browserManager - Needed to map Worker IDs to UI slots.
 */

/**
 * @method start
 * @summary Initializes the Dashboard and attaches event listeners.
 * @description Should be called after the System Init phase when worker count is known.
 */
```

### 5. Integration Point

In `main-cluster.js`:

```javascript
// 1. Init Logger with File Strategy
const logger = Logger.getInstance();
logger.addStrategy(new FileStrategy(config.outputDir));

// 2. Init System
// ... (BrowserInitializer spawns workers) ...

// 3. Init UI
const dashboardCtrl = new DashboardController(SystemEventBus.getInstance(), browserManager);
dashboardCtrl.start(); // Will hijack stdout

// 4. Start Orchestrator
orchestrator.start();
```

### 6. Key Benefits of this Approach

1.  **Decoupling:** The `ClusterOrchestrator` doesn't know a progress bar exists. It just emits events.
2.  **Separation of Concerns:**
    *   `FileStrategy` handles Markdown formatting and I/O.
    *   `TerminalDashboard` handles ANSI codes and bar rendering.
    *   `DashboardController` handles logic (mapping IDs to slots).
3.  **Observability:** You get a beautiful real-time UI *and* a permanent, searchable Markdown log file for debugging.
4.  **Stability:** If the UI crashes, the File Strategy keeps logging. If the User runs in a CI environment (non-interactive), we can simply not initialize `DashboardController` and fall back to `ConsoleStrategy`.

### 7. Update Docs/ARCHITECTURE.md after Implementation completion

### 8. ADD UNIT TESTS for Logger Strategies and Dashboard Controller