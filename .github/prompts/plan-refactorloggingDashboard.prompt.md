This is a complex concurrency and IO handling issue. Based on the logs and the architecture, I have identified the two specific culprits responsible for the failure.

### 1. The Culprits

#### **Culprit A: The "Stdout Leak" (Messy Terminal)**
The terminal output is messy because **Worker processes are logging directly to stdout**.
In `child_process.fork()`, the child's stdout is piped to the parent. When a Worker calls `logger.info()`, it uses the `ConsoleStrategy` inside the *Worker process*. This text bypasses the Master's `Dashboard` and prints raw text over the progress bars.
**The Fix:** Workers must **never** log to console. They must send log data to the Master via IPC, and the Master decides how to display it (Dashboard or File).

#### **Culprit B: The "Re-Init Destructor" (Missing Log File)**
The `main-cluster.js` calls `logger.init({ reinit: true ... })` when the dashboard starts.
Looking at your `Logger.js`:
```javascript
if (options.reinit) {
  this.strategies = []; // <--- DESTRUCTIVE ACTION
  this._initialized = false;
}
```
This **destroys the active FileStrategy** created at startup. It attempts to create a *new* one, but because the previous stream might not have closed gracefully, or the new timestamp generates a different filename, the logic fragments. Furthermore, if the application crashes shortly after (as seen with the SIGINTs), the buffer of the *new* file strategy likely never flushes to disk.

---

### 2. The Refactoring Plan

We need to implement a **Centralized Logging Architecture** where Workers are silent data producers and the Master is the aggregator.

#### Step 1: Create `IpcStrategy` (Worker Side)
A new logging strategy that, instead of `console.log`, does `process.send({ type: 'IPC_LOG', payload: ... })`.

#### Step 2: Update `WorkerProxy` (Master Side)
Update the proxy to listen for `IPC_LOG` messages and forward them to the Master's `Logger` instance.

#### Step 3: Fix `Logger.js` Lifecycle
Remove the destructive `reinit` logic. Add a method `swapConsoleForDashboard()` that removes *only* the `ConsoleStrategy` and adds the `DashboardStrategy`, leaving the `FileStrategy` untouched and continuous.

---

### 3. New & Refactored Code

#### 1. New File: `src/core/logger/IpcStrategy.js`
This runs inside the Worker to send logs to Master.

```javascript
/**
 * @fileoverview IPC Logging Strategy
 * @module core/logger/IpcStrategy
 */

const LogStrategy = require('./LogStrategy');

/**
 * @class IpcStrategy
 * @extends LogStrategy
 * @description Sends log entries to the Master process via Node.js IPC.
 * Used by Workers to prevent stdout pollution in the terminal.
 */
class IpcStrategy extends LogStrategy {
  constructor() {
    super();
  }

  log(level, category, message, meta) {
    // specific check to avoid circular JSON errors or massive payloads
    const safeMeta = meta ? JSON.parse(JSON.stringify(meta, this._replacer)) : null;

    if (process.send) {
      process.send({
        type: 'IPC_LOG',
        payload: {
          level,
          category,
          message,
          meta: safeMeta,
          timestamp: Date.now()
        }
      });
    }
  }

  // Helper to handle circular references in errors
  _replacer(key, value) {
    if (value instanceof Error) {
      return {
        message: value.message,
        stack: value.stack,
        name: value.name
      };
    }
    return value;
  }
}

module.exports = IpcStrategy;
```

#### 2. Refactored `src/core/Logger.js`
Fixing the destructive initialization logic.

```javascript
// ... imports ...

class Logger {
  // ... constructor ...

  /**
   * @method switchMode
   * @param {string} mode - 'console' or 'dashboard'
   * @param {Object} context - { dashboardInstance } if mode is dashboard
   * @description Safely switches UI strategies without killing FileStrategy
   */
  switchMode(mode, context = {}) {
    // 1. Remove existing UI strategies (Console or Dashboard)
    this.strategies = this.strategies.filter(s => 
      !(s instanceof ConsoleStrategy) && 
      !(s.constructor.name === 'DashboardStrategy')
    );

    // 2. Add new UI strategy
    if (mode === 'dashboard' && context.dashboardInstance) {
      const DashboardStrategy = require('./logger/DashboardStrategy');
      this.addStrategy(new DashboardStrategy(context.dashboardInstance));
    } else {
      this.addStrategy(new ConsoleStrategy());
    }
    
    // FileStrategy remains untouched!
  }

  // ... keep existing methods ...
}
```

#### 3. Refactored `src/cluster/WorkerProxy.js`
Handling the incoming log messages.

```javascript
// ... inside _attachListeners ...

childProcess.on('message', (msg) => {
  if (msg.type === 'IPC_LOG') {
    // Forward Worker logs to Master Logger
    // The Master Logger will then write to File and/or Dashboard
    const { level, category, message, meta } = msg.payload;
    const Logger = require('../core/Logger');
    Logger.getInstance().log(level, `Worker-${this.id}:${category}`, message, meta);
    return;
  }
  // ... existing message handling ...
});
```

#### 4. Refactored `src/worker/WorkerEntrypoint.js`
Initializing Workers with `IpcStrategy`.

```javascript
// ... imports ...
const IpcStrategy = require('../core/logger/IpcStrategy');
const Logger = require('../core/Logger');

// ... 

// 2. Wait for INIT message from parent
const initPromise = new Promise((resolve) => {
  process.once('message', (msg) => {
    if (msg.type === 'IPC_INIT') {
      // INITIALIZE LOGGER WITH IPC STRATEGY
      // This ensures no console.log leaks to stdout
      const logger = Logger.getInstance();
      logger.addStrategy(new IpcStrategy());
      
      resolve(msg.payload);
    }
  });
});

// ... rest of file ...
```

#### 5. Refactored `main-cluster.js`
Using the safe switch method instead of reinit.

```javascript
// ... inside initDashboard ...

if (enableUI && orchestrator.browserManager.getTotalCount() > 0) {
  try {
    dashboardCtrl = new DashboardController(
      eventBus,
      orchestrator.browserManager,
      { enabled: true }
    );
    dashboardCtrl.start();
    
    // SAFE SWITCH: Keeps the file logging active!
    logger.switchMode('dashboard', { 
      dashboardInstance: dashboardCtrl.getDashboard() 
    });
    
    // ...
  }
}
```

### Why this fixes it:
1.  **No Ghost Logs:** The `FileStrategy` created at startup stays alive for the entire session. It captures Bootstrap, Discovery, and Execution logs in one file.
2.  **Clean UI:** Workers transmit logs as data objects. The Master receives them and the `DashboardStrategy` formats them into the bottom "ticker" bar, preventing the chaotic overwriting of progress bars.
3.  **Separation of Concerns:** Workers don't know about files or dashboards. They just emit data. The Master handles the presentation.

### Important: Ensure NO emojies are used in any component of the codebase