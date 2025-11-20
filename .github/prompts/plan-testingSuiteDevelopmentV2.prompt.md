# Plan: Comprehensive Testing Suite Development (v2)

This plan transforms the current integration-heavy test suite into a professional 3-tier testing architecture (Unit → Integration → E2E) specifically designed for the **Reactive Event-Driven Micro-Kernel** architecture. It addresses the specific challenges of testing distributed systems, including IPC message passing, process isolation, and event-driven coordination.

## Steps

### 1. Establish Testing Foundation

**Objective**: Set up professional testing infrastructure with Jest ecosystem and reusable mock utilities.

**Tasks**:
- Install Jest ecosystem (`jest`, `@types/jest`, `jest-extended`, `mock-fs`, `nock`)
- Configure `jest.config.js` with separate profiles for unit/integration/e2e tests
- Create mock factory utilities in `tests/helpers/` (`factories.js`, `MockWorkerProxy.js`, `MockSystemEventBus.js`)
- Set up static HTML fixtures in `tests/fixtures/` for pipeline step verification
- Establish CI/CD integration with coverage thresholds (70% minimum)

**Deliverables**:
- Working Jest configuration with 3 test profiles
- `MockSystemEventBus` that mimics the singleton behavior but allows isolation
- `MockWorkerProxy` for simulating IPC messages
- 6 HTML fixtures representing different Notion page types

---

### 2. Build Critical Path Unit Tests

**Objective**: Test core utilities and domain models that form the foundation of the system.

**Tasks**:
- **Protocol Definitions**: Test `ProtocolDefinitions.js` message validation, error serialization, and payload structure. Ensure `DownloadPayload` strictly requires absolute paths.
- **Domain Logic**: Test `PageContext.js` hierarchy management, JSON serialization/deserialization (rehydration), and relative path calculation.
- **Conflict Resolution**: Test `ConflictResolver.js` logic for identifying duplicates, selecting canonical instances (by depth/order), and generating the correct link rewrite map.
- **Graph Analysis**: Test `GraphAnalyzer.js` edge classification (tree/back/forward/cross) using mock graph topologies.
- **Utilities**: Test `FileSystemUtils.js` sanitization and `Logger.js` formatting.

**Deliverables**:
- 100% coverage on `ProtocolDefinitions` and `PageContext` serialization
- Validated graph pruning logic with complex diamond dependency scenarios

---

### 3. Test Master Process Coordination (Micro-Kernel Logic)

**Objective**: Validate Master-side orchestration without spawning real processes.

**Tasks**:
- **Queue Management**: Test `GlobalQueueManager.js` for BFS ordering, URL deduplication, title registry immutability, and dependency blocking (Leaf-First logic).
- **Resource Allocation**: Test `BrowserManager.js` worker pool logic (LIFO stack), allocation of tasks to idle workers, and correct handling of worker death/respawn.
- **Orchestration**: Test `ClusterOrchestrator.js` state machine transitions (Init → Discovery → Pruning → Execution). Verify it triggers `IPC_UPDATE_REGISTRY` after pruning.
- **IPC Translation**: Test `WorkerProxy.js` correctly translates `child_process` messages into `SystemEventBus` events and vice-versa.

**Deliverables**:
- Master process logic tested with 80%+ coverage
- Integration tests verifying the event loop (Bus → Manager → Proxy → Bus)

---

### 4. Test Worker Pipeline Execution

**Objective**: Validate the new Scraping Pipeline and Step logic.

**Tasks**:
- **Pipeline Controller**: Test `ScrapingPipeline.js` executes steps in order and aborts on failure.
- **Pipeline Steps**:
    - `NavigationStep`: Verify `page.goto` calls and error handling.
    - `ExpansionStep`: Verify `ContentExpander` interaction.
    - `AssetDownloadStep`: Verify it identifies assets and calls `AssetDownloader`.
    - `LinkRewriterStep`: Verify DOM manipulation using the provided `linkMap`.
    - `HtmlWriteStep`: Verify it calls `WorkerFileSystem` with the correct absolute path.
- **I/O Adapter**: Test `WorkerFileSystem.js` enforces absolute paths and logs writes.
- **Task Runner**: Test `TaskRunner.js` validates payloads before creating the pipeline.

**Deliverables**:
- Unit tests for each Pipeline Step
- Validation of the "Absolute Path" safety check in `WorkerFileSystem`

---

### 5. Build Integration Test Suite

**Objective**: Validate end-to-end workflows with simulated IPC.

**Tasks**:
- **Discovery Flow**: Simulate a full discovery phase using `MockWorkerProxy` returning fake links. Verify `GlobalQueueManager` builds the correct tree.
- **Pruning & Registry**: Verify that after discovery, the Orchestrator triggers pruning and sends the updated Title Registry to workers.
- **Execution Flow**: Simulate the download phase. Verify that `GlobalQueueManager` only releases parent pages after children are marked complete.
- **Crash Recovery**: Simulate a `WORKER:DIED` event during a task and verify `BrowserManager` respawns a worker and re-queues the task.

**Deliverables**:
- Integration suite running <30s
- Verification of the "Leaf-First" download order

---

## Test Directory Structure

```
tests/
├── unit/                           # Fast unit tests (<10s total)
│   ├── core/
│   │   ├── ProtocolDefinitions.test.js
│   │   └── SystemEventBus.test.js
│   ├── domain/
│   │   └── PageContext.test.js
│   ├── orchestration/
│   │   ├── GlobalQueueManager.test.js
│   │   └── analysis/
│   │       └── ConflictResolver.test.js
│   ├── worker/
│   │   ├── pipeline/
│   │   │   ├── ScrapingPipeline.test.js
│   │   │   └── steps/
│   │   │       ├── NavigationStep.test.js
│   │   │       └── HtmlWriteStep.test.js
│   │   └── io/
│   │       └── WorkerFileSystem.test.js
│   └── utils/
│       └── FileSystemUtils.test.js
│
├── integration/                    # Medium integration tests (<30s total)
│   ├── master/
│   │   ├── ClusterOrchestrator.test.js
│   │   ├── BrowserManager.test.js
│   ├── workflows/
│   │   ├── discovery-phase.test.js
│   │   └── execution-phase.test.js
│
├── helpers/                       # Reusable test utilities
│   ├── factories.js              # Mock data generators
│   ├── MockWorkerProxy.js        # Simulates Worker behavior
│   ├── MockSystemEventBus.js     # Isolated Event Bus
│   ├── MockPuppeteer.js          # Browser automation mock
│   └── MockFileSystem.js         # File I/O mock
│
├── fixtures/                      # Static test data
│   ├── html/
│   └── graphs/
│
└── jest.config.js                 # Jest configuration
```

---

## Critical Test Cases (Architectural Validation)

### 1. Singleton Isolation (SystemEventBus)

**Problem**: `SystemEventBus.getInstance()` persists state between tests.
**Solution**: Implement a `reset()` method for testing.

```javascript
// tests/helpers/MockSystemEventBus.js
const SystemEventBus = require('../../src/core/SystemEventBus');

beforeEach(() => {
  // Reset singleton state before every test
  SystemEventBus.instance = null; 
  SystemEventBus.getInstance().removeAllListeners();
});
```

### 2. IPC Payload Serialization

**Objective**: Ensure `PageContext` and Errors survive the JSON trip.

```javascript
// tests/unit/domain/PageContext.test.js
it('survives JSON serialization round-trip', () => {
  const original = new PageContext('http://url', 'Title', 1, 'parent-id');
  const json = JSON.stringify(original);
  const restored = PageContext.fromJSON(JSON.parse(json));
  
  expect(restored).toBeInstanceOf(PageContext);
  expect(restored.id).toBe(original.id);
  expect(restored.getRelativePath).toBeDefined(); // Methods restored
});
```

### 3. Absolute Path Enforcement

**Objective**: Prevent "Ghost Writes".

```javascript
// tests/unit/worker/io/WorkerFileSystem.test.js
it('throws error on relative paths', async () => {
  const fs = new WorkerFileSystem();
  await expect(fs.safeWrite('relative/path.html', 'content'))
    .rejects.toThrow('Path must be absolute');
});

it('allows absolute paths', async () => {
  const fs = new WorkerFileSystem();
  const absPath = path.resolve('/tmp/test.html');
  await expect(fs.safeWrite(absPath, 'content')).resolves.not.toThrow();
});
```

### 4. Pipeline Error Propagation

**Objective**: Ensure a failed step stops the pipeline.

```javascript
// tests/unit/worker/pipeline/ScrapingPipeline.test.js
it('aborts pipeline if step fails', async () => {
  const step1 = { process: jest.fn().mockResolvedValue() };
  const step2 = { process: jest.fn().mockRejectedValue(new Error('Fail')) };
  const step3 = { process: jest.fn() };
  
  const pipeline = new ScrapingPipeline([step1, step2, step3]);
  
  await expect(pipeline.execute({})).rejects.toThrow('Fail');
  expect(step1.process).toHaveBeenCalled();
  expect(step3.process).not.toHaveBeenCalled(); // Crucial check
});
```

### 5. Worker Respawn & Re-Init

**Objective**: Ensure a respawned worker gets the Title Registry.

```javascript
// tests/integration/master/BrowserManager.test.js
it('re-initializes worker with registry on respawn', async () => {
  const manager = new BrowserManager();
  // Mock the spawn method
  const spawnSpy = jest.spyOn(BrowserInitializer, 'spawnWorker').mockResolvedValue(new MockWorkerProxy());
  
  await manager._handleWorkerExit('worker-1');
  
  expect(spawnSpy).toHaveBeenCalled();
  const newWorker = manager.getWorker('worker-1-replacement');
  // Verify IPC_INIT was sent with registry
  expect(newWorker.lastMessage.type).toBe('IPC_INIT');
  expect(newWorker.lastMessage.payload.titleRegistry).toBeDefined();
});
```