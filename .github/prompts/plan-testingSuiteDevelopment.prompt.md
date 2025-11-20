# Plan: Comprehensive Testing Suite Development

This plan transforms the current integration-heavy test suite into a professional 3-tier testing architecture (Unit → Integration → E2E) aligned with the micro-kernel design. The strategy prioritizes critical path components, introduces proper mocking infrastructure, and establishes fast feedback loops.

## Steps

### 1. Establish Testing Foundation

**Objective**: Set up professional testing infrastructure with Jest ecosystem and reusable mock utilities.

**Tasks**:
- Install Jest ecosystem (`jest`, `@types/jest`, `jest-extended`, `puppeteer-mock`, `mock-fs`, `nock`)
- Configure `jest.config.js` with separate profiles for unit/integration/e2e tests
- Create mock factory utilities in `tests/helpers/` (`factories.js`, `MockWorkerProxy.js`, `MockPuppeteer.js`)
- Set up static HTML fixtures in `tests/fixtures/` for DOM manipulation tests
- Establish CI/CD integration with coverage thresholds (70% minimum)

**Deliverables**:
- Working Jest configuration with 3 test profiles
- Mock factory library for common dependencies
- 6 HTML fixtures representing different Notion page types
- CI pipeline running `npm test` on every commit

---

### 2. Build Critical Path Unit Tests

**Objective**: Test core utilities and domain models that form the foundation of the system.

**Tasks**:
- Test `ProtocolDefinitions.js` message validation, error serialization, and title map conversion
- Test `FileSystemUtils.js` filename sanitization with illegal characters, unicode, and length limits
- Test `PageContext.js` hierarchy management, relative path calculation, and JSON serialization round-trips
- Test `ConflictResolver.js` duplicate detection, canonical selection by depth, and link rewrite map generation
- Test `GraphAnalyzer.js` edge classification (tree/back/forward/cross) with fixture graphs

**Deliverables**:
- 5 fully tested utility modules with 100% coverage
- Documented test patterns for pure functions
- Fixture graph topologies for edge classification testing

---

### 3. Test Master Process Coordination

**Objective**: Validate Master-side orchestration logic with mocked worker dependencies.

**Tasks**:
- Test `GlobalQueueManager.js` BFS depth ordering, URL deduplication, title registry immutability, and download queue dependency blocking
- Test `BrowserManager.js` worker pool allocation (LIFO idle stack), crash recovery, and graceful shutdown with mocked `WorkerProxy` instances
- Test `WorkerProxy.js` IPC message translation, state tracking (IDLE→BUSY→IDLE), and error handling with mocked `ChildProcess`
- Test `ClusterOrchestrator.js` phase transitions, cookie broadcast timing, and user confirmation flow with full dependency mocks

**Deliverables**:
- Master process core tested with 80%+ coverage
- Reusable `MockWorkerProxy` and `MockChildProcess` utilities
- Integration tests for event-driven coordination flows

---

### 4. Test Worker Process Execution

**Objective**: Validate Worker-side task execution with mocked Puppeteer and network layers.

**Tasks**:
- Test `TaskRunner.js` command routing, cookie management, title registry caching, and error serialization with mocked Puppeteer browser/page
- Test `LinkRewriter.js` relative path rewriting and CSS localization using static HTML fixtures
- Test `AssetDownloader.js` retry logic, filename hashing, and DOM src rewriting with `nock` for HTTP mocks
- Create worker crash/recovery integration tests using `MockWorkerProxy` to simulate process exit events

**Deliverables**:
- Worker process tested with mocked browser (no real Puppeteer launches)
- `MockPuppeteer` utility with page navigation and DOM evaluation stubs
- Crash recovery scenarios validated with synthetic IPC errors

---

### 5. Build Integration Test Suite

**Objective**: Validate end-to-end workflows with full Master-Worker coordination.

**Tasks**:
- Create Master-Worker IPC flow tests verifying `IPC_INIT` → `IPC_DISCOVER` → `IPC_RESULT` → `IPC_DOWNLOAD` sequences
- Test full discovery phase workflow with 3-level graph fixture (bootstrap → BFS traversal → title registry population)
- Test download phase dependency ordering (children complete before parents) with diamond dependency graph
- Test conflict resolution workflow (duplicate detection → pruning → link rewrite map propagation)

**Deliverables**:
- Integration test suite completing in <30 seconds
- Documented IPC protocol test patterns
- Fixture-based workflows replacing live Notion tests

---

### 6. Migrate and Refactor Existing Tests

**Objective**: Consolidate existing manual tests into Jest framework and eliminate redundancy.

**Tasks**:
- Convert `test-title-registry.js` assertions to Jest `expect()` and move to `tests/unit/orchestration/GlobalQueueManager.test.js`
- Refactor `test-user-confirmation.js` `MockReadline` into `tests/helpers/MockReadline.js` and convert to Jest format
- Extract reusable patterns from `test-integration.js` `TestHarness` into `tests/helpers/IntegrationHarness.js`
- Archive or delete redundant manual tests (`test-cluster.js`, `test-worker.js`) after migration

**Deliverables**:
- All existing test logic migrated to Jest
- Clean test directory structure following Jest conventions
- Historical tests archived in `tests/archive/` for reference

---

## Test Directory Structure

```
tests/
├── unit/                           # Fast unit tests (<10s total)
│   ├── core/
│   │   └── ProtocolDefinitions.test.js
│   ├── domain/
│   │   └── PageContext.test.js
│   ├── orchestration/
│   │   ├── GlobalQueueManager.test.js
│   │   └── analysis/
│   │       ├── ConflictResolver.test.js
│   │       └── GraphAnalyzer.test.js
│   ├── processing/
│   │   └── LinkRewriter.test.js
│   ├── download/
│   │   └── AssetDownloader.test.js
│   └── utils/
│       └── FileSystemUtils.test.js
│
├── integration/                    # Medium integration tests (<30s total)
│   ├── master/
│   │   ├── ClusterOrchestrator.test.js
│   │   ├── BrowserManager.test.js
│   │   └── WorkerProxy.test.js
│   ├── worker/
│   │   └── TaskRunner.test.js
│   └── workflows/
│       ├── discovery-phase.test.js
│       ├── download-phase.test.js
│       └── conflict-resolution.test.js
│
├── e2e/                           # Optional live tests (--live flag)
│   └── full-scrape.test.js
│
├── helpers/                       # Reusable test utilities
│   ├── factories.js              # Mock data generators
│   ├── MockWorkerProxy.js        # Worker process mock
│   ├── MockPuppeteer.js          # Browser automation mock
│   ├── MockChildProcess.js       # IPC mock
│   ├── MockReadline.js           # User input mock
│   └── IntegrationHarness.js     # E2E test orchestration
│
├── fixtures/                      # Static test data
│   ├── html/
│   │   ├── notion-page-simple.html
│   │   ├── notion-page-toggles.html
│   │   ├── notion-page-database.html
│   │   ├── notion-page-images.html
│   │   └── notion-page-complex.html
│   └── graphs/
│       ├── graph-simple.json
│       ├── graph-cycle.json
│       ├── graph-diamond.json
│       └── graph-deep.json
│
├── archive/                       # Migrated legacy tests
│   ├── test-cluster.js
│   ├── test-worker.js
│   └── test-integration.js
│
└── jest.config.js                 # Jest configuration
```

---

## Jest Configuration Strategy

### `jest.config.js`

```javascript
module.exports = {
  // Base configuration for all tests
  testEnvironment: 'node',
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/**/*.test.js',
    '!**/node_modules/**'
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70
    }
  },
  
  // Separate test patterns
  testMatch: [
    '**/tests/unit/**/*.test.js',
    '**/tests/integration/**/*.test.js'
  ],
  
  // Setup files
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  
  // Module paths
  moduleDirectories: ['node_modules', 'src'],
  
  // Timeout configuration
  testTimeout: 10000, // 10s for most tests
  
  // Projects for separate test profiles
  projects: [
    {
      displayName: 'unit',
      testMatch: ['**/tests/unit/**/*.test.js'],
      testTimeout: 5000 // Fast unit tests
    },
    {
      displayName: 'integration',
      testMatch: ['**/tests/integration/**/*.test.js'],
      testTimeout: 30000 // Slower integration tests
    },
    {
      displayName: 'e2e',
      testMatch: ['**/tests/e2e/**/*.test.js'],
      testTimeout: 120000, // 2 min for live tests
      testEnvironment: 'jest-environment-puppeteer'
    }
  ]
};
```

### Package.json Scripts

```json
{
  "scripts": {
    "test": "jest --projects unit integration",
    "test:unit": "jest --selectProject=unit",
    "test:integration": "jest --selectProject=integration",
    "test:e2e": "jest --selectProject=e2e",
    "test:watch": "jest --watch --projects unit",
    "test:coverage": "jest --coverage --projects unit integration",
    "test:all": "jest --projects unit integration e2e"
  }
}
```

---

## Critical Test Cases (Architectural Validation)

### Functional Gates (from Architecture Doc)

**1. 100% Graph Fidelity**
```javascript
// tests/integration/workflows/discovery-phase.test.js
it('produces exact PageContext tree from controlled mock graph', async () => {
  const fixtureGraph = require('../../fixtures/graphs/graph-simple.json');
  const result = await orchestrator.runDiscovery(fixtureGraph.rootUrl);
  
  expect(result.contexts).toHaveLength(fixtureGraph.nodes.length);
  expect(result.contexts[0].depth).toBe(0); // Root
  expect(result.contexts[1].parentId).toBe(result.contexts[0].id); // Parent link
});
```

**2. Zero Deadlocks**
```javascript
// tests/integration/master/BrowserManager.test.js
it('never starves workers under simulated latency', async () => {
  const manager = new BrowserManager(config, logger, eventBus);
  manager.registerWorkers(mockWorkers);
  
  // Queue 100 tasks with random delays
  const tasks = Array.from({ length: 100 }, (_, i) => 
    manager.execute('DISCOVER', { taskId: i })
  );
  
  await Promise.all(tasks);
  expect(manager.getIdleWorkers()).toHaveLength(mockWorkers.length); // All returned
});
```

**3. Worker Recovery**
```javascript
// tests/integration/master/BrowserManager.test.js
it('re-queues task when worker crashes mid-execution', async () => {
  const manager = new BrowserManager(config, logger, eventBus);
  const taskPromise = manager.execute('DISCOVER', { taskId: 'test-1' });
  
  // Simulate worker death
  setTimeout(() => eventBus.emit('WORKER:DIED', { workerId: 'worker-1' }), 100);
  
  // Task should eventually complete with different worker
  const result = await taskPromise;
  expect(result.success).toBe(true);
  expect(result.completedBy).not.toBe('worker-1');
});
```

**4. Pruning Idempotency**
```javascript
// tests/unit/orchestration/analysis/ConflictResolver.test.js
it('produces identical results when pruning twice', () => {
  const contexts = createDuplicateContexts();
  
  const result1 = ConflictResolver.resolve(contexts);
  const result2 = ConflictResolver.resolve(result1.canonicalContexts);
  
  expect(result2.canonicalContexts).toEqual(result1.canonicalContexts);
  expect(result2.linkRewriteMap).toEqual(result1.linkRewriteMap);
});
```

### Output Gates

**5. Relative Links**
```javascript
// tests/unit/processing/LinkRewriter.test.js
it('converts all internal links to relative paths', async () => {
  const html = '<a href="https://notion.so/page-abc">Link</a>';
  const rewritten = await linkRewriter.rewrite(html, context, urlMap);
  
  expect(rewritten).toMatch(/href="\.\.\/.*\/index\.html"/);
  expect(rewritten).not.toMatch(/https:\/\/notion\.so/);
});
```

**6. Asset Localization**
```javascript
// tests/unit/download/AssetDownloader.test.js
it('rewrites all image sources to local paths', async () => {
  const html = '<img src="https://s3.amazonaws.com/notion/image.png">';
  const result = await assetDownloader.download(html, context);
  
  expect(result.html).toMatch(/src="\.\/images\/[a-f0-9]{8}\.png"/);
});
```

**7. Filesystem Safety**
```javascript
// tests/unit/utils/FileSystemUtils.test.js
it('removes all illegal characters from filenames', () => {
  const dangerous = 'Page<>:"/\\|?*.html';
  const safe = FileSystemUtils.sanitizeFilename(dangerous);
  
  expect(safe).toMatch(/^[a-zA-Z0-9_\-\.]+$/); // Only safe chars
  expect(safe).not.toMatch(/[<>:"/\\|?*]/);
});
```

---

## Mock Factory Examples

### `tests/helpers/factories.js`

```javascript
const PageContext = require('../../src/domain/PageContext');

/**
 * Creates a mock PageContext with sensible defaults
 */
function createMockPageContext(overrides = {}) {
  return new PageContext(
    overrides.url || `https://notion.so/page-${Date.now()}`,
    overrides.title || 'Test Page',
    overrides.depth || 0,
    overrides.parentId || null
  );
}

/**
 * Creates a mock graph structure for testing
 */
function createMockGraph(levels = 3, childrenPerLevel = 2) {
  const contexts = [];
  const root = createMockPageContext({ title: 'Root', depth: 0 });
  contexts.push(root);
  
  let currentLevel = [root];
  for (let depth = 1; depth < levels; depth++) {
    const nextLevel = [];
    for (const parent of currentLevel) {
      for (let i = 0; i < childrenPerLevel; i++) {
        const child = createMockPageContext({
          title: `Child-${depth}-${i}`,
          depth,
          parentId: parent.id
        });
        parent.addChild(child);
        nextLevel.push(child);
        contexts.push(child);
      }
    }
    currentLevel = nextLevel;
  }
  
  return { root, contexts };
}

/**
 * Creates a mock IPC discovery result
 */
function createMockDiscoveryResult(overrides = {}) {
  return {
    success: true,
    pageId: overrides.pageId || 'page-id-123',
    url: overrides.url || 'https://notion.so/page',
    resolvedTitle: overrides.resolvedTitle || 'Page Title',
    links: overrides.links || [],
    cookies: overrides.cookies || null
  };
}

module.exports = {
  createMockPageContext,
  createMockGraph,
  createMockDiscoveryResult
};
```

### `tests/helpers/MockWorkerProxy.js`

```javascript
const EventEmitter = require('events');
const { MESSAGE_TYPES } = require('../../src/core/ProtocolDefinitions');

/**
 * Mock WorkerProxy that simulates IPC without spawning processes
 */
class MockWorkerProxy extends EventEmitter {
  constructor(workerId = 'mock-worker-1', options = {}) {
    super();
    this.workerId = workerId;
    this.state = 'IDLE';
    this.childProcess = { pid: Math.floor(Math.random() * 10000) };
    
    // Configurable behavior
    this.responseDelay = options.responseDelay || 10; // ms
    this.shouldFail = options.shouldFail || false;
    this.crashAfter = options.crashAfter || null; // Crash after N commands
    this.commandCount = 0;
  }
  
  sendCommand(type, payload) {
    this.state = 'BUSY';
    this.commandCount++;
    
    // Simulate crash
    if (this.crashAfter && this.commandCount >= this.crashAfter) {
      setTimeout(() => {
        this.emit('exit', { code: 1, signal: null });
      }, 5);
      return;
    }
    
    // Simulate async IPC response
    setTimeout(() => {
      if (this.shouldFail) {
        this.emit('message', {
          type: MESSAGE_TYPES.RESULT,
          data: {
            success: false,
            error: { message: 'Mock error', name: 'Error' }
          }
        });
      } else {
        this.emit('message', {
          type: MESSAGE_TYPES.RESULT,
          data: {
            success: true,
            pageId: payload.pageId || 'test-page',
            resolvedTitle: 'Mock Title',
            links: []
          }
        });
      }
      this.state = 'IDLE';
    }, this.responseDelay);
  }
  
  sendInitialization(titleRegistry) {
    // Simulate initialization handshake
    setTimeout(() => {
      this.emit('message', {
        type: MESSAGE_TYPES.READY,
        data: { workerId: this.workerId }
      });
    }, 5);
  }
  
  terminate() {
    this.state = 'TERMINATED';
    this.emit('exit', { code: 0, signal: 'SIGTERM' });
  }
}

module.exports = MockWorkerProxy;
```

### `tests/helpers/MockPuppeteer.js`

```javascript
/**
 * Mock Puppeteer browser and page for Worker tests
 */
class MockPage {
  constructor(options = {}) {
    this.url = null;
    this._cookies = options.cookies || [];
    this._content = options.content || '<html><body></body></html>';
    this._evaluateResponses = options.evaluateResponses || [];
    this._evaluateIndex = 0;
  }
  
  async goto(url, options) {
    this.url = url;
    return { status: 200 };
  }
  
  async evaluate(fn) {
    // Return pre-configured responses
    if (this._evaluateResponses.length > 0) {
      return this._evaluateResponses[this._evaluateIndex++ % this._evaluateResponses.length];
    }
    return [];
  }
  
  async content() {
    return this._content;
  }
  
  async cookies() {
    return this._cookies;
  }
  
  async setCookie(...cookies) {
    this._cookies.push(...cookies);
  }
  
  async close() {
    this.closed = true;
  }
}

class MockBrowser {
  constructor(options = {}) {
    this.options = options;
    this._pages = [];
  }
  
  async newPage() {
    const page = new MockPage(this.options.pageDefaults || {});
    this._pages.push(page);
    return page;
  }
  
  async close() {
    this.closed = true;
  }
}

/**
 * Factory function matching puppeteer.launch() signature
 */
async function createMockBrowser(options = {}) {
  return new MockBrowser(options);
}

module.exports = {
  MockPage,
  MockBrowser,
  createMockBrowser
};
```

---

## Success Metrics

### Coverage Targets

| Component Type | Unit Coverage | Integration Coverage | E2E Coverage |
|---------------|---------------|----------------------|--------------|
| Core Utilities | 100% | N/A | N/A |
| Domain Models | 100% | 80% | N/A |
| Master Orchestration | 80% | 90% | 100% |
| Worker Execution | 70% | 80% | 90% |
| IPC Protocol | 100% | 100% | 100% |

### Performance Targets

- **Unit Tests**: <10 seconds total runtime
- **Integration Tests**: <30 seconds total runtime
- **E2E Tests**: <2 minutes total runtime (optional, --live flag only)
- **CI Pipeline**: <45 seconds for unit + integration

### Quality Gates

- ✅ All 7 architectural validation tests passing (4 functional + 3 output gates)
- ✅ No test uses real Puppeteer, filesystem, or network in unit/integration suites
- ✅ Zero flaky tests (tests pass consistently 100 times in a row)
- ✅ All IPC message types validated with schema tests
- ✅ Master-Worker communication tested in both directions
- ✅ Worker crash recovery verified with synthetic failures

---

## Further Considerations

### 1. Fixture Strategy vs Live Testing

**Current State**: Tests use real Notion URLs (slow, brittle, requires network access)

**Recommendation**: 
- **Primary**: Use static HTML fixtures for 95% of tests
- **Secondary**: Record/replay Puppeteer sessions with `puppeteer-har` for edge cases
- **Optional**: `--live` flag for manual E2E validation against real Notion

**Trade-offs**:
- ✅ Fixtures are fast (no network), deterministic, and enable CI without credentials
- ⚠️ Fixtures become stale if Notion changes DOM structure (requires maintenance)
- ❌ Live tests are fragile (rate limits, auth changes, network issues)

**Action**: Start with fixtures, add `npm run test:record` script to refresh fixtures quarterly

---

### 2. Coverage vs Speed Trade-off

**Target Coverage by Priority**:
- **Critical Path** (Master orchestration, IPC, conflict resolution): 85%+
- **Worker Execution** (scraping, rewriting, downloading): 70%+
- **Utilities** (filename sanitization, logging): 90%+
- **Legacy Components** (PageProcessor monolithic code): 40%+ (low priority)

**Performance Budget**:
- Unit tests must complete in <10 seconds (hard limit)
- Integration tests must complete in <30 seconds (hard limit)
- If test takes >5s, it's an integration test, not a unit test

**Action**: Prioritize fast feedback loops over exhaustive coverage. Focus on critical paths first.

---

### 3. SystemEventBus Singleton Test Isolation

**Problem**: `SystemEventBus.getInstance()` creates shared state across tests, causing pollution

**Options**:

**A. Add Reset Method** (Simplest)
```javascript
// src/core/SystemEventBus.js
static reset() {
  if (this.instance) {
    this.instance.removeAllListeners();
    this.instance = null;
  }
}

// tests/setup.js
afterEach(() => {
  SystemEventBus.reset();
});
```

**B. Factory Pattern** (Clean, but requires refactoring)
```javascript
// src/core/SystemEventBus.js
class SystemEventBus extends EventEmitter {
  // Remove getInstance(), make instantiable
}

// Inject via constructor dependency injection
const eventBus = new SystemEventBus();
const orchestrator = new ClusterOrchestrator(config, logger, eventBus);
```

**C. Unique Namespaces** (Quick fix, but hacky)
```javascript
// tests/unit/master/test1.test.js
const eventBus = SystemEventBus.getInstance();
const namespace = `test-${Date.now()}`;
eventBus.on(`${namespace}:DISCOVERY:START`, handler);
```

**Recommendation**: Start with **Option A** (reset method) for immediate test isolation. Consider **Option B** (factory pattern) if refactoring master components later. This preserves singleton pattern in production while enabling test isolation.