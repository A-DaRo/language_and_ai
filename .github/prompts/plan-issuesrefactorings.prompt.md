# Refactoring Plan: System Stability & Integrity

**Status**: Draft
**Target Version**: 1.1.0
**Based on**: `Docs/SystemIssues.md` and `Docs/ARCHITECTURE.md`

## 1. Executive Summary

This document outlines the remediation plan for the four critical issues identified in the system analysis. The goal is to restore system stability, ensure graph integrity, and eliminate deadlocks. The refactoring will strictly adhere to the **Reactive Event-Driven Micro-Kernel** architecture.

## 2. Architectural Principles

The following principles from `Docs/ARCHITECTURE.md` guide this refactoring:

1.  **Single Source of Truth**: State (titles, queue counts, page existence) must be stored in exactly one place and propagated consistently.
2.  **Fail-Fast & Defensive Programming**: Components must validate invariants (e.g., non-negative counters) and fail loudly rather than corrupting state.
3.  **Event-Driven Coordination**: Control flow should rely on `SystemEventBus` events rather than polling loops to prevent deadlocks and reduce CPU usage.
4.  **Canonicalization**: Every entity (Page, URL) must have a single canonical representation in the system.

## 3. Detailed Implementation Plan

### Phase 1: Graph Integrity (Fixing Issue 4)

**Objective**: Prevent the creation of duplicate `PageContext` objects for the same URL, ensuring a valid Directed Acyclic Graph (DAG) (or cyclic graph with correct back-edges).

**Component**: `src/orchestration/GlobalQueueManager.js`

**Analysis**:
Currently, `completeDiscovery` creates a `new PageContext` for every link found, relying on `visitedUrls` to prevent *re-queuing*. However, it fails to check if a `PageContext` already exists in `allContexts`, leading to duplicate objects with different depths.

**Proposed Changes**:
1.  **Deduplication Logic**: In `completeDiscovery`, check `this.allContexts.has(pageId)` before creating a new context.
2.  **Reuse Existing Contexts**: If a context exists, use it to create the edge in `PageGraph`, but do **not** add it to `newContexts` (to avoid re-queuing).
3.  **Edge Classification**: Pass the *canonical* (existing) context to `EdgeClassifier`.

**Code Sketch**:
```javascript
// src/orchestration/GlobalQueueManager.js

completeDiscovery(pageId, discoveredLinks, ...) {
  // ... setup parent ...

  for (const link of discoveredLinks) {
    const childId = this._generateId(link.url); // Helper needed
    let childContext = this.allContexts.get(childId);
    let isNew = false;

    if (!childContext) {
      // Create NEW canonical context
      childContext = new PageContext(link.url, ...);
      this.allContexts.set(childId, childContext);
      isNew = true;
    }

    // Always add edge to graph
    const classification = this.edgeClassifier.classifyEdge(parentContext, childContext);
    this.pageGraph.addEdge(parentContext.id, childContext.id, classification);

    // Only enqueue if it's actually new
    if (isNew) {
      parentContext.addChild(childContext);
      newContexts.push(childContext);
    }
  }
  return newContexts;
}
```

### Phase 2: Queue Stability (Fixing Issue 1)

**Objective**: Eliminate the race condition causing negative "In Queue" counts and ensure the `pendingTasks` counter reflects reality.

**Component**: `src/orchestration/queues/DiscoveryQueue.js`

**Proposed Changes**:
1.  **Guard Clauses**: Add checks in `markComplete` and `markFailed` to ensure `pendingTasks` never drops below zero.
2.  **Idempotency**: Ensure that a task ID can only be marked complete once. Use a `Set` of `pendingTaskIds` instead of a simple counter for absolute truth.

**Code Sketch**:
```javascript
// src/orchestration/queues/DiscoveryQueue.js

class DiscoveryQueue {
  constructor() {
    this.pendingTaskIds = new Set(); // Track IDs instead of just count
    // ...
  }

  next() {
    const task = this.queue.shift();
    if (task) {
      this.pendingTaskIds.add(task.pageContext.id);
    }
    return task;
  }

  markComplete(pageId) {
    if (this.pendingTaskIds.has(pageId)) {
      this.pendingTaskIds.delete(pageId);
    } else {
      this.logger.warn('DiscoveryQueue', `Attempted to complete unknown task: ${pageId}`);
    }
  }
  
  getPendingCount() {
    return this.pendingTaskIds.size;
  }
}
```

### Phase 3: Control Flow & Deadlock Prevention (Fixing Issue 2)

**Objective**: Replace the infinite polling loop in `DiscoveryPhase` with a fully event-driven control flow that eliminates CPU waste and deadlock potential.

**Components**: 
- `src/orchestration/phases/DiscoveryPhase.js`
- `src/orchestration/queues/DiscoveryQueue.js`
- `src/core/ProtocolDefinitions.js` (add new events)

**Architectural Analysis**:

The current implementation violates the **Event-Driven Coordination** principle by using a polling loop:
```javascript
while (!this.queueManager.isDiscoveryComplete()) {
  await new Promise(resolve => setTimeout(resolve, 100));
}
```

This has multiple flaws:
1.  **CPU Waste**: Checks state every 100ms even when no state change has occurred
2.  **Deadlock Vulnerability**: If `isDiscoveryComplete()` condition never becomes true (due to Issue 1), the loop spins forever
3.  **No Timeout**: No upper bound on execution time
4.  **Architectural Violation**: `SystemEventBus` exists but isn't used for state signaling

**Event-Driven Solution**:

The system should use **state transition events** emitted by `DiscoveryQueue` when critical conditions change:

**New Events in `ProtocolDefinitions.js`**:
```javascript
/**
 * @event DISCOVERY:QUEUE_EMPTY
 * @description Emitted when the discovery queue becomes empty (no tasks to dispatch)
 * @payload {Object} { queueLength: 0, pendingCount: number }
 * @when Queue transitions from non-empty to empty
 */

/**
 * @event DISCOVERY:ALL_IDLE
 * @description Emitted when both queue is empty AND all pending tasks complete
 * @payload {Object} { queueLength: 0, pendingCount: 0 }
 * @when System reaches quiescent state (phase can transition)
 */

/**
 * @event DISCOVERY:TASK_COMPLETED
 * @description Emitted when a discovery task finishes (success or failure)
 * @payload {Object} { pageId: string, success: boolean, pendingCount: number }
 * @when Worker returns result
 */
```

**Updated DiscoveryQueue (Emitter)**:
```javascript
// src/orchestration/queues/DiscoveryQueue.js

const SystemEventBus = require('../../core/SystemEventBus');

class DiscoveryQueue {
  constructor() {
    this.eventBus = SystemEventBus.getInstance();
    this.queue = [];
    this.pendingTaskIds = new Set();
    this.visitedUrls = new Set();
  }

  enqueue(pageContext, isFirstPage) {
    // ... existing logic ...
    
    if (enqueued && this.queue.length === 1) {
      // Queue was empty, now has work
      this.eventBus.emit('DISCOVERY:QUEUE_READY', {
        queueLength: this.queue.length
      });
    }
  }

  markComplete(pageId) {
    const wasTracked = this.pendingTaskIds.has(pageId);
    if (wasTracked) {
      this.pendingTaskIds.delete(pageId);
      
      // Emit state change
      this.eventBus.emit('DISCOVERY:TASK_COMPLETED', {
        pageId,
        success: true,
        pendingCount: this.pendingTaskIds.size,
        queueLength: this.queue.length
      });
      
      // Check for quiescent state
      if (this.isComplete()) {
        this.eventBus.emit('DISCOVERY:ALL_IDLE', {
          queueLength: 0,
          pendingCount: 0
        });
      }
    } else {
      this.logger.warn('DiscoveryQueue', `Task ${pageId} not tracked`);
    }
  }

  next() {
    const task = this.queue.shift();
    if (task) {
      this.pendingTaskIds.add(task.pageContext.id);
      
      // Emit queue empty if this was last item
      if (this.queue.length === 0 && this.pendingTaskIds.size > 0) {
        this.eventBus.emit('DISCOVERY:QUEUE_EMPTY', {
          queueLength: 0,
          pendingCount: this.pendingTaskIds.size
        });
      }
    }
    return task;
  }
}
```

**Updated DiscoveryPhase (Listener)**:
```javascript
// src/orchestration/phases/DiscoveryPhase.js

async execute(maxDepth) {
  this.logger.separator('Phase 2: Discovery');
  this.orchestrator.eventBus.emit('PHASE:CHANGED', { phase: 'discovery' });

  // Promise-based event-driven completion
  const completionPromise = new Promise((resolve, reject) => {
    // Listen for quiescent state
    const completeHandler = () => {
      this.logger.debug('DISCOVERY', 'Quiescent state reached');
      cleanup();
      resolve();
    };

    // Safety timeout (fail-fast instead of hang)
    const timeoutMs = 30 * 60 * 1000; // 30 minutes
    const timeoutId = setTimeout(() => {
      this.logger.error('DISCOVERY', `Phase timeout after ${timeoutMs}ms`);
      cleanup();
      reject(new Error('Discovery Phase Timeout: No progress in 30 minutes'));
    }, timeoutMs);

    // Invariant violation detection
    const taskCompleteHandler = ({ pendingCount, queueLength }) => {
      if (pendingCount < 0) {
        this.logger.error('DISCOVERY', `INVARIANT VIOLATION: pendingCount = ${pendingCount}`);
        cleanup();
        reject(new Error('Queue state corrupted: negative pending count'));
      }
    };

    const cleanup = () => {
      clearTimeout(timeoutId);
      this.orchestrator.eventBus.off('DISCOVERY:ALL_IDLE', completeHandler);
      this.orchestrator.eventBus.off('DISCOVERY:TASK_COMPLETED', taskCompleteHandler);
    };

    this.orchestrator.eventBus.on('DISCOVERY:ALL_IDLE', completeHandler);
    this.orchestrator.eventBus.on('DISCOVERY:TASK_COMPLETED', taskCompleteHandler);
  });

  // Main dispatch loop (event-driven, not polling)
  const dispatchLoop = async () => {
    while (!this.queueManager.isDiscoveryComplete()) {
      const task = this.queueManager.nextDiscovery();
      
      if (!task) {
        // Queue empty, wait for event (not polling!)
        await new Promise(resolve => {
          const handler = () => {
            this.orchestrator.eventBus.off('DISCOVERY:QUEUE_READY', handler);
            resolve();
          };
          this.orchestrator.eventBus.once('DISCOVERY:QUEUE_READY', handler);
          
          // Double-check in case event fired before listener registered
          if (this.queueManager.discoveryQueue.getLength() > 0) {
            this.orchestrator.eventBus.off('DISCOVERY:QUEUE_READY', handler);
            resolve();
          }
        });
        continue;
      }

      // ... dispatch task to worker (existing logic) ...
    }
  };

  // Run dispatch loop and wait for completion
  await Promise.all([dispatchLoop(), completionPromise]);

  const stats = this.queueManager.getStatistics();
  this.logger.success('DISCOVERY', `Complete: ${stats.discovered} page(s)`);
}
```

**Benefits of Event-Driven Approach**:

1.  **Eliminates CPU Waste**: No 100ms polling; phase sleeps until woken by events
2.  **Fail-Fast on Corruption**: Detects negative counter immediately via `DISCOVERY:TASK_COMPLETED` listener
3.  **Guaranteed Timeout**: 30-minute safety timeout prevents infinite hang
4.  **Architectural Consistency**: Uses `SystemEventBus` as designed
5.  **Observable State**: All state transitions are explicit events (better debugging/logging)
6.  **Testability**: Events can be mocked for unit testing phase logic

**Invariant Enforcement**:

The event listener actively checks for state corruption:
```javascript
const taskCompleteHandler = ({ pendingCount }) => {
  if (pendingCount < 0) {
    reject(new Error('Queue state corrupted'));
  }
};
```

This ensures that if Issue 1 (negative counter) somehow occurs despite Phase 2 fixes, the system **fails fast** with a clear error instead of hanging silently.

### Phase 4: Data Consistency (Fixing Issue 3)

**Objective**: Ensure `PageContext` titles are consistent with `TitleRegistry`.

**Component**: `src/domain/PageContext.js` & `src/orchestration/GlobalQueueManager.js`

**Proposed Changes**:
1.  **Registry Lookup**: `PageContext` should ideally not store `title` if it's mutable. However, for serialization, it needs it.
2.  **Aggressive Updates**: When `GlobalQueueManager` receives a resolved title, it must update the `PageContext` immediately (already partially implemented, needs verification).
3.  **Fallback Logic**: Remove the "Untitled" fallback in UI if a registry entry exists.

## 4. Architectural Trade-offs and Design Justification

### Polling vs Event-Driven: A Principle-Based Analysis

The original note suggested that "a robust polling loop with timeout is a safer immediate fix." However, this approach contradicts the system's foundational **Event-Driven Coordination** principle. Let's analyze why the event-driven solution is not only architecturally correct but also more robust:

#### Polling Approach Analysis

**Advantages**:
- Simple to implement (fewer moving parts)
- Predictable execution flow (synchronous checks)
- Works even if event emitters fail silently

**Disadvantages** (Architectural Violations):
- **Violates Event-Driven Principle**: The architecture document explicitly states components should "coordinate via events, not direct method calls or polling"
- **Resource Waste**: CPU wakes every 100ms even when no work is available (on a 100-page site with 10-second average discovery time = ~10,000 wasted wake-ups)
- **Tight Coupling**: Phase must actively query queue state, creating dependency on internal queue implementation
- **Missed State Changes**: 100ms window where state changes aren't observed
- **Scaling Issues**: With N concurrent phases, O(N) polling loops waste O(N × 10 cycles/sec) CPU
- **Hidden Failures**: If queue state corrupts, polling continues silently until timeout (30 minutes of wasted execution)

#### Event-Driven Approach Analysis

**Advantages** (Architectural Alignment):
- **Respects Micro-Kernel Pattern**: Master components communicate through `SystemEventBus`, maintaining separation of concerns
- **Zero CPU Waste**: Phase sleeps on `await Promise` until event fires (OS-level blocking, not busy-wait)
- **Immediate Failure Detection**: Invariant violations detected on first occurrence (< 1ms) instead of next poll cycle (up to 100ms)
- **Observable System**: All state transitions are logged events, improving debugging
- **Loose Coupling**: Phase depends on event contract, not queue internals
- **Extensibility**: Additional listeners (metrics, logging, dashboard) can subscribe without modifying phase logic

**Disadvantages**:
- Requires `DiscoveryQueue` refactoring to emit events (manageable, ~30 lines)
- Event registration/cleanup boilerplate (mitigated by using `once` and cleanup functions)
- Potential for missed events if listener registers after event fires (solved with double-check pattern)

#### Verdict: Event-Driven is Architecturally Mandated

The polling approach, while simpler, fundamentally violates the system's stated architecture. The `SystemEventBus` exists precisely to prevent polling-based coordination. Implementing polling would create **architectural debt** where:
1. The event bus is underutilized (why have it if we don't use it for critical control flow?)
2. Future maintainers might add more polling loops, degrading system performance
3. The system becomes a "pseudo-event-driven" architecture (event bus for logging, polling for control)

**The event-driven solution is not an optimization—it is the correct implementation of the declared architecture.**

### Timeout Strategy: Defense in Depth

Both approaches require timeouts, but the event-driven approach provides **layered defense**:

**Layer 1: Immediate Detection**
- Event listener checks `pendingCount < 0` on every task completion
- Detects corruption within milliseconds, not minutes

**Layer 2: Quiescent State Signaling**
- `DISCOVERY:ALL_IDLE` event fires when legitimate completion occurs
- No reliance on polling to detect completion

**Layer 3: Safety Timeout**
- 30-minute timeout catches edge cases (stuck worker, IPC failure, event emitter bug)
- Acts as circuit breaker, not primary completion mechanism

**Layer 4: Logging and Observability**
- All state transitions logged via events
- Post-mortem analysis possible even after timeout

Polling provides only Layer 3, making it a single point of failure.

### Why Not Hybrid?

One might suggest "polling with events as optimization," but this creates worst-case scenarios:
- If events work: Polling is redundant overhead
- If events fail: System relies on slow polling fallback (defeats purpose)
- Hybrid complexity: Must maintain both code paths, test both, handle race conditions between them

**Clean architectures choose one paradigm and execute it well.** This system chose event-driven; we must honor that choice.

### Implementation Risk Mitigation

To address concerns about event-driven complexity:

1. **Double-Check Pattern**: Prevent missed events
   ```javascript
   this.eventBus.once('EVENT', handler);
   if (conditionAlreadyTrue()) { handler(); } // Immediate fire if late
   ```

2. **Event Validation**: Ensure events fire correctly in tests
   ```javascript
   test('DiscoveryQueue emits ALL_IDLE when complete', async () => {
     const emitted = waitForEvent(bus, 'DISCOVERY:ALL_IDLE');
     queue.markComplete(lastTaskId);
     await expect(emitted).resolves.toBeDefined();
   });
   ```

3. **Fallback Timeout**: Safety net remains active
   - Primary: Event-driven completion (fast path)
   - Fallback: 30-minute timeout (circuit breaker)

## 5. Documentation Updates

The following sections of `Docs/ARCHITECTURE.md` require updates to reflect the refactoring:

### 5.1 GlobalQueueManager Section

**Location**: Section 5.2 in ARCHITECTURE.md

**Updates Required**:
- Add description of canonicalization logic in `completeDiscovery`
- Document the `allContexts` Map as the single source of truth for PageContext instances
- Explain deduplication check: "Before creating a new PageContext, the manager checks if a context with the same ID already exists in `allContexts`. If found, the existing context is reused to maintain graph integrity."

### 5.2 DiscoveryQueue Section

**Location**: Section within orchestration/queues package

**Updates Required**:
- Replace counter description with Set-based tracking: "Pending tasks tracked via `pendingTaskIds: Set<string>` for idempotency and accurate state"
- Add event emission documentation:
  ```markdown
  **Events Emitted**:
  - `DISCOVERY:QUEUE_READY`: When queue receives new tasks
  - `DISCOVERY:QUEUE_EMPTY`: When last task is dequeued
  - `DISCOVERY:TASK_COMPLETED`: When a task is marked complete
  - `DISCOVERY:ALL_IDLE`: When queue empty AND pending empty (quiescent state)
  ```

### 5.3 DiscoveryPhase Section

**Location**: Section 5.1 in orchestration/phases

**Updates Required**:
- Update behavioral description from polling to event-driven:
  ```markdown
  **Control Flow**: Event-driven completion via `SystemEventBus`. The phase awaits the `DISCOVERY:ALL_IDLE` event to signal completion rather than polling queue state. A 30-minute timeout provides defense-in-depth against event emitter failures or worker deadlocks.
  ```
- Add invariant detection description:
  ```markdown
  **Invariant Enforcement**: Listens to `DISCOVERY:TASK_COMPLETED` events to detect queue state corruption (e.g., negative pending count). Fails fast with descriptive error instead of hanging.
  ```

### 5.4 ProtocolDefinitions Section

**Location**: Section 1.4 in Core Package

**Updates Required**:
- Add new event types to the "Event Types" documentation:
  ```javascript
  /**
   * Discovery Queue State Events
   */

  /**
   * @event DISCOVERY:QUEUE_READY
   * @summary Queue has tasks ready for dispatch
   * @payload {Object} { queueLength: number }
   */

  /**
   * @event DISCOVERY:QUEUE_EMPTY
   * @summary Queue is empty but tasks still pending
   * @payload {Object} { queueLength: 0, pendingCount: number }
   */

  /**
   * @event DISCOVERY:TASK_COMPLETED
   * @summary A discovery task finished
   * @payload {Object} { pageId: string, success: boolean, pendingCount: number }
   */

  /**
   * @event DISCOVERY:ALL_IDLE
   * @summary Discovery reached quiescent state (can transition)
   * @payload {Object} { queueLength: 0, pendingCount: 0 }
   */
  ```

### 5.5 PageContext Section

**Location**: Section 2.1 in Domain Package

**Updates Required**:
- Add note about canonicalization: "PageContext instances are canonical—only one instance exists per unique page ID. The `GlobalQueueManager` enforces this invariant during discovery."
- Update title management note: "The `title` field is synchronized with `TitleRegistry` when `updateTitleFromRegistry()` is called. All display logic should prefer registry lookup over direct `context.title` access for up-to-date human-readable titles."

### 5.6 Architecture Principles Section

**Location**: Section "Core Architectural Principles"

**Updates Required**:
- Strengthen the "Event-Driven Communication" principle:
  ```markdown
  3. **Event-Driven Communication**: Components coordinate via events, not direct method calls or polling loops. State changes emit events; consumers await events. This eliminates busy-waiting and enables reactive composition.
  ```
- Add new principle:
  ```markdown
  10. **Canonicalization Invariant**: Every entity (Page, URL) has exactly one canonical representation in the system. Duplicate detection occurs at creation time, not retroactively.
  ```

## 6. JSDoc Standards

All new methods must include JSDoc with complete type information and behavioral descriptions:

```javascript
/**
 * @method markComplete
 * @description Marks a discovery task as complete and removes it from pending set.
 *              Idempotent: safe to call multiple times for same ID. Emits state
 *              transition events to notify phase controllers.
 * @param {string} pageId - The ID of the completed page
 * @emits DISCOVERY:TASK_COMPLETED - With completion metadata
 * @emits DISCOVERY:ALL_IDLE - If this completion brings system to quiescent state
 * @returns {boolean} True if task was tracked, false if already completed or unknown
 * @example
 * const wasTracked = queue.markComplete('page-abc123');
 * if (!wasTracked) {
 *   logger.warn('QUEUE', 'Double completion detected');
 * }
 */
markComplete(pageId) {
  // Implementation
}
```

**Required JSDoc Tags**:
- `@method` or `@function`: Method name
- `@description`: Full behavioral description including side effects
- `@param`: All parameters with types and descriptions
- `@returns`: Return value type and meaning
- `@emits`: Any events emitted (critical for event-driven components)
- `@throws`: Any exceptions that can be thrown
- `@example`: Usage example for non-trivial methods

**Event Documentation Standard**:
```javascript
/**
 * @event DISCOVERY:ALL_IDLE
 * @description Emitted when discovery reaches quiescent state (no tasks in queue,
 *              no tasks pending). Signals that the discovery phase can safely
 *              transition to the next phase.
 * @payload {Object} payload - Event data
 * @payload {number} payload.queueLength - Always 0
 * @payload {number} payload.pendingCount - Always 0
 * @when Queue becomes empty AND last pending task completes
 * @listeners DiscoveryPhase (phase transition), DashboardController (UI update)
 */
```

## 7. Testing Strategy for Event-Driven Changes

### 7.1 Unit Tests for DiscoveryQueue Events

**File**: `tests/unit/DiscoveryQueue.test.js`

**Critical Test Cases**:

```javascript
describe('DiscoveryQueue Event Emission', () => {
  test('emits QUEUE_READY when first task enqueued', () => {
    const bus = SystemEventBus.getInstance();
    const queue = new DiscoveryQueue();
    
    const emitted = waitForEvent(bus, 'DISCOVERY:QUEUE_READY');
    queue.enqueue(mockPageContext);
    
    return expect(emitted).resolves.toMatchObject({
      queueLength: 1
    });
  });

  test('emits QUEUE_EMPTY when last task dequeued', () => {
    const queue = new DiscoveryQueue();
    queue.enqueue(mockPageContext);
    
    const emitted = waitForEvent(bus, 'DISCOVERY:QUEUE_EMPTY');
    queue.next(); // Dequeue last task
    
    return expect(emitted).resolves.toMatchObject({
      queueLength: 0,
      pendingCount: 1
    });
  });

  test('emits ALL_IDLE when pending reaches zero with empty queue', () => {
    const queue = new DiscoveryQueue();
    queue.enqueue(mockPageContext);
    const task = queue.next(); // pendingCount = 1
    
    const emitted = waitForEvent(bus, 'DISCOVERY:ALL_IDLE');
    queue.markComplete(task.pageContext.id); // pendingCount = 0
    
    return expect(emitted).resolves.toMatchObject({
      queueLength: 0,
      pendingCount: 0
    });
  });

  test('does not emit ALL_IDLE if queue non-empty', () => {
    const queue = new DiscoveryQueue();
    queue.enqueue(mockPageContext1);
    queue.enqueue(mockPageContext2);
    const task = queue.next(); // Queue still has 1 item
    
    const emitted = waitForEvent(bus, 'DISCOVERY:ALL_IDLE', 100);
    queue.markComplete(task.pageContext.id);
    
    return expect(emitted).rejects.toThrow('timeout');
  });

  test('markComplete is idempotent', () => {
    const queue = new DiscoveryQueue();
    queue.enqueue(mockPageContext);
    const task = queue.next();
    
    expect(queue.markComplete(task.pageContext.id)).toBe(true);
    expect(queue.markComplete(task.pageContext.id)).toBe(false); // Already done
    expect(queue.getPendingCount()).toBe(0); // Not negative!
  });
});
```

### 7.2 Integration Tests for DiscoveryPhase

**File**: `tests/integration/DiscoveryPhase.test.js`

**Critical Test Cases**:

```javascript
describe('DiscoveryPhase Event-Driven Flow', () => {
  test('completes when ALL_IDLE event fires', async () => {
    const orchestrator = new MockOrchestrator();
    const phase = new DiscoveryPhase();
    
    // Simulate: queue has 1 task, completes
    setTimeout(() => {
      orchestrator.queueManager.discoveryQueue.markComplete('page-1');
    }, 100);
    
    await expect(phase.execute(5)).resolves.toBeUndefined();
  });

  test('times out after 30 minutes with no progress', async () => {
    const orchestrator = new MockOrchestrator();
    const phase = new DiscoveryPhase();
    
    // Simulate: queue never completes
    jest.useFakeTimers();
    
    const executePromise = phase.execute(5);
    jest.advanceTimersByTime(30 * 60 * 1000 + 1);
    
    await expect(executePromise).rejects.toThrow('Discovery Phase Timeout');
    jest.useRealTimers();
  });

  test('fails fast on negative pending count', async () => {
    const orchestrator = new MockOrchestrator();
    const phase = new DiscoveryPhase();
    
    const executePromise = phase.execute(5);
    
    // Simulate corruption
    orchestrator.eventBus.emit('DISCOVERY:TASK_COMPLETED', {
      pageId: 'page-1',
      pendingCount: -5 // INVARIANT VIOLATION
    });
    
    await expect(executePromise).rejects.toThrow('Queue state corrupted');
  });

  test('handles rapid completion without missed events', async () => {
    const orchestrator = new MockOrchestrator();
    const phase = new DiscoveryPhase();
    
    // Simulate: 100 tasks complete in rapid succession
    for (let i = 0; i < 100; i++) {
      orchestrator.queueManager.enqueueDiscovery(mockPage(i));
    }
    
    setTimeout(() => {
      for (let i = 0; i < 100; i++) {
        orchestrator.queueManager.discoveryQueue.markComplete(`page-${i}`);
      }
    }, 50);
    
    await expect(phase.execute(5)).resolves.toBeUndefined();
  });
});
```

### 7.3 Test Utilities

**File**: `tests/helpers/eventHelpers.js`

```javascript
/**
 * Wait for an event to be emitted on the SystemEventBus
 * @param {SystemEventBus} bus - Event bus instance
 * @param {string} eventName - Name of event to wait for
 * @param {number} [timeoutMs=1000] - Maximum wait time
 * @returns {Promise<Object>} Resolves with event payload, rejects on timeout
 */
function waitForEvent(bus, eventName, timeoutMs = 1000) {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      bus.off(eventName, handler);
      reject(new Error(`Event ${eventName} not emitted within ${timeoutMs}ms`));
    }, timeoutMs);

    const handler = (payload) => {
      clearTimeout(timeoutId);
      resolve(payload);
    };

    bus.once(eventName, handler);
  });
}

module.exports = { waitForEvent };
```

### 7.4 Test Coverage Requirements

**Critical Edge Cases**:
1. Event fires before listener registers (double-check pattern)
2. Multiple rapid completions (race condition stress test)
3. Worker crash mid-discovery (pending count accuracy)
4. Timeout triggers exactly at 30:00.000 (boundary condition)
5. Negative count detection within 1ms of occurrence (fail-fast verification)

## 8. Implementation Phases and Rollout

### Phase Execution Order (Dependency-Based)

**Phase 1: Graph Integrity** (Issue 4)
- **Duration**: 2-3 hours
- **Risk**: Medium (changes core discovery logic)
- **Blockers**: None
- **Validation**: Integration tests with duplicate link scenarios

**Phase 2: Queue Stability** (Issue 1)
- **Duration**: 2-3 hours
- **Risk**: Low (self-contained queue logic)
- **Blockers**: None (can run parallel to Phase 1)
- **Validation**: Unit tests for idempotency

**Phase 3: Event-Driven Flow** (Issue 2)
- **Duration**: 4-6 hours
- **Risk**: Medium-High (changes control flow paradigm)
- **Blockers**: Phase 2 must complete (relies on accurate pending count)
- **Validation**: Integration tests with timeout and rapid completion scenarios

**Phase 4: Data Consistency** (Issue 3)
- **Duration**: 1-2 hours
- **Risk**: Low (UI/display logic)
- **Blockers**: Phase 1 (relies on canonical contexts)
- **Validation**: Manual testing of dashboard and tree display

### Rollout Strategy

**Step 1: Feature Branch**
```bash
git checkout -b refactor/system-stability
```

**Step 2: Implement Phases in Order**
- Phase 1 → Commit → Test
- Phase 2 → Commit → Test
- Phase 3 → Commit → Test
- Phase 4 → Commit → Test

**Step 3: Integration Testing**
```bash
npm test -- --coverage
npm run test:integration
```

**Step 4: Staging Deployment**
- Run on small test site (10 pages)
- Run on medium test site (50 pages)
- Monitor logs for event emission patterns

**Step 5: Production Merge**
```bash
git checkout main
git merge refactor/system-stability
```

## 9. Appendix: Event Flow Diagrams

### Discovery Phase Event Sequence (Happy Path)

```
TIME  | COMPONENT           | EVENT/ACTION
------|---------------------|------------------------------------------
T0    | DiscoveryPhase      | execute() called
T1    | DiscoveryPhase      | Set up event listeners
T2    | DiscoveryQueue      | enqueue(page1) → QUEUE_READY emitted
T3    | DiscoveryPhase      | Dispatch page1 to worker
T4    | DiscoveryQueue      | next() → queue empty → QUEUE_EMPTY emitted
T5    | Worker              | Discovers 3 child links
T6    | GlobalQueueManager  | completeDiscovery() → enqueues 3 children
T7    | DiscoveryQueue      | markComplete(page1) → TASK_COMPLETED emitted
T8    | DiscoveryQueue      | enqueue(page2,3,4) → QUEUE_READY emitted
T9    | DiscoveryPhase      | Dispatch page2 to worker
...   | ...                 | (repeat for all pages)
T99   | DiscoveryQueue      | markComplete(lastPage)
T100  | DiscoveryQueue      | isComplete() = true → ALL_IDLE emitted
T101  | DiscoveryPhase      | Listener resolves completionPromise
T102  | DiscoveryPhase      | execute() returns
```

### Discovery Phase Event Sequence (Timeout Failure)

```
TIME  | COMPONENT           | EVENT/ACTION
------|---------------------|------------------------------------------
T0    | DiscoveryPhase      | execute() called
T1    | DiscoveryPhase      | Set up listeners + 30min timeout
T2    | DiscoveryQueue      | enqueue(page1) → QUEUE_READY
...   | ...                 | (pages processed normally)
T1800s| DiscoveryQueue      | Still waiting (hung worker? IPC failure?)
T1800s| DiscoveryPhase      | Timeout fires → reject(Error)
T1801s| DiscoveryPhase      | Cleanup listeners
T1802s| ClusterOrchestrator | Catches error, logs, shuts down gracefully
```

### Discovery Phase Event Sequence (Invariant Violation)

```
TIME  | COMPONENT           | EVENT/ACTION
------|---------------------|------------------------------------------
T0    | DiscoveryPhase      | execute() called
T1    | DiscoveryPhase      | Set up listeners (including invariant check)
T2    | DiscoveryQueue      | markComplete(page1) → pendingCount = 0 ✓
T3    | DiscoveryQueue      | markComplete(page1) AGAIN (bug!)
T4    | DiscoveryQueue      | pendingCount = -1 ✗
T5    | DiscoveryQueue      | TASK_COMPLETED emitted with pendingCount=-1
T6    | DiscoveryPhase      | Invariant listener detects corruption
T7    | DiscoveryPhase      | reject(Error('Queue state corrupted'))
T8    | ClusterOrchestrator | Catches error, logs full state, exits
```

---

**End of Refactoring Plan**
