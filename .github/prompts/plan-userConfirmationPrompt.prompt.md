# Implementation Plan: User Confirmation Prompt After Discovery Phase

## Overview
Add an interactive yes/no user confirmation prompt after the discovery phase completes. The system should:
1. Display the discovered site tree structure
2. Wait for user input (y/n)
3. Proceed to download phase on "yes"
4. Abort gracefully on "no"

## Current Architecture Context

### Phase System (Before Changes)
1. **Bootstrap Phase** - Initialize system, browser pool, root URL validation
2. **Discovery Phase** - BFS traversal, link extraction, populate queue
3. **Conflict Resolution Phase** - Deduplicate paths, resolve naming conflicts
4. **Download Phase** - Parallel download of all discovered pages
5. **Complete Phase** - Cleanup, shutdown workers, final reporting

### Key Files to Modify
- `src/utils/UserPrompt.js` (NEW) - Readline-based prompt utility
- `src/orchestration/ClusterOrchestrator.js` - Add USER_CONFIRMATION phase
- `Docs/ARCHITECTURE.md` - Document new phase
- `tests/test-user-confirmation.js` (NEW) - Integration tests

## Implementation Steps

### Step 1: Create UserPrompt Utility Module
**File:** `src/utils/UserPrompt.js`

**Requirements:**
- Use Node.js `readline` module for terminal input
- Support timeout mechanism (default 60 seconds)
- Handle SIGINT (Ctrl+C) gracefully
- Validate y/n input with retry on invalid input
- Support optional default answer
- Provide cleanup method to close readline interface

**Methods:**
```javascript
class UserPrompt {
  /**
   * Prompt user with yes/no question
   * @param {string} question - Question to display
   * @param {boolean|null} defaultAnswer - Default answer (null = no default)
   * @param {number} timeout - Timeout in milliseconds (default 60000)
   * @returns {Promise<boolean>} - true for yes, false for no
   */
  async promptYesNo(question, defaultAnswer = null, timeout = 60000)

  /**
   * Specialized prompt for download confirmation with stats
   * @param {Object} stats - Discovery statistics
   * @returns {Promise<boolean>}
   */
  async promptConfirmDownload(stats)

  /**
   * Close readline interface
   */
  close()
}
```

**Error Handling:**
- Timeout → return false (abort)
- SIGINT → return false (abort)
- Invalid input → re-prompt with error message
- Multiple invalid attempts (3+) → return false (abort)

### Step 2: Add USER_CONFIRMATION Phase to ClusterOrchestrator
**File:** `src/orchestration/ClusterOrchestrator.js`

**Changes Required:**

#### 2.1 Update Phase Enum
```javascript
const Phase = {
  BOOTSTRAP: 'BOOTSTRAP',
  DISCOVERY: 'DISCOVERY',
  USER_CONFIRMATION: 'USER_CONFIRMATION',  // NEW
  CONFLICT_RESOLUTION: 'CONFLICT_RESOLUTION',
  DOWNLOAD: 'DOWNLOAD',
  COMPLETE: 'COMPLETE'
};
```

#### 2.2 Implement _phaseUserConfirmation() Method
**Location:** After `_phaseDiscovery()` method

**Logic:**
1. Display separator: "Phase 3: User Confirmation"
2. Get discovery statistics from queueManager
3. Display site tree using existing `_displayPageTree()` method
4. Display summary statistics (total pages, depth, conflicts if any)
5. Create UserPrompt instance
6. Call `promptConfirmDownload()` with statistics
7. Log user decision
8. Clean up prompt interface
9. Return boolean result

**Pseudo-code:**
```javascript
async _phaseUserConfirmation() {
  this.logger.info('='.repeat(80));
  this.logger.info('Phase 3: User Confirmation');
  this.logger.info('='.repeat(80));

  const stats = {
    totalPages: this.queueManager.getQueue().size,
    maxDepth: this.queueManager.getMaxDepth(),
    conflicts: this.queueManager.getConflicts().length
  };

  this.logger.info('\nDiscovered Site Structure:');
  this._displayPageTree();

  this.logger.info(`\nDiscovery Summary:`);
  this.logger.info(`  Total Pages: ${stats.totalPages}`);
  this.logger.info(`  Maximum Depth: ${stats.maxDepth}`);
  if (stats.conflicts > 0) {
    this.logger.info(`  Path Conflicts: ${stats.conflicts}`);
  }

  const prompt = new UserPrompt();
  const proceed = await prompt.promptConfirmDownload(stats);
  prompt.close();

  if (proceed) {
    this.logger.info('User confirmed. Proceeding to download phase...');
  } else {
    this.logger.warn('User declined. Aborting download process.');
  }

  return proceed;
}
```

#### 2.3 Update start() Method Flow
**Current Flow:**
```javascript
await this._phaseBootstrap(rootUrl);
await this._phaseDiscovery(rootUrl, maxDepth);
const allContexts = await this._phaseConflictResolution();
await this._phaseDownload(allContexts, concurrency);
await this._phaseComplete();
```

**New Flow:**
```javascript
await this._phaseBootstrap(rootUrl);
await this._phaseDiscovery(rootUrl, maxDepth);

// NEW: User confirmation phase
const userConfirmed = await this._phaseUserConfirmation();
if (!userConfirmed) {
  this.logger.warn('Download aborted by user.');
  await this._phaseComplete(true); // true = aborted
  return;
}

const allContexts = await this._phaseConflictResolution();
await this._phaseDownload(allContexts, concurrency);
await this._phaseComplete();
```

#### 2.4 Update _phaseComplete() to Handle Abortion
**Add optional parameter:**
```javascript
async _phaseComplete(aborted = false) {
  this.logger.info('='.repeat(80));
  this.logger.info(`Phase ${aborted ? 6 : 7}: Complete`);
  this.logger.info('='.repeat(80));

  if (aborted) {
    this.logger.info('Scraping process aborted by user. Cleaning up...');
  } else {
    this.logger.info('All pages downloaded successfully!');
    // ... existing success logic
  }

  // ... existing cleanup logic
}
```

### Step 3: Update Phase Numbering
**Files Affected:** `src/orchestration/ClusterOrchestrator.js`

**Renumbering Required:**
- Phase 1: Bootstrap (unchanged)
- Phase 2: Discovery (unchanged)
- Phase 3: User Confirmation (NEW)
- Phase 4: Conflict Resolution (was Phase 3)
- Phase 5: Download (was Phase 4)
- Phase 6: Complete (was Phase 5)

**Update Locations:**
- Separator log messages in each `_phase*()` method
- JSDoc comments
- Method documentation
- Any hardcoded phase references

### Step 4: Update ARCHITECTURE.md Documentation
**File:** `Docs/ARCHITECTURE.md`

**Sections to Update:**

#### 4.1 High-Level Architecture Overview
Add USER_CONFIRMATION phase to workflow description:
```markdown
### Phase 3: User Confirmation
After discovery completes, the system displays the discovered site structure
and prompts the user for confirmation:
- Display complete site tree with titles and depths
- Show summary statistics (total pages, max depth, conflicts)
- Wait for y/n input with 60-second timeout
- Proceed to download on "yes", abort on "no"
- Handle Ctrl+C gracefully as abort signal
```

#### 4.2 Detailed Package Documentation
Add `src/utils/UserPrompt.js` to utils section:
```markdown
#### UserPrompt.js
Terminal interaction utility for user confirmation prompts.

**Key Features:**
- Readline-based interactive prompts
- Timeout handling (default 60s)
- Input validation with retry logic
- SIGINT (Ctrl+C) handling
- Graceful cleanup

**Primary Methods:**
- `promptYesNo(question, defaultAnswer, timeout)` - Generic y/n prompt
- `promptConfirmDownload(stats)` - Download confirmation with context
- `close()` - Cleanup readline interface
```

#### 4.3 Update Phase Flow Diagram
Update ASCII diagram to include USER_CONFIRMATION phase between DISCOVERY and CONFLICT_RESOLUTION.

### Step 5: Create Integration Tests
**File:** `tests/test-user-confirmation.js`

**Test Cases:**
1. **User accepts (y)** - Should proceed to conflict resolution
2. **User declines (n)** - Should abort gracefully
3. **User enters invalid input** - Should re-prompt
4. **Timeout occurs** - Should abort after timeout
5. **SIGINT (Ctrl+C)** - Should abort gracefully
6. **Multiple invalid attempts** - Should abort after 3 failures
7. **Display tree accuracy** - Verify tree matches discovered pages

**Test Strategy:**
- Mock readline interface to simulate user input
- Mock setTimeout for timeout scenarios
- Verify logger output for correct messages
- Verify queueManager state after abort
- Verify worker cleanup on abort

**Pseudo-code Structure:**
```javascript
const assert = require('assert');
const UserPrompt = require('../src/utils/UserPrompt');

// Mock readline
class MockReadline {
  constructor(responses) {
    this.responses = responses;
    this.index = 0;
  }
  
  question(query, callback) {
    const response = this.responses[this.index++];
    callback(response);
  }
  
  close() {}
}

// Test: User accepts
async function testUserAccepts() {
  const mock = new MockReadline(['y']);
  const prompt = new UserPrompt(mock);
  const result = await prompt.promptYesNo('Proceed?');
  assert.strictEqual(result, true);
}

// Test: User declines
async function testUserDeclines() {
  const mock = new MockReadline(['n']);
  const prompt = new UserPrompt(mock);
  const result = await prompt.promptYesNo('Proceed?');
  assert.strictEqual(result, false);
}

// ... additional test cases
```

## Edge Cases to Handle

### 1. Non-Interactive Environment
- Detect if stdin is not a TTY (e.g., running in CI/CD)
- Fall back to default behavior (proceed without prompt)
- Log warning about non-interactive mode

### 2. Empty Discovery Results
- If queueManager has 0 pages, skip confirmation
- Log warning and abort early
- Don't display empty tree

### 3. Very Large Trees
- Limit tree display to first 100 pages
- Add "... and N more pages" message
- Provide statistics summary regardless of size

### 4. Concurrent Access
- Ensure readline interface is not used concurrently
- UserPrompt should be single-use instance
- Create new instance for each prompt

## Rollback Plan
If issues arise, changes can be reverted in reverse order:
1. Delete `tests/test-user-confirmation.js`
2. Revert ARCHITECTURE.md changes
3. Remove `_phaseUserConfirmation()` from ClusterOrchestrator
4. Restore original phase numbering
5. Delete `src/utils/UserPrompt.js`

## Success Criteria
- [ ] UserPrompt utility created with all methods implemented
- [ ] USER_CONFIRMATION phase added to ClusterOrchestrator
- [ ] Phase numbering updated throughout codebase
- [ ] ARCHITECTURE.md documentation updated
- [ ] Integration tests created and passing
- [ ] Manual testing confirms y/n/timeout/Ctrl+C behaviors work
- [ ] Non-interactive environments handled gracefully
- [ ] No breaking changes to existing functionality

## Implementation Order
1. Create `src/utils/UserPrompt.js` (foundation)
2. Update `src/orchestration/ClusterOrchestrator.js` (phase implementation)
3. Update phase numbering in ClusterOrchestrator (consistency)
4. Update `Docs/ARCHITECTURE.md` (documentation)
5. Create `tests/test-user-confirmation.js` (validation)
6. Manual testing (verification)

## Further Considerations

1. Interactive mode flag: Add --no-interactive to bypass prompt for automation
2. Dry-run integration: Skip prompt when --dry-run is set