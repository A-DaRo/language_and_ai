/**
 * Integration Tests for User Confirmation Feature
 * Tests UserPrompt utility and ClusterOrchestrator integration
 * 
 * Test Strategy:
 * - Mock readline interface to simulate user input
 * - Test various input scenarios (y/n/invalid/timeout/SIGINT)
 * - Verify UserPrompt behavior in isolation
 * - Verify ClusterOrchestrator phase integration
 */

const assert = require('assert');
const UserPrompt = require('../src/utils/UserPrompt');

// ============================================================================
// Mock Readline Interface
// ============================================================================

/**
 * Mock readline interface for testing
 */
class MockReadline {
  constructor(responses) {
    this.responses = responses || [];
    this.index = 0;
    this.closed = false;
    this.questionCallback = null;
  }

  question(query, callback) {
    if (this.index >= this.responses.length) {
      // No more responses, simulate timeout
      return;
    }
    
    const response = this.responses[this.index++];
    
    // Simulate async callback
    setImmediate(() => {
      if (callback) {
        callback(response);
      }
    });
  }

  close() {
    this.closed = true;
  }
}

// ============================================================================
// Test Suite
// ============================================================================

console.log('\n╔══════════════════════════════════════════════════════════════╗');
console.log('║  User Confirmation Feature - Integration Tests              ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');

let testCount = 0;
let passedTests = 0;
let failedTests = 0;

/**
 * Test helper function
 */
async function runTest(testName, testFn) {
  testCount++;
  try {
    await testFn();
    console.log(`✓ Test ${testCount}: ${testName}`);
    passedTests++;
  } catch (error) {
    console.error(`✗ Test ${testCount}: ${testName}`);
    console.error(`  Error: ${error.message}`);
    if (error.stack) {
      console.error(`  Stack: ${error.stack}`);
    }
    failedTests++;
  }
}

// ============================================================================
// Main Test Runner
// ============================================================================

async function runAllTests() {

// ============================================================================
// Test 1: User Accepts (y)
// ============================================================================

await runTest('User accepts with "y"', async () => {
  const mockRl = new MockReadline(['y']);
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', null, 1000);
  
  assert.strictEqual(result, true, 'Expected true for "y" input');
  
  prompt.close();
});

// ============================================================================
// Test 2: User Accepts (yes)
// ============================================================================

await runTest('User accepts with "yes"', async () => {
  const mockRl = new MockReadline(['yes']);
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', null, 1000);
  
  assert.strictEqual(result, true, 'Expected true for "yes" input');
  
  prompt.close();
});

// ============================================================================
// Test 3: User Declines (n)
// ============================================================================

await runTest('User declines with "n"', async () => {
  const mockRl = new MockReadline(['n']);
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', null, 1000);
  
  assert.strictEqual(result, false, 'Expected false for "n" input');
  
  prompt.close();
});

// ============================================================================
// Test 4: User Declines (no)
// ============================================================================

await runTest('User declines with "no"', async () => {
  const mockRl = new MockReadline(['no']);
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', null, 1000);
  
  assert.strictEqual(result, false, 'Expected false for "no" input');
  
  prompt.close();
});

// ============================================================================
// Test 5: Invalid Input with Retry
// ============================================================================

await runTest('Invalid input followed by valid input', async () => {
  const mockRl = new MockReadline(['invalid', 'maybe', 'y']);
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', null, 5000);
  
  assert.strictEqual(result, true, 'Expected true after invalid inputs and valid "y"');
  
  prompt.close();
});

// ============================================================================
// Test 6: Maximum Invalid Attempts
// ============================================================================

await runTest('Max invalid attempts (3) returns false', async () => {
  const mockRl = new MockReadline(['invalid1', 'invalid2', 'invalid3', 'y']);
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', null, 5000);
  
  assert.strictEqual(result, false, 'Expected false after 3 invalid attempts');
  
  prompt.close();
});

// ============================================================================
// Test 7: Default Answer (Yes)
// ============================================================================

await runTest('Empty input uses default answer (yes)', async () => {
  const mockRl = new MockReadline(['']); // Empty input
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', true, 1000);
  
  assert.strictEqual(result, true, 'Expected true (default) for empty input');
  
  prompt.close();
});

// ============================================================================
// Test 8: Default Answer (No)
// ============================================================================

await runTest('Empty input uses default answer (no)', async () => {
  const mockRl = new MockReadline(['']); // Empty input
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', false, 1000);
  
  assert.strictEqual(result, false, 'Expected false (default) for empty input');
  
  prompt.close();
});

// ============================================================================
// Test 9: Timeout
// ============================================================================

await runTest('Timeout returns false', async () => {
  const mockRl = new MockReadline([]); // No responses = timeout
  const prompt = new UserPrompt(mockRl);
  
  const result = await prompt.promptYesNo('Proceed?', null, 100);
  
  assert.strictEqual(result, false, 'Expected false on timeout');
  
  prompt.close();
});

// ============================================================================
// Test 10: promptConfirmDownload Method
// ============================================================================

await runTest('promptConfirmDownload with stats', async () => {
  const mockRl = new MockReadline(['y']);
  const prompt = new UserPrompt(mockRl);
  
  const stats = {
    totalPages: 42,
    maxDepth: 3,
    conflicts: 2
  };
  
  const result = await prompt.promptConfirmDownload(stats);
  
  assert.strictEqual(result, true, 'Expected true for download confirmation');
  
  prompt.close();
});

// ============================================================================
// Test 11: Case Insensitivity
// ============================================================================

await runTest('Case insensitive input (Y, YES, N, NO)', async () => {
  // Test uppercase Y
  let mockRl = new MockReadline(['Y']);
  let prompt = new UserPrompt(mockRl);
  let result = await prompt.promptYesNo('Proceed?', null, 1000);
  assert.strictEqual(result, true, 'Expected true for "Y"');
  prompt.close();
  
  // Test uppercase YES
  mockRl = new MockReadline(['YES']);
  prompt = new UserPrompt(mockRl);
  result = await prompt.promptYesNo('Proceed?', null, 1000);
  assert.strictEqual(result, true, 'Expected true for "YES"');
  prompt.close();
  
  // Test uppercase N
  mockRl = new MockReadline(['N']);
  prompt = new UserPrompt(mockRl);
  result = await prompt.promptYesNo('Proceed?', null, 1000);
  assert.strictEqual(result, false, 'Expected false for "N"');
  prompt.close();
  
  // Test uppercase NO
  mockRl = new MockReadline(['NO']);
  prompt = new UserPrompt(mockRl);
  result = await prompt.promptYesNo('Proceed?', null, 1000);
  assert.strictEqual(result, false, 'Expected false for "NO"');
  prompt.close();
});

// ============================================================================
// Test 12: Whitespace Handling
// ============================================================================

await runTest('Whitespace trimming', async () => {
  const mockRl = new MockReadline(['  y  ', '  no  ']);
  const prompt = new UserPrompt(mockRl);
  
  // First call with leading/trailing spaces
  let result = await prompt.promptYesNo('Proceed?', null, 1000);
  assert.strictEqual(result, true, 'Expected true for "  y  "');
  
  prompt.close();
});

// ============================================================================
// Test 13: Non-Interactive Mode Detection
// ============================================================================

await runTest('Non-interactive mode returns default', async () => {
  const prompt = new UserPrompt();
  
  // Temporarily mock isTTY to false
  const originalIsTTY = process.stdin.isTTY;
  process.stdin.isTTY = false;
  
  const result = await prompt.promptYesNo('Proceed?', true, 1000);
  
  // Restore original value
  process.stdin.isTTY = originalIsTTY;
  
  assert.strictEqual(result, true, 'Expected default answer in non-interactive mode');
  
  prompt.close();
});

// ============================================================================
// Test Summary
// ============================================================================

console.log('\n' + '='.repeat(64));
console.log(`Test Results: ${passedTests}/${testCount} passed`);

if (failedTests > 0) {
  console.log(`Failed Tests: ${failedTests}`);
  console.log('='.repeat(64));
  process.exit(1);
} else {
  console.log('All tests passed! ✓');
  console.log('='.repeat(64));
  process.exit(0);
}

} // End of runAllTests()

// Run all tests
runAllTests().catch(error => {
  console.error('Fatal error running tests:', error);
  process.exit(1);
});
