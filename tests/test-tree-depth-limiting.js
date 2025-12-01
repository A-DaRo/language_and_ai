/**
 * Unit Tests for Tree Depth Limiting in UserConfirmationPhase
 * Verifies that the site tree only shows BFS expansion + cycle edges at leaves
 */

const assert = require('assert');
const PageContext = require('../src/domain/PageContext');

// ============================================================================
// Mock Logger
// ============================================================================

class MockLogger {
  info() {}
  warn() {}
  error() {}
  success() {}
  separator() {}
  debug() {}
}

// ============================================================================
// Mock QueueManager (minimal)
// ============================================================================

class MockQueueManager {
  constructor(maxDepth = 2, titleRegistry = {}) {
    this.maxDepth = maxDepth;
    this.titleRegistry = titleRegistry;
  }

  getMaxDepth() {
    return this.maxDepth;
  }

  getTitleRegistry() {
    return this.titleRegistry;
  }
}

// ============================================================================
// UserConfirmationPhase Mock (just the tree display methods)
// ============================================================================

class UserConfirmationPhaseMock {
  constructor(maxDepth = 2, titleRegistry = {}) {
    this.logger = new MockLogger();
    this.queueManager = new MockQueueManager(maxDepth, titleRegistry);
    this.output = [];
  }

  _displayPageTree(rootContext) {
    this.output = [];
    const titleRegistry = this.queueManager.getTitleRegistry();
    const rootLabel = titleRegistry[rootContext.id] || rootContext.title || '(root)';
    const maxBfsDepth = this.queueManager.getMaxDepth();
    
    this.output.push('.');
    this.output.push(`└─ ${rootLabel}`);

    rootContext.children.forEach((child, index) => {
      const isLast = index === rootContext.children.length - 1;
      this._printTreeNode(
        child,
        '   ',
        isLast,
        titleRegistry,
        new Set([rootContext.id]),
        maxBfsDepth
      );
    });
  }

  _printTreeNode(context, prefix, isLast, titleRegistry, pathVisited = new Set(), maxBfsDepth = Infinity) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = titleRegistry[context.id] || context.title || 'Untitled';

    // Detect cycles within the path (visited in current recursion)
    if (pathVisited.has(context.id)) {
      this.output.push(`${prefix}${connector}${title} ↺ (Cycle)`);
      return;
    }

    // Filter to only show discovered children (have titles in registry)
    const exploredChildren = context.children.filter(child => titleRegistry[child.id]);
    const internalRefs = context.children.length - exploredChildren.length;
    const label = internalRefs > 0 ? `${title} [${internalRefs} internal ref${internalRefs > 1 ? 's' : ''}]` : title;

    this.output.push(`${prefix}${connector}${label}`);

    // Stop recursion at BFS depth + 1 to limit tree representation
    // This shows the full BFS expansion plus edges creating cycles at leaves
    if (context.depth >= maxBfsDepth) {
      // We're at or past the BFS leaf level - show children as cycles only
      for (const child of exploredChildren) {
        if (pathVisited.has(child.id)) {
          // Only show explicitly detected cycles
          const connector2 = child === exploredChildren[exploredChildren.length - 1] ? '└─ ' : '├─ ';
          const childTitle = titleRegistry[child.id] || child.title || 'Untitled';
          const childPrefix = prefix + (isLast ? '   ' : '│  ');
          this.output.push(`${childPrefix}${connector2}${childTitle} ↺ (Cycle)`);
        }
        // Don't recurse deeper
      }
      return;
    }

    // Recurse into children (we're within BFS expansion depth)
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    const nextVisited = new Set(pathVisited);
    nextVisited.add(context.id);
    
    exploredChildren.forEach((child, index) => {
      const childIsLast = index === exploredChildren.length - 1;
      this._printTreeNode(
        child,
        childPrefix,
        childIsLast,
        titleRegistry,
        nextVisited,
        maxBfsDepth
      );
    });
  }
}

// ============================================================================
// Test Cases
// ============================================================================

console.log('\n╔══════════════════════════════════════════════════════════════╗');
console.log('║  Tree Depth Limiting - Unit Tests                            ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');

let testCount = 0;
let passedTests = 0;
let failedTests = 0;

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
      console.error(`  Stack: ${error.stack.split('\n').slice(0, 3).join('\n')}`);
    }
    failedTests++;
  }
}

// ============================================================================
// Test 1: Linear tree (no cycles)
// ============================================================================

await runTest('Linear tree with maxDepth=2', async () => {
  // Create: Root -> A -> B -> C (depth 0,1,2,3)
  const root = new PageContext('https://notion.so/root', 'Root', 0, null, null);
  const a = new PageContext('https://notion.so/a', 'Page A', 1, root, root.id);
  const b = new PageContext('https://notion.so/b', 'Page B', 2, a, a.id);
  const c = new PageContext('https://notion.so/c', 'Page C', 3, b, b.id);

  root.addChild(a);
  a.addChild(b);
  b.addChild(c);

  const titleRegistry = {
    [root.id]: 'Root',
    [a.id]: 'Page A',
    [b.id]: 'Page B',
    [c.id]: 'Page C'
  };

  const phase = new UserConfirmationPhaseMock(2, titleRegistry);
  phase._displayPageTree(root);

  // With maxBfsDepth=2, we should see Root -> A -> B
  // But NOT the recursion into C (which is at depth 3)
  // So we expect the tree to stop at B
  const output = phase.output.join('\n');
  
  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A (depth 1)');
  assert(output.includes('Page B'), 'Should show Page B (depth 2)');
  assert(!output.includes('Page C'), 'Should NOT show Page C (depth 3 > maxDepth 2)');
});

// ============================================================================
// Test 2: Back edge (cycle) at leaf level
// ============================================================================

await runTest('Cycle detection at BFS leaves (Page B links back to Root)', async () => {
  // Create: Root -> A -> B, and B -> Root (back edge)
  const root = new PageContext('https://notion.so/root', 'Root', 0, null, null);
  const a = new PageContext('https://notion.so/a', 'Page A', 1, root, root.id);
  const b = new PageContext('https://notion.so/b', 'Page B', 2, a, a.id);

  root.addChild(a);
  a.addChild(b);
  b.addChild(root);  // Back edge: B links to Root

  const titleRegistry = {
    [root.id]: 'Root',
    [a.id]: 'Page A',
    [b.id]: 'Page B'
  };

  const phase = new UserConfirmationPhaseMock(2, titleRegistry);
  phase._displayPageTree(root);

  const output = phase.output.join('\n');
  
  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A');
  assert(output.includes('Page B'), 'Should show Page B at leaf level (depth 2 = maxDepth 2)');
  assert(output.includes('↺ (Cycle)'), 'Should mark back edge as cycle');
});

// ============================================================================
// Test 3: Wide tree (multiple children)
// ============================================================================

await runTest('Wide tree with multiple children at each level', async () => {
  // Create tree:
  // Root (depth 0)
  //  ├─ A (depth 1)
  //  ├─ B (depth 1)
  //  └─ C (depth 1)
  //      └─ D (depth 2)

  const root = new PageContext('https://notion.so/root', 'Root', 0, null, null);
  const a = new PageContext('https://notion.so/a', 'Page A', 1, root, root.id);
  const b = new PageContext('https://notion.so/b', 'Page B', 1, root, root.id);
  const c = new PageContext('https://notion.so/c', 'Page C', 1, root, root.id);
  const d = new PageContext('https://notion.so/d', 'Page D', 2, c, c.id);

  root.addChild(a);
  root.addChild(b);
  root.addChild(c);
  c.addChild(d);

  const titleRegistry = {
    [root.id]: 'Root',
    [a.id]: 'Page A',
    [b.id]: 'Page B',
    [c.id]: 'Page C',
    [d.id]: 'Page D'
  };

  const phase = new UserConfirmationPhaseMock(2, titleRegistry);
  phase._displayPageTree(root);

  const output = phase.output.join('\n');

  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A');
  assert(output.includes('Page B'), 'Should show Page B');
  assert(output.includes('Page C'), 'Should show Page C');
  assert(output.includes('Page D'), 'Should show Page D (depth 2 = maxDepth)');
});

// ============================================================================
// Test 4: Child with back edge at leaf
// ============================================================================

await runTest('Child node at leaf links to ancestor (cross edge)', async () => {
  // Create:
  // Root (depth 0)
  //  └─ A (depth 1)
  //      └─ B (depth 2)
  //          └─ A (back edge - should show as cycle)

  const root = new PageContext('https://notion.so/root', 'Root', 0, null, null);
  const a = new PageContext('https://notion.so/a', 'Page A', 1, root, root.id);
  const b = new PageContext('https://notion.so/b', 'Page B', 2, a, a.id);

  root.addChild(a);
  a.addChild(b);
  b.addChild(a);  // Back edge

  const titleRegistry = {
    [root.id]: 'Root',
    [a.id]: 'Page A',
    [b.id]: 'Page B'
  };

  const phase = new UserConfirmationPhaseMock(2, titleRegistry);
  phase._displayPageTree(root);

  const output = phase.output.join('\n');

  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A');
  assert(output.includes('Page B'), 'Should show Page B');
  assert(output.includes('↺ (Cycle)'), 'Should show back edge as cycle');
});

// ============================================================================
// Test 5: Only discovered children shown
// ============================================================================

await runTest('Only discovered children (with registry entries) shown', async () => {
  // Create:
  // Root -> A -> B, but B has a child C that was NOT discovered
  // (C won't be in titleRegistry)

  const root = new PageContext('https://notion.so/root', 'Root', 0, null, null);
  const a = new PageContext('https://notion.so/a', 'Page A', 1, root, root.id);
  const b = new PageContext('https://notion.so/b', 'Page B', 2, a, a.id);
  const c = new PageContext('https://notion.so/c', 'Page C', 3, b, b.id);

  root.addChild(a);
  a.addChild(b);
  b.addChild(c);  // C not in registry

  const titleRegistry = {
    [root.id]: 'Root',
    [a.id]: 'Page A',
    [b.id]: 'Page B'
    // Note: C is NOT in registry (not discovered)
  };

  const phase = new UserConfirmationPhaseMock(2, titleRegistry);
  phase._displayPageTree(root);

  const output = phase.output.join('\n');

  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A');
  assert(output.includes('Page B'), 'Should show Page B');
  assert(!output.includes('Page C'), 'Should NOT show Page C (not in registry)');
  assert(output.includes('[1 internal ref]'), 'Should show count of undiscovered children');
});

// ============================================================================
// Test Summary
// ============================================================================

async function main() {
  console.log('\n' + '='.repeat(64));
  console.log(`Test Results: ${passedTests}/${testCount} passed`);

  if (failedTests > 0) {
    console.log(`Failed Tests: ${failedTests}`);
    console.log('='.repeat(64));
    process.exit(1);
  } else {
    console.log('All tests passed! ✓');
    console.log('='.repeat(64) + '\n');
    process.exit(0);
  }
}

main().catch(error => {
  console.error('Fatal error running tests:', error);
  process.exit(1);
});
