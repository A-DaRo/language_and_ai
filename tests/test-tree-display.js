#!/usr/bin/env node

const assert = require('assert');
const PageContext = require('../src/domain/PageContext');

// ============================================================================
// Simple implementation of _printTreeNode for testing
// ============================================================================

function captureTreeOutput(context, maxBfsDepth, titleRegistry) {
  const output = [];

  function _printTreeNode(context, prefix, isLast, titleRegistry, pathVisited = new Set(), maxBfsDepth = Infinity) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = titleRegistry[context.id] || context.title || 'Untitled';

    // Detect cycles within the path (visited in current recursion)
    if (pathVisited.has(context.id)) {
      output.push(`${prefix}${connector}${title} ↺ (Cycle)`);
      return;
    }

    // Filter to only show discovered children (have titles in registry)
    const exploredChildren = context.children.filter(child => titleRegistry[child.id]);
    const internalRefs = context.children.length - exploredChildren.length;
    const label = internalRefs > 0 ? `${title} [${internalRefs} internal ref${internalRefs > 1 ? 's' : ''}]` : title;

    output.push(`${prefix}${connector}${label}`);

    // Stop recursion at BFS depth + 1 to limit tree representation
    if (context.depth >= maxBfsDepth) {
      // We're at or past the BFS leaf level - show children as cycles only
      for (const child of exploredChildren) {
        if (pathVisited.has(child.id)) {
          const connector2 = child === exploredChildren[exploredChildren.length - 1] ? '└─ ' : '├─ ';
          const childTitle = titleRegistry[child.id] || child.title || 'Untitled';
          const childPrefix = prefix + (isLast ? '   ' : '│  ');
          output.push(`${childPrefix}${connector2}${childTitle} ↺ (Cycle)`);
        }
      }
      return;
    }

    // Recurse into children (we're within BFS expansion depth)
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    const nextVisited = new Set(pathVisited);
    nextVisited.add(context.id);
    
    exploredChildren.forEach((child, index) => {
      const childIsLast = index === exploredChildren.length - 1;
      _printTreeNode(
        child,
        childPrefix,
        childIsLast,
        titleRegistry,
        nextVisited,
        maxBfsDepth
      );
    });
  }

  const rootLabel = titleRegistry[context.id] || context.title || '(root)';
  output.push(`└─ ${rootLabel}`);

  context.children.forEach((child, index) => {
    const isLast = index === context.children.length - 1;
    _printTreeNode(
      child,
      '   ',
      isLast,
      titleRegistry,
      new Set([context.id]),
      maxBfsDepth
    );
  });

  return output.join('\n');
}

// ============================================================================
// Test Cases
// ============================================================================

console.log('\n╔══════════════════════════════════════════════════════════════╗');
console.log('║  Tree Depth Limiting - Integration Tests                    ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');

let testCount = 0;
let passedTests = 0;

function test(name, fn) {
  testCount++;
  try {
    fn();
    console.log(`✓ Test ${testCount}: ${name}`);
    passedTests++;
  } catch (error) {
    console.error(`✗ Test ${testCount}: ${name}`);
    console.error(`  Error: ${error.message}`);
  }
}

// ============================================================================
// Test 1: Linear tree (no cycles)
// ============================================================================

test('Linear tree with maxDepth=2', () => {
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

  const output = captureTreeOutput(root, 2, titleRegistry);
  
  // With maxBfsDepth=2, we should see Root -> A -> B
  // But NOT C (which is at depth 3)
  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A (depth 1)');
  assert(output.includes('Page B'), 'Should show Page B (depth 2)');
  assert(!output.includes('Page C'), 'Should NOT show Page C (depth 3 > maxDepth 2)');
  
  console.log('  Output:\n' + output.split('\n').map(l => '    ' + l).join('\n'));
});

// ============================================================================
// Test 2: Back edge (cycle) at leaf level
// ============================================================================

test('Cycle detection at BFS leaves (Page B links back to Root)', () => {
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

  const output = captureTreeOutput(root, 2, titleRegistry);
  
  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A');
  assert(output.includes('Page B'), 'Should show Page B at leaf level');
  assert(output.includes('↺ (Cycle)'), 'Should mark back edge as cycle');
  
  console.log('  Output:\n' + output.split('\n').map(l => '    ' + l).join('\n'));
});

// ============================================================================
// Test 3: Wide tree (multiple children)
// ============================================================================

test('Wide tree with multiple children at each level', () => {
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

  const output = captureTreeOutput(root, 2, titleRegistry);

  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A');
  assert(output.includes('Page B'), 'Should show Page B');
  assert(output.includes('Page C'), 'Should show Page C');
  assert(output.includes('Page D'), 'Should show Page D (depth 2 = maxDepth)');
  
  console.log('  Output:\n' + output.split('\n').map(l => '    ' + l).join('\n'));
});

// ============================================================================
// Test 4: Undiscovered children tracking
// ============================================================================

test('Only discovered children shown (with internal ref count)', () => {
  // Create: Root -> A -> B, but B has a child C that was NOT discovered
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

  const output = captureTreeOutput(root, 2, titleRegistry);

  assert(output.includes('Root'), 'Should show root');
  assert(output.includes('Page A'), 'Should show Page A');
  assert(output.includes('Page B'), 'Should show Page B');
  assert(!output.includes('Page C'), 'Should NOT show Page C (not in registry)');
  assert(output.includes('[1 internal ref]'), 'Should show count of undiscovered children');
  
  console.log('  Output:\n' + output.split('\n').map(l => '    ' + l).join('\n'));
});

// ============================================================================
// Test Summary
// ============================================================================

console.log('\n' + '='.repeat(64));
console.log(`Test Results: ${passedTests}/${testCount} passed`);

if (passedTests === testCount) {
  console.log('All tests passed! ✓');
  console.log('='.repeat(64) + '\n');
  process.exit(0);
} else {
  console.log(`Failed Tests: ${testCount - passedTests}`);
  console.log('='.repeat(64) + '\n');
  process.exit(1);
}
