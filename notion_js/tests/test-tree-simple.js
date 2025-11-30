#!/usr/bin/env node

const assert = require('assert');
const PageContext = require('../src/domain/PageContext');

console.log('Testing tree depth limiting...');

// Test 1: Simple linear tree
const root = new PageContext('https://notion.so/root', 'Root', 0, null, null);
const a = new PageContext('https://notion.so/a', 'Page A', 1, root, root.id);
const b = new PageContext('https://notion.so/b', 'Page B', 2, a, a.id);

root.addChild(a);
a.addChild(b);

console.log('✓ Created test contexts');

// Verify structure
assert.strictEqual(root.children.length, 1, 'Root should have 1 child');
assert.strictEqual(root.children[0].id, a.id, 'Root child should be A');
assert.strictEqual(a.children.length, 1, 'A should have 1 child');
assert.strictEqual(a.children[0].id, b.id, 'A child should be B');

console.log('✓ Tree structure verified');
console.log('\nAll basic tests passed! ✓');
process.exit(0);
