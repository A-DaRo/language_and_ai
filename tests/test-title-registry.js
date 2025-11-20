/**
 * @fileoverview Integration test for centralized title registry
 * @description Tests that the ID-to-title mapping remains consistent across
 * discovery, conflict resolution, and download phases
 */

const GlobalQueueManager = require('../src/orchestration/GlobalQueueManager');
const PageContext = require('../src/domain/PageContext');
const ConflictResolver = require('../src/orchestration/analysis/ConflictResolver');

/**
 * Test Suite: Title Registry Synchronization
 */
function testTitleRegistry() {
  console.log('\n=== Test: Title Registry Synchronization ===\n');
  
  let testsPassed = 0;
  let testsFailed = 0;
  
  try {
    // Test 1: Registry initialization
    console.log('Test 1: Registry initialization and basic operations');
    const queueManager = new GlobalQueueManager();
    
    // Create root context
    const rootContext = new PageContext(
      'https://notion.so/29abc123000000000000000000000001',
      'Root Page',
      0,
      null,
      null
    );
    
    // Enqueue and complete discovery with resolved title
    queueManager.enqueueDiscovery(rootContext, true);
    queueManager.completeDiscovery(
      rootContext.id,
      [],
      {},
      'Resolved Root Title'
    );
    
    // Verify title is stored in registry
    const rootTitle = queueManager.getTitleById(rootContext.id);
    assert(rootTitle === 'Resolved Root Title', 'Root title should be stored in registry');
    console.log('✓ Title stored correctly in registry');
    testsPassed++;
    
    // Test 2: Registry serialization
    console.log('\nTest 2: Registry serialization for IPC');
    const titleRegistry = queueManager.getTitleRegistry();
    assert(typeof titleRegistry === 'object', 'Registry should serialize to plain object');
    assert(titleRegistry[rootContext.id] === 'Resolved Root Title', 'Serialized registry should contain title');
    console.log('✓ Registry serializes correctly');
    testsPassed++;
    
    // Test 3: Multiple pages with different titles
    console.log('\nTest 3: Multiple pages with unique titles');
    const child1 = new PageContext(
      'https://notion.so/29abc123000000000000000000000002',
      'Child 1',
      1,
      rootContext,
      rootContext.id
    );
    const child2 = new PageContext(
      'https://notion.so/29abc123000000000000000000000003',
      'Child 2',
      1,
      rootContext,
      rootContext.id
    );
    
    queueManager.enqueueDiscovery(child1, false);
    queueManager.enqueueDiscovery(child2, false);
    
    queueManager.completeDiscovery(child1.id, [], {}, 'Resolved Child 1');
    queueManager.completeDiscovery(child2.id, [], {}, 'Resolved Child 2');
    
    const registry = queueManager.getTitleRegistry();
    assert(Object.keys(registry).length === 3, 'Registry should contain 3 entries');
    assert(registry[child1.id] === 'Resolved Child 1', 'Child 1 title should be correct');
    assert(registry[child2.id] === 'Resolved Child 2', 'Child 2 title should be correct');
    console.log('✓ Multiple titles tracked correctly');
    testsPassed++;
    
    // Test 4: Title immutability
    console.log('\nTest 4: Title immutability after initial resolution');
    queueManager.completeDiscovery(child1.id, [], {}, 'Different Title');
    const unchangedTitle = queueManager.getTitleById(child1.id);
    assert(unchangedTitle === 'Resolved Child 1', 'Title should remain unchanged after initial set');
    console.log('✓ Titles are immutable after first resolution');
    testsPassed++;
    
    // Test 5: ConflictResolver integration with title registry
    console.log('\nTest 5: ConflictResolver uses title registry for logging');
    const allContexts = queueManager.getAllContexts();
    const titleRegistryForResolver = queueManager.getTitleRegistry();
    
    const { canonicalContexts, linkRewriteMap, stats } = ConflictResolver.resolve(
      allContexts,
      titleRegistryForResolver
    );
    
    assert(canonicalContexts.length === allContexts.length, 'All contexts should be canonical (no duplicates in test)');
    assert(linkRewriteMap.size === allContexts.length, 'Link rewrite map should have entries for all pages');
    
    // Verify each context has a mapping
    for (const context of canonicalContexts) {
      const hasMapping = linkRewriteMap.has(context.id);
      assert(hasMapping, `Context ${context.id} should have link rewrite mapping`);
      
      const hasTitle = titleRegistryForResolver[context.id] !== undefined;
      assert(hasTitle, `Context ${context.id} should have title in registry`);
    }
    
    console.log('✓ ConflictResolver integrates correctly with title registry');
    testsPassed++;
    
    // Test 6: Duplicate detection with title registry
    console.log('\nTest 6: Duplicate pages share title mapping');
    const queueManager2 = new GlobalQueueManager();
    
    // Create two contexts pointing to same URL (duplicate scenario)
    const original = new PageContext(
      'https://notion.so/29abc123000000000000000000000010',
      'Original',
      1,
      null,
      null
    );
    const duplicate = new PageContext(
      'https://notion.so/29abc123000000000000000000000010',
      'Duplicate',
      2,
      null,
      null
    );
    
    queueManager2.enqueueDiscovery(original, false);
    queueManager2.completeDiscovery(original.id, [], {}, 'Resolved Original');
    
    // Note: In real scenario, duplicate wouldn't be enqueued (same URL)
    // But simulate for testing purposes
    queueManager2.allContexts.set(duplicate.id, duplicate);
    queueManager2.idToTitleMap.set(duplicate.id, 'Resolved Duplicate');
    
    const allContexts2 = queueManager2.getAllContexts();
    const titleRegistry2 = queueManager2.getTitleRegistry();
    
    const { canonicalContexts: canonical2, linkRewriteMap: map2 } = ConflictResolver.resolve(
      allContexts2,
      titleRegistry2
    );
    
    // Both IDs should map to the same file path
    const originalPath = map2.get(original.id);
    const duplicatePath = map2.get(duplicate.id);
    assert(originalPath === duplicatePath, 'Duplicate contexts should map to same file path');
    console.log('✓ Duplicate detection works with title registry');
    testsPassed++;
    
    // Test 7: PageContext serialization without title fields
    console.log('\nTest 7: PageContext serialization excludes display/original title');
    const testContext = new PageContext(
      'https://notion.so/29abc123000000000000000000000020',
      'Test Page',
      1,
      null,
      null
    );
    
    const json = testContext.toJSON();
    assert(json.originalTitle === undefined, 'originalTitle should not be in serialized JSON');
    assert(json.displayTitle === undefined, 'displayTitle should not be in serialized JSON');
    assert(json.title !== undefined, 'Sanitized title should be present');
    assert(json.id !== undefined, 'ID should be present');
    
    // Verify deserialization works
    const reconstructed = PageContext.fromJSON(json);
    assert(reconstructed.id === testContext.id, 'ID should match after deserialization');
    assert(reconstructed.title === testContext.title, 'Sanitized title should match');
    
    console.log('✓ PageContext serialization simplified correctly');
    testsPassed++;
    
    // Summary
    console.log('\n=== Test Summary ===');
    console.log(`Tests Passed: ${testsPassed}`);
    console.log(`Tests Failed: ${testsFailed}`);
    
    if (testsFailed === 0) {
      console.log('\n✅ All tests passed!');
      return true;
    } else {
      console.log('\n❌ Some tests failed');
      return false;
    }
    
  } catch (error) {
    console.error('❌ Test suite failed with error:', error);
    testsFailed++;
    return false;
  }
}

/**
 * Simple assertion helper
 */
function assert(condition, message) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  const success = testTitleRegistry();
  process.exit(success ? 0 : 1);
}

module.exports = { testTitleRegistry };
