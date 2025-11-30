/**
 * Integration test: Discovery Phase Workflow
 * 
 * Tests the complete discovery workflow:
 * 1. GlobalQueueManager coordinates graph traversal
 * 2. Workers discover pages and return child links
 * 3. GraphAnalyzer classifies edges (tree, back, forward, cross)
 * 4. ConflictResolver detects and prunes duplicates
 */

const GlobalQueueManager = require('../../../src/orchestration/GlobalQueueManager');
const GraphAnalyzer = require('../../../src/orchestration/analysis/GraphAnalyzer');
const ConflictResolver = require('../../../src/orchestration/analysis/ConflictResolver');
const { createMockPageContext, createDuplicateContexts } = require('../../helpers/factories');
const SystemEventBus = require('../../../src/core/SystemEventBus');

describe('Discovery Phase Workflow', () => {
  let queueManager;
  let graphAnalyzer;

  beforeEach(() => {
    queueManager = new GlobalQueueManager();
    graphAnalyzer = new GraphAnalyzer({
      info: jest.fn(),
      debug: jest.fn(),
      error: jest.fn()
    });
    SystemEventBus._reset();
  });

  test('discovers 3-level graph topology', () => {
    // Level 0: Root page
    const root = createMockPageContext({
      pageId: 'root',
      url: 'https://notion.so/Root-29' + 'a'.repeat(30),
      depth: 0,
      parentId: null
    });

    // Level 1: Two children
    const child1 = createMockPageContext({
      pageId: 'child1',
      url: 'https://notion.so/Child1-29' + 'b'.repeat(30),
      depth: 1,
      parentId: 'root'
    });

    const child2 = createMockPageContext({
      pageId: 'child2',
      url: 'https://notion.so/Child2-29' + 'c'.repeat(30),
      depth: 1,
      parentId: 'root'
    });

    // Level 2: Grandchild
    const grandchild = createMockPageContext({
      pageId: 'grandchild',
      url: 'https://notion.so/Grandchild-29' + 'd'.repeat(30),
      depth: 2,
      parentId: 'child1'
    });

    // Simulate discovery sequence
    queueManager.enqueueDiscovery(root, true);
    expect(queueManager.discoveryQueue.length).toBe(1);

    // Root discovered, children added
    queueManager.enqueueDiscovery(child1);
    queueManager.enqueueDiscovery(child2);
    expect(queueManager.discoveryQueue.length).toBe(3);

    // Child1 discovered, grandchild added
    queueManager.enqueueDiscovery(grandchild);
    expect(queueManager.discoveryQueue.length).toBe(4);

    // Analyze graph topology using edge classification
    graphAnalyzer.addTreeEdge(root.url, child1.url, child1.title);
    graphAnalyzer.addTreeEdge(root.url, child2.url, child2.title);
    graphAnalyzer.addTreeEdge(child1.url, grandchild.url, grandchild.title);

    expect(graphAnalyzer.treeEdges.length).toBe(3);
  });

  test('detects back edges (cycles)', () => {
    // Create circular reference: A -> B -> A
    const pageA = createMockPageContext({
      pageId: 'page-a',
      url: 'https://notion.so/PageA-29' + 'e'.repeat(30),
      depth: 0,
      parentId: null
    });

    const pageB = createMockPageContext({
      pageId: 'page-b',
      url: 'https://notion.so/PageB-29' + 'f'.repeat(30),
      depth: 1,
      parentId: 'page-a'
    });

    // Set up parent-child relationship for GraphAnalyzer to detect ancestry
    pageB.parentContext = pageA;

    // First edge: A -> B (tree edge)
    graphAnalyzer.addTreeEdge(pageA.url, pageB.url, 'Page B');

    // Second edge: B -> A (back edge, creates cycle)
    const edgeType = graphAnalyzer.classifyEdge(pageB, pageA, pageA.url, 'Back to Page A');

    expect(edgeType).toBe('back');
    expect(graphAnalyzer.backEdges.length).toBe(1);
    expect(graphAnalyzer.backEdges[0].type).toBe('to-ancestor');
  });

  test('resolves duplicate page discoveries', () => {
    // Simulate scenario: Same child URL discovered from two different parents
    const duplicates = createDuplicateContexts();

    // Resolve conflicts
    const result = ConflictResolver.resolve(duplicates);

    // Verify deduplication
    expect(result.stats.uniquePages).toBe(1);
    expect(result.stats.duplicates).toBe(1);
    expect(result.canonicalContexts.length).toBe(1);

    // Link rewrite map should have entries for both IDs pointing to same canonical path
    expect(result.linkRewriteMap.size).toBeGreaterThanOrEqual(1);
    const rewritePaths = new Set(result.linkRewriteMap.values());
    expect(rewritePaths.size).toBe(1); // All map to same canonical path
  });

  test('maintains queue ordering for breadth-first traversal', () => {
    // Add pages at different depths with valid Notion URLs
    const depth0 = createMockPageContext({ 
      pageId: 'd0', 
      url: 'https://notion.so/D0-29' + 'a'.repeat(30),
      depth: 0 
    });
    const depth1a = createMockPageContext({ 
      pageId: 'd1a', 
      url: 'https://notion.so/D1A-29' + 'b'.repeat(30),
      depth: 1, 
      parentId: 'd0' 
    });
    const depth1b = createMockPageContext({ 
      pageId: 'd1b', 
      url: 'https://notion.so/D1B-29' + 'c'.repeat(30),
      depth: 1, 
      parentId: 'd0' 
    });
    const depth2 = createMockPageContext({ 
      pageId: 'd2', 
      url: 'https://notion.so/D2-29' + 'd'.repeat(30),
      depth: 2, 
      parentId: 'd1a' 
    });

    // Add in discovery order
    queueManager.enqueueDiscovery(depth0, true);
    queueManager.enqueueDiscovery(depth1a);
    queueManager.enqueueDiscovery(depth1b);
    queueManager.enqueueDiscovery(depth2);

    // Verify all enqueued
    expect(queueManager.discoveryQueue.length).toBe(4);

    // Dequeue should follow breadth-first ordering
    const task1 = queueManager.nextDiscovery();
    expect(task1.pageContext.id).toBe('29' + 'a'.repeat(30));

    const task2 = queueManager.nextDiscovery();
    expect(task2.pageContext.id).toBe('29' + 'b'.repeat(30));

    const task3 = queueManager.nextDiscovery();
    expect(task3.pageContext.id).toBe('29' + 'c'.repeat(30));

    const task4 = queueManager.nextDiscovery();
    expect(task4.pageContext.id).toBe('29' + 'd'.repeat(30));
  });

  test('tracks visited URLs to prevent redundant discovery', () => {
    const page = createMockPageContext({
      url: 'https://notion.so/Test-29' + 'a'.repeat(30),
      pageId: 'test-page'
    });

    // First enqueue should succeed
    const enqueued1 = queueManager.enqueueDiscovery(page, false);
    expect(enqueued1).toBe(true);

    // Duplicate enqueue should fail (already visited)
    const enqueued2 = queueManager.enqueueDiscovery(page, false);
    expect(enqueued2).toBe(false);

    // Queue should only have one item
    expect(queueManager.discoveryQueue.length).toBe(1);
  });
});
