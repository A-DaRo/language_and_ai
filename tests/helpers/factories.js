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

/**
 * Creates a mock IPC download result
 */
function createMockDownloadResult(overrides = {}) {
  return {
    success: true,
    pageId: overrides.pageId || 'page-id-123',
    html: overrides.html || '<html><body>Test Content</body></html>',
    assets: overrides.assets || []
  };
}

/**
 * Creates a mock title registry for testing
 */
function createMockTitleRegistry(entries = {}) {
  const registry = new Map();
  for (const [url, title] of Object.entries(entries)) {
    registry.set(url, title);
  }
  return registry;
}

/**
 * Creates a diamond dependency graph for testing download ordering
 * Structure: Root -> [A, B] -> C (where C has both A and B as parents)
 */
function createDiamondGraph() {
  const root = createMockPageContext({ 
    url: 'https://notion.so/root',
    title: 'Root', 
    depth: 0 
  });
  
  const childA = createMockPageContext({ 
    url: 'https://notion.so/child-a',
    title: 'Child A', 
    depth: 1,
    parentId: root.id
  });
  
  const childB = createMockPageContext({ 
    url: 'https://notion.so/child-b',
    title: 'Child B', 
    depth: 1,
    parentId: root.id
  });
  
  const childC = createMockPageContext({ 
    url: 'https://notion.so/child-c',
    title: 'Child C', 
    depth: 2,
    parentId: childA.id
  });
  
  root.addChild(childA);
  root.addChild(childB);
  childA.addChild(childC);
  
  return { root, childA, childB, childC, contexts: [root, childA, childB, childC] };
}

/**
 * Creates duplicate contexts for conflict resolution testing
 */
function createDuplicateContexts() {
  const url = 'https://notion.so/duplicate-page';
  
  const shallow = createMockPageContext({
    url,
    title: 'Duplicate Page',
    depth: 1,
    parentId: 'parent-1'
  });
  
  const deep = createMockPageContext({
    url,
    title: 'Duplicate Page',
    depth: 3,
    parentId: 'parent-2'
  });
  
  return [shallow, deep];
}

module.exports = {
  createMockPageContext,
  createMockGraph,
  createMockDiscoveryResult,
  createMockDownloadResult,
  createMockTitleRegistry,
  createDiamondGraph,
  createDuplicateContexts
};
