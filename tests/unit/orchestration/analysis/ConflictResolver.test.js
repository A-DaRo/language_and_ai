const ConflictResolver = require('../../../../src/orchestration/analysis/ConflictResolver');
const { createDuplicateContexts } = require('../../../helpers/factories');
const PageContext = require('../../../../src/domain/PageContext');

describe('ConflictResolver', () => {
  test('resolves duplicates and is idempotent', () => {
    const duplicates = createDuplicateContexts();
    const titleRegistry = {};

    const result1 = ConflictResolver.resolve(duplicates, titleRegistry);
    const result2 = ConflictResolver.resolve(result1.canonicalContexts, titleRegistry);

    expect(result1.canonicalContexts).toEqual(result2.canonicalContexts);
    // linkRewriteMap is a Map; serialize to object for equality check
    const obj1 = Object.fromEntries(result1.linkRewriteMap);
    const obj2 = Object.fromEntries(result2.linkRewriteMap);
    expect(obj1).toEqual(obj2);
  });

  test('maps duplicate ids to same canonical path', () => {
    const duplicates = createDuplicateContexts();
    const result = ConflictResolver.resolve(duplicates);
    const map = result.linkRewriteMap;
    // Expect resolution statistics to reflect one unique page and one duplicate
    expect(result.stats.uniquePages).toBe(1);
    expect(result.stats.duplicates).toBe(1);

    // All mapped values should equal the canonical context targetFilePath
    const values = Array.from(map.values());
    expect(values.length).toBeGreaterThanOrEqual(1);
    const uniquePaths = new Set(values);
    expect(uniquePaths.size).toBe(1);
    expect(uniquePaths.has(result.canonicalContexts[0].targetFilePath)).toBe(true);
  });

  test('updates context titles from title registry', () => {
    // Create a context with a raw ID as title
    const rawIdUrl = 'https://notion.so/29d979eeca9f8102a85be4dd9007f020';
    const context = new PageContext(rawIdUrl, '29d979eeca9f8102a85be4dd9007f020', 0);
    
    // Title registry with human-readable title
    const titleRegistry = {
      '29d979eeca9f8102a85be4dd9007f020': 'Introduction to AI'
    };
    
    // Before resolution, title should be the raw ID (sanitized)
    expect(context.title).toContain('29d979eeca9f8102a85be4dd9007f020');
    
    // Resolve with title registry
    const result = ConflictResolver.resolve([context], titleRegistry);
    
    // After resolution, title should be updated to sanitized human-readable version
    const resolvedContext = result.canonicalContexts[0];
    expect(resolvedContext.title).toBe('Introduction_to_AI');
    expect(resolvedContext.title).not.toContain('29d979eeca9f8102a85be4dd9007f020');
  });

  test('file paths use human-readable names instead of raw IDs', () => {
    // Create parent and child contexts with raw IDs
    const parentUrl = 'https://notion.so/29d979eeca9f8102a85be4dd9007f020';
    const childUrl = 'https://notion.so/29d979eeca9f8126a7fdc2c749c412ea';
    
    const parentContext = new PageContext(parentUrl, '29d979eeca9f8102a85be4dd9007f020', 0);
    const childContext = new PageContext(childUrl, '29d979eeca9f8126a7fdc2c749c412ea', 1, parentContext, parentContext.id);
    parentContext.addChild(childContext);
    
    // Title registry with human-readable titles
    const titleRegistry = {
      '29d979eeca9f8102a85be4dd9007f020': 'Language and AI Course',
      '29d979eeca9f8126a7fdc2c749c412ea': 'Week 1 Introduction'
    };
    
    // Resolve with title registry
    const result = ConflictResolver.resolve([parentContext, childContext], titleRegistry);
    
    // Root page gets index.html path (prefix removed after resolver consolidation)
    const parentPath = result.linkRewriteMap.get(parentContext.id);
    expect(parentPath).toBe('index.html');
    
    // Child page gets path based on its relative path
    const childPath = result.linkRewriteMap.get(childContext.id);
    // Child contexts have their title as path segment
    expect(childPath).toContain('index.html');
    // After title update, the path should use the sanitized human-readable title
    expect(result.canonicalContexts[1].title).toBe('Week_1_Introduction');
  });
});
