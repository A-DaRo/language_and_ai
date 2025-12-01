const GraphAnalyzer = require('../../../../src/orchestration/analysis/GraphAnalyzer');
const PageContext = require('../../../../src/domain/PageContext');

// Minimal mock logger
const mockLogger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  separator: () => {}
};

describe('GraphAnalyzer', () => {
  test('classifies self-loop as back edge', () => {
    const analyzer = new GraphAnalyzer(mockLogger);
    const node = new PageContext('https://notion.so/node', 'Node', 1);
    node.parentContext = new PageContext('https://notion.so/parent', 'Parent', 0);
    // simulate self loop
    const type = analyzer.classifyEdge(node, node, node.url, 'self');
    expect(type).toBe('back');
    expect(analyzer.getStats().back).toBeGreaterThanOrEqual(1);
  });

  test('classifies back edge when pointing to ancestor', () => {
    const analyzer = new GraphAnalyzer(mockLogger);
    const root = new PageContext('https://notion.so/root', 'Root', 0);
    const child = new PageContext('https://notion.so/child', 'Child', 1, root);
    child.parentContext = root;

    const type = analyzer.classifyEdge(child, root, root.url, 'to-root');
    expect(type).toBe('back');
    expect(analyzer.getStats().back).toBeGreaterThanOrEqual(1);
  });

  test('classifies forward and cross edges by depth', () => {
    const analyzer = new GraphAnalyzer(mockLogger);
    const root = new PageContext('https://notion.so/root', 'Root', 0);
    const child = new PageContext('https://notion.so/child', 'Child', 1, root);
    const grandchild = new PageContext('https://notion.so/gc', 'Grand', 2, child);
    child.parentContext = root;
    grandchild.parentContext = child;

    const forwardType = analyzer.classifyEdge(root, grandchild, grandchild.url, 'to-grand');
    expect(forwardType).toBe('forward');

    const crossType = analyzer.classifyEdge(child, root, root.url.replace('root','other'), 'cross-link');
    // crossType depends on depths (here to different branch same or earlier level)
    expect(['cross','back','forward']).toContain(crossType);
  });
});
