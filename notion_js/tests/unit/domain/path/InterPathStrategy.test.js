/**
 * @file InterPathStrategy.test.js
 * @description Unit tests for InterPathStrategy - cross-page path resolution
 * 
 * Tests simulate real Notion page hierarchies like the JBC090 Language & AI course site.
 * Verifies correct relative path computation across IPC serialization boundary.
 */

const InterPathStrategy = require('../../../../src/domain/path/InterPathStrategy');
const PageContext = require('../../../../src/domain/PageContext');

describe('InterPathStrategy', () => {
  let strategy;

  beforeEach(() => {
    strategy = new InterPathStrategy();
  });

  describe('supports()', () => {
    test('returns true for different internal pages', () => {
      const pageA = createContext('https://notion.so/a123', 'Page A', 1);
      const pageB = createContext('https://notion.so/b456', 'Page B', 1);
      
      expect(strategy.supports(pageA, pageB)).toBe(true);
    });

    test('returns false for same page', () => {
      const page = createContext('https://notion.so/a123', 'Page A', 1);
      
      expect(strategy.supports(page, page)).toBe(false);
    });

    test('returns false for null contexts', () => {
      const page = createContext('https://notion.so/a123', 'Page A', 1);
      
      expect(strategy.supports(null, page)).toBe(false);
      expect(strategy.supports(page, null)).toBe(false);
      expect(strategy.supports(null, null)).toBe(false);
    });
  });

  describe('resolve() - Sibling Navigation', () => {
    /**
     * Simulates JBC090 site structure:
     * Root: JBC090 Language & AI
     * ├── Lab_Session_1
     * ├── Lab_Session_2
     * ├── Material
     * └── Syllabus
     */
    test('navigates correctly between sibling pages at depth 1', () => {
      // Build hierarchy with parent references
      const root = createContext('https://notion.so/root123', 'JBC090_Language_AI', 0);
      const labSession1 = createContext('https://notion.so/lab1', 'Lab_Session_1', 1, root);
      const labSession2 = createContext('https://notion.so/lab2', 'Lab_Session_2', 1, root);
      const material = createContext('https://notion.so/mat1', 'Material', 1, root);

      // Lab_Session_1 → Lab_Session_2
      const path1 = strategy.resolve(labSession1, labSession2, {});
      expect(path1).toBe('../Lab_Session_2/index.html');

      // Lab_Session_1 → Material
      const path2 = strategy.resolve(labSession1, material, {});
      expect(path2).toBe('../Material/index.html');
    });

    test('navigates from child to root (parent) page', () => {
      const root = createContext('https://notion.so/root123', 'JBC090_Language_AI', 0);
      const labSession1 = createContext('https://notion.so/lab1', 'Lab_Session_1', 1, root);

      // Lab_Session_1 → Root
      const path = strategy.resolve(labSession1, root, {});
      expect(path).toBe('../index.html');
    });

    test('navigates from root to child page', () => {
      const root = createContext('https://notion.so/root123', 'JBC090_Language_AI', 0);
      const labSession1 = createContext('https://notion.so/lab1', 'Lab_Session_1', 1, root);

      // Root → Lab_Session_1
      const path = strategy.resolve(root, labSession1, {});
      expect(path).toBe('Lab_Session_1/index.html');
    });
  });

  describe('resolve() - Deep Hierarchy', () => {
    /**
     * Simulates deeper hierarchy:
     * Root
     * └── Section
     *     └── Subsection
     *         └── DeepPage
     */
    test('navigates correctly in 3-level hierarchy', () => {
      const root = createContext('https://notion.so/root', 'Root', 0);
      const section = createContext('https://notion.so/sec', 'Section', 1, root);
      const subsection = createContext('https://notion.so/sub', 'Subsection', 2, section);
      const deepPage = createContext('https://notion.so/deep', 'DeepPage', 3, subsection);

      // DeepPage → Root
      const path1 = strategy.resolve(deepPage, root, {});
      expect(path1).toBe('../../../index.html');

      // DeepPage → Section
      const path2 = strategy.resolve(deepPage, section, {});
      expect(path2).toBe('../../index.html');

      // Root → DeepPage
      const path3 = strategy.resolve(root, deepPage, {});
      expect(path3).toBe('Section/Subsection/DeepPage/index.html');
    });

    test('navigates between cousins in different branches', () => {
      const root = createContext('https://notion.so/root', 'Root', 0);
      const branchA = createContext('https://notion.so/a', 'BranchA', 1, root);
      const branchB = createContext('https://notion.so/b', 'BranchB', 1, root);
      const leafA = createContext('https://notion.so/la', 'LeafA', 2, branchA);
      const leafB = createContext('https://notion.so/lb', 'LeafB', 2, branchB);

      // LeafA → LeafB (cousins)
      const path = strategy.resolve(leafA, leafB, {});
      expect(path).toBe('../../BranchB/LeafB/index.html');
    });
  });

  describe('resolve() - IPC Serialization Survival', () => {
    /**
     * CRITICAL: Tests that path resolution works AFTER IPC serialization.
     * This is the core fix for the inter-page path resolution bug.
     */
    test('resolves correctly after JSON serialization (IPC simulation)', () => {
      // Build hierarchy with parent references
      const root = createContext('https://notion.so/root123', 'JBC090_Language_AI', 0);
      const labSession1 = createContext('https://notion.so/lab1', 'Lab_Session_1', 1, root);
      const material = createContext('https://notion.so/mat1', 'Material', 1, root);

      // Simulate IPC: serialize to JSON
      const labSession1Json = labSession1.toJSON();
      const materialJson = material.toJSON();
      const rootJson = root.toJSON();

      // Simulate IPC: deserialize (without parent context reference)
      const labSession1Deserialized = PageContext.fromJSON(labSession1Json);
      const materialDeserialized = PageContext.fromJSON(materialJson);

      // CRITICAL: parentContext is now null after deserialization
      expect(labSession1Deserialized.parentContext).toBeNull();

      // BUT pathSegments should survive serialization
      expect(labSession1Deserialized.pathSegments).toEqual(['Lab_Session_1']);
      expect(materialDeserialized.pathSegments).toEqual(['Material']);

      // Path resolution should still work using pathSegments
      const path = strategy.resolve(labSession1Deserialized, materialDeserialized, {});
      expect(path).toBe('../Material/index.html');
    });

    test('preserves pathSegments through toJSON/fromJSON cycle', () => {
      const root = createContext('https://notion.so/root', 'Root', 0);
      const section = createContext('https://notion.so/sec', 'Section', 1, root);
      const page = createContext('https://notion.so/page', 'Page', 2, section);

      // Original pathSegments
      expect(page.pathSegments).toEqual(['Section', 'Page']);

      // Serialize and deserialize
      const json = page.toJSON();
      expect(json.pathSegments).toEqual(['Section', 'Page']);

      const restored = PageContext.fromJSON(json);
      expect(restored.pathSegments).toEqual(['Section', 'Page']);
    });

    test('handles deep hierarchy after serialization', () => {
      const root = createContext('https://notion.so/root', 'Root', 0);
      const l1 = createContext('https://notion.so/l1', 'Level1', 1, root);
      const l2 = createContext('https://notion.so/l2', 'Level2', 2, l1);
      const l3 = createContext('https://notion.so/l3', 'Level3', 3, l2);

      // Serialize and deserialize
      const l3Json = l3.toJSON();
      const rootJson = root.toJSON();
      const l3Restored = PageContext.fromJSON(l3Json);
      const rootRestored = PageContext.fromJSON(rootJson);

      // Path from deep page to root should still work
      const path = strategy.resolve(l3Restored, rootRestored, {});
      expect(path).toBe('../../../index.html');
    });
  });

  describe('resolve() - Block Anchors', () => {
    test('appends anchor hash when blockId provided', () => {
      const root = createContext('https://notion.so/root', 'Root', 0);
      const page = createContext('https://notion.so/page', 'Page', 1, root);

      const path = strategy.resolve(page, root, { blockId: '29d979eeca9f4abc' });
      expect(path).toMatch(/^\.\.\/index\.html#/);
    });
  });

  describe('getType()', () => {
    test('returns inter type', () => {
      expect(strategy.getType()).toBe('inter');
    });
  });
});

/**
 * Helper to create PageContext with proper hierarchy.
 * Automatically sets parentId and computes pathSegments.
 */
function createContext(url, title, depth, parentContext = null) {
  const context = new PageContext(
    url,
    title,
    depth,
    parentContext,
    parentContext?.id || null
  );
  
  if (parentContext) {
    parentContext.addChild(context);
  }
  
  return context;
}
