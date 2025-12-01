/**
 * @file path-resolution-integration.test.js
 * @description Integration tests for path resolution across IPC boundary
 * 
 * These tests simulate the complete Master → IPC → Worker flow for path resolution,
 * verifying that path resolution correctly handles:
 * 1. Inter-page path resolution with pathSegments serialization
 * 2. Intra-page anchor resolution with anchor-only detection
 * 
 * Uses real-world examples from the JBC090 Language & AI Notion site.
 */

const PageContext = require('../../../src/domain/PageContext');
const { 
  PathResolverFactory,
  InterPageResolver,
  IntraPageResolver
} = require('../../../src/domain/path');

describe('Path Resolution Integration - JBC090 Site Simulation', () => {
  // Shared test data: simulated JBC090 site hierarchy
  let siteHierarchy;
  
  beforeEach(() => {
    siteHierarchy = createJBC090Hierarchy();
  });

  describe('Inter-Page Path Resolution after IPC', () => {
    test('Lab_Session_1 → Root breadcrumb link resolves correctly', () => {
      const { root, labSession1 } = siteHierarchy;
      
      // Simulate IPC serialization
      const labSession1Wire = JSON.stringify(labSession1.toJSON());
      const rootWire = JSON.stringify(root.toJSON());
      
      // Simulate Worker receiving contexts
      const labSession1Worker = PageContext.fromJSON(JSON.parse(labSession1Wire));
      const rootWorker = PageContext.fromJSON(JSON.parse(rootWire));
      
      // Verify parentContext is null (expected after IPC)
      expect(labSession1Worker.parentContext).toBeNull();
      
      // But pathSegments should be intact
      expect(labSession1Worker.pathSegments).toEqual(['Lab_Session_1']);
      expect(rootWorker.pathSegments).toEqual([]);
      
      // Path resolution should work
      const resolver = new InterPageResolver();
      const path = resolver.resolve({ source: labSession1Worker, target: rootWorker });
      
      // CRITICAL: Should be '../index.html', not './index.html'
      expect(path).toBe('../index.html');
    });

    test('Lab_Session_1 → Material sibling link resolves correctly', () => {
      const { labSession1, material } = siteHierarchy;
      
      // Simulate IPC
      const labSession1Worker = simulateIPC(labSession1);
      const materialWorker = simulateIPC(material);
      
      const resolver = new InterPageResolver();
      const path = resolver.resolve({ source: labSession1Worker, target: materialWorker });
      
      // CRITICAL: Should be '../Material/index.html', not 'Material/index.html'
      expect(path).toBe('../Material/index.html');
    });

    test('Root → Lab_Session_1 child link resolves correctly', () => {
      const { root, labSession1 } = siteHierarchy;
      
      const rootWorker = simulateIPC(root);
      const labSession1Worker = simulateIPC(labSession1);
      
      const resolver = new InterPageResolver();
      const path = resolver.resolve({ source: rootWorker, target: labSession1Worker });
      
      expect(path).toBe('Lab_Session_1/index.html');
    });

    test('Deep page → Root resolves with correct ../ count', () => {
      // Create a deeper hierarchy
      const root = createContext('https://notion.so/root', 'JBC090_Language_AI', 0);
      const section = createContext('https://notion.so/sec', 'Course_Material', 1, root);
      const topic = createContext('https://notion.so/topic', 'Week_1', 2, section);
      const page = createContext('https://notion.so/page', 'Lecture_Notes', 3, topic);
      
      // Simulate IPC
      const pageWorker = simulateIPC(page);
      const rootWorker = simulateIPC(root);
      
      const resolver = new InterPageResolver();
      const path = resolver.resolve({ source: pageWorker, target: rootWorker });
      
      expect(path).toBe('../../../index.html');
    });
  });

  describe('Intra-Page Anchor Resolution', () => {
    test('ToC entry with #block-id resolves to anchor', () => {
      const { labSession1 } = siteHierarchy;
      
      const labSession1Worker = simulateIPC(labSession1);
      
      const resolver = new IntraPageResolver();
      
      // ToC link format: #29d979ee-ca9f-81cf-b69d-f8e08f3ff10d
      const tocHref = '#29d979ee-ca9f-81cf-b69d-f8e08f3ff10d';
      
      // Should be recognized as intra-page
      expect(resolver.supports({ source: labSession1Worker, href: tocHref })).toBe(true);
      
      // Should resolve to anchor (preserves existing anchor format)
      const path = resolver.resolve({ 
        source: labSession1Worker, 
        target: labSession1Worker,
        href: tocHref
      });
      
      expect(path).toMatch(/^#/);
      expect(path).not.toContain('Lab_Session_1');
      expect(path).not.toContain('index.html');
    });

    test('Multiple ToC entries on same page all resolve to anchors', () => {
      const { syllabus } = siteHierarchy;
      
      const syllabusWorker = simulateIPC(syllabus);
      const resolver = new IntraPageResolver();
      
      const tocEntries = [
        '#course-overview',
        '#29d979ee-ca9f-8126-a7fd-c2c749c412ea',
        '#learning-objectives',
        '#assessment-information',
        '#weekly-schedule'
      ];
      
      for (const href of tocEntries) {
        expect(resolver.supports({ source: syllabusWorker, href })).toBe(true);
        
        const path = resolver.resolve({ 
          source: syllabusWorker, 
          target: syllabusWorker, 
          href 
        });
        expect(path).toMatch(/^#/);
      }
    });

    test('Same-page link by ID match still works', () => {
      const { labSession1 } = siteHierarchy;
      
      const labSession1Worker = simulateIPC(labSession1);
      
      const resolver = new IntraPageResolver();
      
      // When sourceContext.id === targetContext.id
      expect(resolver.supports({ 
        source: labSession1Worker, 
        target: labSession1Worker 
      })).toBe(true);
      
      // Resolve with blockId option
      const path = resolver.resolve({ 
        source: labSession1Worker, 
        target: labSession1Worker,
        blockId: '29d979eeca9f81cfb69df8e08f3ff10d'
      });
      
      expect(path).toMatch(/^#29d979ee/);
    });
  });

  describe('PathResolverFactory Integration', () => {
    let factory;
    
    beforeEach(() => {
      factory = new PathResolverFactory({}, mockLogger());
    });

    test('IntraPageResolver handles anchor-only href directly', () => {
      const { labSession1 } = siteHierarchy;
      const labSession1Worker = simulateIPC(labSession1);
      
      // IntraPageResolver.supports() accepts anchor-only hrefs
      const resolver = new IntraPageResolver();
      expect(resolver.supports({ source: labSession1Worker, href: '#block-id' })).toBe(true);
      
      // Resolver returns the anchor as-is for non-UUID anchors
      const path = resolver.resolve({ 
        source: labSession1Worker, 
        target: labSession1Worker, 
        href: '#block-id' 
      });
      expect(path).toBe('#block-id');
    });

    test('correctly selects InterPageResolver for cross-page link', () => {
      const { labSession1, material } = siteHierarchy;
      
      const labSession1Worker = simulateIPC(labSession1);
      const materialWorker = simulateIPC(material);
      
      const pathType = factory.getPathType({ 
        source: labSession1Worker, 
        target: materialWorker 
      });
      
      // Path type uses lowercase enum values
      expect(pathType).toBe('inter');
    });

    test('full resolution flow produces correct paths', () => {
      const { root, labSession1, material, syllabus } = siteHierarchy;
      
      // Simulate batch IPC (all contexts sent to worker)
      const contexts = {
        root: simulateIPC(root),
        labSession1: simulateIPC(labSession1),
        material: simulateIPC(material),
        syllabus: simulateIPC(syllabus)
      };
      
      // Test various navigation scenarios
      const testCases = [
        // [source, target, expectedPath]
        [contexts.labSession1, contexts.root, '../index.html'],
        [contexts.labSession1, contexts.material, '../Material/index.html'],
        [contexts.labSession1, contexts.syllabus, '../Syllabus/index.html'],
        [contexts.root, contexts.labSession1, 'Lab_Session_1/index.html'],
        [contexts.material, contexts.syllabus, '../Syllabus/index.html'],
      ];
      
      for (const [source, target, expected] of testCases) {
        const path = factory.resolve({ source, target });
        expect(path).toBe(expected);
      }
    });
  });

  describe('Edge Cases', () => {
    test('handles context with special characters in title', () => {
      const root = createContext('https://notion.so/root', 'Root', 0);
      const page = createContext(
        'https://notion.so/page', 
        'Week 1: Intro & Overview (Part 1)',
        1,
        root
      );
      
      const pageWorker = simulateIPC(page);
      const rootWorker = simulateIPC(root);
      
      const resolver = new InterPageResolver();
      const path = resolver.resolve({ source: pageWorker, target: rootWorker });
      
      expect(path).toBe('../index.html');
      expect(pageWorker.pathSegments[0]).not.toContain(':');
      expect(pageWorker.pathSegments[0]).not.toContain('&');
    });

    test('handles deeply nested pages (5+ levels)', () => {
      const root = createContext('https://notion.so/root', 'Root', 0);
      const l1 = createContext('https://notion.so/l1', 'L1', 1, root);
      const l2 = createContext('https://notion.so/l2', 'L2', 2, l1);
      const l3 = createContext('https://notion.so/l3', 'L3', 3, l2);
      const l4 = createContext('https://notion.so/l4', 'L4', 4, l3);
      const l5 = createContext('https://notion.so/l5', 'L5', 5, l4);
      
      const l5Worker = simulateIPC(l5);
      const rootWorker = simulateIPC(root);
      
      expect(l5Worker.pathSegments).toEqual(['L1', 'L2', 'L3', 'L4', 'L5']);
      
      const resolver = new InterPageResolver();
      const path = resolver.resolve({ source: l5Worker, target: rootWorker });
      
      expect(path).toBe('../../../../../index.html');
    });

    test('handles cousin navigation (different branches)', () => {
      const root = createContext('https://notion.so/root', 'Root', 0);
      const branchA = createContext('https://notion.so/a', 'BranchA', 1, root);
      const branchB = createContext('https://notion.so/b', 'BranchB', 1, root);
      const leafA = createContext('https://notion.so/la', 'LeafA', 2, branchA);
      const leafB = createContext('https://notion.so/lb', 'LeafB', 2, branchB);
      
      const leafAWorker = simulateIPC(leafA);
      const leafBWorker = simulateIPC(leafB);
      
      const resolver = new InterPageResolver();
      const path = resolver.resolve({ source: leafAWorker, target: leafBWorker });
      
      expect(path).toBe('../../BranchB/LeafB/index.html');
    });
  });
});

/**
 * Creates the JBC090 Language & AI site hierarchy for testing.
 */
function createJBC090Hierarchy() {
  const root = createContext(
    'https://notion.so/JBC090-Language-AI-29d979eeca9f8102a85be4dd9007f020',
    'JBC090_Language_AI',
    0
  );
  
  const labSession1 = createContext(
    'https://notion.so/Lab-Session-1-abc123',
    'Lab_Session_1',
    1,
    root
  );
  
  const labSession2 = createContext(
    'https://notion.so/Lab-Session-2-def456',
    'Lab_Session_2',
    1,
    root
  );
  
  const material = createContext(
    'https://notion.so/Material-ghi789',
    'Material',
    1,
    root
  );
  
  const syllabus = createContext(
    'https://notion.so/Syllabus-jkl012',
    'Syllabus',
    1,
    root
  );
  
  return { root, labSession1, labSession2, material, syllabus };
}

/**
 * Helper to create PageContext with proper hierarchy.
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

/**
 * Simulates IPC serialization/deserialization.
 */
function simulateIPC(context) {
  const wireFormat = JSON.stringify(context.toJSON());
  return PageContext.fromJSON(JSON.parse(wireFormat));
}

/**
 * Creates mock logger.
 */
function mockLogger() {
  return {
    info: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn()
  };
}
