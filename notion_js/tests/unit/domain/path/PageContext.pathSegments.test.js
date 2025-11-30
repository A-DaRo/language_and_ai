/**
 * @file PageContext.pathSegments.test.js
 * @description Unit tests for PageContext pathSegments - IPC serialization survival
 * 
 * These tests verify that path segments are correctly computed at construction time
 * and survive JSON serialization for cross-process communication.
 */

const PageContext = require('../../../../src/domain/PageContext');

describe('PageContext - pathSegments', () => {
  describe('_computePathSegments()', () => {
    test('root page has empty pathSegments', () => {
      const root = new PageContext('https://notion.so/root123', 'Root_Page', 0);
      
      expect(root.pathSegments).toEqual([]);
      expect(root.getPathSegments()).toEqual([]);
    });

    test('depth-1 page has single segment', () => {
      const root = new PageContext('https://notion.so/root123', 'Root_Page', 0);
      const child = new PageContext('https://notion.so/child456', 'Lab_Session_1', 1, root);
      
      expect(child.pathSegments).toEqual(['Lab_Session_1']);
    });

    test('depth-2 page has two segments', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      const section = new PageContext('https://notion.so/sec', 'Section', 1, root);
      const page = new PageContext('https://notion.so/page', 'Page', 2, section);
      
      expect(page.pathSegments).toEqual(['Section', 'Page']);
    });

    test('depth-3 page has three segments', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      const l1 = new PageContext('https://notion.so/l1', 'Level_1', 1, root);
      const l2 = new PageContext('https://notion.so/l2', 'Level_2', 2, l1);
      const l3 = new PageContext('https://notion.so/l3', 'Level_3', 3, l2);
      
      expect(l3.pathSegments).toEqual(['Level_1', 'Level_2', 'Level_3']);
    });

    test('sanitizes special characters in path segments', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      const child = new PageContext(
        'https://notion.so/child', 
        'Week 1: Intro & Overview (Part 1)', 
        1, 
        root
      );
      
      // FileSystemUtils.sanitizeFilename should handle special chars
      expect(child.pathSegments[0]).not.toContain(':');
      expect(child.pathSegments[0]).not.toContain('&');
      expect(child.pathSegments[0]).not.toContain('(');
    });
  });

  describe('toJSON() - pathSegments serialization', () => {
    test('includes pathSegments in JSON output', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      const child = new PageContext('https://notion.so/child', 'Lab_Session_1', 1, root);
      
      const json = child.toJSON();
      
      expect(json).toHaveProperty('pathSegments');
      expect(json.pathSegments).toEqual(['Lab_Session_1']);
    });

    test('serializes empty pathSegments for root', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      
      const json = root.toJSON();
      
      expect(json.pathSegments).toEqual([]);
    });

    test('serializes multi-level pathSegments', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      const section = new PageContext('https://notion.so/sec', 'Section', 1, root);
      const page = new PageContext('https://notion.so/page', 'Deep_Page', 2, section);
      
      const json = page.toJSON();
      
      expect(json.pathSegments).toEqual(['Section', 'Deep_Page']);
    });
  });

  describe('fromJSON() - pathSegments restoration', () => {
    test('restores pathSegments from JSON', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      const child = new PageContext('https://notion.so/child', 'Lab_Session_1', 1, root);
      
      // Serialize
      const json = child.toJSON();
      
      // Deserialize (simulating IPC receive)
      const restored = PageContext.fromJSON(json);
      
      expect(restored.pathSegments).toEqual(['Lab_Session_1']);
      expect(restored.getPathSegments()).toEqual(['Lab_Session_1']);
    });

    test('handles missing pathSegments in legacy JSON', () => {
      const legacyJson = {
        id: 'abc123',
        url: 'https://notion.so/page',
        title: 'Page',
        depth: 1,
        parentId: 'root123'
        // No pathSegments field (legacy data)
      };
      
      const restored = PageContext.fromJSON(legacyJson);
      
      // Should not crash, returns computed (possibly empty) segments
      expect(restored.getPathSegments()).toBeDefined();
    });

    test('restores empty pathSegments correctly', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      
      const json = root.toJSON();
      const restored = PageContext.fromJSON(json);
      
      expect(restored.pathSegments).toEqual([]);
    });
  });

  describe('IPC Round-Trip Simulation', () => {
    /**
     * These tests simulate the full Master → Worker IPC cycle.
     */
    test('pathSegments survive JSON.stringify/parse cycle', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      const child = new PageContext('https://notion.so/child', 'Lab_Session_1', 1, root);
      
      // Simulate IPC: Master serializes
      const wireFormat = JSON.stringify(child.toJSON());
      
      // Simulate IPC: Worker receives and parses
      const parsed = JSON.parse(wireFormat);
      const restored = PageContext.fromJSON(parsed);
      
      expect(restored.pathSegments).toEqual(['Lab_Session_1']);
    });

    test('deep hierarchy survives IPC', () => {
      const root = new PageContext('https://notion.so/root', 'JBC090_Language_AI', 0);
      const section = new PageContext('https://notion.so/sec', 'Course_Material', 1, root);
      const subsection = new PageContext('https://notion.so/sub', 'Week_1', 2, section);
      const page = new PageContext('https://notion.so/page', 'Lecture_Notes', 3, subsection);
      
      // Full IPC cycle
      const wireFormat = JSON.stringify(page.toJSON());
      const parsed = JSON.parse(wireFormat);
      const restored = PageContext.fromJSON(parsed);
      
      expect(restored.pathSegments).toEqual([
        'Course_Material',
        'Week_1',
        'Lecture_Notes'
      ]);
      
      // parentContext should be null after deserialization
      expect(restored.parentContext).toBeNull();
      
      // But pathSegments should have all the hierarchy info
      expect(restored.pathSegments.length).toBe(3);
    });

    test('multiple pages from same hierarchy maintain correct segments', () => {
      const root = new PageContext('https://notion.so/root', 'Root', 0);
      const sectionA = new PageContext('https://notion.so/a', 'Section_A', 1, root);
      const sectionB = new PageContext('https://notion.so/b', 'Section_B', 1, root);
      const pageA = new PageContext('https://notion.so/pa', 'Page_A', 2, sectionA);
      const pageB = new PageContext('https://notion.so/pb', 'Page_B', 2, sectionB);
      
      // Serialize all
      const pages = [root, sectionA, sectionB, pageA, pageB];
      const serialized = pages.map(p => JSON.stringify(p.toJSON()));
      
      // Deserialize all (as separate operations, like IPC)
      const restored = serialized.map(s => PageContext.fromJSON(JSON.parse(s)));
      
      expect(restored[0].pathSegments).toEqual([]); // root
      expect(restored[1].pathSegments).toEqual(['Section_A']);
      expect(restored[2].pathSegments).toEqual(['Section_B']);
      expect(restored[3].pathSegments).toEqual(['Section_A', 'Page_A']);
      expect(restored[4].pathSegments).toEqual(['Section_B', 'Page_B']);
    });
  });

  describe('Real Notion Site Examples', () => {
    /**
     * These tests use real page names from the JBC090 Language & AI site.
     */
    test('JBC090 site hierarchy', () => {
      const root = new PageContext(
        'https://notion.so/JBC090-Language-AI-29d979eeca9f8102a85be4dd9007f020',
        'JBC090_Language_AI',
        0
      );
      
      const labSession1 = new PageContext(
        'https://notion.so/Lab-Session-1-abc123',
        'Lab_Session_1',
        1,
        root
      );
      
      const material = new PageContext(
        'https://notion.so/Material-def456',
        'Material',
        1,
        root
      );
      
      const syllabus = new PageContext(
        'https://notion.so/Syllabus-ghi789',
        'Syllabus',
        1,
        root
      );
      
      expect(root.pathSegments).toEqual([]);
      expect(labSession1.pathSegments).toEqual(['Lab_Session_1']);
      expect(material.pathSegments).toEqual(['Material']);
      expect(syllabus.pathSegments).toEqual(['Syllabus']);
      
      // After IPC serialization
      const labJson = JSON.stringify(labSession1.toJSON());
      const labRestored = PageContext.fromJSON(JSON.parse(labJson));
      
      expect(labRestored.pathSegments).toEqual(['Lab_Session_1']);
    });
  });
});
