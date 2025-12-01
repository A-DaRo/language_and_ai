/**
 * @fileoverview Unit tests for EdgeClassifier
 * @module tests/unit/orchestration/analysis/EdgeClassifier.test.js
 */

const EdgeClassifier = require('../../../../src/orchestration/analysis/EdgeClassifier');
const PageContext = require('../../../../src/domain/PageContext');

describe('EdgeClassifier', () => {
  let classifier;

  beforeEach(() => {
    classifier = new EdgeClassifier();
  });

  describe('classifyEdge', () => {
    it('should classify FORWARD edge when target is deeper', () => {
      const source = new PageContext('http://example.com/page1-29abc', 'Page 1', 0);
      const target = new PageContext('http://example.com/page2-29def', 'Page 2', 1);

      const classification = classifier.classifyEdge(source, target);

      expect(classification.type).toBe('FORWARD');
      expect(classification.depthDelta).toBe(1);
      expect(classification.isAncestor).toBe(false);
    });

    it('should classify BACK edge when target is at same level', () => {
      const source = new PageContext('http://example.com/page1-29abc', 'Page 1', 1);
      const target = new PageContext('http://example.com/page2-29def', 'Page 2', 1);

      const classification = classifier.classifyEdge(source, target);

      expect(classification.type).toBe('BACK');
      expect(classification.depthDelta).toBe(0);
    });

    it('should classify BACK edge when target is shallower', () => {
      const source = new PageContext('http://example.com/page1-29abc', 'Page 1', 2);
      const target = new PageContext('http://example.com/page2-29def', 'Page 2', 1);

      const classification = classifier.classifyEdge(source, target);

      expect(classification.type).toBe('BACK');
      expect(classification.depthDelta).toBe(1);
    });

    it('should identify ancestor relationships', () => {
      const root = new PageContext('http://example.com/root-29root', 'Root', 0);
      const parent = new PageContext('http://example.com/parent-29parent', 'Parent', 1, root, root.id);
      const child = new PageContext('http://example.com/child-29child', 'Child', 2, parent, parent.id);

      const contextMap = new Map([
        [root.id, root],
        [parent.id, parent],
        [child.id, child]
      ]);

      classifier.setContextMap(contextMap);

      const classification = classifier.classifyEdge(child, root);

      expect(classification.type).toBe('BACK');
      expect(classification.isAncestor).toBe(true);
    });

    it('should not identify non-ancestor as ancestor', () => {
      const page1 = new PageContext('http://example.com/page1-29a', 'Page 1', 1);
      const page2 = new PageContext('http://example.com/page2-29b', 'Page 2', 1);

      const classification = classifier.classifyEdge(page1, page2);

      expect(classification.isAncestor).toBe(false);
    });

    it('should handle null contexts gracefully', () => {
      const classification = classifier.classifyEdge(null, null);

      expect(classification.type).toBe('UNKNOWN');
      expect(classification.depthDelta).toBe(0);
      expect(classification.isAncestor).toBe(false);
    });

    it('should handle multiple depth levels', () => {
      const source = new PageContext('http://example.com/page1-29abc', 'Page 1', 0);
      const target = new PageContext('http://example.com/page2-29def', 'Page 2', 5);

      const classification = classifier.classifyEdge(source, target);

      expect(classification.type).toBe('FORWARD');
      expect(classification.depthDelta).toBe(5);
    });
  });

  describe('setContextMap', () => {
    it('should set context map for ancestor lookups', () => {
      const contextMap = new Map();
      classifier.setContextMap(contextMap);

      expect(classifier.contextMap).toBe(contextMap);
    });
  });

  describe('Ancestor chain walking', () => {
    it('should find ancestor multiple levels up', () => {
      const level0 = new PageContext('http://example.com/l0-29l0', 'L0', 0);
      const level1 = new PageContext('http://example.com/l1-29l1', 'L1', 1, level0, level0.id);
      const level2 = new PageContext('http://example.com/l2-29l2', 'L2', 2, level1, level1.id);
      const level3 = new PageContext('http://example.com/l3-29l3', 'L3', 3, level2, level2.id);

      const contextMap = new Map([
        [level0.id, level0],
        [level1.id, level1],
        [level2.id, level2],
        [level3.id, level3]
      ]);

      classifier.setContextMap(contextMap);

      // Level 3 should be able to reach level 0
      const classification = classifier.classifyEdge(level3, level0);

      expect(classification.isAncestor).toBe(true);
    });
  });
});
