/**
 * @fileoverview Unit tests for PageGraph
 * @module tests/unit/orchestration/PageGraph.test.js
 */

const PageGraph = require('../../../src/orchestration/PageGraph');
const PageContext = require('../../../src/domain/PageContext');

describe('PageGraph', () => {
  let graph;

  beforeEach(() => {
    graph = new PageGraph();
  });

  describe('addNode and hasNode', () => {
    it('should add and find nodes', () => {
      const context = new PageContext('http://example.com/page-29abc', 'Test Page', 0);
      graph.addNode(context.id, context);

      expect(graph.hasNode(context.id)).toBe(true);
      expect(graph.getNode(context.id)).toBe(context);
    });

    it('should handle null nodes gracefully', () => {
      graph.addNode(null, null);
      expect(graph.hasNode(null)).toBe(false);
    });
  });

  describe('addEdge and getEdgeClassification', () => {
    it('should add edge with classification', () => {
      const sourceId = '29abc';
      const targetId = '29def';
      const classification = { type: 'FORWARD', depthDelta: 1, isAncestor: false };

      graph.addEdge(sourceId, targetId, classification);

      expect(graph.hasEdge(sourceId, targetId)).toBe(true);
      expect(graph.getEdgeClassification(sourceId, targetId)).toEqual(classification);
    });

    it('should handle edges between same nodes', () => {
      const nodeId = '29abc';
      const classification = { type: 'BACK', depthDelta: 0, isAncestor: false };

      graph.addEdge(nodeId, nodeId, classification);

      expect(graph.hasEdge(nodeId, nodeId)).toBe(true);
    });
  });

  describe('getOutgoingEdges', () => {
    it('should return all outgoing edges from a node', () => {
      const sourceId = '29abc';
      const targets = ['29def', '29ghi', '29jkl'];

      for (const targetId of targets) {
        graph.addEdge(sourceId, targetId, { type: 'FORWARD' });
      }

      const outgoing = graph.getOutgoingEdges(sourceId);

      expect(outgoing.size).toBe(3);
      for (const targetId of targets) {
        expect(outgoing.has(targetId)).toBe(true);
      }
    });

    it('should return empty set for node with no outgoing edges', () => {
      const outgoing = graph.getOutgoingEdges('nonexistent');
      expect(outgoing.size).toBe(0);
    });
  });

  describe('getIncomingEdges', () => {
    it('should return all incoming edges to a node', () => {
      const targetId = '29jkl';
      const sources = ['29abc', '29def', '29ghi'];

      for (const sourceId of sources) {
        graph.addEdge(sourceId, targetId, { type: 'FORWARD' });
      }

      const incoming = graph.getIncomingEdges(targetId);

      expect(incoming.size).toBe(3);
      for (const sourceId of sources) {
        expect(incoming.has(sourceId)).toBe(true);
      }
    });

    it('should return empty set for node with no incoming edges', () => {
      const incoming = graph.getIncomingEdges('isolated');
      expect(incoming.size).toBe(0);
    });
  });

  describe('getStatistics', () => {
    it('should return accurate statistics', () => {
      const nodes = ['29a', '29b', '29c'];
      for (const nodeId of nodes) {
        graph.addNode(nodeId, { id: nodeId, title: `Node ${nodeId}` });
      }

      graph.addEdge('29a', '29b', { type: 'FORWARD' });
      graph.addEdge('29b', '29c', { type: 'FORWARD' });
      graph.addEdge('29c', '29a', { type: 'BACK', isAncestor: false });

      const stats = graph.getStatistics();

      expect(stats.nodeCount).toBe(3);
      expect(stats.edgeCount).toBe(3);
      expect(stats.forwardEdges).toBe(2);
      expect(stats.backEdges).toBe(1);
    });
  });

  describe('clear', () => {
    it('should clear all data', () => {
      graph.addNode('29a', { id: '29a' });
      graph.addEdge('29a', '29b', { type: 'FORWARD' });

      graph.clear();

      expect(graph.nodes.size).toBe(0);
      expect(graph.edges.size).toBe(0);
      expect(graph.edgeMetadata.size).toBe(0);
    });
  });

  describe('Serialization', () => {
    it('should serialize and deserialize graph', () => {
      const context = new PageContext('http://example.com/page-29abc', 'Test Page', 0);
      graph.addNode(context.id, context);
      graph.addEdge(context.id, '29def', { type: 'FORWARD', depthDelta: 1 });

      const json = graph.toJSON();
      const restored = PageGraph.fromJSON(json);

      expect(restored.hasNode(context.id)).toBe(true);
      expect(restored.hasEdge(context.id, '29def')).toBe(true);
      expect(restored.getEdgeClassification(context.id, '29def')).toEqual({
        type: 'FORWARD',
        depthDelta: 1
      });
    });

    it('should handle empty graph serialization', () => {
      const json = graph.toJSON();
      const restored = PageGraph.fromJSON(json);

      expect(restored.nodes.size).toBe(0);
      expect(restored.edges.size).toBe(0);
      expect(restored.edgeMetadata.size).toBe(0);
    });
  });

  describe('Complex scenarios', () => {
    it('should track page hierarchy with multiple levels', () => {
      // Create a hierarchy: root -> parent -> child -> grandchild
      const root = '29root';
      const parent = '29parent';
      const child = '29child';
      const grandchild = '29grandchild';

      graph.addNode(root, { id: root });
      graph.addNode(parent, { id: parent });
      graph.addNode(child, { id: child });
      graph.addNode(grandchild, { id: grandchild });

      graph.addEdge(root, parent, { type: 'FORWARD', depthDelta: 1 });
      graph.addEdge(parent, child, { type: 'FORWARD', depthDelta: 1 });
      graph.addEdge(child, grandchild, { type: 'FORWARD', depthDelta: 1 });
      
      // Add a back edge from grandchild to root
      graph.addEdge(grandchild, root, { type: 'BACK', isAncestor: true });

      expect(graph.getOutgoingEdges(root).size).toBe(1);
      expect(graph.getOutgoingEdges(grandchild).size).toBe(1);
      expect(graph.getIncomingEdges(root).size).toBe(1);
      expect(graph.getStatistics().edgeCount).toBe(4);
    });
  });
});
