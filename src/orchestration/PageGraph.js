/**
 * @fileoverview Page graph with edge classification metadata
 * @module orchestration/PageGraph
 * @description Maintains the discovered page hierarchy with edge classification metadata.
 */

/**
 * @class PageGraph
 * @classdesc Stores the discovered page hierarchy with edges and their classifications.
 * Built during discovery phase, passed to execution phase for context-aware link rewriting.
 */
class PageGraph {
  constructor() {
    // Map: pageId → PageContext
    this.nodes = new Map();

    // Map: pageId → Set<targetPageId>
    this.edges = new Map();

    // Map: "sourceId-targetId" → EdgeClassification
    // EdgeClassification = { type, depthDelta, isAncestor }
    this.edgeMetadata = new Map();
  }

  /**
   * Add a node (discovered page) to the graph
   * @param {string} pageId - Page identifier
   * @param {PageContext} context - Page context object
   */
  addNode(pageId, context) {
    if (pageId && context) {
      this.nodes.set(pageId, context);
    }
  }

  /**
   * Add an edge with classification metadata
   * @param {string} sourceId - Source page ID
   * @param {string} targetId - Target page ID
   * @param {Object} classification - EdgeClassifier result
   *   @param {string} classification.type - 'FORWARD', 'BACK', or 'UNKNOWN'
   *   @param {number} classification.depthDelta - Depth difference
   *   @param {boolean} classification.isAncestor - Is target an ancestor of source
   */
  addEdge(sourceId, targetId, classification) {
    if (!sourceId || !targetId) {
      return;
    }

    // Add to edge set
    if (!this.edges.has(sourceId)) {
      this.edges.set(sourceId, new Set());
    }
    this.edges.get(sourceId).add(targetId);

    // Store classification metadata
    const key = `${sourceId}-${targetId}`;
    this.edgeMetadata.set(key, classification);
  }

  /**
   * Get classification for an edge
   * @param {string} sourceId - Source page ID
   * @param {string} targetId - Target page ID
   * @returns {Object|null} EdgeClassification or null if edge not found
   */
  getEdgeClassification(sourceId, targetId) {
    const key = `${sourceId}-${targetId}`;
    return this.edgeMetadata.get(key) || null;
  }

  /**
   * Get all outgoing edges from a page
   * @param {string} sourceId - Source page ID
   * @returns {Set<string>} Set of target page IDs, or empty set
   */
  getOutgoingEdges(sourceId) {
    return this.edges.get(sourceId) || new Set();
  }

  /**
   * Get all incoming edges to a page
   * @param {string} targetId - Target page ID
   * @returns {Set<string>} Set of source page IDs
   */
  getIncomingEdges(targetId) {
    const incoming = new Set();
    for (const [source, targets] of this.edges) {
      if (targets.has(targetId)) {
        incoming.add(source);
      }
    }
    return incoming;
  }

  /**
   * Get page context by ID
   * @param {string} pageId - Page identifier
   * @returns {PageContext|null} Page context or null if not found
   */
  getNode(pageId) {
    return this.nodes.get(pageId) || null;
  }

  /**
   * Check if node exists
   * @param {string} pageId - Page identifier
   * @returns {boolean} True if node exists
   */
  hasNode(pageId) {
    return this.nodes.has(pageId);
  }

  /**
   * Check if edge exists
   * @param {string} sourceId - Source page ID
   * @param {string} targetId - Target page ID
   * @returns {boolean} True if edge exists
   */
  hasEdge(sourceId, targetId) {
    const edges = this.edges.get(sourceId);
    return edges ? edges.has(targetId) : false;
  }

  /**
   * Get all nodes
   * @returns {Map<string, PageContext>} All nodes
   */
  getAllNodes() {
    return new Map(this.nodes);
  }

  /**
   * Get statistics
   * @returns {Object} Graph statistics
   */
  getStatistics() {
    return {
      nodeCount: this.nodes.size,
      edgeCount: this.edgeMetadata.size,
      forwardEdges: Array.from(this.edgeMetadata.values())
        .filter(e => e.type === 'FORWARD').length,
      backEdges: Array.from(this.edgeMetadata.values())
        .filter(e => e.type === 'BACK').length,
      crossEdges: Array.from(this.edgeMetadata.values())
        .filter(e => e.type === 'CROSS').length
    };
  }

  /**
   * Clear the graph
   */
  clear() {
    this.nodes.clear();
    this.edges.clear();
    this.edgeMetadata.clear();
  }

  /**
   * Serialize to JSON for IPC transfer
   * @returns {Object} Serializable representation
   */
  toJSON() {
    const nodesObj = {};
    for (const [id, context] of this.nodes) {
      nodesObj[id] = context.toJSON ? context.toJSON() : context;
    }

    const edgesObj = {};
    for (const [source, targets] of this.edges) {
      edgesObj[source] = Array.from(targets);
    }

    const metadataObj = {};
    for (const [key, value] of this.edgeMetadata) {
      metadataObj[key] = value;
    }

    return {
      nodes: nodesObj,
      edges: edgesObj,
      edgeMetadata: metadataObj
    };
  }

  /**
   * Deserialize from JSON
   * @static
   * @param {Object} json - Serialized data
   * @returns {PageGraph} Reconstructed instance
   */
  static fromJSON(json) {
    const graph = new PageGraph();

    if (json.nodes) {
      for (const [id, nodeData] of Object.entries(json.nodes)) {
        graph.nodes.set(id, nodeData);
      }
    }

    if (json.edges) {
      for (const [source, targets] of Object.entries(json.edges)) {
        graph.edges.set(source, new Set(targets));
      }
    }

    if (json.edgeMetadata) {
      for (const [key, value] of Object.entries(json.edgeMetadata)) {
        graph.edgeMetadata.set(key, value);
      }
    }

    return graph;
  }
}

module.exports = PageGraph;
