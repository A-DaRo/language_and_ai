/**
 * Analyzes and classifies edges in the page graph (tree, back, forward, cross) according to BFS traversal rules.
 * 
 * @classdesc During BFS traversal of a page graph, edges are classified into four types:
 * - **Tree edges**: Parent → Child relationships (first discovery)
 * - **Back edges**: Point to ancestors, creating cycles (including self-loops)
 * - **Forward edges**: Skip levels, pointing to descendants
 * - **Cross edges**: Connect different branches or nodes at the same level
 * 
 * This classification is essential for understanding the page graph structure and detecting
 * cycles that could cause infinite traversal loops.
 * 
 * @see RecursiveScraper
 * @see PageContext
 * 
 * @typedef {Object} EdgeInfo
 * @property {string} from - Source page URL
 * @property {string} to - Target page URL
 * @property {string} title - Link title
 * @property {string} [type] - Edge subtype (e.g., 'self-loop', 'to-ancestor')
 */
class GraphAnalyzer {
  /**
   * @param {Logger} logger - Logger instance for edge classification logging.
   */
  constructor(logger) {
    this.logger = logger;
    
    // Edge classification arrays
    this.treeEdges = [];       // Parent -> Child (first discovery)
    this.backEdges = [];       // To ancestor (creates cycle)
    this.forwardEdges = [];    // To descendant (shortcut)
    this.crossEdges = [];      // Between branches (same or earlier level)
  }

  /**
   * @summary Classify an edge when encountering an already-discovered node.
   * 
   * Determines the edge type based on the relationship between source and target nodes:
   * - Back edge if target is an ancestor or self-loop
   * - Forward edge if target is at a deeper level
   * - Cross edge if target is in a different branch or same level
   * 
   * @param {PageContext} fromContext - Source page context.
   * @param {PageContext} toContext - Target page context (already discovered).
   * @param {string} toUrl - Target page URL.
   * @param {string} linkTitle - Title of the link.
   * @returns {('tree'|'back'|'forward'|'cross')} The classified edge type.
   * 
   * @example
   * const edgeType = analyzer.classifyEdge(parentContext, childContext, childUrl, 'Child Page');
   * // Returns: 'back' if child points back to parent
   * // Returns: 'forward' if parent points to grandchild
   * // Returns: 'cross' if between different branches
   */
  classifyEdge(fromContext, toContext, toUrl, linkTitle) {
    const fromDepth = fromContext.depth;
    const toDepth = toContext.depth;
    
    // Check if toContext is an ancestor of fromContext
    const isAncestor = this._isAncestor(fromContext, toUrl);
    
    if (isAncestor || fromContext.url === toUrl) {
      // Back edge (including self-loops)
      const edgeInfo = { 
        from: fromContext.url, 
        to: toUrl, 
        title: linkTitle,
        type: fromContext.url === toUrl ? 'self-loop' : 'to-ancestor'
      };
      this.backEdges.push(edgeInfo);
      this.logger.debug('DISCOVERY', `    [BACK EDGE] ${fromContext.title} -> ${linkTitle} (${edgeInfo.type === 'self-loop' ? 'SELF-LOOP' : 'ancestor'})`);
      return 'back';
      
    } else if (toDepth > fromDepth) {
      // Forward edge (points to deeper level)
      this.forwardEdges.push({ from: fromContext.url, to: toUrl, title: linkTitle });
      this.logger.debug('DISCOVERY', `    [FORWARD EDGE] ${fromContext.title} -> ${linkTitle}`);
      return 'forward';
      
    } else {
      // Cross edge (points to same level or different branch)
      this.crossEdges.push({ from: fromContext.url, to: toUrl, title: linkTitle });
      this.logger.debug('DISCOVERY', `    [CROSS EDGE] ${fromContext.title} -> ${linkTitle}`);
      return 'cross';
    }
  }

  /**
   * @summary Add a tree edge to the classification.
   * 
   * Called when a new node is discovered for the first time, establishing
   * a parent-child relationship.
   * 
   * @param {string} fromUrl - Source page URL.
   * @param {string} toUrl - Target page URL.
   * @param {string} title - Link title.
   */
  addTreeEdge(fromUrl, toUrl, title) {
    this.treeEdges.push({ from: fromUrl, to: toUrl, title });
  }

  /**
   * @summary Log edge classification statistics for current BFS level.
   * 
   * Displays counts of each edge type discovered at the specified level.
   * 
   * @param {number} level - Current BFS level (depth).
   */
  logEdgeStatistics(level) {
    const tree = this.treeEdges.length;
    const back = this.backEdges.length;
    const forward = this.forwardEdges.length;
    const cross = this.crossEdges.length;
    const total = tree + back + forward + cross;
    
    if (total > 0) {
      this.logger.info('DISCOVERY', `  Level ${level} edges: ${tree} tree, ${back} back, ${forward} forward, ${cross} cross`);
    }
  }

  /**
   * @summary Log final edge classification summary.
   * 
   * Displays comprehensive statistics for all edges discovered during traversal,
   * including warnings for cycles (back edges) and self-loops.
   */
  logFinalEdgeStatistics() {
    this.logger.separator('Edge Classification Summary');
    this.logger.info('EDGES', `Tree edges (parent->child): ${this.treeEdges.length}`);
    this.logger.info('EDGES', `Back edges (to ancestor/self): ${this.backEdges.length}`);
    this.logger.info('EDGES', `Forward edges (skip levels): ${this.forwardEdges.length}`);
    this.logger.info('EDGES', `Cross edges (between branches): ${this.crossEdges.length}`);
    
    if (this.backEdges.length > 0) {
      const selfLoops = this.backEdges.filter(e => e.type === 'self-loop').length;
      if (selfLoops > 0) {
        this.logger.warn('EDGES', `  Found ${selfLoops} self-loop(s) - pages linking to themselves`);
      }
      const cycles = this.backEdges.length - selfLoops;
      if (cycles > 0) {
        this.logger.warn('EDGES', `  Found ${cycles} cycle(s) - back edges to ancestors`);
      }
    }
  }

  /**
   * @summary Reset all edge classification arrays.
   * 
   * Clears all stored edge information. Should be called before starting
   * a new discovery phase.
   */
  resetEdgeClassification() {
    this.treeEdges = [];
    this.backEdges = [];
    this.forwardEdges = [];
    this.crossEdges = [];
  }

  /**
   * @summary Check if targetUrl is an ancestor of fromContext.
   * 
   * Walks up the parent chain from fromContext to determine if the target URL
   * appears in the ancestry path.
   * 
   * @param {PageContext} fromContext - Starting context.
   * @param {string} targetUrl - URL to search for in ancestors.
   * @returns {boolean} True if targetUrl is an ancestor of fromContext.
   * @private
   */
  _isAncestor(fromContext, targetUrl) {
    let current = fromContext.parentContext;
    while (current) {
      if (current.url === targetUrl) {
        return true;
      }
      current = current.parentContext;
    }
    return false;
  }

  /**
   * @summary Get current edge statistics.
   * 
   * @returns {{tree: number, back: number, forward: number, cross: number}} Edge counts by type.
   */
  getStats() {
    return {
      tree: this.treeEdges.length,
      back: this.backEdges.length,
      forward: this.forwardEdges.length,
      cross: this.crossEdges.length
    };
  }
}

module.exports = GraphAnalyzer;
