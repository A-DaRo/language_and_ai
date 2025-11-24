/**
 * @fileoverview Phase 6: Completion - Cleanup and statistics
 * @module orchestration/phases/CompletionPhase
 * @description Finalizes the scraping process and displays results.
 */

const PhaseStrategy = require('./PhaseStrategy');

/**
 * @class CompletionPhase
 * @extends PhaseStrategy
 * @classdesc Handles process finalization and statistics reporting.
 */
class CompletionPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Cleanup and final reporting
   * @description
   * 1. Displays statistics
   * 2. Shows final page tree if needed
   * 3. Performs cleanup operations
   * @param {boolean} aborted - Whether process was aborted
   * @returns {Promise<Object>} Final result with stats
   */
  async execute(aborted = false) {
    if (aborted) {
      this.logger.separator('Phase 6: Complete (Aborted)');
    } else {
      this.logger.separator('Phase 6: Complete');
    }

    const stats = this.queueManager.getStatistics();

    if (aborted) {
      this.logger.info('COMPLETE', 'Scraping process aborted. Cleaning up...');
      this.logger.info('STATS', `Discovered: ${stats.discovered} pages`);

      const allContexts = this.queueManager.getAllContexts();
      const rootContext = allContexts.find(ctx => ctx.depth === 0);
      if (rootContext) {
        this._displayPageTree(rootContext);
      }
    } else {
      this.logger.success('COMPLETE', 'All operations completed successfully');
      this.logger.info('STATS',
        `Discovered: ${stats.discovered} | Downloaded: ${stats.downloaded} | Failed: ${stats.failed}`
      );
    }

    return {
      stats,
      allContexts: this.queueManager.getAllContexts()
    };
  }

  /**
   * @private
   * @param {PageContext} rootContext - Root page context
   */
  _displayPageTree(rootContext) {
    this.logger.separator('Page Tree');
    console.log('.');

    const titleRegistry = this.queueManager.getTitleRegistry();
    const rootLabel = titleRegistry[rootContext.id] || rootContext.title || '(root)';
    const maxBfsDepth = this.queueManager.getMaxDepth();
    
    console.log(`└─ ${rootLabel}`);

    rootContext.children.forEach((child, index) => {
      const isLast = index === rootContext.children.length - 1;
      this._printTreeNode(
        child,
        '   ',
        isLast,
        titleRegistry,
        new Set([rootContext.id]),
        maxBfsDepth
      );
    });

    this.logger.separator();
  }

  /**
   * @private
   * @param {PageContext} context - Page context
   * @param {string} prefix - Line prefix
   * @param {boolean} isLast - Is this the last child?
   * @param {Object} titleRegistry - ID-to-title map
   * @param {Set<string>} pathVisited - Visited nodes in current path (for cycle detection)
   * @param {number} maxBfsDepth - Maximum depth of BFS expansion (limit recursion to maxBfsDepth)
   */
  _printTreeNode(context, prefix, isLast, titleRegistry, pathVisited = new Set(), maxBfsDepth = Infinity) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = titleRegistry[context.id] || context.title || 'Untitled';

    // Detect cycles within the path (visited in current recursion)
    if (pathVisited.has(context.id)) {
      console.log(`${prefix}${connector}${title} ↺ (Cycle)`);
      return;
    }

    // Filter to only show discovered children (have titles in registry)
    const exploredChildren = context.children.filter(child => titleRegistry[child.id]);
    const internalRefs = context.children.length - exploredChildren.length;
    const label = internalRefs > 0 ? `${title} [${internalRefs} internal ref${internalRefs > 1 ? 's' : ''}]` : title;

    console.log(`${prefix}${connector}${label}`);

    // Stop recursion at BFS depth to limit tree representation
    // This shows the full BFS expansion plus edges creating cycles at leaves
    if (context.depth >= maxBfsDepth) {
      // We're at or past the BFS leaf level - don't recurse deeper
      return;
    }

    // Recurse into children (we're within BFS expansion depth)
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    const nextVisited = new Set(pathVisited);
    nextVisited.add(context.id);
    
    exploredChildren.forEach((child, index) => {
      const childIsLast = index === exploredChildren.length - 1;
      
      // Only recurse if this is a tree edge (child's parent is current context)
      if (child.parentContext === context) {
        this._printTreeNode(
          child,
          childPrefix,
          childIsLast,
          titleRegistry,
          nextVisited,
          maxBfsDepth
        );
      } else {
        // Non-tree edge (Cycle/Cross-link) - Print leaf and stop
        const connector = childIsLast ? '└─ ' : '├─ ';
        const title = titleRegistry[child.id] || child.title || 'Untitled';
        console.log(`${childPrefix}${connector}${title} ↺ (Cycle)`);
      }
    });
  }
}

module.exports = CompletionPhase;
