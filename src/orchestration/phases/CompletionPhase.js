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
    console.log(`└─ ${rootLabel}`);

    rootContext.children.forEach((child, index) => {
      const isLast = index === rootContext.children.length - 1;
      this._printTreeNode(child, '   ', isLast, titleRegistry);
    });

    this.logger.separator();
  }

  /**
   * @private
   * @param {PageContext} context - Page context
   * @param {string} prefix - Line prefix
   * @param {boolean} isLast - Is this the last child?
   * @param {Object} titleRegistry - ID-to-title map
   */
  _printTreeNode(context, prefix, isLast, titleRegistry) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = titleRegistry[context.id] || context.title || 'Untitled';
    const exploredChildren = context.children.filter(child => titleRegistry[child.id]);
    const internalRefs = context.children.length - exploredChildren.length;
    const label = internalRefs > 0 ? `${title} [${internalRefs} internal ref${internalRefs > 1 ? 's' : ''}]` : title;

    console.log(`${prefix}${connector}${label}`);

    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    exploredChildren.forEach((child, index) => {
      const childIsLast = index === exploredChildren.length - 1;
      this._printTreeNode(child, childPrefix, childIsLast, titleRegistry);
    });
  }
}

module.exports = CompletionPhase;
