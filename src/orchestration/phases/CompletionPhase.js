/**
 * @fileoverview Phase 6: Completion - Cleanup and statistics
 * @module orchestration/phases/CompletionPhase
 * @description Finalizes the scraping process and displays results.
 */

const PageTreeRenderer = require('../../utils/PageTreeRenderer');
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
        // Render tree via PageTreeRenderer (renders to both console and logs)
        const renderer = new PageTreeRenderer();
        renderer.renderToConsoleAndLog(
          rootContext,
          this.queueManager.getTitleRegistry(),
          this.queueManager.getMaxDepth()
        );
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

}

module.exports = CompletionPhase;
