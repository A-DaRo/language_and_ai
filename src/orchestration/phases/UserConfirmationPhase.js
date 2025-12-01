/**
 * @fileoverview Phase 3: User Confirmation - Display tree and prompt for confirmation
 * @module orchestration/phases/UserConfirmationPhase
 * @description Allows user to review discovered structure before download.
 */

const UserPrompt = require('../../utils/UserPrompt');
const PageTreeRenderer = require('../../utils/PageTreeRenderer');
const PhaseStrategy = require('./PhaseStrategy');

/**
 * @class UserConfirmationPhase
 * @extends PhaseStrategy
 * @classdesc Displays site hierarchy and prompts user for confirmation.
 */
class UserConfirmationPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Displays the site tree and prompts for confirmation
   * @description
   * 1. Generates visual representation of discovered hierarchy
   * 2. Displays summary statistics (total pages, depth)
   * 3. Prompts user (Y/N) via UserPrompt
   * 4. Aborts if declined
   * @returns {Promise<boolean>} True to proceed, false to abort
   */
  async execute() {
    this.orchestrator.eventBus.emit('PHASE:STOPPING_DASHBOARD', {});
    await new Promise(resolve => setTimeout(resolve, 100));

    this.logger.separator('Phase 3: User Confirmation');

    const stats = this.queueManager.getStatistics();
    if (stats.discovered === 0) {
      this.logger.warn('USER_CONFIRMATION', 'No pages discovered. Aborting.');
      return false;
    }

    // Display tree via PageTreeRenderer (renders to both console and logs)
    this.logger.info('USER_CONFIRMATION', 'Discovered Site Structure:');
    const allContexts = this.queueManager.getAllContexts();
    const rootContext = allContexts.find(ctx => ctx.depth === 0);
    if (rootContext) {
      const renderer = new PageTreeRenderer();
      renderer.renderToConsoleAndLog(
        rootContext,
        this.queueManager.getTitleRegistry(),
        this.queueManager.getMaxDepth()
      );
    }

    // Display summary
    this.logger.info('USER_CONFIRMATION', '\nDiscovery Summary:');
    this.logger.info('USER_CONFIRMATION', `  Total Pages: ${stats.discovered}`);
    this.logger.info('USER_CONFIRMATION', `  Maximum Depth: ${this.queueManager.getMaxDepth()}`);

    // Prompt user
    const prompt = new UserPrompt();
    const proceed = await prompt.promptConfirmDownload({
      totalPages: stats.discovered,
      maxDepth: this.queueManager.getMaxDepth()
    });
    prompt.close();

    if (proceed) {
      this.logger.success('USER_CONFIRMATION', 'User confirmed. Proceeding...');
    } else {
      this.logger.warn('USER_CONFIRMATION', 'User declined. Aborting.');
    }

    return proceed;
  }

}

module.exports = UserConfirmationPhase;
