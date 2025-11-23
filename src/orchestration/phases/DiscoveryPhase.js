/**
 * @fileoverview Phase 2: Discovery - Parallel BFS page discovery
 * @module orchestration/phases/DiscoveryPhase
 * @description Performs lightweight traversal to map site structure.
 */

const { MESSAGE_TYPES } = require('../../core/ProtocolDefinitions');
const PhaseStrategy = require('./PhaseStrategy');

/**
 * @class DiscoveryPhase
 * @extends PhaseStrategy
 * @classdesc Manages the BFS discovery of the site structure.
 */
class DiscoveryPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Performs parallel BFS traversal to map the site
   * @description
   * 1. Consumes tasks from DiscoveryQueue
   * 2. Dispatches IPC_DISCOVER tasks to idle workers
   * 3. Processes results and enqueues child pages
   * 4. Updates TitleRegistry with discovered page titles
   * 5. Continues until discovery frontier is empty
   * @param {number} maxDepth - Maximum discovery depth
   * @returns {Promise<void>}
   */
  async execute(maxDepth) {
    this.logger.separator('Phase 2: Discovery');
    this.orchestrator.eventBus.emit('PHASE:CHANGED', { phase: 'discovery', data: {} });

    while (!this.queueManager.isDiscoveryComplete()) {
      const task = this.queueManager.nextDiscovery();
      if (!task) {
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }

      // Emit progress
      const stats = this.queueManager.getStatistics();
      const queueLength = this.queueManager.discoveryQueue.getLength();
      const pendingCount = this.queueManager.discoveryQueue.getPendingCount();
      this.orchestrator.eventBus.emit('DISCOVERY:PROGRESS', {
        pagesFound: stats.discovered,
        inQueue: queueLength + pendingCount,
        conflicts: stats.conflicts || 0,
        currentDepth: task.pageContext.depth
      });

      // Check depth limit
      if (task.pageContext.depth >= maxDepth) {
        this.logger.info('DISCOVERY', `Depth limit reached for: ${task.pageContext.title}`);
        this.queueManager.failDiscovery(task.pageContext.id, new Error('Depth limit reached'));
        continue;
      }

      // Execute discovery
      const workerId = await this.browserManager.execute(MESSAGE_TYPES.DISCOVER, {
        url: task.pageContext.url,
        pageId: task.pageContext.id,
        parentId: task.pageContext.parentId,
        depth: task.pageContext.depth,
        isFirstPage: false,
        cookies: this.orchestrator.cookies
      });

      const titleRegistry = this.queueManager.getTitleRegistry();
      let title = titleRegistry[task.pageContext.id] || task.pageContext.title || 'Untitled';
      if (title.match(/^[a-f0-9]{32}$/i)) {
        title = `Page ${title.substring(0, 6)}...`;
      }

      this.orchestrator.eventBus.emit('WORKER:BUSY', {
        workerId,
        task: { description: `Discovering '${title}'...` }
      });
    }

    // Wait for all pending discoveries
    while (!this.queueManager.isDiscoveryComplete()) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    const stats = this.queueManager.getStatistics();
    this.logger.success('DISCOVERY', `Complete: ${stats.discovered} page(s) discovered`);

    // Update workers with complete registry
    this.logger.info('DISCOVERY', 'Updating workers with discovered titles...');
    const titleRegistry = this.queueManager.getTitleRegistry();
    await this.browserManager.initializeWorkers(titleRegistry);
  }
}

module.exports = DiscoveryPhase;
