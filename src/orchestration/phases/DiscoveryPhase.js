/**
 * @fileoverview Phase 2: Discovery - Parallel BFS page discovery
 * @module orchestration/phases/DiscoveryPhase
 * @description Performs lightweight traversal to map site structure.
 */

const { MESSAGE_TYPES } = require('../../core/ProtocolDefinitions');
const PhaseStrategy = require('./PhaseStrategy');

const DISCOVERY_PHASE_TIMEOUT_MS = 30 * 60 * 1000;

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

    const bus = this.orchestrator.eventBus;
    let isPhaseActive = true;
    let pendingQueueResolver = null;

    const releaseQueueWait = () => {
      if (pendingQueueResolver) {
        pendingQueueResolver();
      }
    };

    const completionPromise = new Promise((resolve, reject) => {
      let timeoutId;
      let cleanup;

      const handleAllIdle = () => {
        this.logger.debug('DISCOVERY', 'Discovery queue reached quiescent state');
        releaseQueueWait();
        cleanup();
        resolve();
      };

      const handleTaskCompleted = ({ pendingCount }) => {
        if (pendingCount < 0) {
          this.logger.error('DISCOVERY', `INVARIANT VIOLATION: pendingCount = ${pendingCount}`);
          releaseQueueWait();
          cleanup();
          reject(new Error('Queue state corrupted: negative pending count'));
        }
      };

      cleanup = () => {
        if (!isPhaseActive) {
          return;
        }
        isPhaseActive = false;
        clearTimeout(timeoutId);
        bus.off('DISCOVERY:ALL_IDLE', handleAllIdle);
        bus.off('DISCOVERY:TASK_COMPLETED', handleTaskCompleted);
      };

      timeoutId = setTimeout(() => {
        releaseQueueWait();
        cleanup();
        const error = new Error(`Discovery Phase Timeout after ${DISCOVERY_PHASE_TIMEOUT_MS}ms`);
        this.logger.error('DISCOVERY', error.message);
        reject(error);
      }, DISCOVERY_PHASE_TIMEOUT_MS);

      bus.on('DISCOVERY:ALL_IDLE', handleAllIdle);
      bus.on('DISCOVERY:TASK_COMPLETED', handleTaskCompleted);
    });

    const waitForQueueReady = async () => {
      if (!isPhaseActive || this.queueManager.discoveryQueue.getLength() > 0) {
        return;
      }

      await new Promise(resolve => {
        const handler = () => {
          if (pendingQueueResolver) {
            pendingQueueResolver();
          }
        };

        pendingQueueResolver = () => {
          pendingQueueResolver = null;
          bus.off('DISCOVERY:QUEUE_READY', handler);
          resolve();
        };

        bus.once('DISCOVERY:QUEUE_READY', handler);

        if (this.queueManager.discoveryQueue.getLength() > 0) {
          handler();
        }
      });
    };

    const dispatchLoop = async () => {
      while (isPhaseActive && !this.queueManager.isDiscoveryComplete()) {
        const task = this.queueManager.nextDiscovery();
        if (!task) {
          await waitForQueueReady();
          continue;
        }

        const stats = this.queueManager.getStatistics();
        const queueLength = this.queueManager.discoveryQueue.getLength();
        const pendingCount = this.queueManager.discoveryQueue.getPendingCount();
        this.orchestrator.eventBus.emit('DISCOVERY:PROGRESS', {
          pagesFound: stats.discovered,
          inQueue: queueLength + pendingCount,
          conflicts: stats.conflicts || 0,
          currentDepth: task.pageContext.depth
        });

        if (task.pageContext.depth >= maxDepth) {
          this.logger.info('DISCOVERY', `Depth limit reached for: ${task.pageContext.title}`);
          this.queueManager.failDiscovery(task.pageContext.id, new Error('Depth limit reached'));
          continue;
        }

        const titleRegistry = this.queueManager.getTitleRegistry();
        const pageTitle = task.pageContext.getDisplayTitle(titleRegistry);

        const workerId = await this.browserManager.execute(MESSAGE_TYPES.DISCOVER, {
          url: task.pageContext.url,
          pageId: task.pageContext.id,
          parentId: task.pageContext.parentId,
          depth: task.pageContext.depth,
          isFirstPage: false,
          cookies: this.orchestrator.cookies,
          metadata: { pageTitle, pageId: task.pageContext.id }
        });
      }
    };

    await Promise.all([dispatchLoop(), completionPromise]);

    const stats = this.queueManager.getStatistics();
    this.logger.success('DISCOVERY', `Complete: ${stats.discovered} page(s) discovered`);

    this.logger.info('DISCOVERY', 'Updating workers with discovered titles...');
    const titleRegistry = this.queueManager.getTitleRegistry();
    await this.browserManager.initializeWorkers(titleRegistry);
  }
}

module.exports = DiscoveryPhase;
