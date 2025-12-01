/**
 * @fileoverview Phase 1: Bootstrap - Worker initialization and cookie capture
 * @module orchestration/phases/BootstrapPhase
 * @description Spawns worker pool and captures initial authentication cookies.
 */

const { MESSAGE_TYPES } = require('../../core/ProtocolDefinitions');
const BrowserInitializer = require('../../cluster/BrowserInitializer');
const PageContext = require('../../domain/PageContext');
const PhaseStrategy = require('./PhaseStrategy');

/**
 * @class BootstrapPhase
 * @extends PhaseStrategy
 * @classdesc Initializes the worker cluster and captures cookies from root page.
 */
class BootstrapPhase extends PhaseStrategy {
  /**
   * @async
   * @method execute
   * @summary Spawns workers and captures initial session state
   * @description
   * 1. Calculates optimal worker count based on system resources
   * 2. Spawns the initial worker pool
   * 3. Navigates to root URL to capture authentication cookies
   * 4. Broadcasts cookies to all workers
   * 5. Initializes workers with the TitleRegistry
   * @param {string} rootUrl - Root URL to scrape
   * @returns {Promise<void>}
   */
  async execute(rootUrl) {
    this.logger.separator('Phase 1: Bootstrap');

    // Calculate capacity
    const capacity = BrowserInitializer.calculateCapacity(1, 8);
    this.logger.info('BOOTSTRAP', `System capacity: ${capacity.workerCount} worker(s)`);

    // Spawn initial worker for cookie capture
    this.logger.info('BOOTSTRAP', 'Spawning bootstrap worker...');
    const bootstrapWorker = await BrowserInitializer.spawnWorkerPool(1);
    this.browserManager.registerWorkers(bootstrapWorker);
    await this._waitForWorkers(1);

    // Normalize root URL
    const cleanRootUrl = rootUrl.split('?')[0];
    const rootRawIdMatch = cleanRootUrl.match(/29[a-f0-9]{30}/i);
    const normalizedRootUrl = rootRawIdMatch
      ? this.config.getBaseUrl() + '/' + rootRawIdMatch[0]
      : rootUrl;

    const rootTitle = this.config.extractPageNameFromUrl(rootUrl) || 'Main_Page';
    const rootContext = new PageContext(normalizedRootUrl, rootTitle, 0, null, null);
    this.queueManager.enqueueDiscovery(rootContext, true);

    // Execute first discovery to capture cookies
    this.logger.info('BOOTSTRAP', 'Capturing cookies...');
    const task = this.queueManager.nextDiscovery();
    await this.browserManager.execute(MESSAGE_TYPES.DISCOVER, {
      url: task.pageContext.url,
      pageId: task.pageContext.id,
      parentId: task.pageContext.parentId,
      depth: task.pageContext.depth,
      isFirstPage: true
    });

    // Wait for cookies
    await this._waitForCookies();

    // Spawn remaining workers
    if (capacity.workerCount > 1) {
      this.logger.info('BOOTSTRAP', `Spawning ${capacity.workerCount - 1} additional worker(s)...`);
      const additionalWorkers = await BrowserInitializer.spawnWorkerPool(capacity.workerCount - 1, 2);
      this.browserManager.registerWorkers(additionalWorkers);
      await this._waitForWorkers(capacity.workerCount);
    }

    // Broadcast cookies
    if (this.orchestrator.cookies && this.orchestrator.cookies.length > 0) {
      this.logger.info('BOOTSTRAP', `Broadcasting ${this.orchestrator.cookies.length} cookie(s)...`);
      await this.browserManager.broadcastCookies(this.orchestrator.cookies);
    }

    // Initialize workers with title registry
    this.logger.info('BOOTSTRAP', 'Initializing workers...');
    const titleRegistry = this.queueManager.getTitleRegistry();
    await this.browserManager.initializeWorkers(titleRegistry);

    this.logger.success('BOOTSTRAP', `Complete with ${this.browserManager.getTotalCount()} worker(s)`);
    
    // Emit BOOTSTRAP:COMPLETE event to trigger dashboard initialization
    this.eventBus.emit('BOOTSTRAP:COMPLETE', {
      workerCount: this.browserManager.getTotalCount()
    });
  }

  /**
   * @private
   * @async
   * @param {number} count - Number of workers to wait for
   */
  async _waitForWorkers(count) {
    const startTime = Date.now();
    const timeout = 60000;
    while (this.browserManager.getTotalCount() < count) {
      if (Date.now() - startTime > timeout) throw new Error('Timeout waiting for workers');
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  /**
   * @private
   * @async
   */
  async _waitForCookies() {
    const startTime = Date.now();
    const timeout = 60000;
    while (!this.orchestrator.cookies) {
      if (Date.now() - startTime > timeout) {
        this.logger.warn('BOOTSTRAP', 'Cookie timeout, proceeding without auth');
        this.orchestrator.cookies = [];
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
}

module.exports = BootstrapPhase;
