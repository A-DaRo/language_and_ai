/**
 * @fileoverview Master State Machine
 * @module orchestration/ClusterOrchestrator
 * @description Coordinates the entire distributed scraping workflow via phase strategies.
 * Delegates work to specialized phase implementations.
 * 
 * **CRITICAL CONTEXT**: Runs in MASTER process. This is the brain of the system.
 */

const SystemEventBus = require('../core/SystemEventBus');
const BrowserManager = require('../cluster/BrowserManager');
const GlobalQueueManager = require('./GlobalQueueManager');
const Logger = require('../core/Logger');
const Config = require('../core/Config');

// Import phase strategies
const BootstrapPhase = require('./phases/BootstrapPhase');
const DiscoveryPhase = require('./phases/DiscoveryPhase');
const UserConfirmationPhase = require('./phases/UserConfirmationPhase');
const ConflictResolutionPhase = require('./phases/ConflictResolutionPhase');
const DownloadPhase = require('./phases/DownloadPhase');
const CompletionPhase = require('./phases/CompletionPhase');

const Phase = {
  BOOTSTRAP: 'BOOTSTRAP',
  DISCOVERY: 'DISCOVERY',
  USER_CONFIRMATION: 'USER_CONFIRMATION',
  CONFLICT_RESOLUTION: 'CONFLICT_RESOLUTION',
  DOWNLOAD: 'DOWNLOAD',
  COMPLETE: 'COMPLETE'
};

/**
 * @class ClusterOrchestrator
 * @classdesc Main state machine coordinating distributed scraping via phase strategies
 */
class ClusterOrchestrator {
  /**
   * @param {Object} config - Configuration object
   * @param {Object} logger - Logger instance
   */
  constructor(config, logger) {
    this.config = config || new Config();
    this.logger = logger || new Logger();
    this.eventBus = SystemEventBus.getInstance();

    this.browserManager = new BrowserManager();
    this.queueManager = new GlobalQueueManager();

    this.currentPhase = null;
    this.cookies = null;
    this.linkRewriteMap = null;

    this._setupEventListeners();
  }

  /**
   * Setup event listeners for task completion
   * @private
   */
  _setupEventListeners() {
    this.eventBus.on('TASK:COMPLETE', async ({ workerId, taskType, result }) => {
      await this._handleTaskComplete(workerId, taskType, result);
    });

    this.eventBus.on('TASK:FAILED', async ({ workerId, taskType, error }) => {
      await this._handleTaskFailed(workerId, taskType, error);
    });

    this.eventBus.on('QUEUE:DISCOVERY_ENQUEUED', ({ url, depth }) => {
      // Silent listener
    });
  }
  
  /**
   * Start the orchestration workflow
   * @async
   * @param {string} rootUrl - Root URL to scrape
   * @param {number} maxDepth - Maximum discovery depth
   * @param {boolean} dryRun - If true, only perform discovery phase (no downloads)
   * @returns {Promise<Object>} Scraping result
   */
  async start(rootUrl, maxDepth, dryRun = false) {
    try {
      this.logger.separator('Cluster Orchestrator Starting');

      // Phase 1: Bootstrap
      const bootstrapPhase = new BootstrapPhase(this);
      await bootstrapPhase.execute(rootUrl);

      // Phase 2: Discovery
      const discoveryPhase = new DiscoveryPhase(this);
      await discoveryPhase.execute(maxDepth);

      if (dryRun) {
        this.logger.info('ORCHESTRATOR', 'Dry run mode: skipping download phases');
        const completionPhase = new CompletionPhase(this);
        return await completionPhase.execute(true);
      }

      // Phase 3: User Confirmation
      const userConfPhase = new UserConfirmationPhase(this);
      const userConfirmed = await userConfPhase.execute();
      if (!userConfirmed) {
        const completionPhase = new CompletionPhase(this);
        return await completionPhase.execute(true);
      }

      // Phase 4: Conflict Resolution
      const conflictPhase = new ConflictResolutionPhase(this);
      const { canonicalContexts, linkRewriteMap } = await conflictPhase.execute();

      // Phase 5: Download
      const downloadPhase = new DownloadPhase(this);
      await downloadPhase.execute(canonicalContexts, linkRewriteMap);

      // Phase 6: Completion
      const completionPhase = new CompletionPhase(this);
      return await completionPhase.execute();

    } catch (error) {
      this.logger.error('ORCHESTRATOR', 'Fatal error during orchestration', error);
      throw error;
    }
  }
  /**
   * Handle task completion
   * @private
   * @async
   */
  async _handleTaskComplete(workerId, taskType, result) {
    const { MESSAGE_TYPES } = require('../core/ProtocolDefinitions');

    if (taskType === MESSAGE_TYPES.DISCOVER) {
      const titleRegistry = this.queueManager.getTitleRegistry();
      const displayTitle = result.resolvedTitle || titleRegistry[result.pageId] || 'Untitled';
      this.logger.info('TASK', `✓ Discovered: ${displayTitle}`);

      if (result.cookies !== null && result.cookies !== undefined && this.cookies === null) {
        this.cookies = result.cookies;
        if (this.cookies.length > 0) {
          this.logger.success('TASK', `Captured ${this.cookies.length} cookie(s)`);
        } else {
          this.logger.warn('TASK', 'No cookies captured (possible bot detection)');
        }
      }

      const newContexts = this.queueManager.completeDiscovery(
        result.pageId,
        result.links || [],
        result.metadata,
        result.resolvedTitle
      );

      for (const context of newContexts) {
        this.queueManager.enqueueDiscovery(context, false);
      }

    } else if (taskType === MESSAGE_TYPES.DOWNLOAD) {
      this.logger.info('TASK', `✓ Downloaded: ${result.savedPath}`);
      this.queueManager.markDownloadComplete(result.pageId, result.savedPath);
    }
  }

  /**
   * Handle task failure
   * @private
   * @async
   */
  async _handleTaskFailed(workerId, taskType, error) {
    this.logger.error('TASK', `✗ Task failed: ${error.message}`);

    const { MESSAGE_TYPES } = require('../core/ProtocolDefinitions');

    if (taskType === MESSAGE_TYPES.DISCOVER) {
      this.queueManager.failDiscovery('unknown', new Error(error.message));
    } else if (taskType === MESSAGE_TYPES.DOWNLOAD) {
      this.queueManager.failDownload('unknown', new Error(error.message));
    }
  }
  
  /**
   * Shutdown orchestrator and cleanup resources
   * @async
   */
  async shutdown() {
    this.logger.info('ORCHESTRATOR', 'Shutting down...');
    await this.browserManager.shutdown();
    this.logger.success('ORCHESTRATOR', 'Shutdown complete');
  }
  
  /**
   * Get current status
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      phase: this.currentPhase,
      workers: this.browserManager.getStatistics(),
      queues: this.queueManager.getStatistics()
    };
  }
}

module.exports = { ClusterOrchestrator, Phase };
