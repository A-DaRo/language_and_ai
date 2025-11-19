/**
 * @fileoverview Master State Machine
 * @module orchestration/ClusterOrchestrator
 * @description Coordinates the entire distributed scraping workflow.
 * Acts as the "conductor" orchestrating workers, queues, and phases.
 * 
 * **CRITICAL CONTEXT**: Runs in MASTER process. This is the brain of the system.
 * 
 * Workflow Phases:
 * 1. Bootstrap: Spawn workers, capture cookies from first page
 * 2. Discovery: Parallel discovery of all pages (metadata only)
 * 3. Conflict Resolution: Detect duplicates, calculate file paths
 * 4. Download: Parallel download with link rewriting
 * 5. Completion: Statistics and cleanup
 */

const { MESSAGE_TYPES } = require('../core/ProtocolDefinitions');
const SystemEventBus = require('../core/SystemEventBus');
const BrowserInitializer = require('../cluster/BrowserInitializer');
const BrowserManager = require('../cluster/BrowserManager');
const GlobalQueueManager = require('./GlobalQueueManager');
const ConflictResolver = require('./analysis/ConflictResolver');
const PageContext = require('../domain/PageContext');
const Logger = require('../core/Logger');
const Config = require('../core/Config');

/**
 * Orchestration phases
 * @enum {string}
 */
const Phase = {
  BOOTSTRAP: 'BOOTSTRAP',
  DISCOVERY: 'DISCOVERY',
  CONFLICT_RESOLUTION: 'CONFLICT_RESOLUTION',
  DOWNLOAD: 'DOWNLOAD',
  COMPLETE: 'COMPLETE'
};

/**
 * @class ClusterOrchestrator
 * @classdesc Main state machine coordinating the distributed scraping workflow
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
    // Task completed successfully
    this.eventBus.on('TASK:COMPLETE', async ({ workerId, taskType, result }) => {
      await this._handleTaskComplete(workerId, taskType, result);
    });
    
    // Task failed
    this.eventBus.on('TASK:FAILED', async ({ workerId, taskType, error }) => {
      await this._handleTaskFailed(workerId, taskType, error);
    });
    
    // Queue events (silent listener to prevent warnings)
    this.eventBus.on('QUEUE:DISCOVERY_ENQUEUED', ({ url, depth }) => {
      // Silent listener - no action needed
    });
  }
  
  /**
   * Start the orchestration workflow
   * @async
   * @param {string} rootUrl - Root URL to scrape
   * @param {number} maxDepth - Maximum discovery depth
   * @param {boolean} dryRun - If true, only perform discovery phase (no downloads)
   * @returns {Promise<Object>} Scraping result
   * @example
   * const result = await orchestrator.start('https://notion.so/page', 3);
   * console.log(`Downloaded ${result.stats.downloaded} pages`);
   */
  async start(rootUrl, maxDepth, dryRun = false) {
    try {
      this.logger.separator('Cluster Orchestrator Starting');
      
      // Phase 1: Bootstrap
      await this._phaseBootstrap(rootUrl);
      
      // Phase 2: Discovery
      await this._phaseDiscovery(rootUrl, maxDepth);
      
      if (dryRun) {
        // Dry run mode: skip download phases
        this.logger.info('ORCHESTRATOR', 'Dry run mode: skipping conflict resolution and download phases');
        return await this._phaseComplete(true);
      }
      
      // Phase 3: Conflict Resolution
      const { canonicalContexts, linkRewriteMap } = await this._phaseConflictResolution();
      
      // Phase 4: Download
      await this._phaseDownload(canonicalContexts, linkRewriteMap);
      
      // Phase 5: Complete
      return await this._phaseComplete();
      
    } catch (error) {
      this.logger.error('ORCHESTRATOR', 'Fatal error during orchestration', error);
      throw error;
    }
  }
  
  /**
   * Phase 1: Bootstrap - Spawn workers and capture cookies
   * @private
   * @async
   * @param {string} rootUrl - Root URL
   */
  async _phaseBootstrap(rootUrl) {
    this.currentPhase = Phase.BOOTSTRAP;
    this.logger.separator('Phase 1: Bootstrap');
    
    // Calculate capacity
    const capacity = BrowserInitializer.calculateCapacity(1, 8);
    this.logger.info('BOOTSTRAP', `System capacity: ${capacity.workerCount} worker(s) (${capacity.freeMemoryMB}MB free)`);
    
    // Spawn initial worker for cookie capture (to avoid race condition)
    this.logger.info('BOOTSTRAP', 'Spawning bootstrap worker for cookie capture...');
    const bootstrapWorker = await BrowserInitializer.spawnWorkerPool(1);
    this.browserManager.registerWorkers(bootstrapWorker);
    
    // Wait for worker to be ready
    await this._waitForWorkers(1);
    
    // Convert root URL to raw ID format for consistency with queue management
    // Step 1: Remove everything after '?' (query params)
    const cleanRootUrl = rootUrl.split('?')[0];
    
    // Step 2: Extract raw page ID and reconstruct URL
    const rootRawIdMatch = cleanRootUrl.match(/29[a-f0-9]{30}/i);
    const normalizedRootUrl = rootRawIdMatch 
      ? this.config.getBaseUrl() + '/' + rootRawIdMatch[0]
      : rootUrl;  // Fallback to original if no ID found
    
    // Extract page name from URL (will be resolved after first discovery)
    const rootTitle = this.config.extractPageNameFromUrl(rootUrl) || 'Main_Page';
    const rootContext = new PageContext(normalizedRootUrl, rootTitle, 0, null, null);
    this.queueManager.enqueueDiscovery(rootContext, true);
    
    // Execute first discovery task to capture cookies
    this.logger.info('BOOTSTRAP', 'Capturing cookies from root page...');
    const task = this.queueManager.nextDiscovery();
    
    await this.browserManager.execute(MESSAGE_TYPES.DISCOVER, {
      url: task.pageContext.url,
      pageId: task.pageContext.id,
      parentId: task.pageContext.parentId,
      depth: task.pageContext.depth,
      isFirstPage: true
    });
    
    // Wait for cookies to be captured
    await this._waitForCookies();
    
    // Now spawn remaining workers
    if (capacity.workerCount > 1) {
      this.logger.info('BOOTSTRAP', `Spawning ${capacity.workerCount - 1} additional worker(s)...`);
      const additionalWorkers = await BrowserInitializer.spawnWorkerPool(capacity.workerCount - 1, 2);
      this.browserManager.registerWorkers(additionalWorkers);
      
      // Wait for all workers to be ready
      await this._waitForWorkers(capacity.workerCount);
    }
    
    // Broadcast cookies to all workers
    if (this.cookies && this.cookies.length > 0) {
      this.logger.info('BOOTSTRAP', `Broadcasting ${this.cookies.length} cookie(s) to all workers`);
      await this.browserManager.broadcastCookies(this.cookies);
    }
    
    this.logger.success('BOOTSTRAP', `Bootstrap complete with ${this.browserManager.getTotalCount()} worker(s)`);
  }
  
  /**
   * Phase 2: Discovery - Parallel page discovery
   * @private
   * @async
   * @param {string} rootUrl - Root URL
   * @param {number} maxDepth - Maximum depth
   */
  async _phaseDiscovery(rootUrl, maxDepth) {
    this.currentPhase = Phase.DISCOVERY;
    this.logger.separator('Phase 2: Discovery');
    
    // Process discovery queue
    while (!this.queueManager.isDiscoveryComplete()) {
      const task = this.queueManager.nextDiscovery();
      
      if (!task) {
        // No tasks available, wait a bit
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }
      
      // Check depth limit
      if (task.pageContext.depth >= maxDepth) {
        this.logger.info('DISCOVERY', `Depth limit reached for: ${task.pageContext.title}`);
        this.queueManager.failDiscovery(task.pageContext.id, new Error('Depth limit reached'));
        continue;
      }
      
      // Execute discovery task
      await this.browserManager.execute(MESSAGE_TYPES.DISCOVER, {
        url: task.pageContext.url,
        pageId: task.pageContext.id,
        parentId: task.pageContext.parentId,
        depth: task.pageContext.depth,
        isFirstPage: false,
        cookies: this.cookies
      });
    }
    
    // Wait for all pending discoveries to complete
    while (!this.queueManager.isDiscoveryComplete()) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    const stats = this.queueManager.getStatistics();
    this.logger.success('DISCOVERY', `Discovery complete: ${stats.discovered} page(s) discovered`);
  }
  
  /**
   * Phase 3: Conflict Resolution - Detect duplicates and generate paths
   * @private
   * @async
   * @returns {Object} Resolution result
   */
  async _phaseConflictResolution() {
    this.currentPhase = Phase.CONFLICT_RESOLUTION;
    this.logger.separator('Phase 3: Conflict Resolution');
    
    const allContexts = this.queueManager.getAllContexts();
    const { canonicalContexts, linkRewriteMap, stats } = ConflictResolver.resolve(allContexts);
    
    this.linkRewriteMap = linkRewriteMap;
    
    this.logger.success('CONFLICT_RESOLUTION', 
      `Resolved ${stats.uniquePages} unique page(s) (${stats.duplicates} duplicate(s) removed)`);
    
    return { canonicalContexts, linkRewriteMap };
  }
  
  /**
   * Phase 4: Download - Parallel download with link rewriting
   * @private
   * @async
   * @param {Array<PageContext>} canonicalContexts - Contexts to download
   * @param {Map} linkRewriteMap - NotionID -> FilePath map
   */
  async _phaseDownload(canonicalContexts, linkRewriteMap) {
    this.currentPhase = Phase.DOWNLOAD;
    this.logger.separator('Phase 4: Download');
    
    // Build download queue
    this.queueManager.buildDownloadQueue(canonicalContexts);
    
    // Process download queue
    while (!this.queueManager.isDownloadComplete()) {
      const context = this.queueManager.nextDownload();
      
      if (!context) {
        // No tasks available, wait a bit
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }
      
      // Execute download task
      await this.browserManager.execute(MESSAGE_TYPES.DOWNLOAD, {
        url: context.url,
        pageId: context.id,
        parentId: context.parentId,
        depth: context.depth,
        targetFilePath: context.targetFilePath,
        cookies: this.cookies,
        linkRewriteMap: Object.fromEntries(linkRewriteMap)
      });
    }
    
    // Wait for all pending downloads to complete
    while (!this.queueManager.isDownloadComplete()) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    const stats = this.queueManager.getStatistics();
    this.logger.success('DOWNLOAD', `Download complete: ${stats.downloaded} page(s) downloaded`);
  }
  
  /**
   * Phase 5: Complete - Cleanup and statistics
   * @private
   * @async
   * @param {boolean} isDryRun - Whether this was a dry run
   * @returns {Object} Final result
   */
  async _phaseComplete(isDryRun = false) {
    this.currentPhase = Phase.COMPLETE;
    
    if (isDryRun) {
      this.logger.separator('Phase 3: Discovery Complete (Dry Run)');
    } else {
      this.logger.separator('Phase 5: Complete');
    }
    
    const stats = this.queueManager.getStatistics();
    
    if (isDryRun) {
      this.logger.success('COMPLETE', 'Discovery phase completed successfully (dry run mode)');
      this.logger.info('STATS', `Discovered: ${stats.discovered} pages`);
      
      // Display page tree
      const allContexts = this.queueManager.getAllContexts();
      const rootContext = allContexts.find(ctx => ctx.depth === 0);
      if (rootContext) {
        this._displayPageTree(rootContext);
      }
    } else {
      this.logger.success('COMPLETE', 'All operations completed successfully');
      this.logger.info('STATS', `Discovered: ${stats.discovered} | Downloaded: ${stats.downloaded} | Failed: ${stats.failed}`);
    }
    
    return {
      stats,
      allContexts: this.queueManager.getAllContexts()
    };
  }
  
  /**
   * Display page tree hierarchy
   * @private
   * @param {PageContext} rootContext - Root page context
   */
  _displayPageTree(rootContext) {
    this.logger.separator('Page Tree');
    console.log('.');
    const rootLabel = rootContext.displayTitle || rootContext.title || '(root)';
    console.log(`└─ ${rootLabel}`);
    
    rootContext.children.forEach((child, index) => {
      const isLast = index === rootContext.children.length - 1;
      this._printTreeNode(child, '   ', isLast);
    });
    
    this.logger.separator();
  }
  
  /**
   * Recursively print tree nodes
   * @private
   * @param {PageContext} context - Page context
   * @param {string} prefix - Line prefix
   * @param {boolean} isLast - Is this the last child?
   */
  _printTreeNode(context, prefix, isLast) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = context.displayTitle || context.title || 'Untitled';
    
    // Filter explored children: those with displayTitle set (resolved after discovery)
    // Unexplored children have raw IDs as titles (29abc... format)
    const exploredChildren = context.children.filter(child => child.displayTitle);
    const internalRefs = context.children.length - exploredChildren.length;
    
    // Display title with internal reference count if any
    const label = internalRefs > 0 ? `${title} [${internalRefs} internal reference${internalRefs > 1 ? 's' : ''}]` : title;
    console.log(`${prefix}${connector}${label}`);
    
    // Only recurse into explored children (those with resolved names)
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    exploredChildren.forEach((child, index) => {
      const childIsLast = index === exploredChildren.length - 1;
      this._printTreeNode(child, childPrefix, childIsLast);
    });
  }
  
  /**
   * Handle task completion
   * @private
   * @async
   */
  async _handleTaskComplete(workerId, taskType, result) {
    if (taskType === MESSAGE_TYPES.DISCOVER) {
      // Discovery task completed
      this.logger.info('TASK', `✓ Discovered: ${result.title}`);
      
      // Capture cookies from first page (even if empty)
      if (result.cookies !== null && result.cookies !== undefined && this.cookies === null) {
        this.cookies = result.cookies;
        if (this.cookies.length > 0) {
          this.logger.success('TASK', `Captured ${this.cookies.length} cookie(s) from first page`);
        } else {
          this.logger.warn('TASK', 'No cookies captured from first page (possible bot detection)');
        }
      }
      
      // Process discovered links and enqueue children
      const newContexts = this.queueManager.completeDiscovery(
        result.pageId,
        result.links || [],
        result.metadata,
        result.title  // Pass resolved title
      );
      
      // Enqueue new discoveries
      for (const context of newContexts) {
        this.queueManager.enqueueDiscovery(context, false);
      }
      
    } else if (taskType === MESSAGE_TYPES.DOWNLOAD) {
      // Download task completed
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
    this.logger.error('TASK', `✗ Task failed on ${workerId}: ${error.message}`);
    
    if (taskType === MESSAGE_TYPES.DISCOVER) {
      // Mark discovery as failed (pageId not available in error case)
      this.queueManager.failDiscovery('unknown', new Error(error.message));
    } else if (taskType === MESSAGE_TYPES.DOWNLOAD) {
      this.queueManager.failDownload('unknown', new Error(error.message));
    }
  }
  
  /**
   * Wait for workers to become ready
   * @private
   * @async
   * @param {number} count - Number of workers to wait for
   */
  async _waitForWorkers(count) {
    const startTime = Date.now();
    const timeout = 60000; // 60 seconds
    
    while (this.browserManager.getTotalCount() < count) {
      if (Date.now() - startTime > timeout) {
        throw new Error('Timeout waiting for workers to become ready');
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  /**
   * Wait for cookies to be captured
   * @private
   * @async
   */
  async _waitForCookies() {
    const startTime = Date.now();
    const timeout = 60000; // 60 seconds
    
    while (!this.cookies) {
      if (Date.now() - startTime > timeout) {
        this.logger.warn('BOOTSTRAP', 'Timeout waiting for cookies, proceeding without authentication');
        this.cookies = [];
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
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
