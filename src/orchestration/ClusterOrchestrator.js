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
 * 3. User Confirmation: Display site tree and prompt for confirmation
 * 4. Conflict Resolution: Detect duplicates, calculate file paths
 * 5. Download: Parallel download with link rewriting
 * 6. Completion: Statistics and cleanup
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
const UserPrompt = require('../utils/UserPrompt');

/**
 * Orchestration phases
 * @enum {string}
 */
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
        this.logger.info('ORCHESTRATOR', 'Dry run mode: skipping user confirmation, conflict resolution and download phases');
        return await this._phaseComplete(true);
      }
      
      // Phase 3: User Confirmation
      const userConfirmed = await this._phaseUserConfirmation();
      if (!userConfirmed) {
        this.logger.warn('ORCHESTRATOR', 'Download aborted by user.');
        return await this._phaseComplete(true);
      }
      
      // Phase 4: Conflict Resolution
      const { canonicalContexts, linkRewriteMap } = await this._phaseConflictResolution();
      
      // Phase 5: Download
      await this._phaseDownload(canonicalContexts, linkRewriteMap);
      
      // Phase 6: Complete
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
    
    // Initialize all workers with title registry (sent once, eliminates IPC overhead)
    this.logger.info('BOOTSTRAP', 'Initializing workers with title registry...');
    const titleRegistry = this.queueManager.getTitleRegistry();
    await this.browserManager.initializeWorkers(titleRegistry);
    
    this.logger.success('BOOTSTRAP', `Bootstrap complete with ${this.browserManager.getTotalCount()} worker(s)`);
    this.eventBus.emit('BOOTSTRAP:COMPLETE', { 
      workerCount: this.browserManager.getTotalCount() 
    });
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
    
    // Emit phase change for dashboard
    this.eventBus.emit('PHASE:CHANGED', { 
      phase: 'discovery', 
      data: {} 
    });
    
    // Process discovery queue
    while (!this.queueManager.isDiscoveryComplete()) {
      const task = this.queueManager.nextDiscovery();
      
      if (!task) {
        // No tasks available, wait a bit
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }
      
      // Emit discovery progress
      const stats = this.queueManager.getStatistics();
      const queueLength = this.queueManager.discoveryQueue ? this.queueManager.discoveryQueue.length : 0;
      const pendingTasks = stats.pending && stats.pending.discovery ? stats.pending.discovery : 0;
      this.eventBus.emit('DISCOVERY:PROGRESS', {
        pagesFound: stats.discovered,
        inQueue: queueLength + pendingTasks,
        conflicts: stats.conflicts || 0,
        currentDepth: task.pageContext.depth
      });
      
      // Check depth limit
      if (task.pageContext.depth >= maxDepth) {
        this.logger.info('DISCOVERY', `Depth limit reached for: ${task.pageContext.title}`);
        this.queueManager.failDiscovery(task.pageContext.id, new Error('Depth limit reached'));
        continue;
      }
      
      // Execute discovery task
      const workerId = await this.browserManager.execute(MESSAGE_TYPES.DISCOVER, {
        url: task.pageContext.url,
        pageId: task.pageContext.id,
        parentId: task.pageContext.parentId,
        depth: task.pageContext.depth,
        isFirstPage: false,
        cookies: this.cookies
      });
      
      // Emit worker busy event
      const titleRegistry = this.queueManager.getTitleRegistry();
      let title = titleRegistry[task.pageContext.id] || task.pageContext.title || 'Untitled';
      
      // Format raw IDs to be more readable if title resolution failed
      if (title.match(/^[a-f0-9]{32}$/i)) {
        title = `Page ${title.substring(0, 6)}...`;
      }

      this.eventBus.emit('WORKER:BUSY', {
        workerId,
        task: { description: `Discovering '${title}'...` }
      });
    }
    
    // Wait for all pending discoveries to complete
    while (!this.queueManager.isDiscoveryComplete()) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    const stats = this.queueManager.getStatistics();
    this.logger.success('DISCOVERY', `Discovery complete: ${stats.discovered} page(s) discovered`);
    
    // CRITICAL: Update workers with complete title registry after discovery
    this.logger.info('DISCOVERY', 'Updating workers with discovered titles...');
    await this._updateWorkerTitleRegistry();
  }
  
  /**
   * Update all workers with the complete title registry after discovery
   * @private
   * @async
   */
  async _updateWorkerTitleRegistry() {
    const titleRegistry = this.queueManager.getTitleRegistry();
    const titleCount = Object.keys(titleRegistry).length;
    
    this.logger.info('REGISTRY', `Broadcasting ${titleCount} title(s) to workers...`);
    
    // Re-initialize all workers with the complete title registry
    await this.browserManager.initializeWorkers(titleRegistry);
    
    this.logger.success('REGISTRY', 'Workers updated with complete title registry');
  }
  
  /**
   * Phase 3: User Confirmation - Display site tree and prompt for confirmation
   * @private
   * @async
   * @returns {Promise<boolean>} True to proceed, false to abort
   */
  async _phaseUserConfirmation() {
    this.currentPhase = Phase.USER_CONFIRMATION;
    
    // Signal to stop dashboard and switch back to console mode
    this.eventBus.emit('PHASE:STOPPING_DASHBOARD', {});
    
    // Allow dashboard to stop and terminal to clear
    await new Promise(resolve => setTimeout(resolve, 100));
    
    this.logger.separator('Phase 3: User Confirmation');
    
    // Get discovery statistics
    const stats = this.queueManager.getStatistics();
    
    // Handle edge case: no pages discovered
    if (stats.discovered === 0) {
      this.logger.warn('USER_CONFIRMATION', 'No pages discovered. Aborting.');
      return false;
    }
    
    // Display site tree
    this.logger.info('USER_CONFIRMATION', 'Discovered Site Structure:');
    const allContexts = this.queueManager.getAllContexts();
    const rootContext = allContexts.find(ctx => ctx.depth === 0);
    if (rootContext) {
      this._displayPageTree(rootContext);
    }
    
    // Display summary statistics
    this.logger.info('USER_CONFIRMATION', '\nDiscovery Summary:');
    this.logger.info('USER_CONFIRMATION', `  Total Pages: ${stats.discovered}`);
    this.logger.info('USER_CONFIRMATION', `  Maximum Depth: ${this.queueManager.getMaxDepth()}`);
    
    // Prompt user for confirmation
    const prompt = new UserPrompt();
    const proceed = await prompt.promptConfirmDownload({
      totalPages: stats.discovered,
      maxDepth: this.queueManager.getMaxDepth()
    });
    prompt.close();
    
    if (proceed) {
      this.logger.success('USER_CONFIRMATION', 'User confirmed. Proceeding to download phase...');
    } else {
      this.logger.warn('USER_CONFIRMATION', 'User declined. Aborting download process.');
    }
    
    return proceed;
  }
  
  /**
   * Phase 4: Conflict Resolution - Detect duplicates and generate paths
   * @private
   * @async
   * @returns {Object} Resolution result
   */
  async _phaseConflictResolution() {
    this.currentPhase = Phase.CONFLICT_RESOLUTION;
    this.logger.separator('Phase 4: Conflict Resolution');
    
    const allContexts = this.queueManager.getAllContexts();
    const titleRegistry = this.queueManager.getTitleRegistry();
    const { canonicalContexts, linkRewriteMap, stats } = ConflictResolver.resolve(allContexts, titleRegistry);
    
    this.linkRewriteMap = linkRewriteMap;
    
    this.logger.success('CONFLICT_RESOLUTION', 
      `Resolved ${stats.uniquePages} unique page(s) (${stats.duplicates} duplicate(s) removed)`);
    
    return { canonicalContexts, linkRewriteMap };
  }
  
  /**
   * Phase 5: Download - Parallel download with link rewriting
   * @private
   * @async
   * @param {Array<PageContext>} canonicalContexts - Contexts to download
   * @param {Map} linkRewriteMap - NotionID -> FilePath map
   */
  async _phaseDownload(canonicalContexts, linkRewriteMap) {
    this.currentPhase = Phase.DOWNLOAD;
    this.logger.separator('Phase 5: Download');
    
    // Emit phase change for dashboard
    this.eventBus.emit('PHASE:CHANGED', { 
      phase: 'download', 
      data: { total: canonicalContexts.length } 
    });
    
    // Build download queue
    this.queueManager.buildDownloadQueue(canonicalContexts);
    
    // Process download queue
    while (!this.queueManager.isDownloadComplete()) {
      const downloadTask = this.queueManager.nextDownload(this.config.OUTPUT_DIR);
      
      if (!downloadTask) {
        // No tasks available, wait a bit
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }
      
      // Emit execution progress
      const stats = this.queueManager.getStatistics();
      this.eventBus.emit('EXECUTION:PROGRESS', {
        pending: stats.pending.download || 0,
        active: this.browserManager.getAllocatedCount(),
        completed: stats.downloaded,
        total: canonicalContexts.length,
        failed: stats.failed
      });
      
      const { context, savePath } = downloadTask;
      
      // Execute download task with absolute savePath
      const workerId = await this.browserManager.execute(MESSAGE_TYPES.DOWNLOAD, {
        url: context.url,
        pageId: context.id,
        parentId: context.parentId,
        depth: context.depth,
        savePath: savePath, // Absolute path from GlobalQueueManager
        cookies: this.cookies,
        linkRewriteMap: Object.fromEntries(linkRewriteMap)
      });
      
      // Emit worker busy event
      const titleRegistry = this.queueManager.getTitleRegistry();
      let title = titleRegistry[context.id] || context.title || 'Untitled';
      
      // Format raw IDs to be more readable
      if (title.match(/^[a-f0-9]{32}$/i)) {
        title = `Page ${title.substring(0, 6)}...`;
      }

      this.eventBus.emit('WORKER:BUSY', {
        workerId,
        task: { description: `Downloading '${title}'...` }
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
   * Phase 6: Complete - Cleanup and statistics
   * @private
   * @async
   * @param {boolean} aborted - Whether the process was aborted (dry run or user declined)
   * @returns {Object} Final result
   */
  async _phaseComplete(aborted = false) {
    this.currentPhase = Phase.COMPLETE;
    
    if (aborted) {
      this.logger.separator('Phase 6: Complete (Aborted)');
    } else {
      this.logger.separator('Phase 6: Complete');
    }
    
    const stats = this.queueManager.getStatistics();
    
    if (aborted) {
      this.logger.info('COMPLETE', 'Scraping process aborted. Cleaning up...');
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
   * Recursively print tree nodes
   * @private
   * @param {PageContext} context - Page context
   * @param {string} prefix - Line prefix
   * @param {boolean} isLast - Is this the last child?
   * @param {Object} titleRegistry - ID-to-title map
   */
  _printTreeNode(context, prefix, isLast, titleRegistry) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = titleRegistry[context.id] || context.title || 'Untitled';
    
    // Filter explored children: those with titles in registry
    const exploredChildren = context.children.filter(child => titleRegistry[child.id]);
    const internalRefs = context.children.length - exploredChildren.length;
    
    // Display title with internal reference count if any
    const label = internalRefs > 0 ? `${title} [${internalRefs} internal reference${internalRefs > 1 ? 's' : ''}]` : title;
    console.log(`${prefix}${connector}${label}`);
    
    // Only recurse into explored children (those with resolved names)
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    exploredChildren.forEach((child, index) => {
      const childIsLast = index === exploredChildren.length - 1;
      this._printTreeNode(child, childPrefix, childIsLast, titleRegistry);
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
      const titleRegistry = this.queueManager.getTitleRegistry();
      const displayTitle = result.resolvedTitle || titleRegistry[result.pageId] || 'Untitled';
      this.logger.info('TASK', `✓ Discovered: ${displayTitle}`);
      
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
        result.resolvedTitle  // Pass resolved title to registry
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
