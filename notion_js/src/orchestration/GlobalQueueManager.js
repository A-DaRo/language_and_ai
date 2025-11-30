/**
 * @fileoverview Centralized Task Queue Manager (Facade)
 * @module orchestration/GlobalQueueManager
 * @description Lightweight facade delegating to specialized queue and registry components.
 * 
 * **CRITICAL CONTEXT**: Runs in MASTER process. Coordinates queue lifecycle.
 * 
 * **Deadlock Prevention Features:**
 * - Leaf-first download ordering (Solution D)
 * - Global hidden file registry (Solution E)
 */

const DiscoveryQueue = require('./queues/DiscoveryQueue');
const ExecutionQueue = require('./queues/ExecutionQueue');
const TitleRegistry = require('./queues/TitleRegistry');
const PageGraph = require('./PageGraph');
const EdgeClassifier = require('./analysis/EdgeClassifier');
const GlobalHiddenFileRegistry = require('./GlobalHiddenFileRegistry');
const PageContext = require('../domain/PageContext');
const Logger = require('../core/Logger');

/**
 * @class GlobalQueueManager
 * @classdesc Facade for discovery, execution, and title management with deadlock prevention.
 */
class GlobalQueueManager {
  constructor() {
    this.logger = Logger.getInstance();

    // Delegate to specialized components
    this.discoveryQueue = new DiscoveryQueue();
    this.executionQueue = new ExecutionQueue();
    this.titleRegistry = new TitleRegistry();

    // Page graph for tracking hierarchy and edge classifications
    this.pageGraph = new PageGraph();
    this.edgeClassifier = new EdgeClassifier();
    
    // Global hidden file registry for cross-page deduplication (Solution E)
    this.hiddenFileRegistry = new GlobalHiddenFileRegistry();

    // Context registry
    this.allContexts = new Map();

    // Statistics
    this.stats = {
      discovered: 0,
      downloaded: 0,
      failed: 0
    };
    
    /**
     * Configuration for download queue logging
     * @type {{logDir: string}}
     */
    this.downloadConfig = {
      logDir: './logs'
    };
  }

  /**
   * Configure download queue behavior.
   * 
   * @param {Object} config - Configuration options
   * @param {string} [config.logDir='./logs'] - Directory for queue logs
   */
  configureDownload(config) {
    this.downloadConfig = { ...this.downloadConfig, ...config };
    this.logger.info('GlobalQueueManager', 
      `Download configured: logDir=${this.downloadConfig.logDir}`);
  }

  /**
   * Enqueue a page for discovery
   * @param {PageContext} pageContext - Page context to discover
   * @param {boolean} [isFirstPage=false] - Whether this is the root page
   * @returns {boolean} True if enqueued, false if already visited
   */
  enqueueDiscovery(pageContext, isFirstPage = false) {
    const enqueued = this.discoveryQueue.enqueue(pageContext, isFirstPage);

    if (enqueued) {
      this.allContexts.set(pageContext.id, pageContext);
    }

    return enqueued;
  }

  /**
   * Get next discovery task from queue
   * @returns {Object|null} Next task or null if queue is empty
   */
  nextDiscovery() {
    return this.discoveryQueue.next();
  }

  /**
   * Mark a discovery task as complete and process discovered links
   * @param {string} pageId - ID of the discovered page
   * @param {Array<Object>} discoveredLinks - Array of discovered links
   * @param {Object} [metadata] - Additional metadata
   * @param {string} [resolvedTitle] - Resolved page title from server
   * @returns {Array<PageContext>} New page contexts created from links
   */
  completeDiscovery(pageId, discoveredLinks, metadata = {}, resolvedTitle = null) {
    this.discoveryQueue.markComplete(pageId);
    this.stats.discovered++;

    const parentContext = this.allContexts.get(pageId);
    if (!parentContext) {
      this.logger.warn('GlobalQueueManager', `Parent context not found: ${pageId}`);
      return [];
    }

    // Add parent node to page graph
    this.pageGraph.addNode(pageId, parentContext);

    if (resolvedTitle) {
      this.titleRegistry.register(pageId, resolvedTitle);
      parentContext.updateTitleFromRegistry(resolvedTitle);
    }

    const newContexts = [];
    this.edgeClassifier.setContextMap(this.allContexts);

    for (const link of discoveredLinks) {
      if (this.discoveryQueue.visitedUrls.has(link.url)) {
        continue;
      }

      const childId = this._derivePageId(link.url);
      let childContext = this.allContexts.get(childId);
      const isNewContext = !childContext;

      if (isNewContext) {
        childContext = new PageContext(
          link.url,
          link.text || 'Untitled',
          parentContext.depth + 1,
          parentContext,
          parentContext.id
        );

        if (link.section) {
          childContext.setSection(link.section);
        }
        if (link.subsection) {
          childContext.setSubsection(link.subsection);
        }

        this.allContexts.set(childId, childContext);
        newContexts.push(childContext);
      }

      // Ensure parent-child relation exists even for previously seen contexts
      parentContext.addChild(childContext);

      const classification = this.edgeClassifier.classifyEdge(parentContext, childContext);
      this.pageGraph.addEdge(pageId, childContext.id, classification);
      this.pageGraph.addNode(childContext.id, childContext);

      this.logger.debug(
        'EDGE-CLASSIFICATION',
        `${parentContext.title} -> ${childContext.title}: ${classification.type}`
      );
    }

    return newContexts;
  }

  /**
   * Derive a canonical page ID from a Notion URL
   * @private
   * @param {string} url - Notion URL to normalize
   * @returns {string} Extracted page ID or original URL
   */
  _derivePageId(url) {
    try {
      const { normalizedUrl, normalizedPath } = this._normalizeUrlParts(url);
      const match = normalizedPath.match(/([a-f0-9]{32})$/i);
      return match ? match[1] : normalizedUrl;
    } catch (error) {
      this.logger.debug('GlobalQueueManager', `Failed to normalize URL for ID: ${url} (${error.message})`);
      return url;
    }
  }

  _normalizeUrlParts(url) {
    const urlObj = new URL(url);
    let cleanPath = (urlObj.pathname || '').replace(/\/$/, '');
    if (!cleanPath) {
      cleanPath = '/';
    }
    const normalizedUrl = `${urlObj.host}${cleanPath}`.toLowerCase();
    return {
      normalizedUrl,
      normalizedPath: cleanPath.toLowerCase()
    };
  }

  /**
   * Mark a discovery task as failed
   * @param {string} pageId - ID of the failed page
   * @param {Error} error - Error that occurred
   */
  failDiscovery(pageId, error) {
    this.discoveryQueue.markFailed(pageId);
    this.stats.failed++;
    this.logger.error('GlobalQueueManager', `Discovery failed for ${pageId}: ${error.message}`, error);
  }

  /**
   * Build download queue from discovered contexts with configurable ordering strategy.
   * 
   * **Deadlock Prevention (Solution D):**
   * Uses leaf-first ordering which processes deepest pages first.
   * This ensures hidden files are discovered on leaf pages before their parents,
   * enabling natural deduplication across the tree.
   * 
   * @param {Array<PageContext>} contexts - All discovered contexts
   * @returns {number} Number of pages queued for download
   */
  buildDownloadQueue(contexts) {
    // Build queue with leaf-first ordering
    this.executionQueue.build(contexts, this.pageGraph);
    
    this.logger.info('GlobalQueueManager', 
      `Download queue built with leaf-first ordering: ${this.executionQueue.getTotalCount()} pages`);
    
    return this.executionQueue.getTotalCount();
  }

  /**
   * Get the hidden file registry for cross-page deduplication.
   * Workers should query this before processing hidden files.
   * 
   * @returns {GlobalHiddenFileRegistry} The hidden file registry
   */
  getHiddenFileRegistry() {
    return this.hiddenFileRegistry;
  }

  /**
   * Get next download task from queue with absolute path calculation
   * @param {string} outputDir - The system's root output directory (absolute)
   * @returns {Object|null} Download payload with absolute savePath or null
   */
  nextDownload(outputDir) {
    return this.executionQueue.next(outputDir);
  }

  /**
   * Mark a download task as complete
   * @param {string} pageId - ID of the downloaded page
   * @param {string} savedPath - Path where the file was saved
   */
  markDownloadComplete(pageId, savedPath) {
    this.executionQueue.markComplete(pageId, savedPath);
    this.stats.downloaded++;
  }

  /**
   * Mark a download task as failed
   * @param {string} pageId - ID of the failed page
   * @param {Error} error - Error that occurred
   */
  failDownload(pageId, error) {
    this.executionQueue.markFailed(pageId, error);
    this.stats.failed++;
    this.logger.error('GlobalQueueManager', `Download failed for ${pageId}: ${error.message}`, error);
  }

  /**
   * Check if discovery phase is complete
   * @returns {boolean} True if all discovery tasks are done
   */
  isDiscoveryComplete() {
    return this.discoveryQueue.isComplete();
  }

  /**
   * Check if download phase is complete
   * @returns {boolean} True if all download tasks are done
   */
  isDownloadComplete() {
    return this.executionQueue.isComplete();
  }

  /**
   * Get all discovered contexts
   * @returns {Array<PageContext>} Array of all page contexts
   */
  getAllContexts() {
    return Array.from(this.allContexts.values());
  }

  /**
   * Get title registry as serializable object for IPC
   * @returns {Object} Plain object representation of ID-to-title map
   */
  getTitleRegistry() {
    return this.titleRegistry.serialize();
  }

  /**
   * Get page graph built during discovery
   * @returns {PageGraph} The page graph with all discovered pages and edges
   */
  getPageGraph() {
    return this.pageGraph;
  }

  /**
   * Get maximum depth discovered
   * @returns {number} Maximum depth (0 if no pages discovered)
   */
  getMaxDepth() {
    return this.discoveryQueue.getMaxDepth();
  }

  /**
   * Get statistics
   * @returns {Object} Current statistics
   */
  getStatistics() {
    return {
      discovered: this.stats.discovered,
      downloaded: this.stats.downloaded,
      failed: this.stats.failed,
      pending: {
        discovery: this.discoveryQueue.getPendingCount(),
        download: this.executionQueue.getLength()
      },
      total: {
        contexts: this.allContexts.size,
        visited: this.discoveryQueue.getVisitedCount()
      }
    };
  }

  /**
   * Reset all queues and state (for testing or restart)
   */
  reset() {
    this.discoveryQueue.reset();
    this.executionQueue.reset();
    this.titleRegistry = new TitleRegistry();
    this.pageGraph = new PageGraph();
    this.edgeClassifier = new EdgeClassifier();
    this.hiddenFileRegistry.reset();
    this.allContexts.clear();
    this.stats = { discovered: 0, downloaded: 0, failed: 0 };
  }

  /**
   * Get combined statistics including hidden file registry.
   * @returns {Object} Extended statistics
   */
  getExtendedStatistics() {
    return {
      ...this.getStatistics(),
      hiddenFiles: this.hiddenFileRegistry.getStatistics()
    };
  }
}

module.exports = GlobalQueueManager;
