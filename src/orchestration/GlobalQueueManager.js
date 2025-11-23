/**
 * @fileoverview Centralized Task Queue Manager (Facade)
 * @module orchestration/GlobalQueueManager
 * @description Lightweight facade delegating to specialized queue and registry components.
 * 
 * **CRITICAL CONTEXT**: Runs in MASTER process. Coordinates queue lifecycle.
 */

const DiscoveryQueue = require('./queues/DiscoveryQueue');
const ExecutionQueue = require('./queues/ExecutionQueue');
const TitleRegistry = require('./queues/TitleRegistry');
const PageContext = require('../domain/PageContext');
const Logger = require('../core/Logger');

/**
 * @class GlobalQueueManager
 * @classdesc Facade for discovery, execution, and title management
 */
class GlobalQueueManager {
  constructor() {
    this.logger = Logger.getInstance();

    // Delegate to specialized components
    this.discoveryQueue = new DiscoveryQueue();
    this.executionQueue = new ExecutionQueue();
    this.titleRegistry = new TitleRegistry();

    // Context registry
    this.allContexts = new Map();

    // Statistics
    this.stats = {
      discovered: 0,
      downloaded: 0,
      failed: 0
    };
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
    const task = this.discoveryQueue.next();

    if (task) {
      this.discoveryQueue.markComplete(task.pageContext.id);
    }

    return task;
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

    if (resolvedTitle && !this.titleRegistry.has(pageId)) {
      this.titleRegistry.register(pageId, resolvedTitle);
      parentContext.updateTitleFromRegistry(resolvedTitle);
    }

    const newContexts = [];

    for (const link of discoveredLinks) {
      if (this.discoveryQueue.visitedUrls.has(link.url)) {
        continue;
      }

      const childContext = new PageContext(
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

      parentContext.addChild(childContext);
      newContexts.push(childContext);
    }

    return newContexts;
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
   * Build download queue from discovered contexts
   * @param {Array<PageContext>} contexts - All discovered contexts
   * @returns {number} Number of pages queued for download
   */
  buildDownloadQueue(contexts) {
    this.executionQueue.build(contexts);
    return this.executionQueue.getTotalCount();
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
    this.allContexts.clear();
    this.stats = { discovered: 0, downloaded: 0, failed: 0 };
  }
}

module.exports = GlobalQueueManager;

module.exports = GlobalQueueManager;
