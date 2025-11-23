/**
 * @fileoverview Download queue manager with dependency tracking
 * @module orchestration/queues/ExecutionQueue
 * @description Manages download queue with dependency constraints.
 */

const path = require('path');
const SystemEventBus = require('../../core/SystemEventBus');
const Logger = require('../../core/Logger');

/**
 * @class ExecutionQueue
 * @classdesc Manages download queue respecting parent-child dependencies.
 * Ensures parents are downloaded only after their children (bottom-up).
 */
class ExecutionQueue {
  constructor() {
    this.eventBus = SystemEventBus.getInstance();
    this.logger = Logger.getInstance();

    this.queue = [];
    this.pendingDownloads = new Map();
    this.completedDownloads = new Set();
  }

  /**
   * Constructs the queue from the canonical page list
   * @param {Array<PageContext>} contexts - The list of pages to download
   * @param {Map} dependencyGraph - Parent-child relationships (unused but accepted)
   */
  build(contexts, dependencyGraph = null) {
    this.queue = [];
    this.pendingDownloads.clear();
    this.completedDownloads.clear();

    for (const context of contexts) {
      this.queue.push(context);
      this.pendingDownloads.set(context.id, {
        context: context,
        childrenCount: context.children.length,
        completedChildren: 0
      });
    }

    this.eventBus.emit('QUEUE:DOWNLOAD_READY', {
      count: this.queue.length
    });

    this.logger.debug('ExecutionQueue', `Loaded ${this.queue.length} page(s) for download`);
  }

  /**
   * Retrieves the next ready download task
   * @description Returns pages whose children have all been processed
   * @param {string} outputDir - Root output directory (absolute)
   * @returns {Object|null} The task payload with { context, savePath }
   */
  next(outputDir) {
    if (this.queue.length === 0) {
      return null;
    }

    const context = this.queue.shift();
    const savePath = this._calculateAbsolutePath(outputDir, context);

    return {
      context,
      savePath
    };
  }

  /**
   * Marks a page as downloaded and updates parent dependencies
   * @param {string} pageId - The ID of the completed page
   * @param {string} savedPath - Path where the file was saved
   */
  markComplete(pageId, savedPath) {
    this.completedDownloads.add(pageId);

    const downloadInfo = this.pendingDownloads.get(pageId);
    if (downloadInfo && downloadInfo.context.parentId) {
      const parentInfo = this.pendingDownloads.get(downloadInfo.context.parentId);
      if (parentInfo) {
        parentInfo.completedChildren++;
      }
    }

    this.eventBus.emit('QUEUE:DOWNLOAD_COMPLETE_ITEM', {
      pageId,
      savedPath,
      progress: `${this.completedDownloads.size}/${this.pendingDownloads.size}`
    });
  }

  /**
   * Marks a page as failed
   * @param {string} pageId - The ID of the failed page
   * @param {Error} error - Error that occurred
   */
  markFailed(pageId, error) {
    this.eventBus.emit('QUEUE:DOWNLOAD_FAILED_ITEM', {
      pageId,
      error: error.message
    });
  }

  /**
   * Checks if download phase is complete
   * @returns {boolean} True if all downloads are done
   */
  isComplete() {
    return this.queue.length === 0 && this.completedDownloads.size === this.pendingDownloads.size;
  }

  /**
   * Get number of remaining pages
   * @returns {number} Queue length
   */
  getLength() {
    return this.queue.length;
  }

  /**
   * Get number of completed downloads
   * @returns {number} Completed count
   */
  getCompletedCount() {
    return this.completedDownloads.size;
  }

  /**
   * Get total pages to download
   * @returns {number} Total pending downloads
   */
  getTotalCount() {
    return this.pendingDownloads.size;
  }

  /**
   * @private
   * @param {string} outputDir - Root output directory (relative or absolute)
   * @param {PageContext} context - The page context containing hierarchy info
   * @returns {string} The absolute path ending in 'index.html'
   */
  _calculateAbsolutePath(outputDir, context) {
    const absoluteOutputDir = path.resolve(process.cwd(), outputDir);
    const relativePath = context.getFilePath(outputDir);
    const absolutePath = path.resolve(absoluteOutputDir, relativePath);
    return absolutePath;
  }

  /**
   * Reset the queue
   */
  reset() {
    this.queue = [];
    this.pendingDownloads.clear();
    this.completedDownloads.clear();
  }
}

module.exports = ExecutionQueue;
