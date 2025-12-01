/**
 * @fileoverview Download queue manager with leaf-first ordering
 * @module orchestration/queues/ExecutionQueue
 * @description Manages download queue with dependency constraints using leaf-first ordering.
 * 
 * **Ordering Strategy:**
 * **Leaf-First (Depth Sort)**: Orders by depth descending, processing deepest pages first.
 * This naturally processes leaf nodes before their parents, allowing hidden file
 * deduplication across the tree.
 */

const path = require('path');
const fs = require('fs');
const SystemEventBus = require('../../core/SystemEventBus');
const Logger = require('../../core/Logger');

/**
 * @class ExecutionQueue
 * @classdesc Manages download queue with leaf-first ordering to prevent worker deadlock.
 * 
 * **Deadlock Prevention Strategy:**
 * By processing leaf pages first (those with no children), hidden files are discovered
 * and registered globally before their parent pages are processed. This enables:
 * 1. Natural deduplication of hidden files across the page tree
 * 2. Shorter worker blocking times (leaf pages typically have fewer hidden elements)
 * 3. Better worker pool utilization through progressive task completion
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
   * Build queue with leaf-first (depth descending) ordering.
   * 
   * @description Sorts contexts by depth descending so that deepest pages (leaves)
   * are processed first. For pages at the same depth, those with fewer children
   * are prioritized to complete faster.
   * 
   * **Algorithm:** O(n log n) sort by depth
   * 
   * @param {Array<PageContext>} contexts - Discovered pages to download
   * @param {PageGraph} [pageGraph=null] - Optional page graph for edge metadata (unused)
   */
  build(contexts, pageGraph = null) {
    this.queue = [];
    this.pendingDownloads.clear();
    this.completedDownloads.clear();
    
    // Sort by depth descending (deepest/leaves first)
    const sortedContexts = [...contexts].sort((a, b) => {
      // Primary: depth descending (leaves first)
      if (b.depth !== a.depth) {
        return b.depth - a.depth;
      }
      // Secondary: pages with fewer children first (faster completion)
      return (a.children?.length || 0) - (b.children?.length || 0);
    });
    
    for (const context of sortedContexts) {
      this.queue.push(context);
      this.pendingDownloads.set(context.id, {
        context: context,
        childrenCount: context.children?.length || 0,
        completedChildren: 0
      });
    }
    
    this.eventBus.emit('QUEUE:DOWNLOAD_READY', {
      count: this.queue.length,
      strategy: 'leaf-first'
    });
    
    this.logger.info('ExecutionQueue', `Built queue with leaf-first ordering: ${this.queue.length} page(s)`);
  }

  /**
   * Retrieves the next ready download task
   * @description Returns pages from the queue in the pre-computed order
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
