/**
 * @fileoverview BFS discovery queue manager
 * @module orchestration/queues/DiscoveryQueue
 * @description Manages the BFS frontier for the discovery phase.
 */

const SystemEventBus = require('../../core/SystemEventBus');
const Logger = require('../../core/Logger');

/**
 * @class DiscoveryQueue
 * @classdesc Manages BFS queue and visited URL tracking for discovery phase.
 */
class DiscoveryQueue {
  constructor() {
    this.eventBus = SystemEventBus.getInstance();
    this.logger = Logger.getInstance();

    this.queue = [];
    this.visitedUrls = new Set(); // Public for GlobalQueueManager access
    this.pendingTaskIds = new Set();
    this.maxDepth = 0;
  }

  /**
   * Adds a page to the discovery queue
   * @param {PageContext} context - The page context
   * @param {boolean} isRoot - Whether this is the root page
   * @returns {boolean} True if enqueued, false if skipped
   */
  enqueue(context, isRoot = false) {
    if (!context || !context.url) {
      this.logger.warn('DiscoveryQueue', 'Attempted to enqueue invalid context (missing context or URL)');
      return false;
    }

    const rawPageId = this._extractPageId(context.url);

    if (!rawPageId) {
      this.logger.warn('DiscoveryQueue', `Failed to extract page ID from URL: ${context.url}`);
      return false;
    }

    if (this.visitedUrls.has(rawPageId)) {
      this.logger.debug('DiscoveryQueue', `Skipping already-visited page: ${rawPageId.substring(0, 8)}...`);
      return false;
    }

    this.visitedUrls.add(rawPageId);

    if (context.depth > this.maxDepth) {
      this.maxDepth = context.depth;
    }

    this.queue.push({ pageContext: context, isFirstPage: isRoot });

    if (this.queue.length === 1) {
      this.eventBus.emit('DISCOVERY:QUEUE_READY', {
        queueLength: this.queue.length
      });
    }

    this.eventBus.emit('QUEUE:DISCOVERY_ENQUEUED', {
      url: context.url,
      depth: context.depth
    });

    return true;
  }

  /**
   * Retrieves the next available discovery task
   * @returns {Object|null} The task payload or null if empty
   */
  next() {
    if (this.queue.length === 0) {
      return null;
    }

    const task = this.queue.shift();
    this.pendingTaskIds.add(task.pageContext.id);

    if (this.queue.length === 0 && this.pendingTaskIds.size > 0) {
      this.eventBus.emit('DISCOVERY:QUEUE_EMPTY', {
        queueLength: 0,
        pendingCount: this.pendingTaskIds.size
      });
    }

    return task;
  }

  /**
   * @method markComplete
   * @description Marks a discovery task as complete and publishes the updated queue state.
   *              Safe to call multiple times for the same page ID (idempotent).
   * @param {string} pageId - The page ID to finalize
   * @emits DISCOVERY:TASK_COMPLETED - Emitted every time a tracked task finishes
   * @emits DISCOVERY:ALL_IDLE - Emitted when this completion brings the system to quiescent state
   * @returns {boolean} True if the task was tracked and finalized
   * @example
   * queue.markComplete('page-abc123');
   */
  markComplete(pageId) {
    return this._finalizeTask(pageId, true);
  }

  /**
   * @method markFailed
   * @description Marks a discovery task as failed and emits the failure event for monitoring.
   * @param {string} pageId - The page ID that failed
   * @emits DISCOVERY:TASK_COMPLETED - Emitted with success=false
   * @emits DISCOVERY:ALL_IDLE - Emitted when pending and queue counts reach zero
   * @returns {boolean} True if the task was tracked and finalized
   * @example
   * queue.markFailed('page-bad123');
   */
  markFailed(pageId) {
    return this._finalizeTask(pageId, false);
  }

  /**
   * Checks if discovery is finished
   * @returns {boolean} True if queues are empty and no pending tasks
   */
  isComplete() {
    return this.queue.length === 0 && this.pendingTaskIds.size === 0;
  }

  /**
   * Get current queue length
   * @returns {number} Number of items in queue
   */
  getLength() {
    return this.queue.length;
  }

  /**
   * Get maximum depth discovered
   * @returns {number} Maximum depth
   */
  getMaxDepth() {
    return this.maxDepth;
  }

  /**
   * Get pending task count
   * @returns {number} Number of pending tasks
   */
  getPendingCount() {
    return this.pendingTaskIds.size;
  }

  /**
   * Get count of visited URLs
   * @returns {number} Number of visited URLs
   */
  getVisitedCount() {
    return this.visitedUrls.size;
  }

  /**
   * Reset the queue
   */
  reset() {
    this.queue = [];
    this.visitedUrls.clear();
    this.pendingTaskIds.clear();
    this.maxDepth = 0;
  }

  /**
   * Normalize Notion page id from URL
   * @private
   * @param {string} url - Notion URL
   * @returns {string|null} Normalized id
   */
  _extractPageId(url) {
    const match = url && url.match(/([a-f0-9]{32})$/i);
    return match ? match[1] : null;
  }

  /**
   * Finalize a pending task (success or failure)
   * @private
   * @param {string} pageId - Completed page ID
   * @param {boolean} success - Whether task succeeded
   * @returns {boolean} True if task was tracked, false otherwise
   */
  /**
   * @method _finalizeTask
   * @description Handles shared completion logic for success/failure paths.
   * @private
   * @param {string} pageId - The page ID that finished
   * @param {boolean} success - Whether the task succeeded
   * @returns {boolean} True if the task was tracked and finalized
   */
  _finalizeTask(pageId, success) {
    if (!pageId) {
      return false;
    }

    if (!this.pendingTaskIds.has(pageId)) {
      this.logger.warn('DiscoveryQueue', `Attempted to finalize unknown task: ${pageId}`);
      return false;
    }

    this.pendingTaskIds.delete(pageId);

    this.eventBus.emit('DISCOVERY:TASK_COMPLETED', {
      pageId,
      success,
      pendingCount: this.pendingTaskIds.size,
      queueLength: this.queue.length
    });

    if (this.isComplete()) {
      this.eventBus.emit('DISCOVERY:ALL_IDLE', {
        queueLength: 0,
        pendingCount: 0
      });
    }

    return true;
  }
}

module.exports = DiscoveryQueue;
