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
    this.pendingTasks = 0;
    this.maxDepth = 0;
  }

  /**
   * Adds a page to the discovery queue
   * @param {PageContext} context - The page context
   * @param {boolean} [isRoot=false] - Whether this is the root page
   * @returns {boolean} True if enqueued, false if already visited
   */
  enqueue(context, isRoot = false) {
    const rawIdMatch = context.url.match(/29[a-f0-9]{30}/i);
    const rawPageId = rawIdMatch ? rawIdMatch[0] : null;

    if (!rawPageId || this.visitedUrls.has(rawPageId)) {
      return false;
    }

    this.visitedUrls.add(rawPageId);

    if (context.depth > this.maxDepth) {
      this.maxDepth = context.depth;
    }

    this.queue.push({ pageContext: context, isFirstPage: isRoot });

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
    this.pendingTasks++;
    return task;
  }

  /**
   * Marks a discovery task as complete
   * @param {string} pageId - The page ID
   */
  markComplete(pageId) {
    this.pendingTasks--;
  }

  /**
   * Marks a discovery task as failed
   * @param {string} pageId - The page ID
   */
  markFailed(pageId) {
    this.pendingTasks--;
  }

  /**
   * Checks if discovery is finished
   * @returns {boolean} True if queues are empty and no pending tasks
   */
  isComplete() {
    return this.queue.length === 0 && this.pendingTasks === 0;
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
    return this.pendingTasks;
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
    this.pendingTasks = 0;
    this.maxDepth = 0;
  }
}

module.exports = DiscoveryQueue;
