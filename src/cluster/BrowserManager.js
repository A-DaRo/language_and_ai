/**
 * @fileoverview Worker Pool State Manager
 * @module cluster/BrowserManager
 * @description Manages the lifecycle and allocation of Worker processes.
 * Maintains idle/busy state and provides task execution interface.
 * 
 * **CRITICAL CONTEXT**: Runs in MASTER process. Coordinates worker allocation.
 */

const SystemEventBus = require('../core/SystemEventBus');
const { WorkerState } = require('./WorkerProxy');
const Logger = require('../core/Logger');

/**
 * @class BrowserManager
 * @classdesc Manages a pool of Worker processes, handling allocation and lifecycle
 */
class BrowserManager {
  constructor() {
    this.workers = new Map(); // workerId -> WorkerProxy
    this.idleWorkers = []; // Array of idle worker IDs (LIFO stack)
    this.busyWorkers = new Map(); // workerId -> taskInfo
    this.eventBus = SystemEventBus.getInstance();
    this.logger = Logger.getInstance();
    this.cachedTitleRegistry = {}; // Cache for respawned workers
    
    this._setupEventListeners();
  }
  
  /**
   * Setup event listeners for worker state changes
   * @private
   */
  _setupEventListeners() {
    // Worker becomes ready (initial or after task)
    this.eventBus.on('WORKER:READY', ({ workerId }) => {
      if (!this.idleWorkers.includes(workerId)) {
        this.idleWorkers.push(workerId);
      }
    });
    
    // Worker is idle (after task completion or failure)
    this.eventBus.on('WORKER:IDLE', ({ workerId }) => {
      this.busyWorkers.delete(workerId);
      if (!this.idleWorkers.includes(workerId)) {
        this.idleWorkers.push(workerId);
      }
    });
    
    // Worker started a task
    this.eventBus.on('TASK:STARTED', ({ workerId, taskId, taskType, metadata }) => {
      const index = this.idleWorkers.indexOf(workerId);
      if (index > -1) {
        this.idleWorkers.splice(index, 1);
      }
      this.busyWorkers.set(workerId, { taskId, taskType });
      
      // Emit WORKER:BUSY for UI dashboard with human-readable description
      const description = this._getTaskDescription(taskType, metadata);
      this.eventBus.emit('WORKER:BUSY', {
        workerId,
        task: { description }
      });
    });
    
    // Worker crashed
    this.eventBus.on('WORKER:CRASHED', ({ workerId }) => {
      this._handleWorkerCrash(workerId);
    });
  }
  
  /**
   * Register workers with the manager
   * @param {Array<WorkerProxy>} workerProxies - Array of worker proxies
   * @returns {void}
   */
  registerWorkers(workerProxies) {
    for (const proxy of workerProxies) {
      this.workers.set(proxy.workerId, proxy);
      this.logger.debug('BrowserManager', `Registered worker: ${proxy.workerId}`);
    }
    
    this.logger.info('BrowserManager', `Registered ${this.workers.size} worker(s)`);
  }
  
  /**
   * Initialize all workers with titleRegistry (sent once at startup, then after discovery)
   * @async
   * @param {Object} titleRegistry - ID-to-title map
   * @returns {Promise<void>}
   */
  async initializeWorkers(titleRegistry) {
    this.logger.debug('BrowserManager', `Initializing ${this.workers.size} worker(s) with title registry`);
    
    // Cache the title registry for respawned workers
    this.cachedTitleRegistry = titleRegistry || {};
    
    const initPromises = [];
    for (const worker of this.workers.values()) {
      initPromises.push(worker.sendInitialization(titleRegistry));
    }
    
    await Promise.all(initPromises);
    this.logger.info('BrowserManager', `All workers initialized with ${Object.keys(this.cachedTitleRegistry).length} title(s)`);
  }
  
  /**
   * Execute a task on an available worker
   * @async
   * @param {string} messageType - Message type (DISCOVER or DOWNLOAD)
   * @param {Object} payload - Task payload
   * @returns {Promise<string>} Worker ID that received the task
   * @throws {Error} If no workers are available
   * @example
   * const workerId = await browserManager.execute(MESSAGE_TYPES.DISCOVER, {
   *   url: 'https://notion.so/page',
   *   pageId: 'abc123',
   *   depth: 1
   * });
   */
  async execute(messageType, payload) {
    const workerId = await this._allocateWorker();
    const worker = this.workers.get(workerId);
    
    if (!worker) {
      throw new Error(`Worker ${workerId} not found in registry`);
    }
    
    await worker.sendCommand(messageType, payload, payload.metadata || {});
    return workerId;
  }
  
  /**
   * Generate human-readable task description for UI
   * @private
   * @param {string} taskType - Task type (IPC_DISCOVER or IPC_DOWNLOAD)
   * @param {Object} metadata - Task metadata (pageTitle, etc.)
   * @returns {string} Task description
   */
  _getTaskDescription(taskType, metadata = {}) {
    const pageTitle = metadata.pageTitle || 'page';
    
    if (taskType === 'IPC_DISCOVER') {
      const id = metadata.pageId || 'unknown';
      return `Discovering [${id.substring(0, 8)}...]`;
    } else if (taskType === 'IPC_DOWNLOAD') {
      return `Downloading '${pageTitle}'...`;
    }
    
    return `Processing ${taskType}...`;
  }
  
  /**
   * Allocate an idle worker for a task
   * @private
   * @async
   * @returns {Promise<string>} Worker ID
   * @throws {Error} If no workers are available
   */
  async _allocateWorker() {
    // Wait for an idle worker (with timeout)
    const timeout = 60000; // 60 seconds
    const startTime = Date.now();
    
    while (this.idleWorkers.length === 0) {
      if (Date.now() - startTime > timeout) {
        throw new Error('Timeout waiting for available worker');
      }
      
      // Wait 100ms before checking again
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Pop worker from idle stack (LIFO - reuse most recently freed worker)
    const workerId = this.idleWorkers.pop();
    
    return workerId;
  }
  
  /**
   * Handle worker crash - remove from pool and log
   * @private
   * @param {string} workerId - ID of crashed worker
   */
  _handleWorkerCrash(workerId) {
    this.logger.error('BrowserManager', `Worker ${workerId} crashed`);
    
    // Remove from idle workers
    const idleIndex = this.idleWorkers.indexOf(workerId);
    if (idleIndex > -1) {
      this.idleWorkers.splice(idleIndex, 1);
    }
    
    // Remove from busy workers
    this.busyWorkers.delete(workerId);
    
    // NOTE: For production, implement worker respawn logic here:
    // const newWorker = await BrowserInitializer.spawnWorker(workerId);
    // await newWorker.sendInitialization(this.cachedTitleRegistry);
    // this.workers.set(workerId, newWorker);
    
    // For now, just log the crash
    this.logger.warn('BrowserManager', `Worker pool now has ${this.getAvailableCount()} available workers`);
  }
  
  /**
   * Broadcast cookies to all workers
   * @async
   * @param {Array<Object>} cookies - Cookie objects
   * @returns {Promise<void>}
   */
  async broadcastCookies(cookies) {
    this.logger.debug('BrowserManager', `Broadcasting cookies to ${this.workers.size} worker(s)`);
    
    const promises = [];
    for (const worker of this.workers.values()) {
      promises.push(worker.broadcastCookies(cookies));
    }
    
    await Promise.all(promises);
  }
  
  /**
   * Get count of available (idle) workers
   * @returns {number} Number of idle workers
   */
  getAvailableCount() {
    return this.idleWorkers.length;
  }
  
  /**
   * Get count of allocated (busy) workers
   * @returns {number} Number of busy workers
   */
  getAllocatedCount() {
    return this.busyWorkers.size;
  }
  
  /**
   * Get total worker count
   * @returns {number} Total number of workers
   */
  getTotalCount() {
    return this.workers.size;
  }

  /**
   * Get all worker IDs
   * @returns {Array<string>} Array of worker IDs
   */
  getAllWorkerIds() {
    return Array.from(this.workers.keys());
  }
  
  /**
   * Get pool statistics
   * @returns {Object} Pool statistics
   */
  getStatistics() {
    return {
      total: this.getTotalCount(),
      idle: this.getAvailableCount(),
      busy: this.getAllocatedCount()
    };
  }
  
  /**
   * Shutdown all workers gracefully
   * @async
   * @returns {Promise<void>}
   */
  async shutdown() {
    this.logger.info('BrowserManager', `Shutting down ${this.workers.size} worker(s)...`);
    
    const promises = [];
    for (const worker of this.workers.values()) {
      promises.push(worker.terminate());
    }
    
    await Promise.all(promises);
    
    this.workers.clear();
    this.idleWorkers = [];
    this.busyWorkers.clear();
    
    this.logger.info('BrowserManager', 'All workers terminated');
  }
}

module.exports = BrowserManager;
