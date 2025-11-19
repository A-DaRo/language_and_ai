/**
 * @fileoverview Central Event Bus for Master Process Coordination
 * @module core/SystemEventBus
 * @description Provides a singleton EventEmitter for coordinating communication between
 * Master-side components (Orchestrator, BrowserManager, GlobalQueueManager). This is the
 * nervous system of the Master process, enabling loose coupling between subsystems.
 * 
 * **CRITICAL**: This runs ONLY in the Master process. Workers use IPC, not this bus.
 */

const EventEmitter = require('events');

/**
 * @class SystemEventBus
 * @extends EventEmitter
 * @classdesc Central event bus for Master process coordination.
 * Implements the Singleton pattern to ensure only one event bus exists.
 * 
 * All events follow a namespace convention:
 * - SYSTEM:* - System lifecycle events
 * - WORKER:* - Worker process events
 * - TASK:* - Task execution events
 * - QUEUE:* - Queue state events
 * - CMD:* - Command events
 * 
 * @example
 * const bus = SystemEventBus.getInstance();
 * bus.on('WORKER:READY', (workerId) => {
 *   console.log(`Worker ${workerId} is ready`);
 * });
 * bus.emit('WORKER:READY', 'worker-1');
 */
class SystemEventBus extends EventEmitter {
  /**
   * @private
   * @static
   * @type {SystemEventBus}
   * @description Singleton instance
   */
  static _instance = null;

  /**
   * @private
   * @description Private constructor to enforce singleton pattern
   */
  constructor() {
    if (SystemEventBus._instance) {
      throw new Error('SystemEventBus is a singleton. Use getInstance() instead.');
    }
    super();
    this.setMaxListeners(50); // Increase limit for complex orchestration
  }

  /**
   * Get the singleton instance of the SystemEventBus
   * @static
   * @returns {SystemEventBus} The singleton instance
   * @example
   * const bus = SystemEventBus.getInstance();
   */
  static getInstance() {
    if (!SystemEventBus._instance) {
      SystemEventBus._instance = new SystemEventBus();
    }
    return SystemEventBus._instance;
  }

  /**
   * Reset the singleton instance (for testing only)
   * @static
   * @private
   */
  static _reset() {
    if (SystemEventBus._instance) {
      SystemEventBus._instance.removeAllListeners();
      SystemEventBus._instance = null;
    }
  }

  /**
   * Emit an event with structured logging
   * @param {string} eventName - Name of the event
   * @param {...*} args - Event arguments
   * @returns {boolean} True if the event had listeners
   * @example
   * bus.emit('TASK:COMPLETE', { taskId: '123', result: {...} });
   */
  emit(eventName, ...args) {
    const hasListeners = this.listenerCount(eventName) > 0;
    if (!hasListeners && !eventName.startsWith('SYSTEM:')) {
      console.warn(`[SystemEventBus] No listeners for event: ${eventName}`);
    }
    return super.emit(eventName, ...args);
  }
}

// ============================================================================
// Event Type Definitions
// ============================================================================

/**
 * @event SystemEventBus#SYSTEM:INIT
 * @type {Object}
 * @property {Object} config - System configuration
 * @description System initialization event. Triggered by main.js to bootstrap all subsystems.
 */

/**
 * @event SystemEventBus#SYSTEM:READY
 * @type {Object}
 * @property {number} workerCount - Number of workers initialized
 * @description Emitted when all workers are spawned and ready. Signals that task execution can begin.
 */

/**
 * @event SystemEventBus#SYSTEM:SHUTDOWN
 * @description Graceful shutdown signal. All components should clean up and terminate.
 */

/**
 * @event SystemEventBus#WORKER:SPAWNED
 * @type {Object}
 * @property {string} workerId - Unique worker identifier
 * @property {number} pid - Process ID
 * @description Emitted when a new worker process is spawned.
 */

/**
 * @event SystemEventBus#WORKER:READY
 * @type {Object}
 * @property {string} workerId - Unique worker identifier
 * @description Emitted when a worker sends the READY signal after initialization.
 */

/**
 * @event SystemEventBus#WORKER:IDLE
 * @type {Object}
 * @property {string} workerId - Unique worker identifier
 * @description Emitted when a worker becomes idle and available for new tasks.
 */

/**
 * @event SystemEventBus#WORKER:BUSY
 * @type {Object}
 * @property {string} workerId - Unique worker identifier
 * @property {string} taskType - Type of task (DISCOVER or DOWNLOAD)
 * @description Emitted when a worker starts executing a task.
 */

/**
 * @event SystemEventBus#WORKER:CRASHED
 * @type {Object}
 * @property {string} workerId - Unique worker identifier
 * @property {number} exitCode - Process exit code
 * @property {string} [signal] - Signal that caused the crash
 * @description Emitted when a worker process crashes unexpectedly.
 */

/**
 * @event SystemEventBus#TASK:QUEUED
 * @type {Object}
 * @property {string} taskId - Unique task identifier
 * @property {string} taskType - Type of task (DISCOVER or DOWNLOAD)
 * @property {string} url - Target URL
 * @description Emitted when a new task is added to the queue.
 */

/**
 * @event SystemEventBus#TASK:STARTED
 * @type {Object}
 * @property {string} taskId - Unique task identifier
 * @property {string} workerId - Worker assigned to the task
 * @property {string} taskType - Type of task
 * @description Emitted when a task starts execution on a worker.
 */

/**
 * @event SystemEventBus#TASK:COMPLETE
 * @type {Object}
 * @property {string} taskId - Unique task identifier
 * @property {string} workerId - Worker that completed the task
 * @property {string} taskType - Type of task
 * @property {Object} result - Task result data
 * @description Emitted when a task completes successfully.
 */

/**
 * @event SystemEventBus#TASK:FAILED
 * @type {Object}
 * @property {string} taskId - Unique task identifier
 * @property {string} workerId - Worker that failed the task
 * @property {string} taskType - Type of task
 * @property {Error} error - Error that caused the failure
 * @property {number} retryCount - Number of retry attempts
 * @description Emitted when a task fails.
 */

/**
 * @event SystemEventBus#QUEUE:DISCOVERY_COMPLETE
 * @description Emitted when the discovery queue is empty and all discovery tasks are complete.
 */

/**
 * @event SystemEventBus#QUEUE:DOWNLOAD_COMPLETE
 * @description Emitted when the download queue is empty and all download tasks are complete.
 */

/**
 * @event SystemEventBus#CMD:SHUTDOWN
 * @type {Object}
 * @property {string} reason - Reason for shutdown
 * @description Graceful shutdown command. All workers should save state and exit.
 */

module.exports = SystemEventBus;
